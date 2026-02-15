import os
import time
import torch
import random
import numpy as np
from timm.utils import get_state_dict

from losses import RankLossPack
from nnformer.utils import *
from config import parse_args
from nnformer.data_process import init_dataloader
from tqdm import *


def format_second(secs):
    return "{:0>2}:{:0>2}:{:0>2}".format(
        int(secs / 3600), int((secs % 3600) / 60), int(secs % 60)
    )


def save_tau_curve(tau_history, config, logger):
    if len(tau_history) == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.info(f"Skip saving tau curve because matplotlib is unavailable: {e}")
        return
    os.makedirs("./results/tau", exist_ok=True)
    ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
    model = getattr(config, "model", "model")
    percent = getattr(config, "percent", "na")
    filename = f"{ts}_{model}_{percent}.pdf"
    path = os.path.join("./results/tau", filename)
    epochs = [x[0] for x in tau_history]
    taus = [x[1] for x in tau_history]
    plt.figure()
    plt.plot(epochs, taus)
    plt.xlabel("epoch")
    plt.ylabel("tau")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()
    logger.info(f"Saved tau curve to {path}")


def normalize_save_path(path):
    if path is None:
        return path
    if path.startswith("./output/"):
        return "./results/" + path[len("./output/") :]
    if path.startswith("output/"):
        return "results/" + path[len("output/") :]
    return path


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return True


def train(config, logger):
    # Load Dataset
    train_loader, val_loader = init_dataloader(config, logger)
    n_batches = len(train_loader)
    # Init Model
    net, model_ema, criterion = init_layers(config, logger)

    config.save_path = normalize_save_path(config.save_path)
    os.makedirs(config.save_path, exist_ok=True)

    # Optimizer
    optimizer, scheduler = init_optim(config, net, n_batches)
    # Auto Resume
    start_epoch_idx = auto_load_model(config, net, model_ema, optimizer, scheduler)

    # Init Value
    best_tau, best_mape, best_error = -99, 1e5, 0
    best_epoch = -1

    rank_loss_function = RankLossPack(config)
    early_stop_counter = 0
    stop_training = False
    tau_history = []
    train_start_time = time.time()
    epoch_iter = (
        trange(start_epoch_idx, config.epochs)
        if config.tqdm
        else range(start_epoch_idx, config.epochs)
    )
    for epoch_idx in epoch_iter:
        metric = Metric()
        t0 = time.time()

        net.train()
        for batch_idx, batch_data in enumerate(train_loader):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            optimizer.zero_grad()
            if "nasbench" in config.dataset:
                if config.lambda_consistency > 0:
                    data_0, data_1 = batch_data
                    batch_data = {
                        key: torch.cat([data_0[key], data_1[key]], dim=0)
                        for key in data_0.keys()
                    }
                for k, v in batch_data.items():
                    batch_data[k] = v.to(config.device)
                gt = batch_data["val_acc_avg"]
                logits = net(batch_data, None)

            elif config.dataset == "nnlqp":
                codes, gt, sf = (
                    batch_data[0]["netcode"],
                    batch_data[0]["cost"],
                    batch_data[1],
                )
                codes, gt, sf = (
                    codes.to(config.device),
                    gt.to(config.device),
                    sf.to(config.device),
                )
                logits = net(None, None, codes, sf)

            loss_dict = criterion(logits, gt)
            loss = loss_dict["loss"]

            # rank_loss = rank_loss_function(logits, gt)
            # loss += rank_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            if model_ema is not None:
                model_ema.update(net)

            ps = logits.detach().cpu().numpy()[:, 0].tolist()
            gs = gt.detach().cpu().numpy()[:, 0].tolist()
            metric.update(ps, gs)
            acc, err, tau = metric.get()

        t1 = time.time()
        speed = n_batches * config.batch_size / (t1 - t0)
        exp_time = format_second((t1 - t0) * (config.epochs - epoch_idx - 1))

        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        display_tau = tau

        if (epoch_idx + 1) % config.test_freq == 0:
            acc, err, tau = infer(val_loader, net, config.dataset, config.device)
            display_tau = tau
            if tau > best_tau:
                best_mape, best_error, best_tau = acc, err, tau
                best_epoch = epoch_idx + 1
                early_stop_counter = 0
                save_check_point(
                    epoch_idx + 1,
                    batch_idx + 1,
                    config,
                    net.state_dict(),
                    None,
                    None,
                    False,
                    config.dataset + "_model_best.pth.tar",
                )
            elif config.patience > 0:
                early_stop_counter += 1
                if early_stop_counter >= config.patience:
                    logger.info(
                        "Early stopping at epoch {} | Best KT:{:.5f} MAPE:{:.5f} ErrB:{:.5f}".format(
                            epoch_idx + 1, best_tau, best_mape, best_error
                        )
                    )
                    stop_training = True

            if (epoch_idx + 1) % config.print_freq == 0:
                logger.info(
                    "Epoch[{}/{}] lr:{:.7f} loss:{:.7f} "
                    "KT:{:.5f} MAPE:{:.5f} ErrB:{:.5f} | Best KT:{:.5f} MAPE:{:.5f} ErrB:{:.5f} | "
                    "Speed:{:.0f}/s ETA:{}".format(
                        epoch_idx,
                        config.epochs,
                        lr,
                        loss,
                        tau,
                        acc,
                        err,
                        best_tau,
                        best_mape,
                        best_error,
                        speed,
                        exp_time,
                    )
                )

        if config.tqdm:
            patience_info = (
                f"{early_stop_counter}/{config.patience}"
                if config.patience > 0
                else "off"
            )
            epoch_iter.set_postfix(
                best=f"{best_tau:.5f}",
                patience=patience_info,
                KT=f"{display_tau:.5f}",
            )
        tau_history.append((epoch_idx + 1, float(display_tau)))

        if (epoch_idx + 1) % config.save_epoch_freq == 0:
            save_check_point(
                epoch_idx + 1,
                batch_idx + 1,
                config,
                net.state_dict(),
                optimizer,
                scheduler,
                False,
                config.dataset + "_checkpoint_Epoch" + str(epoch_idx + 1) + ".pth.tar",
            )
        save_check_point(
            epoch_idx + 1,
            batch_idx + 1,
            config,
            net.state_dict(),
            optimizer,
            scheduler,
            False,
            config.dataset + "_latest.pth.tar",
        )
        if stop_training:
            break
    save_tau_curve(tau_history, config, logger)
    train_total_time = time.time() - train_start_time
    logger.info(
        f"Training Finished | Best KT {best_tau:.5f} | Best MAPE {best_mape:11.8f} | Best ErrB {best_error:11.8f}"
    )
    return {
        "best_epoch": best_epoch,
        "train_time_sec": train_total_time,
    }


@torch.no_grad()
def infer(dataloader, net, dataset, device=None, isTest=False):
    metric = Metric()
    net.eval()
    for bid, batch_data in enumerate(dataloader):
        if "nasbench" in dataset:
            gt = batch_data["test_acc_avg"] if isTest else batch_data["val_acc_avg"]
            if device != None:
                for k, v in batch_data.items():
                    batch_data[k] = v.to(device)
            logits = net(batch_data, None)
        elif dataset == "nnlqp":
            codes, gt, sf = (
                batch_data[0]["netcode"],
                batch_data[0]["cost"],
                batch_data[1],
            )
            logits = (
                net(None, None, codes.to(device), sf.to(device))
                if device != None
                else net(None, None, codes, None)
            )
        pre = (
            torch.cat([r.to(gt.device) for r in logits], dim=0)
            if isinstance(logits, list)
            else logits
        )
        ps = pre.data.cpu().numpy()[:, 0].tolist()
        gs = gt.data.cpu().numpy()[:, 0].tolist()
        metric.update(ps, gs)
        acc, err, tau = metric.get()
    return acc, err, tau


def eval_with_loader(
    config, logger, test_loader, *, pretrained_path=None, isTest=False
):
    # 构建模型
    net = init_layers(config, logger)

    # 临时指定要加载的 ckpt（不污染外层）
    old = getattr(config, "pretrained_path", None)
    if pretrained_path is not None:
        config.pretrained_path = pretrained_path

    auto_load_model(config, net)

    net = net.to(config.device)

    # 还原
    if pretrained_path is not None:
        config.pretrained_path = old

    if torch.cuda.is_available():
        net = net.cuda(config.device)

    acc, err, tau = infer(
        test_loader, net, config.dataset, config.device, isTest=isTest
    )
    return acc, err, tau


def run_exp(runid, config):
    args = parse_args()
    args = merge_config_into_args(args, config)

    results_dir = f"results/{args.model}/nasbench101/{args.model}_{args.percent}"
    args.save_path = f"{results_dir}/"
    os.makedirs(args.save_path, exist_ok=True)

    logname = "train" if args.do_train else "test"
    logname = os.path.join(args.save_path, f"{logname}.log")
    logger = setup_logger(logname)

    if torch.cuda.is_available():
        logger.info(f"Totally {torch.cuda.device_count()} GPUs are available.")
    else:
        args.n_workers = 1
        if args.device != "mps":
            args.device = "cpu"

    # 1) 训练（可选）
    if args.do_train:
        logger.info("Configs: %s" % (args))
        train_stats = train(args, logger)
        args.do_train = False
    else:
        train_stats = {}

    # ✅ 2) 只初始化一次数据
    args.batch_size = 2048
    test_loader = init_dataloader(args, logger)

    # 3) 跑两个 ckpt：best 和 best_ema
    best_ckpt = os.path.join(results_dir, "nasbench101_model_best.pth.tar")

    # base
    if os.path.isfile(best_ckpt):
        acc, err, tau = eval_with_loader(
            args, logger, test_loader, pretrained_path=best_ckpt, isTest=False
        )
    else:
        acc = err = tau = None
        logger.warning(f"[WARN] best ckpt not found: {best_ckpt}")

    # 日志
    if tau is not None:
        logger.info(
            f"[EVAL-best]  KT {tau:8.5f}, MAPE {acc:8.5f}, ErrBnd(0.01) {err:8.5f}"
        )

    return {
        "Tau": tau,
        "MAPE": acc,
        "ErrBnd": err,
        "BestEpoch": train_stats.get("best_epoch"),
        "TrainTimeSec": train_stats.get("train_time_sec"),
    }


def merge_config_into_args(
    args, config, *, only_existing=False, skip_none=True, verbose=False
):
    """
    用 config 覆盖 args，并打印：
      - overwritten: args 原来有该字段且值发生变化
      - added: args 原来没有该字段，新添加
      - unchanged: args 原来有该字段，但值相同（可选打印）
      - skipped_none: config 中为 None 被跳过（可选打印）
      - skipped_missing: only_existing=True 且 args 没有该字段被跳过（可选打印）
    """

    def to_dict(x):
        if isinstance(x, dict):
            return x
        if hasattr(x, "__dict__"):
            return vars(x)
        raise TypeError(f"Unsupported type: {type(x)}")

    def has_key(obj, k):
        return (k in obj) if isinstance(obj, dict) else hasattr(obj, k)

    def get_val(obj, k, default=None):
        return (
            obj.get(k, default) if isinstance(obj, dict) else getattr(obj, k, default)
        )

    def set_val(obj, k, v):
        if isinstance(obj, dict):
            obj[k] = v
        else:
            setattr(obj, k, v)

    cdict = to_dict(config)

    overwritten = []  # (k, old, new)
    added = []  # (k, new)
    unchanged = []  # (k, val)
    skipped_none = []  # (k)
    skipped_missing = []  # (k)

    for k, v in cdict.items():
        if skip_none and v is None:
            skipped_none.append(k)
            continue

        exists = has_key(args, k)
        if only_existing and not exists:
            skipped_missing.append(k)
            continue

        if exists:
            old = get_val(args, k)
            if old != v:
                overwritten.append((k, old, v))
            else:
                unchanged.append((k, v))
        else:
            added.append((k, v))

        set_val(args, k, v)

    if verbose:
        if overwritten:
            print(f"[merge] overwritten ({len(overwritten)}):")
            for k, old, new in overwritten:
                print(f"  - {k}: {old} -> {new}")
        else:
            print("[merge] overwritten (0)")

        if added:
            print(f"[merge] added ({len(added)}):")
            for k, new in added:
                print(f"  + {k}: {new}")
        else:
            print("[merge] added (0)")

    summary = {
        "overwritten": overwritten,
        "added": added,
        "unchanged": unchanged,
        "skipped_none": skipped_none,
        "skipped_missing": skipped_missing,
    }
    return args


if __name__ == "__main__":
    # run_exp(args)
    print("Done!")
