import os
import time
import torch
import random
import numpy as np
from timm.utils import get_state_dict

from nnformer.utils import *
from config import parse_args
from nnformer.data_process import init_dataloader
from tqdm import *

def format_second(secs):
    return "{:0>2}:{:0>2}:{:0>2}".format(
        int(secs / 3600), int((secs % 3600) / 60), int(secs % 60)
    )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(config, logger):
    # Load Dataset
    train_loader, val_loader = init_dataloader(config, logger)
    n_batches = len(train_loader)
    # Init Model
    net, model_ema, criterion = init_layers(config, logger)

    # Optimizer
    optimizer, scheduler = init_optim(config, net, n_batches)
    # Auto Resume
    start_epoch_idx = auto_load_model(config, net, model_ema, optimizer, scheduler)

    # Init Value
    best_tau, best_mape, best_error = -99, 1e5, 0
    if config.model_ema and config.model_ema_eval:
        best_tau_ema, best_mape_ema, best_error_ema = -99, 1e5, 0
        
    # 4000
    for epoch_idx in trange(start_epoch_idx, config.epochs):
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

        if (epoch_idx + 1) % config.test_freq == 0:
            acc, err, tau = infer(val_loader, net, config.dataset, config.device)
            if tau > best_tau:
                best_mape, best_error, best_tau = acc, err, tau
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

            if config.model_ema and config.model_ema_eval:
                acc_ema, err_ema, tau_ema = infer(
                    val_loader, model_ema.ema, config.dataset, config.device
                )
                if tau_ema > best_tau_ema:
                    best_mape_ema, best_error_ema, best_tau_ema = (
                        acc_ema,
                        err_ema,
                        tau_ema,
                    )
                    save_check_point(
                        epoch_idx + 1,
                        batch_idx + 1,
                        config,
                        get_state_dict(model_ema),
                        None,
                        None,
                        False,
                        config.dataset + "_model_best_ema.pth.tar",
                    )

            if (epoch_idx + 1) % config.print_freq == 0:
                logger.info(
                    "Epoch[{}/{}] Lr:{:.7f} Loss:{:.7f} L_MSE:{:.7f} L_rank:{:.7f} L_con:{:.7f} KT:{:.5f} MAPE:{:.5f} "
                    "ErrBnd(0.01):{:.5f} Speed:{:.0f}/s Exa(h:m:s):{}".format(
                        epoch_idx,
                        config.epochs,
                        lr,
                        loss,
                        loss_dict["loss_mse"],
                        loss_dict["loss_rank"],
                        loss_dict["loss_consist"],
                        tau,
                        acc,
                        err,
                        speed,
                        exp_time,
                    )
                )

                logger.info(
                    "CheckPoint_TEST: KT {:.5f}, Best_KT {:.5f}, EMA_KT {:.5f}, Best_EMA_KT {:.5f} "
                    "MAPE {:.5f}, Best_MAPE {:.5f}, EMA_MAPE {:.5f}, Best_EMA_MAPE {:.5f}, "
                    "ErrBnd(0.01) {:.5f}, Best_ErrB {:.5f}, EMA_ErrBnd(0.01) {:.5f}, Best_EMA_ErrB {:.5f}, ".format(
                        tau,
                        best_tau,
                        tau_ema,
                        best_tau_ema,
                        acc,
                        best_mape,
                        acc_ema,
                        best_mape_ema,
                        err,
                        best_error,
                        err_ema,
                        best_error_ema,
                    )
                )

        if (epoch_idx + 1) % config.save_epoch_freq == 0:
            logger.info("Saving Model after %d-th Epoch." % (epoch_idx + 1))
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
    logger.info(
        f"Training Finished! Best MAPE: {best_mape:11.8f}, "
        f"Best MAPE on EMA: {best_mape_ema:11.8f}, "
        f"Best ErrBnd(0.01): {best_error:11.8f}; "
        f"Best ErrBond(0.05) on EMA: {best_error_ema:11.8f}"
    )
    logger.info(
        f"CheckPoint_TEST: Best_KT {best_tau:.5f}, Best_EMA_KT {best_tau_ema:.5f}"
    )

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


def eval_with_loader(config, logger, test_loader, *, pretrained_path=None, isTest=False):
    # 构建模型
    net = init_layers(config, logger)

    # 临时指定要加载的 ckpt（不污染外层）
    old = getattr(config, "pretrained_path", None)
    if pretrained_path is not None:
        config.pretrained_path = pretrained_path

    auto_load_model(config, net)

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

    # ✅ 正确的 Python f-string
    output_dir = f"output/{args.model}/nasbench101/{args.model}_{args.percent}"
    args.save_path = f"{output_dir}/"
    os.makedirs(args.save_path, exist_ok=True)

    logname = "train" if args.do_train else "test"
    logname = os.path.join(args.save_path, f"{logname}.log")
    logger = setup_logger(logname)

    if torch.cuda.is_available():
        print("Totally", torch.cuda.device_count(), "GPUs are available.")
        torch.cuda.set_device(args.device)
        print("Device:", args.device, "Name:", torch.cuda.get_device_name(args.device))
    else:
        args.n_workers = 1
        if args.device != 'mps':
            args.device = 'cpu'


    # 1) 训练（可选）
    if args.do_train:
        logger.info("Configs: %s" % (args))
        train(args, logger)
        args.do_train = False

    # ✅ 2) 只初始化一次数据
    args.batch_size = 2048
    test_loader = init_dataloader(args, logger)

    # 3) 跑两个 ckpt：best 和 best_ema
    best_ckpt = os.path.join(output_dir, "nasbench101_model_best.pth.tar")
    ema_ckpt  = os.path.join(output_dir, "nasbench101_model_best_ema.pth.tar")

    # base
    if os.path.isfile(best_ckpt):
        acc, err, tau = eval_with_loader(args, logger, test_loader, pretrained_path=best_ckpt, isTest=False)
    else:
        acc = err = tau = None
        logger.warning(f"[WARN] best ckpt not found: {best_ckpt}")

    # ema
    if os.path.isfile(ema_ckpt):
        acc_ema, err_ema, tau_ema = eval_with_loader(args, logger, test_loader, pretrained_path=ema_ckpt, isTest=False)
    else:
        acc_ema = err_ema = tau_ema = None
        logger.warning(f"[WARN] ema ckpt not found: {ema_ckpt}")

    # 日志
    if tau is not None:
        logger.info(f"[EVAL-best]     KT {tau:8.5f}, MAPE {acc:8.5f}, ErrBnd(0.01) {err:8.5f}")
    if tau_ema is not None:
        logger.info(f"[EVAL-best_ema] KT {tau_ema:8.5f}, MAPE {acc_ema:8.5f}, ErrBnd(0.01) {err_ema:8.5f}")

    return {
        "Tau": tau,
        "MAPE": acc,
        "ErrBnd": err,
        "Tau_EMA": tau_ema,
        "MAPE_EMA": acc_ema,
        "ErrBnd_EMA": err_ema,
    }


def merge_config_into_args(
        args, config, *, only_existing=False, skip_none=True, verbose=True
):
    """
    用 config 覆盖 args，并打印：
      - overwritten: args 原来有该字段且值发生变化
      - added: args 原来没有该字段，新添加
      - unchanged: args 原来有该字段，但值相同（可选打印）
      - skipped_none: config 中为 None 被跳过（可选打印）
      - skipped_missing: only_existing=True 且 args 没有该字段被跳过（可选打印）
    """
    print(f"*" * 80)

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

        # 下面这些你不想太吵可以注释掉
        if unchanged:
            print(
                f"[merge] unchanged ({len(unchanged)}): "
                + ", ".join(k for k, _ in unchanged)
            )
        else:
            print("[merge] unchanged (0)")

        if skipped_none:
            print(
                f"[merge] skipped_none ({len(skipped_none)}): "
                + ", ".join(skipped_none)
            )
        if skipped_missing:
            print(
                f"[merge] skipped_missing ({len(skipped_missing)}): "
                + ", ".join(skipped_missing)
            )

    summary = {
        "overwritten": overwritten,
        "added": added,
        "unchanged": unchanged,
        "skipped_none": skipped_none,
        "skipped_missing": skipped_missing,
    }
    print(f"*" * 80)
    return args


if __name__ == "__main__":
    # run_exp(args)
    print('Done!')