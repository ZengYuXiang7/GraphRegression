import torch
from timm.utils import ModelEma
from timm.optim import create_optimizer_v2, optimizer_kwargs

from nnformer.optim.scheduler import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from nnformer.models.encoders import NNFormer
from nnformer.models.losses import NARLoss
from .utils import model_info

# from parallel import DataParallelModel, DataParallelCriterion
from mytools.registry import get_model


def init_layers(args, logger):
    # Model
    net = get_model(args)
    print(f"Model : {args.model}")
    if not args.do_train:
        return net

    loss = NARLoss(
        args.lambda_mse,
        args.lambda_rank,
        args.lambda_consistency,
    )
    # model_info(net, logger)
    net = net.to(args.device)
    loss = loss.to(args.device)

    return net, loss


#
def init_optim(args, net, nbatches, warm_step=0.1):
    # 2 4 6 35
    optimizer = create_optimizer_v2(net, **optimizer_kwargs(args))
    configured_warmup_steps = getattr(args, "warmup_steps", -1)
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_step * nbatches * args.epochs,
        num_training_steps=nbatches * args.epochs,
        num_cycles=1,
        min_ratio=args.min_ratio,
    )
    print(f"warm_step={warm_step}, nbatches={nbatches}, epochs={args.epochs}")
    print(f"warmup_steps = {warm_step * nbatches * args.epochs}")
    print(f"total_steps = {nbatches * args.epochs}")
    return optimizer, lr_scheduler


def auto_load_model(args, model, optimizer=None, scheduler=None):
    if args.do_train:
        if args.resume:
            if args.resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(
                    args.resume, map_location="cpu", weights_only=True
                )
            model.load_state_dict(checkpoint["state_dict"])
            print("Resume checkpoint %s" % args.resume)
            if args.finetuning:
                print("Start fine-tuning from 0-th epoch/iter!")
                return 0
            else:
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    scheduler.load_state_dict(checkpoint["scheduler"])
                print("With optim & ached!")
                if "epoch" in checkpoint.keys():
                    start_id = checkpoint["epoch"]
                elif "iter" in checkpoint.keys():
                    start_id = checkpoint["iter"]
        else:
            start_id = 0
        # print("Start training from %d-th epoch/iter!" % (start_id))
        return start_id

    if args.pretrained_path:
        checkpoint = torch.load(
            args.pretrained_path,
            map_location="{}".format(args.device),
            weights_only=False,
        )
        pretrained_dict = {
            key.replace("module.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(pretrained_dict)
        # print(
        # torch.load(
        # args.pretrained_path,
        # map_location="{}".format(args.device),
        # weights_only=False,
        # )["config"]
        # )
