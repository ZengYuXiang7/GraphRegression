# coding : utf-8
# Author : Yuxiang Zeng
import os
import sys
import shutil
import torch
import random
import time
import argparse
import collections
import numpy as np
import pickle
from main import run_exp

torch.set_default_dtype(torch.float32)


# 在这里写下任何实验模板所产生的metrics
def RunOnce(runid, config):
    remove_pycache_dirs()
    # 这里写单次实验的代码
    # results = {"accuracy": random.random(), "mape": random.random()}
    results = run_exp(runid=runid, config=config)
    return results


def build_parser() -> argparse.ArgumentParser:
    """只负责声明已知参数"""
    parser = argparse.ArgumentParser()
    default_device = "mps" if sys.platform == "darwin" else "cuda"
    parser.add_argument("--rounds", default=2, type=int)
    parser.add_argument("--dataset", default="nasbench101", type=str)
    parser.add_argument("--model", default="model49", type=str)
    parser.add_argument("--device", default=default_device, type=str)
    parser.add_argument("--debug", default=0, type=int)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def remove_pycache_dirs(project_root=None):
    root = project_root or os.path.dirname(os.path.abspath(__file__))
    removed = []
    for current_root, dirnames, _ in os.walk(root, topdown=True):
        pycache_paths = [
            os.path.join(current_root, name)
            for name in dirnames
            if name == "__pycache__"
        ]
        for path in pycache_paths:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
        if pycache_paths:
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    return removed


def inject_unknown_args(
    args: argparse.Namespace, unknown: list[str]
) -> argparse.Namespace:
    """只负责把 parse_known_args() 的 unknown 写进 args（支持 bool/int/float/str 自动推断）"""
    if len(unknown) % 2 != 0:
        raise ValueError(f"Unknown args must be key-value pairs, got: {unknown}")

    it = iter(unknown)
    for k in it:
        if not k.startswith("--"):
            raise ValueError(f"Unknown arg key must start with '--', got: {k}")
        key = k[2:]
        val = next(it)

        low = val.lower()
        if low in ("true", "false"):
            parsed = low == "true"
        else:
            try:
                parsed = int(val)
            except ValueError:
                try:
                    parsed = float(val)
                except ValueError:
                    parsed = val

        setattr(args, key, parsed)

    return args


def get_config():
    args, unknown_args = build_parser()
    args = inject_unknown_args(args, unknown_args)
    return args


def get_experiment_name(config):
    exclude = {"rounds", "track"}  # 不写进文件名的字段

    # 收集字段
    detail_fields = {}
    for k, v in vars(config).items():
        if k in exclude:
            continue
        detail_fields[k] = v

    # 文件名清洗：避免空格/特殊字符
    def safe(x):
        s = str(x)
        return "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in s)

    # 1) 固定前缀顺序：dataset、model（你要哪个在前就调换这个列表顺序）
    front_keys = ["dataset", "model"]
    front_items = [(k, detail_fields.pop(k)) for k in front_keys if k in detail_fields]

    # 2) 其余字段按 key 字典序排序
    rest_items = sorted(detail_fields.items(), key=lambda kv: str(kv[0]))

    # 3) 合并
    items = front_items + rest_items

    exper_detail = ", ".join(f"{k} : {v}" for k, v in items)
    log_filename = "|".join(f"{k.replace('_','')}__{safe(v)}" for k, v in items)

    return log_filename, exper_detail


# 日志记录函数
def log(message, logfile="run.log"):
    os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)

    msg_str = str(message)

    # 判断：是否只包含空白/换行
    is_pure_newline = msg_str.strip() == ""

    if is_pure_newline:
        # 只输出空行：不加时间戳
        print(msg_str, end="")
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(msg_str + "\n\n")
        return True

    # 正常日志：加时间戳
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {msg_str}"

    print(msg)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

    return True


# 保存结果到pickle文件
def save_result(metrics, log_filename, config):
    os.makedirs("./results/metrics/", exist_ok=True)
    config_copy = {k: v for k, v in config.__dict__.items() if k != "log"}
    result = {
        "config": config_copy,
        "dataset": config.dataset,
        "model": config.model,
        **{k: metrics[k] for k in metrics},
        **{f"{k}_mean": np.mean(metrics[k]) for k in metrics},
        **{f"{k}_std": np.std(metrics[k]) for k in metrics},
    }
    with open(f"./results/metrics/{log_filename}.pkl", "wb") as f:
        pickle.dump(result, f)


# 时间种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True


def RunExperiments(config):
    if config.debug:
        config.rounds = 1
        config.epochs = 1
    log("")
    log_filename, exper_detail = get_experiment_name(config)
    metrics = collections.defaultdict(list)

    for runid in range(config.rounds):
        set_seed(runid)
        results = RunOnce(runid, config)
        for key in results:
            metrics[key].append(results[key])

    log("*" * 20 + "Experiment Results" + "*" * 20)
    log(f"log Filename: {log_filename}")
    log(f"Experiment Detail: {exper_detail}")
    log("-" * 60)
    for key in metrics:
        log(f"{key:10s}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}")

    save_result(metrics, log_filename, config)
    log("*" * 20 + "Experiment Success" + "*" * 20)
    return metrics


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
    return args, summary


if __name__ == "__main__":
    config = get_config()
    metrics = RunExperiments(config)
