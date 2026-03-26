import os
import shutil


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



def get_experiment_name(config):
    exclude = {"rounds", "track", "debug", "epochs"}  # 不写进文件名的字段

    # 收集字段
    source = config if isinstance(config, dict) else vars(config)
    detail_fields = {}
    for k, v in source.items():
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
    print(f"\033[1;31m{log_filename}\033[0m")
    return log_filename, exper_detail
