# coding : utf-8
# Author : yuxiang Zeng
from datetime import datetime
import time
import subprocess
import numpy as np
from datetime import datetime
import pickle
from Experiment import get_experiment_name

RUN_FILE = "./predictor/Experiment.py"

# 在这里写下超参数探索空间
hyper_dict = {
    "dataset": ["nnlqp"],
    "model": ["model22"],
    "mp_type": ["gcn", "sage", "gat"],
    # GNN 层数
    "gcn_layers": [1, 2],
    # 融合方式
    "fuse_method": ["sum", "weighted", "local_only", "global_only"],
    # pooling
    "pool": ["sum", "mean", "cls"],
    # 是否使用 FFN（你用 0/1）
    "use_ffn": [1, 0],
    # 归一化
    "norm_type": ["layernorm", "l2", "batchnorm", "none"],
}


# 这里是总执行实验顺序！！！！！！！！
def experiment_run():
    Our_model(hyper_dict)
    return True


def Our_model(hyper):
    # monitor_metric = NMAE KendallTau
    once_experiment(
        hyper,
        monitor_metric="mape",
        reverse=False,
        debug=0,
    )
    return True


######################################################################################################


def write_and_print(string):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("./run.log", "a") as f:
        print(string)
        f.write(f"[{timestamp}] {string}\n")
    return True


# 搜索最佳超参数然后取最佳
def add_parameter(command: str, params: dict) -> str:
    for param_name, param_value in params.items():
        command += f" --{param_name} {param_value}"
    return command


def once_experiment(
    hyper_dict,
    monitor_metric,
    reverse=False,
    debug=0,
    run_again=False,
):
    # 先进行超参数探索
    best_hyper = hyper_search(
        hyper_dict,
        monitor_metric=monitor_metric,
        reverse=reverse,
        debug=debug,
    )

    if run_again:
        # 再跑最佳参数实验
        commands = []
        command = f"python {RUN_FILE}"
        commands.append(command)

        commands = [add_parameter(command, best_hyper) for command in commands]

        # 执行所有命令
        for command in commands:
            run_command(command)
    return True


def hyper_search(
    hyper_dict,
    monitor_metric,
    reverse=False,
    debug=0,
):
    """
    入口函数：选择使用网格搜索还是逐步搜索
    """
    return sequential_hyper_search(hyper_dict, monitor_metric, reverse, debug)


def run_and_get_metric(cmd_str, chosen_hyper, monitor_metric, debug=False):
    """
    运行训练命令，并提取 metric
    """
    timestamp = time.strftime("|%Y-%m-%d %H:%M:%S| ")
    print(
        f"\033[1;38;2;151;200;129m{timestamp}\033[0m \033[1;38;2;100;149;237m{cmd_str}\033[0m"
    )
    log_filename = get_experiment_name_from_dict(chosen_hyper)
    print(log_filename)
    print(chosen_hyper)
    subprocess.run(cmd_str, shell=True)

    metric_file_address = (
        f"./results/metrics/" + get_experiment_name_from_dict(chosen_hyper)[0]
    )
    this_expr_metrics = pickle.load(open(metric_file_address + ".pkl", "rb"))

    # 选择最优 metric
    best_value = np.mean(this_expr_metrics[monitor_metric])
    return best_value


def get_experiment_name_from_dict(d: dict):
    exclude = {"rounds", "track"}

    detail_fields = {k: v for k, v in d.items() if k not in exclude}

    def safe(x):
        s = str(x)
        return "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in s)

    front_keys = ["dataset", "model"]
    front_items = [(k, detail_fields.pop(k)) for k in front_keys if k in detail_fields]
    rest_items = sorted(detail_fields.items(), key=lambda kv: str(kv[0]))
    items = front_items + rest_items

    exper_detail = ", ".join(f"{k} : {v}" for k, v in items)

    # 强烈建议别用 '|'
    log_filename = "|".join(f"{k.replace('_','')}__{safe(v)}" for k, v in items)

    return log_filename, exper_detail


def sequential_hyper_search(hyper_dict, monitor_metric, reverse, debug):
    """
    逐步搜索超参数，每次调整一个参数，并保持其他最优值
    - 修复：避免后续超参的第一个值重复执行
    - 修复：不再把未探索超参写进 best_hyper（避免副作用）
    - 新增：evaluated_cache，相同配置只跑一次
    """
    log_file = f"./run.log"
    best_hyper = {}

    # 缓存：同一组超参组合 -> metric
    evaluated_cache = {}

    def make_key(d: dict):
        return tuple(sorted(d.items()))

    def run_once_with_cache(chosen_dict: dict):
        key = make_key(chosen_dict)
        if key in evaluated_cache:
            return evaluated_cache[key]

        command = f"python {RUN_FILE} "
        for k, v in chosen_dict.items():
            command += f"--{k} {v} "

        if debug:
            command += "--debug 1 "

        current_metric = run_and_get_metric(command, chosen_dict, monitor_metric, debug)
        evaluated_cache[key] = current_metric
        return current_metric

    with open(log_file, "a") as f:
        for hyper_name, hyper_values in hyper_dict.items():
            if len(hyper_values) == 1:
                best_hyper[hyper_name] = hyper_values[0]
                continue

            print(f"{hyper_name} => {hyper_values}")

            # 先构造“固定参数”：已确定的 best_hyper + 其他未搜索参数的默认值（但不写入 best_hyper）
            fixed_params = dict(best_hyper)
            for other_name, other_values in hyper_dict.items():
                if other_name == hyper_name:
                    continue
                if other_name not in fixed_params:
                    fixed_params[other_name] = other_values[0]

            # baseline（默认取当前超参的第一个值）
            baseline_val = hyper_values[0]
            baseline_dict = dict(fixed_params)
            baseline_dict[hyper_name] = baseline_val

            # 先确保 baseline 有 metric（有缓存就直接用；没有就跑一次）
            baseline_metric = run_once_with_cache(baseline_dict)

            # 初始化本轮最优为 baseline（这样就可以“直接从第二个值开始试”，但仍保留 baseline 作为比较基准）
            local_best_metric = baseline_metric
            current_best_value = baseline_val
            write_and_print(
                f"{hyper_name}: {baseline_val}, Metric: {baseline_metric:5.4f} (baseline)"
            )

            # 直接从第二个值开始（避免重复）
            for value in hyper_values[1:]:
                chosen_dict = dict(fixed_params)
                chosen_dict[hyper_name] = value

                current_metric = run_once_with_cache(chosen_dict)

                if reverse:
                    if current_metric > local_best_metric:
                        local_best_metric = current_metric
                        current_best_value = value
                else:
                    if current_metric < local_best_metric:
                        local_best_metric = current_metric
                        current_best_value = value

                write_and_print(f"{hyper_name}: {value}, Metric: {current_metric:5.4f}")

            # 更新该超参的最优值
            best_hyper[hyper_name] = current_best_value

            write_and_print(
                f"==> Best {hyper_name}: {current_best_value}, local_best_metric: {local_best_metric:5.4f}\n"
            )
            write_and_print('*' * 180 + '\n')

        write_and_print(f"The Best Hyperparameters: {best_hyper}\n")

    return best_hyper


def run_command(command, log_file="./run.log", retry_count=0):
    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 如果是重试的命令，标记为 "Retrying"
    if retry_count > 0:
        retry_message = "Retrying"
    else:
        retry_message = "Running"

    # 将执行的命令和时间写入日志文件

    write_and_print(f"{retry_message} at {current_time}: {command}\n")

    # 直接执行命令，将输出和错误信息打印到终端
    process = subprocess.run(
        f"echo {command} &&" + command,
        shell=True,
    )

    if process.returncode != 0:
        with open(log_file, "a") as f:
            f.write(f"Command failed, retrying in 3 seconds: {command}\n")


def log_message(message):
    log_file = "run.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


if __name__ == "__main__":
    try:
        log_message("Experiment Start!!!")
        experiment_run()
    except KeyboardInterrupt as e:
        log_message("Experiment interrupted by user.")
    finally:
        log_message("All commands executed successfully.\n")
