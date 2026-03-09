import argparse
import os
import shlex
import signal
import subprocess


def run_nohup(script_path: str, workdir: str) -> None:
    command = (
        f"nohup bash {shlex.quote(script_path)} > train.log 2>&1 & echo $! > train.pid"
    )
    subprocess.Popen(["bash", "-lc", command], cwd=workdir)


def cli_select(workdir: str) -> str:
    candidates = [
        f for f in os.listdir(workdir) if os.path.isfile(os.path.join(workdir, f))
    ]
    sh_files = [f for f in candidates if f.endswith(".sh")]
    files = sh_files if sh_files else candidates

    if not files:
        raise FileNotFoundError("当前目录没有可选择的文件")

    print("选择要运行的脚本：")
    for idx, name in enumerate(files, start=1):
        print(f"{idx}. {name}")

    while True:
        choice = input("请输入序号: ").strip()
        if not choice.isdigit():
            print("请输入数字序号")
            continue
        index = int(choice)
        if 1 <= index <= len(files):
            return os.path.join(workdir, files[index - 1])
        print("序号超出范围")


def parse_ps_table() -> list[dict]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,comm=,args="],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            continue
        pid = int(parts[0])
        ppid = int(parts[1])
        comm = parts[2]
        args = parts[3] if len(parts) > 3 else ""
        rows.append({"pid": pid, "ppid": ppid, "comm": comm, "args": args})
    return rows


def load_train_pid(workdir: str) -> int | None:
    pid_file = os.path.join(workdir, "train.pid")
    if not os.path.isfile(pid_file):
        return None
    try:
        with open(pid_file, "r") as f:
            content = f.read().strip()
        return int(content)
    except Exception:
        return None


def find_bash_by_pid(rows: list[dict], pid: int) -> dict | None:
    for row in rows:
        if row["pid"] == pid and row["comm"] == "bash":
            return row
    return None


def find_bash_processes(rows: list[dict], script_path: str | None) -> list[dict]:
    result = []
    for row in rows:
        if row["comm"] != "bash":
            continue
        if script_path:
            if script_path in row["args"]:
                result.append(row)
        else:
            result.append(row)
    return result


def find_python_children(rows: list[dict], bash_pid: int) -> list[dict]:
    result = []
    for row in rows:
        if row["ppid"] != bash_pid:
            continue
        if row["comm"].startswith("python"):
            result.append(row)
        elif "python" in row["args"]:
            result.append(row)
    return result


def confirm_kill(target_bash: list[dict], target_python: list[dict]) -> bool:
    print("即将终止以下进程：")
    for row in target_python:
        print(f"python pid={row['pid']} args={row['args']}")
    for row in target_bash:
        print(f"bash pid={row['pid']} args={row['args']}")
    choice = input("确认终止？输入 yes 继续: ").strip().lower()
    return choice == "yes"


def kill_processes(pids: list[int]) -> list[int]:
    killed = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    return killed


def kill_running(workdir: str, script_path: str | None) -> None:
    rows = parse_ps_table()

    train_pid = load_train_pid(workdir)
    target_bash = []
    if train_pid is not None:
        bash_row = find_bash_by_pid(rows, train_pid)
        if bash_row:
            target_bash.append(bash_row)

    if not target_bash and script_path:
        target_bash = find_bash_processes(rows, script_path)

    if not target_bash:
        print("未找到匹配的 bash，未执行终止")
        return

    target_python = []
    for bash_row in target_bash:
        target_python.extend(find_python_children(rows, bash_row["pid"]))

    if not target_python:
        print("未发现 python 子进程，未执行终止")
        return

    if not confirm_kill(target_bash, target_python):
        print("已取消终止")
        return

    killed_python = kill_processes([p["pid"] for p in target_python])
    killed_bash = kill_processes([b["pid"] for b in target_bash])

    if killed_python:
        print("已终止 python 进程:", " ".join(str(pid) for pid in killed_python))
    if killed_bash:
        print("已终止 bash 进程:", " ".join(str(pid) for pid in killed_bash))


def choose_action_cli() -> str:
    print("选择操作：")
    print("1. 运行脚本")
    print("2. 终止运行")
    print("3. 退出程序")
    while True:
        choice = input("请输入序号: ").strip()
        if choice == "1":
            return "run"
        if choice == "2":
            return "kill"
        if choice == "3":
            return "exit"
        print("请输入 1/2/3")


def run_flow(workdir: str, args_script: str) -> None:
    if args_script:
        script_path = args_script
    else:
        script_path = cli_select(workdir)

    if not script_path:
        return

    if not os.path.isabs(script_path):
        script_path = os.path.join(workdir, script_path)

    if not os.path.isfile(script_path):
        raise FileNotFoundError("未找到文件")

    run_nohup(script_path, workdir)
    print("已在后台启动，输出: train.log，进程号: train.pid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", nargs="?", default="")
    parser.add_argument("--kill", action="store_true")
    args = parser.parse_args()

    workdir = os.getcwd()

    if args.kill:
        script_path = args.script if args.script else None
        kill_running(workdir, script_path)
        return

    while True:
        if args.script:
            action = "run"
        else:
            action = choose_action_cli()

        if action == "exit":
            return
        if action == "kill":
            script_path = args.script if args.script else None
            kill_running(workdir, script_path)
            continue

        run_flow(workdir, args.script)

        if args.script:
            return


if __name__ == "__main__":
    main()
