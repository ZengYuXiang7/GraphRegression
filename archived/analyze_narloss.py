"""
解析 narloss 搜索实验结果，从 results/metrics/ 的 pkl 文件中
找出最优的 alpha / lambda_rank / lambda_consistency，
打印汇总表并输出 stage4 shell 命令。
"""
import os
import pickle
import glob
import numpy as np
from collections import defaultdict


def load_narloss_results(metrics_dir="./results/metrics"):
    results = []
    for pkl_path in glob.glob(os.path.join(metrics_dir, "*.pkl")):
        fname = os.path.basename(pkl_path).replace(".pkl", "")
        # 只处理 narloss 搜索实验（含 lambdarank 字段）
        if "lambdarank__" not in fname:
            continue
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue

        # 解析 filename 中的关键字段
        fields = {}
        for part in fname.split("|"):
            if "__" in part:
                k, v = part.split("__", 1)
                fields[k] = v

        tau_vals = data.get("Tau", data.get("tau", []))
        if not tau_vals:
            continue

        results.append({
            "fname": fname,
            "model": fields.get("model", "?"),
            "percent": fields.get("percent", "?"),
            "graphreadout": fields.get("graphreadout", "?"),
            "ranklosstype": fields.get("ranklosstype", "l1"),
            "rankalpha": float(fields.get("rankalpha", 10.0)),
            "lambdamse": float(fields.get("lambdamse", 1.0)),
            "lambdarank": float(fields.get("lambdarank", 0.2)),
            "lambdaconsistency": float(fields.get("lambdaconsistency", 1.0)),
            "tau_mean": float(np.mean(tau_vals)),
            "tau_std": float(np.std(tau_vals)),
        })

    return results


def print_table(title, rows, key_cols, sort_by="tau_mean"):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if not rows:
        print("  (无数据)")
        return
    rows = sorted(rows, key=lambda r: r[sort_by], reverse=True)
    header = "  " + "  ".join(f"{c:<18}" for c in key_cols + ["tau_mean", "tau_std"])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        vals = [str(r.get(c, "?"))[:18] for c in key_cols]
        vals += [f"{r['tau_mean']:.5f}", f"{r['tau_std']:.5f}"]
        print("  " + "  ".join(f"{v:<18}" for v in vals))


def get_top_n(rows, key, n=3):
    """按 tau_mean 排序，取 top-n 的 key 值（去重）"""
    sorted_rows = sorted(rows, key=lambda r: r["tau_mean"], reverse=True)
    seen, top = [], []
    for r in sorted_rows:
        v = r[key]
        if v not in seen:
            seen.append(v)
            top.append(v)
        if len(top) >= n:
            break
    return top


def main():
    results = load_narloss_results()
    if not results:
        print("[ERROR] 没有找到 narloss 实验结果，请确认 results/metrics/ 目录下有对应 pkl 文件")
        return

    print(f"\n共加载 {len(results)} 条 narloss 实验结果")

    # ── 阶段0：l1 vs soft_kt alpha 扫描 ──────────────────────────────
    stage0 = [r for r in results
              if abs(r["lambdarank"] - 0.2) < 1e-4
              and abs(r["lambdaconsistency"] - 1.0) < 1e-4]
    print_table("阶段0: rank_loss_type 对比 (lambda_rank=0.2, consist=1.0)",
                stage0, ["ranklosstype", "rankalpha"])

    # ── 阶段1：消融 ───────────────────────────────────────────────────
    stage1_mse_only = [r for r in results
                       if abs(r["lambdarank"]) < 1e-4
                       and abs(r["lambdaconsistency"]) < 1e-4]
    stage1_rank_only = [r for r in results
                        if r["lambdarank"] > 1e-4
                        and abs(r["lambdaconsistency"]) < 1e-4
                        and r["ranklosstype"] == "l1"]
    stage1_consist_only = [r for r in results
                           if abs(r["lambdarank"]) < 1e-4
                           and r["lambdaconsistency"] > 1e-4
                           and r["ranklosstype"] == "l1"]
    print_table("阶段1a: 纯MSE基线", stage1_mse_only, ["lambdamse"])
    print_table("阶段1b: MSE+Rank (consist=0)", stage1_rank_only, ["lambdarank"])
    print_table("阶段1c: MSE+Consistency (rank=0)", stage1_consist_only, ["lambdaconsistency"])

    # ── 阶段2：lambda_rank 扫描 ───────────────────────────────────────
    stage2_l1 = [r for r in results
                 if abs(r["lambdaconsistency"] - 1.0) < 1e-4
                 and r["ranklosstype"] == "l1"
                 and r["lambdarank"] > 1e-4]
    stage2_skt = [r for r in results
                  if abs(r["lambdaconsistency"] - 1.0) < 1e-4
                  and r["ranklosstype"] == "soft_kt"]
    print_table("阶段2a: lambda_rank 扫描 (l1, consist=1.0)", stage2_l1, ["lambdarank"])
    print_table("阶段2b: lambda_rank 扫描 (soft_kt, consist=1.0)", stage2_skt,
                ["rankalpha", "lambdarank"])

    # ── 阶段3：lambda_consistency 扫描 ────────────────────────────────
    stage3 = [r for r in results
              if abs(r["lambdarank"] - 0.2) < 1e-4
              and r["ranklosstype"] == "soft_kt"
              and abs(r["rankalpha"] - 10.0) < 1e-4]
    print_table("阶段3: lambda_consistency 扫描 (soft_kt, rank=0.2, alpha=10)",
                stage3, ["lambdaconsistency"])

    # ── 自动推断 stage4 参数 ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Stage 4 推荐参数（基于当前结果）")
    print(f"{'='*70}")

    best_alphas = get_top_n(stage2_skt or stage0, "rankalpha", n=3)
    best_lrs = get_top_n(stage2_skt, "lambdarank", n=3) if stage2_skt else [0.1, 0.2, 0.4]
    best_lcs = get_top_n(stage3, "lambdaconsistency", n=3) if stage3 else [0.5, 1.0, 2.0]

    print(f"  best_alphas        = {best_alphas}")
    print(f"  best_lambdas_rank  = {best_lrs}")
    print(f"  best_lambdas_consist = {best_lcs}")

    total = len(best_alphas) * len(best_lrs) * len(best_lcs)
    print(f"  → 共 {total} 组 stage4 实验\n")

    print("  生成的 shell 命令（复制到 run_exp.sh 阶段4处）：")
    print()
    base = '--model model56 --percent 100 --d_model 180 --gcn_layers 10 --graph_n_head 6 --embed_type onehot_op --graph_readout att'
    print(f'BASE=({base})')
    print(f'best_alphas=({" ".join(str(a) for a in best_alphas)})')
    print(f'lambdas_rank=({" ".join(str(r) for r in best_lrs)})')
    print(f'lambdas_consistency=({" ".join(str(c) for c in best_lcs)})')
    print('for alpha in "${best_alphas[@]}"; do')
    print('    for lr in "${lambdas_rank[@]}"; do')
    print('        for lc in "${lambdas_consistency[@]}"; do')
    print('            python Experiment.py "${BASE[@]}" \\')
    print('                --lambda_mse 1.0 --lambda_rank "$lr" --lambda_consistency "$lc" \\')
    print('                --rank_loss_type soft_kt --rank_alpha "$alpha"')
    print('        done')
    print('    done')
    print('done')


if __name__ == "__main__":
    main()
