"""
自动实验运行器 -- 顺序执行 experiments_config.py 中定义的实验

功能:
    - 逐个执行实验，一个完成后再跑下一个
    - 每个实验的日志独立保存到 logs/<exp_name>/ 下
    - 命名规则沿用 main.py: {dataset}_{model}_{partition}_a{alpha}_{timestamp}
    - stdout+stderr 实时写入文件和终端
    - status.json 记录 pending/running/done/failed
    - 任意实验中断不影响其他实验的日志
    - 支持 --exp_ids 指定只跑某些实验
    - 支持 --dry_run 只打印命令不执行

用法:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --exp_ids 0 2
    python scripts/run_experiments.py --dry_run
    python scripts/run_experiments.py --log_tag alpha_ablation
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 缓存默认参数
_DEFAULT_ARGS_CACHE = None


def get_default_args() -> dict:
    """从 args.py 获取默认参数值（带缓存）。"""
    global _DEFAULT_ARGS_CACHE
    if _DEFAULT_ARGS_CACHE is not None:
        return _DEFAULT_ARGS_CACHE
    
    # 添加项目根目录到 path
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # 临时修改 sys.argv 避免解析命令行参数
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    
    try:
        from args import get_args
        args = get_args()
        _DEFAULT_ARGS_CACHE = vars(args)
        return _DEFAULT_ARGS_CACHE
    finally:
        sys.argv = original_argv


def load_experiments():
    """从 experiments_config.py 加载实验列表。"""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from experiments_config import EXPERIMENTS
    return EXPERIMENTS


def make_run_name(params: dict, timestamp: str) -> str:
    """沿用 main.py 的命名逻辑: {dataset}_{model}_{partition}_a{alpha}_{timestamp}
    
    优先使用 params 中的值，否则使用 args.py 的默认值。
    """
    defaults = get_default_args()
    
    # 合并：params 优先，否则用 defaults
    dataset = params.get("dataset", defaults.get("dataset", "cifar10"))
    model = params.get("model", defaults.get("model", "resnet18"))
    partition = params.get("partition", defaults.get("partition", "dirichlet"))
    alpha = params.get("alpha", defaults.get("alpha", 0.1))
    
    return f"{dataset}_{model}_{partition}_a{alpha}_{timestamp}"


def build_command(exp: dict, log_dir: Path) -> list[str]:
    """把实验配置转成 python main.py 的命令行参数列表。"""
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py")]

    params = exp.get("params", {})
    for key, value in params.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    # 把 log_dir 和 save_dir 指向该实验自己的目录
    cmd.extend(["--log_dir", str(log_dir)])
    cmd.extend(["--save_dir", str(log_dir / "checkpoints")])

    return cmd


def run_one_experiment(exp_id: int, exp: dict, global_log_dir: Path) -> dict:
    """执行单个实验，返回结果 dict。"""
    params = exp.get("params", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = make_run_name(params, timestamp)

    # 如果 config 里指定了 name，用它覆盖（一般不需要）
    if "name" in exp:
        run_name = exp["name"]

    log_dir = global_log_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # 构建命令
    cmd = build_command(exp, log_dir)
    cmd_str = " ".join(cmd)

    # 保存命令
    with open(log_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write(cmd_str + "\n")

    # 保存 config
    with open(log_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(exp, f, indent=2, ensure_ascii=False)

    # 初始化 status
    status_file = log_dir / "status.json"
    result = {
        "exp_id": exp_id,
        "name": run_name,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "log_dir": str(log_dir),
    }
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'#' * 70}")
    print(f"  Experiment {exp_id}: {run_name}")
    print(f"  Log dir: {log_dir}")
    print(f"  Command: {cmd_str}")
    print(f"{'#' * 70}\n")

    # 执行
    start_time = time.time()
    try:
        with open(log_dir / "stdout_stderr.log", "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            # 实时输出并写入文件
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

            proc.wait()
            elapsed = round(time.time() - start_time, 1)

            if proc.returncode == 0:
                result["status"] = "done"
            else:
                result["status"] = "failed"
                result["returncode"] = proc.returncode

    except KeyboardInterrupt:
        elapsed = round(time.time() - start_time, 1)
        result["status"] = "interrupted"
        print(f"\n[!] Experiment {exp_id} interrupted by user.")

    except Exception as e:
        elapsed = round(time.time() - start_time, 1)
        result["status"] = "error"
        result["error"] = str(e)
        print(f"\n[!] Experiment {exp_id} error: {e}")

    result["end_time"] = datetime.now().isoformat()
    result["elapsed_sec"] = elapsed

    # 更新 status.json
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run experiments sequentially")
    parser.add_argument("--exp_ids", type=int, nargs="*", default=None,
                        help="Run only these experiment IDs (default: all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--log_tag", type=str, default=None,
                        help="Tag for batch log directory")
    args = parser.parse_args()

    # 加载实验
    experiments = load_experiments()
    if not experiments:
        print("[!] No experiments defined in experiments_config.py")
        return

    # 筛选实验
    if args.exp_ids is not None:
        experiments = [(i, exp) for i, exp in enumerate(experiments) if i in args.exp_ids]
        if not experiments:
            print(f"[!] No experiments match IDs: {args.exp_ids}")
            return
    else:
        experiments = list(enumerate(experiments))

    # 创建批次日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.log_tag:
        batch_name = f"batch_{args.log_tag}_{timestamp}"
    else:
        batch_name = f"batch_{timestamp}"

    global_log_dir = PROJECT_ROOT / "logs" / batch_name
    global_log_dir.mkdir(parents=True, exist_ok=True)

    # 保存批次信息
    batch_info = {
        "batch_name": batch_name,
        "start_time": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "exp_ids": [i for i, _ in experiments],
        "dry_run": args.dry_run,
    }
    with open(global_log_dir / "batch_info.json", "w", encoding="utf-8") as f:
        json.dump(batch_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  DualOrtho Experiment Runner")
    print("=" * 70)
    print(f"  Batch: {batch_name}")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Log dir: {global_log_dir}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70 + "\n")

    if args.dry_run:
        print("[DRY RUN] Commands that would be executed:\n")
        for exp_id, exp in experiments:
            params = exp.get("params", {})
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = make_run_name(params, timestamp)
            log_dir = global_log_dir / run_name
            cmd = build_command(exp, log_dir)
            print(f"  [{exp_id}] {' '.join(cmd)}")
        print("\n[DRY RUN] No experiments executed.")
        return

    # 顺序执行
    batch_start = time.time()
    results = []

    for exp_id, exp in experiments:
        print(f"\n{'#' * 70}")
        print(f"{'#' * 70}")

        result = run_one_experiment(exp_id, exp, global_log_dir)
        results.append(result)

    batch_elapsed = round(time.time() - batch_start, 1)

    # 汇总报告
    print("\n")
    print("=" * 70)
    print("  批次执行完成 -- 汇总报告")
    print("=" * 70)
    print(f"  总耗时: {batch_elapsed}s")
    print(f"  日志目录: {global_log_dir}")
    print("")
    print(f"  {'ID':>3s}  {'状态':<12s}  {'耗时':>8s}  {'名称'}")
    print(f"  {'---':>3s}  {'----':<12s}  {'----':>8s}  {'----'}")

    done_count = 0
    fail_count = 0
    for r in results:
        status = r["status"]
        elapsed = f"{r.get('elapsed_sec', 0):.0f}s" if r.get('elapsed_sec') else "-"
        name = r["name"]
        icon = {"done": "[OK]", "failed": "[FAIL]", "interrupted": "[STOP]", "error": "[ERR]"}.get(status, "[?]")
        print(f"  {r['exp_id']:3d}  {icon:<12s}  {elapsed:>8s}  {name}")
        if status == "done":
            done_count += 1
        else:
            fail_count += 1

    print(f"\n  完成: {done_count}  失败/中断: {fail_count}")
    print("=" * 70)

    # 保存汇总
    summary = {
        "batch_info": batch_info,
        "total_elapsed_sec": batch_elapsed,
        "done": done_count,
        "failed": fail_count,
        "results": results,
    }
    with open(global_log_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n汇总已保存: {global_log_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
