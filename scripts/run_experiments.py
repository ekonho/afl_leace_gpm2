"""
自动实验运行器 -- 顺序执行 experiments_config.py 中定义的实验

每个实验的保存逻辑与 main.py 完全一致：
    logs/{run_name}.log
    logs/{run_name}.json
    checkpoints/{run_name}_best.pt

用法:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --exp_ids 0 2
    python scripts/run_experiments.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_ARGS_CACHE = None


def get_default_args() -> dict:
    global _DEFAULT_ARGS_CACHE
    if _DEFAULT_ARGS_CACHE is not None:
        return _DEFAULT_ARGS_CACHE
    sys.path.insert(0, str(PROJECT_ROOT))
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        from args import get_args
        _DEFAULT_ARGS_CACHE = vars(get_args())
        return _DEFAULT_ARGS_CACHE
    finally:
        sys.argv = original_argv


def load_experiments():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from experiments_config import EXPERIMENTS
    return EXPERIMENTS


def build_command(exp: dict, batch_log_dir: Path) -> list[str]:
    """?? python main.py ???????? batch ????"""
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py")]
    for key, value in exp.get("params", {}).items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    # Override log/save dirs to scripts/logs/<batch_name>
    cmd.extend(["--log_dir", str(batch_log_dir)])
    cmd.extend(["--save_dir", str(batch_log_dir / "checkpoints")])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run experiments sequentially")
    parser.add_argument("--exp_ids", type=int, nargs="*", default=None,
                        help="Run only these experiment IDs")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    experiments = load_experiments()
    if not experiments:
        print("[!] No experiments defined in experiments_config.py")
        return

    if args.exp_ids is not None:
        experiments = [(i, e) for i, e in enumerate(experiments) if i in args.exp_ids]
    else:
        experiments = list(enumerate(experiments))

    if not experiments:
        print(f"[!] No experiments match IDs: {args.exp_ids}")
        return

    print("\n" + "=" * 70)
    print("  DualOrtho Experiment Runner")
    print("=" * 70)
    print(f"  Experiments: {len(experiments)}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70 + "\n")

    if args.dry_run:
        for exp_id, exp in experiments:
            cmd = build_command(exp)
            print(f"  [{exp_id}] {' '.join(cmd)}")
        print("\n[DRY RUN] No experiments executed.")
        return

    # Create batch output directory under scripts/logs/
    script_dir = Path(__file__).resolve().parent
    batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_log_dir = script_dir / "logs" / batch_name
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Batch log dir: {batch_log_dir}")

    batch_start = time.time()
    results = []

    for exp_id, exp in experiments:
        cmd = build_command(exp, batch_log_dir)
        cmd_str = " ".join(cmd)

        print(f"\n{'#' * 70}")
        print(f"  Experiment {exp_id}: {cmd_str}")
        print(f"{'#' * 70}\n")

        start = time.time()
        status = "done"
        try:
            proc = subprocess.run(cmd, encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                status = "failed"
        except KeyboardInterrupt:
            status = "interrupted"
            print(f"\n[!] Experiment {exp_id} interrupted.")
        except Exception as e:
            status = "error"
            print(f"\n[!] Experiment {exp_id} error: {e}")

        elapsed = round(time.time() - start, 1)
        results.append({"exp_id": exp_id, "status": status, "elapsed_sec": elapsed})

    batch_elapsed = round(time.time() - batch_start, 1)

    # 汇总
    print("\n" + "=" * 70)
    print("  汇总报告")
    print("=" * 70)
    print(f"  总耗时: {batch_elapsed}s\n")
    print(f"  {'ID':>3s}  {'状态':<12s}  {'耗时':>8s}")
    print(f"  {'---':>3s}  {'----':<12s}  {'----':>8s}")

    done = fail = 0
    for r in results:
        icon = {"done": "[OK]", "failed": "[FAIL]", "interrupted": "[STOP]", "error": "[ERR]"}.get(r["status"], "[?]")
        print(f"  {r['exp_id']:3d}  {icon:<12s}  {r['elapsed_sec']:>6.0f}s")
        if r["status"] == "done":
            done += 1
        else:
            fail += 1

    print(f"\n  完成: {done}  失败: {fail}")
    print("=" * 70)

    # 保存汇总到 logs/
    summary_path = batch_log_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"total_elapsed_sec": batch_elapsed, "results": results}, f, indent=2)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
