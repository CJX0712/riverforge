"""RiverForge 命令行入口。"""

from __future__ import annotations

import argparse
import sys

from .core.config import Config
from .data.stream import default_datasets
from .domain.streaming.registry import available_entries, list_algorithms
from .pipeline.pipeline import StreamPipeline


def _print_rows(rows) -> None:
    hdr = ["algo", "scenario", "acc", "kappa", "auc", "driftR", "driftFA", "n", "sec"]
    print("  ".join(f"{h:<12}" for h in hdr))
    for r in rows:
        if r.skipped:
            print(f"{r.algorithm:<12}{r.scenario:<12}{'--':<12}skip: {r.note[:40]}")
            continue
        print(
            "  ".join(
                f"{r.algorithm[:12]:<12}{r.scenario[:12]:<12}"
                f"{r.prequential_accuracy:<12.3f}{r.kappa:<12.3f}{r.auc:<12.3f}"
                f"{r.drift_recall:<12.2f}{r.drift_false_alarm:<12.2f}"
                f"{r.n_seen:<12}{r.runtime_s:<12.3f}"
            )
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="riverforge", description="RiverForge 在线学习系统")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("benchmark", help="跨场景跨算法 prequential 基准")
    sub.add_parser("datasets", help="列出流式场景")
    sub.add_parser("algorithms", help="列出可用算法")

    run_p = sub.add_parser("run", help="在单个场景跑全部算法")
    run_p.add_argument("--scenario", default="sudden_drift")
    run_p.add_argument("--drift-type", default="sudden")

    args = p.parse_args(argv)
    cfg = Config.from_env()

    if args.cmd == "datasets":
        for d in default_datasets(cfg):
            print(f"{d.name:<20} drift={d.drift_type:<12} n={d.n_samples} note={d.note}")
        return 0

    if args.cmd == "algorithms":
        for name, sd in list_algorithms(cfg):
            print(f"{name:<24} supports_drift={sd}")
        print(f"\n可用条目: {len(available_entries(cfg))}")
        return 0

    pipe = StreamPipeline(cfg)
    if args.cmd == "run":
        from .data.stream import make_dataset

        ds = make_dataset(args.scenario, args.drift_type, cfg)
        rows = pipe.benchmark(seeds=(cfg.random_state,), datasets=[ds])
    else:
        rows = pipe.benchmark()

    _print_rows(rows)
    summary = pipe.summarize(rows)
    print("\n=== 聚合排名（按 mean prequential accuracy） ===")
    for i, r in enumerate(summary["ranking"], 1):
        print(
            f"{i:>2}. {r['algorithm']:<22} meanAcc={r['mean_acc']:.3f}  "
            f"minAcc={r['min_acc']:.3f}  auc={r['mean_auc']:.3f}  "
            f"runs={r['n_runs']}  mean_sec={r['mean_runtime_s']:.3f}"
        )
    print(f"\n最佳算法: {summary['best_algorithm']}")
    print(f"评测条数: {summary['n_rows']}  跳过: {summary['n_skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
