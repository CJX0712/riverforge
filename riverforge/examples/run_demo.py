"""端到端演示：流式合成数据（含概念漂移）→ 跨算法 prequential 基准 → 打印表 + 落盘 benchmark.json。

零下载可跑（核心依赖 numpy；sklearn / river 可用时自动纳入，缺失自动跳过）。
"""

from __future__ import annotations

import json
import os
import time

from ..core.config import Config
from ..pipeline.pipeline import StreamPipeline


def run(out_path: str | None = None) -> dict:
    cfg = Config.from_env()
    pipe = StreamPipeline(cfg)
    t0 = time.perf_counter()
    rows = pipe.benchmark(seeds=cfg.benchmark_seeds)
    elapsed = time.perf_counter() - t0
    summary = pipe.summarize(rows)

    payload = {
        "system": "RiverForge",
        "version": "0.1.0",
        "author": "晨星",
        "config": cfg.as_dict(),
        "generated_at_note": "确定性可复现基准（固定 random_state）",
        "elapsed_s": round(elapsed, 3),
        "summary": summary,
        "rows": [r.as_dict() for r in rows],
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _print_table(payload: dict) -> None:
    print("  ".join(f"{h:<12}" for h in ["algo", "scenario", "acc", "kappa", "auc", "sec"]))
    for r in payload["rows"][:40]:
        if r.get("skipped"):
            note = str(r.get("note", ""))[:40]
            print(f"{r['algorithm'][:12]:<12}{r['scenario'][:12]:<12}-- skip: {note}")
            continue
        print(
            "  ".join(
                f"{r['algorithm'][:12]:<12}{r['scenario'][:12]:<12}"
                f"{r['prequential_accuracy']:<12.3f}{r['kappa']:<12.3f}"
                f"{r['auc']:<12.3f}{r['runtime_s']:<12.3f}"
            )
        )
    if len(payload["rows"]) > 40:
        print(f"... 共 {len(payload['rows'])} 条（仅显示前 40）")

    print("\n=== 聚合排名（按 mean prequential accuracy） ===")
    for i, r in enumerate(payload["summary"]["ranking"], 1):
        print(
            f"{i:>2}. {r['algorithm']:<22} meanAcc={r['mean_acc']:.3f}  "
            f"minAcc={r['min_acc']:.3f}  auc={r['mean_auc']:.3f}  "
            f"runs={r['n_runs']}  mean_sec={r['mean_runtime_s']:.3f}"
        )
    print(f"\n最佳算法: {payload['summary']['best_algorithm']}")
    print(
        f"总耗时: {payload['elapsed_s']}s  评测条数: {payload['summary']['n_rows']}  "
        f"跳过: {payload['summary']['n_skipped']}"
    )


def _print_drift_table(payload: dict) -> None:
    print("\n=== 漂移场景召回（仅支持漂移探测的算法） ===")
    seen = set()
    for r in payload["rows"]:
        if r.get("skipped") or not r.get("supports_drift"):
            continue
        key = (r["algorithm"], r["scenario"])
        if key in seen or r["scenario"] == "stationary":
            continue
        seen.add(key)
        print(
            f"{r['algorithm'][:20]:<20}{r['scenario'][:18]:<18}"
            f"recall={r['drift_recall']:<8.2f}false_alarm={r['drift_false_alarm']:<8.2f}"
        )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "benchmark.json")
    payload = run(out)
    _print_table(payload)
    _print_drift_table(payload)
    print(f"\n已落盘: {out}")
