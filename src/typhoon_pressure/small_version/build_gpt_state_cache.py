from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from typhoon_pressure.dataset import TyphoonPressureDataset

from .config import GPTStateConfig
from .gpt_state import (
    GPTStateRecord,
    OpenAIStateExtractor,
    build_gpt_state_summary,
    save_gpt_state,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build cached GPT synoptic-state features")
    parser.add_argument("--integrated", required=True)
    parser.add_argument("--output-dir", default="data/gpt_states")
    parser.add_argument("--model", default="gpt-5.6")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--track-steps", type=int, default=20)
    parser.add_argument("--max-highs", type=int, default=3)
    parser.add_argument("--on-error", choices=["mask", "raise"], default="mask")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    integrated = pd.read_parquet(args.integrated) if args.integrated.endswith(".parquet") else pd.read_csv(args.integrated)
    integrated["time"] = pd.to_datetime(integrated["time"])
    base = TyphoonPressureDataset(
        integrated, history=args.history, horizon=args.track_steps, max_highs=args.max_highs
    )
    extractor = OpenAIStateExtractor(GPTStateConfig(model=args.model))
    output = Path(args.output_dir)
    existing = set()
    manifest = output / "manifest.csv"
    if manifest.exists() and not args.overwrite:
        current = pd.read_csv(manifest)
        existing = set(zip(current["storm_id"].astype(str), current["init_time_ns"].astype("int64")))

    completed, masked, skipped = 0, 0, 0
    for index in range(len(base)):
        sample = base[index]
        key = (str(sample["storm_id"]), int(sample["init_time_ns"]))
        if key in existing:
            skipped += 1
            continue
        summary = build_gpt_state_summary(sample, base.feature_cols)
        try:
            record = extractor.extract(summary)
            status = "ok"
            completed += 1
        except Exception as exc:
            if args.on_error == "raise":
                raise
            record = GPTStateRecord.missing()
            status = f"masked:{type(exc).__name__}"
            masked += 1
        save_gpt_state(
            record, output, storm_id=key[0], init_time_ns=key[1], status=status
        )
    print({"completed": completed, "masked": masked, "skipped": skipped, "output_dir": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
