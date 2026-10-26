"""Evalúa un checkpoint de PhysNet con el protocolo común (configs/eval.yaml).

Uso:
    uv run python scripts/evaluate_dl.py checkpoints/physnet_v1/best.pt
    uv run python scripts/evaluate_dl.py checkpoints/physnet_v1/best.pt --split val

Escribe en results/dl/<run_name>/:
- <split>_windows.csv: HR estimada y de referencia por ventana y sujeto;
- <split>_summary.csv: MAE, RMSE y Pearson global ("ALL") y por sujeto;
- <split>_eval_info.json: checkpoint evaluado, commit con que se entrenó el
  modelo y commit con que se evaluó.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from rppg.dl.dataset import load_subject
from rppg.dl.train import evaluate_clips, load_model, resolve_device
from rppg.eval.splits import load_splits
from rppg.utils.config import load_yaml
from rppg.utils.provenance import provenance

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--eval-config", default=ROOT / "configs" / "eval.yaml", type=Path)
    parser.add_argument("--processed-root", type=Path, help="Override data.processed_root")
    parser.add_argument("--raw-root", type=Path, help="Override data.raw_root")
    parser.add_argument("--output-dir", default=ROOT / "results" / "dl", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, meta = load_model(args.checkpoint, device)
    trained_with = (meta["provenance"] or {}).get("git", {})
    print(
        f"Model: epoch {meta['epoch']}, trained with "
        f"{trained_with.get('describe') or 'unknown commit (checkpoint without provenance)'}"
    )
    data_cfg = cfg["data"]
    eval_cfg = load_yaml(args.eval_config)
    subjects = load_splits(resolve(eval_cfg["splits"]))[args.split]
    processed = args.processed_root or resolve(data_cfg["processed_root"])
    raw = args.raw_root or resolve(data_cfg["raw_root"])
    clips = [
        load_subject(s, processed, raw, data_cfg["input_size"])
        for s in tqdm(subjects, desc=f"Loading {args.split}")
    ]

    windows, summary = evaluate_clips(model, clips, data_cfg["chunk_length"], eval_cfg, device)
    out = args.output_dir / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    windows.to_csv(out / f"{args.split}_windows.csv", index=False)
    summary.to_csv(out / f"{args.split}_summary.csv")
    eval_info = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "subjects": subjects,
        "model": meta,
        "evaluation": provenance(),
        "eval_config": eval_cfg,
    }
    (out / f"{args.split}_eval_info.json").write_text(
        json.dumps(eval_info, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    overall = summary.loc["ALL"]
    print(summary.round(2).to_string())
    print(
        f"\n{args.split}: MAE {overall['mae']:.2f} bpm | RMSE {overall['rmse']:.2f} bpm | "
        f"Pearson {overall['pearson']:.3f} | n={int(overall['n'])} windows"
    )
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
