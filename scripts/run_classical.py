"""Evalúa los métodos clásicos (GREEN, CHROM, POS) con el protocolo común.

Uso:
    uv run python scripts/run_classical.py                       # desarrollo (train + val)
    uv run python scripts/run_classical.py --methods pos chrom
    uv run python scripts/run_classical.py --subset test         # solo al final

Subconjuntos (particiones de configs/splits.yaml):
- dev: train + val (33 sujetos). Es el único que se usa para elegir método y
  parámetros. El clásico no entrena, pero elegir mirando test filtraría
  información, igual que en el DL.
- test: los 9 sujetos de test. Se evalúa una sola vez, junto con PhysNet
  (tarea 3.5), con la configuración ya cerrada.
- all: los 42 sujetos (el plan pide reportar también el dataset completo).
  Incluye test, así que vale la misma regla.

Escribe en results/classical/:
- <subset>_windows.csv: HR estimada y de referencia por método, sujeto y ventana;
- <subset>_summary.csv: MAE, RMSE y Pearson por método, global ("ALL") y por sujeto;
- <subset>_eval_info.json: configuración, sujetos y commit con que se evaluó.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rppg.classical.pipeline import evaluate_subject
from rppg.data.ubfc import UBFCrPPG
from rppg.eval.protocol import summarize_windows
from rppg.eval.splits import load_splits
from rppg.face.extract import load_traces
from rppg.utils.config import load_yaml
from rppg.utils.provenance import provenance

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def subset_subjects(splits: dict[str, list[str]], subset: str) -> list[str]:
    if subset == "dev":
        return splits["train"] + splits["val"]
    if subset == "test":
        return list(splits["test"])
    if subset == "all":
        return splits["train"] + splits["val"] + splits["test"]
    raise ValueError(f"Unknown subset {subset!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=ROOT / "configs" / "classical.yaml", type=Path)
    parser.add_argument("--subset", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--methods", nargs="+", help="Subset of configured methods")
    parser.add_argument("--subjects", nargs="+", help="Restrict to these subjects (debugging)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    eval_cfg = load_yaml(resolve(cfg["eval_config"]))
    methods = args.methods or list(cfg["methods"])
    unknown = set(methods) - set(cfg["methods"])
    if unknown:
        parser.error(f"methods not in {args.config.name}: {sorted(unknown)}")
    subjects = subset_subjects(load_splits(resolve(eval_cfg["splits"])), args.subset)
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]
    if args.subset != "dev":
        print(f"WARNING: evaluating on '{args.subset}', which includes test subjects.")

    ds = UBFCrPPG(resolve(cfg["raw_root"]))
    processed = resolve(cfg["processed_root"])
    windows = []
    for subject in tqdm(subjects, desc=f"Classical ({args.subset})"):
        track = load_traces(processed / subject)
        gt = ds.ground_truth(subject)
        for method in methods:
            windows.append(
                evaluate_subject(
                    subject,
                    track,
                    gt.t,
                    gt.ppg,
                    method,
                    eval_cfg,
                    params=cfg["methods"][method],
                    rois=cfg.get("rois"),
                )
            )
    windows = pd.concat(windows, ignore_index=True)
    summary = pd.concat(
        {m: summarize_windows(w) for m, w in windows.groupby("method", sort=False)},
        names=["method", "subject"],
    )

    out = resolve(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    windows.to_csv(out / f"{args.subset}_windows.csv", index=False)
    summary.to_csv(out / f"{args.subset}_summary.csv")
    info = {
        "subset": args.subset,
        "subjects": subjects,
        "methods": {m: cfg["methods"][m] for m in methods},
        "rois": cfg.get("rois"),
        "evaluation": provenance(),
        "eval_config": eval_cfg,
    }
    (out / f"{args.subset}_eval_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    overall = summary.xs("ALL", level="subject")
    print(overall.round(3).to_string())
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
