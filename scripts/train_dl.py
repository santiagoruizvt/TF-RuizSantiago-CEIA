"""Entrena PhysNet sobre las particiones de configs/splits.yaml.

Uso:
    uv run python scripts/train_dl.py                         # config por defecto
    uv run python scripts/train_dl.py --resume                # retoma desde last.pt
    uv run python scripts/train_dl.py --smoke                 # prueba rápida en CPU
    uv run python scripts/train_dl.py --set train.epochs=50 data.input_size=128

En Colab, las rutas se pisan con --set (ver notebooks/03_train_physnet_colab.ipynb).
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from rppg.dl.dataset import load_subject
from rppg.dl.train import train
from rppg.eval.splits import load_splits
from rppg.utils.config import apply_overrides, config_hash, load_yaml

ROOT = Path(__file__).resolve().parents[1]

# Prueba de humo: pocos sujetos, recortes chicos y tramos cortos, para validar
# en CPU que todo corre y la pérdida baja antes de gastar GPU.
SMOKE_OVERRIDES = [
    "data.input_size=32",
    "data.chunk_length=64",
    "data.train_stride=32",
    "train.epochs=3",
    "train.batch_size=4",
    "run_name=smoke",
]
SMOKE_SUBJECTS = {"train": 3, "val": 2}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=ROOT / "configs" / "physnet.yaml", type=Path)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    apply_overrides(cfg, (SMOKE_OVERRIDES if args.smoke else []) + args.set)
    eval_cfg = load_yaml(resolve(cfg["eval_config"]))
    splits = load_splits(resolve(cfg["data"]["splits"]))
    if args.smoke:
        splits = {k: v[: SMOKE_SUBJECTS.get(k, 0)] for k, v in splits.items()}

    data_cfg = cfg["data"]

    def load(subjects, desc):
        processed, raw = resolve(data_cfg["processed_root"]), resolve(data_cfg["raw_root"])
        return [
            load_subject(s, processed, raw, data_cfg["input_size"])
            for s in tqdm(subjects, desc=desc)
        ]

    train_clips = load(splits["train"], "Loading train")
    val_clips = load(splits["val"], "Loading val")

    run_dir = resolve(cfg["output_dir"]) / cfg["run_name"]
    print(f"Run: {run_dir} | config hash {config_hash(cfg)}")
    history = train(cfg, eval_cfg, train_clips, val_clips, run_dir, resume=args.resume)
    select_by = cfg["train"]["select_by"]
    best = history.loc[history[select_by].idxmin()]
    print(
        f"\nBest epoch by {select_by}: {int(best['epoch'])} | val loss {best['val_loss']:.3f} | "
        f"val MAE {best['val_mae']:.2f} bpm | Pearson {best['val_pearson']:.2f} "
        f"-> {run_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
