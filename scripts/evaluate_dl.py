"""Evalúa un checkpoint de PhysNet con el protocolo común (configs/eval.yaml).

Uso:
    uv run python scripts/evaluate_dl.py checkpoints/physnet_v1/best.pt
    uv run python scripts/evaluate_dl.py checkpoints/physnet_v1/best.pt --split val
    uv run python scripts/evaluate_dl.py checkpoints/physnet_v2/best.pt --split val \\
        --stride 64 --flip-average
    uv run python scripts/evaluate_dl.py checkpoints/physnet_v4/best.pt --split val \\
        --speed 0.7 1.0 1.4

Escribe en results/dl/<run_name>/:
- <split>_windows.csv: HR estimada y de referencia por ventana y sujeto;
- <split>_summary.csv: MAE, RMSE y Pearson global ("ALL") y por sujeto;
- <split>_eval_info.json: checkpoint evaluado, opciones de inferencia, commit
  con que se entrenó el modelo y commit con que se evaluó.

Por defecto se usan las opciones de inferencia de la config del checkpoint
(sección `infer`; sin ella, tramos sin solapamiento). Si --stride o
--flip-average las cambian, los archivos llevan un sufijo, p. ej.
`val_s64_flip_summary.csv`, para no pisar la evaluación de referencia.

--speed reproduce cada video (y su PPG) a otra velocidad antes de evaluar, con
`rppg.dl.dataset.time_stretch_clip`: la HR queda multiplicada por ese factor.
Sirve para probar la robustez a HR que validación no tiene (p. ej. 0,7 lleva a
subject12 de ~66 a ~46 bpm). Es sintético: no reemplaza a sujetos reales con
esas HR. Los archivos llevan el sufijo `_x<speed>`. No se permite sobre test,
que se evalúa una sola vez y sin modificar.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from rppg.dl.dataset import load_subject, time_stretch_clip
from rppg.dl.train import evaluate_clips, inference_options, load_model, resolve_device
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
    parser.add_argument(
        "--stride",
        type=int,
        help="Step between inference chunks; below chunk_length, overlapping chunks are averaged",
    )
    parser.add_argument(
        "--flip-average",
        action="store_true",
        help="Average with the prediction on the horizontally flipped video",
    )
    parser.add_argument(
        "--speed",
        type=float,
        nargs="+",
        default=[1.0],
        help="Playback speed(s) applied to video and PPG before evaluating (HR scales by it)",
    )
    args = parser.parse_args()
    if args.split == "test" and args.speed != [1.0]:
        parser.error("--speed is not allowed on the test split")

    device = resolve_device(args.device)
    model, cfg, meta = load_model(args.checkpoint, device)
    trained_with = (meta["provenance"] or {}).get("git", {})
    print(
        f"Model: epoch {meta['epoch']}, trained with "
        f"{trained_with.get('describe') or 'unknown commit (checkpoint without provenance)'}"
    )
    data_cfg = cfg["data"]
    chunk_length = data_cfg["chunk_length"]
    # Por defecto, las opciones con que se entrenó (sección `infer` de la config
    # guardada en el checkpoint); los flags las pisan.
    infer = inference_options(cfg)
    stride = infer["stride"] if args.stride is None else args.stride
    flip_average = infer["flip_average"] or args.flip_average
    eval_cfg = load_yaml(args.eval_config)
    subjects = load_splits(resolve(eval_cfg["splits"]))[args.split]
    processed = args.processed_root or resolve(data_cfg["processed_root"])
    raw = args.raw_root or resolve(data_cfg["raw_root"])
    clips = [
        load_subject(s, processed, raw, data_cfg["input_size"])
        for s in tqdm(subjects, desc=f"Loading {args.split}")
    ]

    out = args.output_dir / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    base_tag = args.split
    if (stride, flip_average) != (infer["stride"], infer["flip_average"]):
        base_tag += f"_s{stride}" + ("_flip" if flip_average else "")
    for speed in args.speed:
        tag = base_tag if speed == 1.0 else f"{base_tag}_x{speed:g}"
        stretched = [time_stretch_clip(c, speed) for c in clips]
        windows, summary = evaluate_clips(
            model,
            stretched,
            chunk_length,
            eval_cfg,
            device,
            stride=stride,
            flip_average=flip_average,
        )
        windows.to_csv(out / f"{tag}_windows.csv", index=False)
        summary.to_csv(out / f"{tag}_summary.csv")
        eval_info = {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "speed": speed,
            "inference": {
                "chunk_length": chunk_length,
                "stride": stride,
                "flip_average": flip_average,
            },
            "subjects": subjects,
            "model": meta,
            "evaluation": provenance(),
            "eval_config": eval_cfg,
        }
        (out / f"{tag}_eval_info.json").write_text(
            json.dumps(eval_info, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

        overall = summary.loc["ALL"]
        print(summary.round(2).to_string())
        print(
            f"\n{tag}: MAE {overall['mae']:.2f} bpm | RMSE {overall['rmse']:.2f} bpm | "
            f"Pearson {overall['pearson']:.3f} | n={int(overall['n'])} windows\n"
        )
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
