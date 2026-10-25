"""Extrae trazas RGB por ROI y recortes de cara de cada video del dataset.

Uso:
    uv run python scripts/extract_traces.py                      # todos los sujetos
    uv run python scripts/extract_traces.py --subjects subject1 subject3
    uv run python scripts/extract_traces.py --overwrite          # ignora la cache

Los sujetos ya procesados con la misma configuración se saltean. Al final
escribe un resumen por sujeto (tasa de detección y velocidad) en
<output_dir>/<dataset>/extraction_summary.csv.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rppg.data.ubfc import UBFCrPPG
from rppg.face.extract import cached_hash, extract_video, save_track
from rppg.face.landmarker import VideoLandmarker, ensure_model
from rppg.utils.config import config_hash, load_yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=ROOT / "configs" / "face.yaml", type=Path)
    parser.add_argument("--subjects", nargs="*", help="Subset of subjects (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if cached")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg_hash = config_hash(cfg)
    model = ensure_model(ROOT / cfg["model"]["path"], cfg["model"]["url"], cfg["model"]["sha256"])
    ds = UBFCrPPG(ROOT / cfg["dataset"]["root"])
    out_root = ROOT / cfg["output_dir"] / cfg["dataset"]["name"]
    det_cfg, skin_cfg, crop_cfg = cfg["detector"], cfg["skin_mask"], cfg["crops"]
    skin_ranges = (skin_cfg["cr_range"], skin_cfg["cb_range"]) if skin_cfg["enabled"] else None

    subjects = args.subjects or ds.subjects()
    rows = []
    for subject in tqdm(subjects, desc="Extracting"):
        folder = out_root / subject
        if not args.overwrite and cached_hash(folder) == cfg_hash:
            continue
        gt = ds.ground_truth(subject)
        start = time.perf_counter()
        with VideoLandmarker(
            model,
            det_cfg["min_detection_confidence"],
            det_cfg["min_tracking_confidence"],
            det_cfg["landmark_smoothing"],
        ) as detector:
            track = extract_video(
                ds.frames(subject),
                gt.t,
                detector,
                cfg["rois"],
                skin_ranges=skin_ranges,
                crop_size=crop_cfg["size"] if crop_cfg["enabled"] else None,
                crop_scale=crop_cfg["scale"],
            )
        elapsed = time.perf_counter() - start
        save_track(track, folder, cfg_hash)
        video_s = track.t[-1] - track.t[0]
        rows.append(
            {
                "subject": subject,
                "frames": len(track.t),
                "detection_rate": track.detection_rate,
                "video_s": video_s,
                "processing_s": elapsed,
                "processing_s_per_video_min": 60 * elapsed / video_s,
                "config_hash": cfg_hash,
            }
        )

    if not rows:
        print("Nothing to do: all subjects are cached with the current config.")
        return
    summary_path = out_root / "extraction_summary.csv"
    summary = pd.DataFrame(rows).set_index("subject")
    if summary_path.is_file():
        previous = pd.read_csv(summary_path, index_col="subject")
        summary = pd.concat([previous.drop(summary.index, errors="ignore"), summary])
    summary.sort_index(key=lambda s: s.str.extract(r"(\d+)")[0].astype(int)).to_csv(summary_path)
    print(summary.loc[[r["subject"] for r in rows]].round(3).to_string())
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
