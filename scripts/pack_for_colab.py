"""Empaqueta lo que necesita el entrenamiento en Colab en un único zip.

Incluye, por sujeto, `crops.npz` (recortes de cara) y `ground_truth.txt` (PPG
de referencia). Pesa ~3 GB en lugar de los ~65 GB de video crudo, así que entra
en el Google Drive gratuito.

Uso:
    uv run python scripts/pack_for_colab.py            # -> data/colab/ubfc_crops.zip

Después, subir el zip a Google Drive (p. ej. a MyDrive/tesis/) y seguir
notebooks/03_train_physnet_colab.ipynb.
"""

import argparse
import zipfile
from pathlib import Path

from tqdm import tqdm

from rppg.data.ubfc import UBFCrPPG

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-root", default=ROOT / "data" / "raw", type=Path)
    parser.add_argument(
        "--processed-root", default=ROOT / "data" / "processed" / "ubfc_rppg", type=Path
    )
    parser.add_argument("--output", default=ROOT / "data" / "colab" / "ubfc_crops.zip", type=Path)
    args = parser.parse_args()

    subjects = UBFCrPPG(args.raw_root).subjects()
    missing = [s for s in subjects if not (args.processed_root / s / "crops.npz").is_file()]
    if missing:
        raise SystemExit(f"Missing crops for {missing}; run scripts/extract_traces.py first")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Los .npz ya están comprimidos: se guardan sin recomprimir (ZIP_STORED).
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as zf:
        for s in tqdm(subjects, desc="Packing"):
            zf.write(args.processed_root / s / "crops.npz", f"processed/ubfc_rppg/{s}/crops.npz")
            zf.write(args.raw_root / s / "ground_truth.txt", f"raw/{s}/ground_truth.txt")
    print(f"{args.output} ({args.output.stat().st_size / 1e9:.2f} GB, {len(subjects)} subjects)")


if __name__ == "__main__":
    main()
