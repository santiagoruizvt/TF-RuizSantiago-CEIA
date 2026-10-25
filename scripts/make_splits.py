"""Genera configs/splits.yaml (particiones por sujeto) con semilla fija.

Se corre una sola vez. El archivo resultante se versiona y es la fuente de
verdad de las particiones: no regenerarlo una vez que haya resultados, porque
cambiaría el conjunto de test.

Uso:
    uv run python scripts/make_splits.py --n-test 9 --n-val 5 --seed 2026
"""

import argparse
from pathlib import Path

import yaml

from rppg.data.ubfc import UBFCrPPG
from rppg.eval.splits import assert_disjoint, make_splits

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=ROOT / "data" / "raw", type=Path)
    parser.add_argument("--n-test", type=int, default=9)
    parser.add_argument("--n-val", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=ROOT / "configs" / "splits.yaml", type=Path)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; use --force to regenerate it")

    subjects = UBFCrPPG(args.data_root).subjects()
    splits = make_splits(subjects, args.n_test, args.n_val, args.seed)
    assert_disjoint(splits)

    header = (
        "# Particiones por sujeto de UBFC-rPPG (decisión 10.5). Generado por\n"
        f"# scripts/make_splits.py --n-test {args.n_test} --n-val {args.n_val} "
        f"--seed {args.seed}\n"
        "# No regenerar una vez que haya resultados: cambiaría el conjunto de test.\n"
    )
    body = {"dataset": "ubfc_rppg", "seed": args.seed, **splits}
    args.output.write_text(
        header + yaml.safe_dump(body, sort_keys=False, default_flow_style=None), encoding="utf-8"
    )
    for name in ("train", "val", "test"):
        print(f"{name:5s} ({len(splits[name]):2d}): {', '.join(splits[name])}")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
