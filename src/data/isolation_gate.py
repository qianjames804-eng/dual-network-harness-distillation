"""CLI gate required by formal runners before any model/trace operation."""
from __future__ import annotations
import argparse
from pathlib import Path
from src.data.splits import assert_manifest_disjoint
def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--manifest",type=Path,required=True); args=parser.parse_args()
    assert_manifest_disjoint(args.manifest); print(f"isolation gate passed: {args.manifest}")
if __name__ == "__main__": main()
