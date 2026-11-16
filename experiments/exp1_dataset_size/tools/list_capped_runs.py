"""
Jobs that continue the runs stopped by the step cap instead of by the learning-rate criterion: one line
"<n_data> <model_type> <param> <seed> <min_lr> <max_steps>" per run with stop_reason "max_steps", with the
new cap (default 6e6). Training resumes from checkpoints/<run>_last.pt. Used by run_all.sh.

Usage: python -m experiments.exp1_dataset_size.tools.list_capped_runs [new_cap] [--results DIR] > jobs_extend.txt
"""
import argparse
import json
from pathlib import Path

from ..train import RESULTS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("new_cap", nargs="?", default="6e6")
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args(argv)
    for path in sorted(Path(args.results).glob("*_seed*.json")):
        cfg = json.load(open(path))["config"]
        if cfg["stop_reason"] == "max_steps":
            print(f"{cfg['n_data']} {cfg['model_type']} {cfg['param']:g} {cfg['seed']} 1e-6 {args.new_cap}")


if __name__ == "__main__":
    main()
