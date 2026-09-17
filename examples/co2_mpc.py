"""Run from repo root: python -m examples.co2_mpc --target 50000 --duration 3600."""
import argparse
from src.config import Config
from src.co2_control import run_standalone

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=float, required=True, help='ppm')
    p.add_argument('--duration', type=float, required=True, help='seconds; 0 runs until stopped (profile must permit indefinite control)')
    p.add_argument('--log', default='co2-mpc.jsonl')
    args = p.parse_args()
    run_standalone(Config(), args.target, args.duration, args.log)
