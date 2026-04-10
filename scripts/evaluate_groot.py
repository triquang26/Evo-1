import argparse
from pathlib import Path
import sys

# Thêm đường dẫn src để import package nội bộ
workspace = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace))

from src.evo.evaluation.groot_evaluator import GrootEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate GR00T or GR00T-Mini using Isaac-GR00T official rollout")
    parser.add_argument("--config", type=str, default="configs/eval/groot.yaml",
                        help="Path to the evaluation config file")
    args = parser.parse_args()
    
    evaluator = GrootEvaluator(args.config)
    evaluator.run()

if __name__ == "__main__":
    main()

