import asyncio
import argparse
import os
import sys
from pathlib import Path
import logging
import random

import numpy as np
import yaml
import websockets

# ── Robosuite log path patch (must run before libero imports) ──────
CUSTOM_LOG_DIR = Path("/mnt/data/sftp/data/quangpt3/Evo-1/Evo_1/logs")
CUSTOM_LOG_DIR.mkdir(parents=True, exist_ok=True)

_original_file_handler = logging.FileHandler

def _patched_file_handler(filename, mode='a', encoding=None, delay=False):
    if filename == "/tmp/robosuite.log":
        filename = str(CUSTOM_LOG_DIR / "robosuite.log")
    return _original_file_handler(filename, mode, encoding, delay)

logging.FileHandler = _patched_file_handler

os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evo.evaluation.benchmarks.libero import LiberoBenchmark



def setup_logger(log_dir: str, ckpt_name: str) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("libero_client")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(
            str(log_path / f"{ckpt_name}.txt"), mode="a"
        )
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def load_config(config_path: str = None) -> dict:
    """Load config from YAML or return defaults."""
    defaults = {
        "evaluation": {
            "horizon": 14,
            "max_steps": [25, 25, 25, 95],
            "task_suites": [
                "libero_spatial",
                "libero_object",
                "libero_goal",
                "libero_10",
            ],
            "num_episodes": 1,
            "seed": 42,
            "camera_res": 448,
        },
        "server": {"url": "ws://0.0.0.0:9000"},
        "checkpoint": {"name": "Evo1_libero_all"},
        "logging": {
            "log_dir": "./log_file",
            "video_dir": "./video_log_file",
        },
    }

    if config_path:
        p = Path(config_path)
        config_file = p if p.is_absolute() else (Path(__file__).resolve().parent.parent / p)
    else:
        config_file = None

    if config_file and config_file.exists():
        with config_file.open("r") as f:
            user_cfg = yaml.safe_load(f) or {}

        for key, value in user_cfg.items():
            if key in defaults and isinstance(value, dict) and isinstance(defaults[key], dict):
                defaults[key].update(value)
            else:
                defaults[key] = value

    return defaults

async def main():
    parser = argparse.ArgumentParser(description="LIBERO evaluation client")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval/libero.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg["evaluation"]
    server_cfg = cfg["server"]
    log_cfg = cfg["logging"]
    ckpt_cfg = cfg["checkpoint"]
    debug_cfg = cfg["debug"]
    np.random.seed(eval_cfg["seed"])
    random.seed(eval_cfg["seed"])

    logger = setup_logger(log_cfg["log_dir"], ckpt_cfg["name"])

    bench = LiberoBenchmark(
        horizon=eval_cfg["horizon"],
        camera_res=eval_cfg["camera_res"],
        seed=eval_cfg["seed"],
        num_episodes=eval_cfg["num_episodes"],
        video_dir=log_cfg["video_dir"],
        ckpt_name=ckpt_cfg["name"],
        logger=logger,
        debug=debug_cfg,
    )

    async with websockets.connect(server_cfg["url"]) as ws:
        for suite_name, max_steps in zip(
            eval_cfg["task_suites"], eval_cfg["max_steps"]
        ):
            await bench.run_suite(suite_name, max_steps, ws)


if __name__ == "__main__":
    asyncio.run(main())