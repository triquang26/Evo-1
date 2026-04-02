import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmarks")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation config YAML",
    )
    args = parser.parse_args()

    import yaml

    with Path(args.config).open("r") as f:
        cfg = yaml.safe_load(f)

    benchmark_name = cfg.get("evaluation", {}).get("benchmark", "libero")

    if benchmark_name == "libero":
        sys.argv = ["libero_client.py", "--config", args.config]
        from clients.libero_client import main as libero_main
        asyncio.run(libero_main())
    else:
        print(f"Unknown benchmark: {benchmark_name}")
        print("Available: libero")
        sys.exit(1)


if __name__ == "__main__":
    main()
