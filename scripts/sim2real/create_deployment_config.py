#!/usr/bin/env python3

from __future__ import annotations

import argparse

from Isaaclab_alicia.deployment import DeploymentConfig, save_deployment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deployment config template for sim-to-real.")
    parser.add_argument(
        "--output",
        type=str,
        default="configs/sim2real/deployment.generated.yaml",
        help="Output path for deployment config.",
    )
    args = parser.parse_args()

    cfg = DeploymentConfig()
    save_deployment_config(cfg, args.output)
    print(f"[INFO] Deployment config saved to: {args.output}")


if __name__ == "__main__":
    main()
