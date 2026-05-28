#!/usr/bin/env python3

from __future__ import annotations

import argparse

import numpy as np
import torch

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("onnxruntime is required. Install it before running this script.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate consistency between exported TorchScript and ONNX policies.")
    parser.add_argument("--torch_policy", type=str, required=True, help="Path to exported policy.pt")
    parser.add_argument("--onnx_policy", type=str, required=True, help="Path to exported policy.onnx")
    parser.add_argument("--obs_dim", type=int, default=30, help="Observation dimension.")
    parser.add_argument("--samples", type=int, default=1024, help="Random sample count for comparison.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for random observations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance.")
    return parser.parse_args()


def run_torch_policy(policy: torch.jit.ScriptModule, obs: np.ndarray) -> np.ndarray:
    obs_tensor = torch.from_numpy(obs).to(torch.float32)
    with torch.inference_mode():
        out = policy(obs_tensor)
    return out.detach().cpu().numpy()


def run_onnx_policy(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: obs.astype(np.float32, copy=False)})
    return outputs[0]


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = torch.jit.load(args.torch_policy, map_location="cpu")
    policy.eval()

    session = ort.InferenceSession(args.onnx_policy, providers=["CPUExecutionProvider"])

    max_abs_error = 0.0
    mean_abs_error = 0.0
    checked = 0

    for start in range(0, args.samples, args.batch_size):
        count = min(args.batch_size, args.samples - start)
        obs = np.random.randn(count, args.obs_dim).astype(np.float32)
        out_torch = run_torch_policy(policy, obs)
        out_onnx = run_onnx_policy(session, obs)

        abs_err = np.abs(out_torch - out_onnx)
        max_abs_error = max(max_abs_error, float(abs_err.max()))
        mean_abs_error += float(abs_err.mean()) * count
        checked += count

        if not np.allclose(out_torch, out_onnx, atol=args.atol, rtol=args.rtol):
            raise SystemExit(
                f"Validation failed at batch starting {start}: max_abs_err={float(abs_err.max()):.6e}, "
                f"mean_abs_err={float(abs_err.mean()):.6e}"
            )

    mean_abs_error /= max(checked, 1)
    print(f"[INFO] Validation passed: samples={checked}")
    print(f"[INFO] max_abs_err={max_abs_error:.6e}, mean_abs_err={mean_abs_error:.6e}")


if __name__ == "__main__":
    main()
