#!/usr/bin/env python3
"""bench_inference.py - measure inference compute & latency of the headline policy.

Reviewer item (ii): "inference compute and latency numbers on a UAV-class platform."

We do NOT have a Jetson/embedded board in this environment, so we measure on a single
x86 CPU thread (OMP/MKL pinned to 1) to *emulate* a UAV companion-computer core, and
report parameter count and per-inference MACs/FLOPs (computed analytically from the
network definition in shared_slam_policy.py). A conservative embedded projection is
reported alongside (see the paper's compute table).

No retraining and no env-var reward knobs are touched, so the headline policy behaviour
is unchanged. Outputs a JSON + a markdown table for the paper.

Usage
-----
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_STEPS_OVERRIDE=600 \
  .venv/bin/python lightweight_train/bench_inference.py \
    --model   lightweight_train/results_3d_shape_delta_800k_maxsteps600/models/slam3d_180000_steps.zip \
    --vecnorm lightweight_train/results_3d_shape_delta_800k_maxsteps600/models/slam3d_vecnormalize_180000_steps.pkl \
    --iters 2000 --warmup 200 --threads 1
"""
from __future__ import annotations

# --- thread pinning + episode horizon MUST be set before importing torch/env ---
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MAX_STEPS_OVERRIDE", "600")  # headline regime (600-step episodes)

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# numpy 2.x <-> 1.x pickle shim (mirrors gazebo_transfer_eval.py:44-52) - no-op on host.
if not hasattr(np, "_core"):  # pragma: no cover
    import numpy.core  # noqa: F401
    import numpy.core.multiarray  # noqa: F401
    import numpy.core.numeric  # noqa: F401
    import numpy.core.umath  # noqa: F401

    for _modname in list(sys.modules):
        if _modname == "numpy.core" or _modname.startswith("numpy.core."):
            sys.modules[_modname.replace("numpy.core", "numpy._core")] = sys.modules[_modname]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def analytic_macs() -> dict:
    """Per-inference multiply-accumulate counts, derived from shared_slam_policy.py.

    Conv MACs = C_out * H_out * W_out * C_in * K_h * K_w.
    Linear MACs = in * out.  LSTM(1 layer) MACs ~= 4 * (in*hidden + hidden*hidden).
    GroupNorm / ReLU / AdaptiveAvgPool contribute negligible MACs and are omitted.
    Map input is 5 channels; CoordConv appends 2 coordinate channels -> 7 effective.
    The predict() path uses the actor LSTM + policy (pi) head only (value head unused).
    """
    layers = {
        # CoordConv2d(5->32, k5, s2, p2):  48x48 -> 24x24, in=5+2 coord = 7
        "coordconv_5to32_k5s2": 32 * 24 * 24 * 7 * 25,
        # Conv2d(32->64, k3, s2, p1): 24x24 -> 12x12
        "conv_32to64_k3s2": 64 * 12 * 12 * 32 * 9,
        # Conv2d(64->64, k3, s2, p1): 12x12 -> 6x6
        "conv_64to64_k3s2": 64 * 6 * 6 * 64 * 9,
        # Conv2d(64->64, k3, s1, p1): 6x6 -> 6x6
        "conv_64to64_k3s1": 64 * 6 * 6 * 64 * 9,
        # scalar_net: Linear(10->64) + Linear(64->32)
        "scalar_mlp": 10 * 64 + 64 * 32,
        # combine: Linear(2304+32 -> 512) + Linear(512 -> 256)
        "combine_mlp": (2304 + 32) * 512 + 512 * 256,
        # actor LSTM(256 -> 128), 1 layer
        "lstm_actor": 4 * (256 * 128 + 128 * 128),
        # pi head MLP(128->256->256) + action mean Linear(256->3)
        "pi_head": 128 * 256 + 256 * 256 + 256 * 3,
    }
    total = int(sum(layers.values()))
    return {
        "per_layer_macs": {k: int(v) for k, v in layers.items()},
        "total_macs": total,
        "total_flops": 2 * total,
        "note": "MACs for the predict() path (actor LSTM + pi head); norm/relu/pool omitted.",
    }


def count_params(policy) -> dict:
    total = int(sum(p.numel() for p in policy.parameters()))
    fe = int(sum(p.numel() for p in policy.features_extractor.parameters()))
    return {"total_params": total, "feature_extractor_params": fe, "head_and_lstm_params": total - fe}


def summarize(latencies_ms: list[float]) -> dict:
    s = sorted(latencies_ms)
    n = len(s)

    def pct(p):
        return s[min(n - 1, int(round(p / 100.0 * n)))]

    mean = statistics.mean(s)
    return {
        "mean_ms": mean,
        "median_ms": statistics.median(s),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "min_ms": s[0],
        "max_ms": s[-1],
        "std_ms": statistics.pstdev(s),
        "throughput_hz": 1000.0 / mean if mean > 0 else float("inf"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=200, help="held-out map seed for the probe env")
    ap.add_argument("--out", default=None, help="output JSON path (default: <model_dir>/../summary/inference_benchmark.json)")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from env_3d import MAX_STEPS, AltitudeActiveSLAMEnv

    print(f"[bench] torch {torch.__version__}  threads={torch.get_num_threads()}  MAX_STEPS={MAX_STEPS}")

    env = DummyVecEnv([lambda: AltitudeActiveSLAMEnv(map_seed=args.seed, difficulty="hard")])
    env = VecNormalize.load(args.vecnorm, env)
    env.training = False
    env.norm_reward = False

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }
    model = RecurrentPPO.load(args.model, env=env, device="cpu", custom_objects=custom_objects)
    model.policy.set_training_mode(False)
    print("[bench] model loaded on CPU")

    params = count_params(model.policy)
    macs = analytic_macs()
    print(f"[bench] params: total={params['total_params']:,}  fe={params['feature_extractor_params']:,}")
    print(f"[bench] MACs/inference={macs['total_macs']:,}  FLOPs={macs['total_flops']:,}")

    # ---- Policy inference latency (single fixed obs, advancing LSTM state) ----
    obs = env.reset()
    lstm_states = None
    starts = np.ones((1,), dtype=bool)
    for _ in range(args.warmup):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=starts, deterministic=True)
        starts[:] = False
    lat_pred = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=starts, deterministic=True)
        lat_pred.append((time.perf_counter() - t0) * 1e3)
        starts[:] = False
    pred_stats = summarize(lat_pred)
    print(f"[bench] policy inference: mean={pred_stats['mean_ms']:.3f} ms  "
          f"p99={pred_stats['p99_ms']:.3f} ms  ~{pred_stats['throughput_hz']:.0f} Hz")

    # ---- Full control step (observation construction + policy inference) ----
    raw = AltitudeActiveSLAMEnv(map_seed=args.seed, difficulty="hard")
    raw.reset()
    obs2 = env.reset()
    lstm2 = None
    starts2 = np.ones((1,), dtype=bool)
    lat_step = []
    rng = np.random.default_rng(0)
    for _ in range(min(args.iters, MAX_STEPS - 1)):
        t0 = time.perf_counter()
        action, lstm2 = model.predict(obs2, state=lstm2, episode_start=starts2, deterministic=True)
        starts2[:] = False
        obs2, _, done, _ = env.step(action)
        lat_step.append((time.perf_counter() - t0) * 1e3)
        if done[0]:
            obs2 = env.reset()
            lstm2 = None
            starts2[:] = True
    step_stats = summarize(lat_step)
    print(f"[bench] full control step (predict+sense+obs): mean={step_stats['mean_ms']:.3f} ms  "
          f"~{step_stats['throughput_hz']:.0f} Hz")

    # ---- Observation construction only ----
    lat_obs = []
    raw.reset()
    for _ in range(min(args.iters, 2000)):
        a = rng.uniform(-1, 1, size=3).astype(np.float32)
        t0 = time.perf_counter()
        o, _r, term, trunc, _i = raw.step(a)
        lat_obs.append((time.perf_counter() - t0) * 1e3)
        if term or trunc:
            raw.reset()
    obs_stats = summarize(lat_obs)

    result = {
        "platform": {
            "cpu": platform.processor() or platform.machine(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": int(torch.get_num_threads()),
            "note": "single CPU thread - UAV companion-computer emulation (no CUDA/Jetson available here)",
        },
        "model": {
            "checkpoint": str(args.model),
            "policy_class": type(model.policy).__name__,
            **params,
        },
        "compute": macs,
        "latency_policy_inference_ms": pred_stats,
        "latency_full_control_step_ms": step_stats,
        "latency_obs_construction_ms": obs_stats,
        "iters": args.iters,
        "warmup": args.warmup,
    }

    out_path = Path(args.out) if args.out else (Path(args.model).resolve().parent.parent / "summary" / "inference_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[bench] wrote {out_path}")

    # markdown table for the paper
    md = [
        "| Metric | Value |",
        "|---|---|",
        f"| Total parameters | {params['total_params']/1e6:.2f} M |",
        f"| Feature-extractor parameters | {params['feature_extractor_params']/1e6:.2f} M |",
        f"| MACs / inference | {macs['total_macs']/1e6:.2f} M |",
        f"| FLOPs / inference | {macs['total_flops']/1e6:.2f} M |",
        f"| Policy inference latency (mean) | {pred_stats['mean_ms']:.2f} ms |",
        f"| Policy inference latency (p99) | {pred_stats['p99_ms']:.2f} ms |",
        f"| Policy inference throughput | {pred_stats['throughput_hz']:.0f} Hz |",
        f"| Full control step (predict+sense) | {step_stats['mean_ms']:.2f} ms ({step_stats['throughput_hz']:.0f} Hz) |",
    ]
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n")
    print(f"[bench] wrote {md_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
