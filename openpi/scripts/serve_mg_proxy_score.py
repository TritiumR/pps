"""MG proxy-score service: the container-side half of eval_mg's steering seam.

eval_mg (MuJoCo host env) cannot load ProxyScorePytorch: the model needs the patched
transformers_replace install plus the local dinov3 torch.hub package, both of which live only in
the pps container. This script runs there (docker exec -i, CPU or GPU) as a persistent process;
the host adapter (pps-mg/steering/mg_proxy.py) speaks a JSON-line protocol over stdin/stdout.

Per request the server runs the proxy's OWN reverse DDIM chain from Gaussian noise -- the exact
distribution its MPC score labels were generated on -- and returns the Tweedie clean action
x0 = (x_t + beta * score) / sqrt(alpha) of every level (the same x0 eval_steering's proposal
injection computes), unnormalized to REAL absolute joint actions. One round trip per replan.

Protocol (stdin request -> stdout reply, replies prefixed "MGPX " so Isaac-python chatter on
stdout is ignored; arrays travel as base64 .npy):
  {"cmd": "ping"} -> {"kind": "pong"}
  {"cmd": "chain", "seed": int, "num_iterations": int, "joint_pos": [7], "gripper_pos": float,
   "table_b64": u8[224,224,3], "wrist_b64": u8[224,224,3]}
      -> {"kind": "chain_ok", "x0_real_b64": f32[L,H,8], "wall_s": ...}
  {"cmd": "embed", <same obs fields as chain>}
      -> {"kind": "embed_ok", "obs": "<token>"}   # prefix cached for score_batch
  {"cmd": "score_batch", "obs": "<token>", "iteration": int, "num_iterations": int,
   "x_b64": f32[N,H,D] proxy-normalized noisy chunks}
      -> {"kind": "score_ok", "score_b64": f32[N,H,D]}   # score field per candidate
  {"cmd": "quit"} -> {"kind": "bye"}

The additive-steering path (eval_mg --steer additive) sends one embed per replan and one
score_batch per denoise level; the ready payload carries action_q01/action_q99 so the host can
invert the quantile bridge. The chain path is untouched.

Observation preprocessing is byte-identical to training: the raw sample takes the trainer's
_demo_sample schema and goes through the same _build_data_pipeline input transform.

Run (inside container pps-jeremysiburian, cwd /workspace/pps/_merge_wt/openpi):
  PYTHONPATH=src:/workspace/pps/_merge_wt /isaac-sim/python.sh scripts/serve_mg_proxy_score.py \
      --checkpoint /workspace/pps/data/mg_stack/checkpoints/score_task_stack/task_eps/5000 \
      --prompt 'stack the red block on the green block'
train-bc (x0-head) checkpoints need --prediction_mode x0; the default ("config") keeps the
registry config's prediction_type (epsilon for score_task_capsule), the current behavior.
A train-bc --action_norm demo_delta checkpoint carries an action_norm_stats.json (step dir or
run root); it is picked up automatically and used for BOTH the input transform and the chain's
unnormalize, so the round trip cannot drift from training. --action_norm_stats overrides it.

Speedup flags (both off by default, so the shipped numerics are untouched):
  --kv_cache   embed the image prefix K/V once per replan; each level/candidate then runs a
               16-token forward instead of a 408-token one (the expert is causal).
  --fp16       fp16 autocast on the chain/embed forwards (score_batch already autocasts).
Random-weights checkpoint for the dry-run gate (writes model.safetensors and exits):
  ... serve_mg_proxy_score.py --make_random_checkpoint <dir> --seed 0
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import os
import sys
import time

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENPI_DIR = os.path.dirname(_SCRIPTS_DIR)
_REPO_DIR = os.path.dirname(_OPENPI_DIR)
for _p in (_REPO_DIR, os.path.join(_OPENPI_DIR, "src"), _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax                    # noqa: E402
import numpy as np            # noqa: E402
import safetensors.torch      # noqa: E402
import torch                  # noqa: E402

from openpi.models import model as _model                              # noqa: E402
import openpi.models_pytorch.proxy_score_pytorch as _proxy_score       # noqa: E402
import openpi.training.config as _config                               # noqa: E402
from train_mpc_proxy_score_pytorch import (                            # noqa: E402
    _build_data_pipeline,
    find_action_norm_stats,
    load_action_norm_stats,
    unnormalize_chunk_actions,
)


def _emit(obj) -> None:
    sys.stdout.write("MGPX " + json.dumps(obj) + "\n")
    sys.stdout.flush()


def _b64_to_array(payload: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False)


def _array_to_b64(array: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, array)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _unnormalize_quantile(values: np.ndarray, stats) -> np.ndarray:
    """openpi transforms.Unnormalize quantile formula (stats sliced/passthrough as there)."""
    q01 = np.asarray(stats.q01, dtype=np.float32)
    q99 = np.asarray(stats.q99, dtype=np.float32)
    stats_dim = q01.shape[-1]
    data_dim = values.shape[-1]
    if stats_dim < data_dim:
        out = np.array(values, dtype=np.float32, copy=True)
        out[..., :stats_dim] = ((values[..., :stats_dim] + 1.0) / 2.0
                                * (q99 - q01 + 1e-6) + q01)
        return out
    q01, q99 = q01[..., :data_dim], q99[..., :data_dim]
    return ((values + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01).astype(np.float32)


class ProxyChainServer:
    """Loads the score proxy once; answers per-replan chain requests."""

    def __init__(self, args):
        config = _config.get_config(args.config)
        if args.prediction_mode != "config":
            # e.g. train-bc checkpoints predict x0 directly; predict_score_from_prefix then
            # returns the score whose Tweedie x0 is exactly the model output.
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(config.model, prediction_type=args.prediction_mode),
            )
        if getattr(args, "action_expert_variant", None):
            # train-bc --action_expert_variant changes the architecture, so the served
            # config has to be told about it or the state dict will not load.
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(
                    config.model, action_expert_variant=args.action_expert_variant),
            )
        self.model_config = config.model
        # A train-bc --action_norm demo_delta checkpoint ships its own [H, D] action scale;
        # serving MUST unnormalize with those exact numbers, so they are read from the
        # checkpoint rather than re-derived here.
        self.action_norm_stats, self.action_norm_path = (
            (load_action_norm_stats(args.action_norm_stats), args.action_norm_stats)
            if args.action_norm_stats
            else find_action_norm_stats(args.checkpoint)
        )
        if args.action_norm_stats and self.action_norm_stats is None:
            raise ValueError(f"No action norm stats at {args.action_norm_stats}.")
        data_config, self.input_transform = _build_data_pipeline(
            config, action_norm_stats=self.action_norm_stats
        )
        if data_config.norm_stats is None:
            raise ValueError("No norm stats resolved; cannot unnormalize actions.")
        if not data_config.use_quantile_norm:
            raise ValueError("Expected quantile norm (the trained proxies use it).")
        self.action_stats = data_config.norm_stats.get("actions")
        if self.action_stats is None and self.action_norm_stats is None:
            raise ValueError("No action stats available; cannot unnormalize actions.")
        self.prediction_mode = args.prediction_mode
        self.prompt = args.prompt
        self.device = torch.device(args.device)
        # Opt-in speedups; both default off so the shipped path is unchanged.
        self.fp16 = bool(getattr(args, "fp16", False))
        self.kv_cache = bool(getattr(args, "kv_cache", False))
        self.score_chunk = int(getattr(args, "score_chunk", 0) or 0)
        self.model = _proxy_score.ProxyScorePytorch(config.model)
        path = args.checkpoint
        if os.path.isdir(path):
            path = os.path.join(path, "model.safetensors")
        safetensors.torch.load_model(self.model, path, device="cpu")
        self.model = self.model.to(self.device, dtype=torch.float32).eval()
        self.checkpoint_path = path
        self._obs_cache = None      # (state, prefix_embs, prefix_pad_masks) for score_batch
        self._obs_token = None
        self._obs_serial = 0

    def _embed_observation(self, req: dict, *, kv_fp16: bool = False):
        """Input transform + prefix embedding for one host observation (chain/embed twins)."""
        raw = {
            "exterior_image_1_left": _b64_to_array(req["table_b64"]),
            "wrist_image_left": _b64_to_array(req["wrist_b64"]),
            "joint_position": np.asarray(req["joint_pos"], dtype=np.float32),
            "gripper_position": np.asarray([req["gripper_pos"]], dtype=np.float32),
            "actions": np.zeros(
                (self.model_config.action_horizon, self.model_config.action_dim),
                dtype=np.float32),
            "prompt": self.prompt,
        }
        inputs = self.input_transform(raw)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.asarray(x)).to(self.device)[None, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        amp = self.fp16 and self.device.type == "cuda"
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            images, img_masks, state = self.model._preprocess_observation(
                observation, train=False)
            prefix_embs, prefix_pad_masks, _ = self.model.embed_prefix(images, img_masks)
            # One 392-token prefix forward per replan; every level/candidate then costs
            # a 16-token forward instead of a 408-token one.
            prefix_kv = (
                self.model.prefix_kv_cache(
                    prefix_embs, prefix_pad_masks,
                    dtype=torch.float16 if kv_fp16 else None)
                if self.kv_cache else None)
        return state, prefix_embs, prefix_pad_masks, prefix_kv

    def embed(self, req: dict) -> dict:
        """Cache the observation prefix once per replan; score_batch reuses it per level."""
        # score_batch always autocasts on cuda, so its prefix K/V must be fp16 too.
        self._obs_cache = self._embed_observation(
            req, kv_fp16=self.device.type == "cuda")
        self._obs_token = str(self._obs_serial)
        self._obs_serial += 1
        return {"obs": self._obs_token}

    def score_batch(self, req: dict) -> dict:
        """Proxy score field of a batch of proxy-normalized noisy chunks at one DDIM level."""
        if self._obs_cache is None or req.get("obs") != self._obs_token:
            raise ValueError(f"unknown obs token {req.get('obs')!r}; send embed first")
        x = _b64_to_array(req["x_b64"]).astype(np.float32)
        horizon = self.model_config.action_horizon
        action_dim = self.model_config.action_dim
        if x.ndim != 3 or x.shape[1] != horizon or x.shape[2] != action_dim:
            raise ValueError(
                f"x_b64 must be [N, {horizon}, {action_dim}], got {list(x.shape)}")
        state, prefix_embs, prefix_pad_masks, prefix_kv = self._obs_cache
        x_t = torch.from_numpy(x).to(self.device)
        chunk = max(int(req.get("chunk", self.score_chunk or 256)), 1)
        # fp16 autocast: 2.4s -> 1.2s for 512x15x8 on the RTX 8000, rel err 1e-3 (measured);
        # a NEW command, so no precision-compat constraint with the chain path. chunk 256
        # halves the eager-attention peak vs 512 at ~1.9s -- the GPU is shared, stay modest.
        amp = self.device.type == "cuda"
        scores = []
        with torch.no_grad():
            _, _, time_cond = _proxy_score.ddim_iteration_alphas(
                iteration=int(req["iteration"]),
                num_iterations=int(req["num_iterations"]),
                num_train_timesteps=self.model_config.ddim_num_train_timesteps,
                device=self.device,
                dtype=x_t.dtype)
            for lo in range(0, x_t.shape[0], chunk):
                x_part = x_t[lo:lo + chunk]
                n = x_part.shape[0]
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    score = self.model.predict_score_from_prefix(
                        state.expand(n, -1),
                        prefix_embs.expand(n, -1, -1),
                        prefix_pad_masks.expand(n, -1),
                        x_part,
                        time_cond.expand(n),
                        prefix_kv=prefix_kv)
                scores.append(score.float().cpu())
        if amp and not self.kv_cache:
            # Varying batch sizes fragment the caching allocator (measured: 7-12 GB held
            # per server); return the blocks -- re-mallocs cost ms against a ~1 s forward.
            # With a cached prefix the transients are ~25x smaller: peak reserved stays at
            # 0.7 GB either way and the call costs ~12 ms per level, so it is skipped.
            torch.cuda.empty_cache()
        out = torch.cat(scores).numpy().astype(np.float32)
        return {"score_b64": _array_to_b64(out), "shape": list(out.shape)}

    def chain(self, req: dict) -> dict:
        state, prefix_embs, prefix_pad_masks, prefix_kv = self._embed_observation(
            req, kv_fp16=self.fp16 and self.device.type == "cuda")
        num_iterations = int(req.get("num_iterations", 11))
        generator = torch.Generator().manual_seed(int(req.get("seed", 0)))
        # Autocast covers the expert forward only; the DDIM update stays fp32.
        amp = self.fp16 and self.device.type == "cuda"
        with torch.no_grad():
            x_t = torch.randn(
                (1, self.model_config.action_horizon, self.model_config.action_dim),
                generator=generator).to(self.device)
            x0_levels = []
            for iteration in range(num_iterations):
                alpha, alpha_prev, time_cond = _proxy_score.ddim_iteration_alphas(
                    iteration=iteration,
                    num_iterations=num_iterations,
                    num_train_timesteps=self.model_config.ddim_num_train_timesteps,
                    device=self.device,
                    dtype=x_t.dtype)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    score = self.model.predict_score_from_prefix(
                        state, prefix_embs, prefix_pad_masks, x_t, time_cond.expand(1),
                        prefix_kv=prefix_kv)
                score = score.float()
                beta = torch.clamp(1.0 - alpha, min=1e-6)
                x0 = (x_t + beta * score) / torch.clamp(alpha, min=1e-6).sqrt()
                x0_levels.append(x0[0])
                if self.prediction_mode == "x0":
                    # train-bc noises with standard DDPM (x_t = sqrt(abar)*a + sqrt(beta)*eps),
                    # so its reverse chain is standard DDIM, not the MPC-label twin.
                    eps = -beta.sqrt() * score
                    x_t = (torch.clamp(alpha_prev, min=0.0).sqrt() * x0
                           + torch.clamp(1.0 - alpha_prev, min=0.0).sqrt() * eps)
                else:
                    # label-twin mbd_score update (planner.step_from_score), not DDIM
                    alpha_step = torch.clamp(alpha / torch.clamp(alpha_prev, min=1e-6), min=1e-6)
                    x_t = (x_t + beta * score) / alpha_step.sqrt()
        x0_norm = torch.stack(x0_levels).cpu().numpy().astype(np.float32)
        x0_real = (
            unnormalize_chunk_actions(x0_norm, self.action_norm_stats)
            if self.action_norm_stats is not None
            else _unnormalize_quantile(x0_norm, self.action_stats)
        )
        # model space is delta-from-q0 (DeltaActions mask): re-add q0 to honor
        # the absolute-actions return contract
        x0_real[..., :7] += np.asarray(req["joint_pos"], dtype=np.float32)[:7]
        return {"x0_real_b64": _array_to_b64(x0_real), "shape": list(x0_real.shape)}


def make_random_checkpoint(args) -> None:
    config = _config.get_config(args.config)
    torch.manual_seed(args.seed)
    model = _proxy_score.ProxyScorePytorch(config.model)
    out_dir = args.make_random_checkpoint
    os.makedirs(out_dir, exist_ok=True)
    safetensors.torch.save_model(model, os.path.join(out_dir, "model.safetensors"))
    torch.save({"global_step": 0, "random_init_seed": args.seed},
               os.path.join(out_dir, "metadata.pt"))
    print(f"random-init checkpoint -> {out_dir}", flush=True)


def serve(args) -> None:
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    server = ProxyChainServer(args)
    _emit({
        "kind": "ready",
        "checkpoint": server.checkpoint_path,
        "config": args.config,
        "prompt": args.prompt,
        "device": args.device,
        "action_horizon": server.model_config.action_horizon,
        "action_dim": server.model_config.action_dim,
        "prediction_type": server.model_config.prediction_type,
        "fp16": server.fp16,
        "kv_cache": server.kv_cache,
        "ddim_num_train_timesteps": server.model_config.ddim_num_train_timesteps,
        # Quantile bridge for the additive-steering host: the same stats chain/unnormalize use.
        # Under demo_delta the bridge is [H, D] affine, not flat quantiles, so the flat keys
        # are withheld and the host must read action_norm.
        "action_norm": "demo_delta" if server.action_norm_stats is not None else "droid_quantile",
        "action_norm_stats_path": (
            str(server.action_norm_path) if server.action_norm_stats is not None else None
        ),
        "action_q01": (
            None if server.action_norm_stats is not None
            else np.asarray(server.action_stats.q01, dtype=np.float32).tolist()
        ),
        "action_q99": (
            None if server.action_norm_stats is not None
            else np.asarray(server.action_stats.q99, dtype=np.float32).tolist()
        ),
        "action_mean_rows": (
            np.asarray(server.action_norm_stats["mean"], dtype=np.float32).tolist()
            if server.action_norm_stats is not None else None
        ),
        "action_std_rows": (
            np.asarray(server.action_norm_stats["std"], dtype=np.float32).tolist()
            if server.action_norm_stats is not None else None
        ),
        "load_s": round(time.perf_counter() - t0, 1),
    })
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _emit({"kind": "error", "error": "request is not valid JSON"})
            continue
        cmd = req.get("cmd")
        if cmd == "quit":
            _emit({"kind": "bye"})
            break
        if cmd == "ping":
            _emit({"kind": "pong"})
            continue
        handlers = {"chain": (server.chain, "chain_ok"),
                    "embed": (server.embed, "embed_ok"),
                    "score_batch": (server.score_batch, "score_ok")}
        if cmd in handlers:
            handler, ok_kind = handlers[cmd]
            t_req = time.perf_counter()
            try:
                reply = handler(req)
                reply["kind"] = ok_kind
                reply["wall_s"] = round(time.perf_counter() - t_req, 3)
            except Exception as exc:  # protocol survives a bad request
                reply = {"kind": "error", "error": f"{type(exc).__name__}: {exc}"}
            _emit(reply)
            continue
        _emit({"kind": "error", "error": f"unknown cmd {cmd!r}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="score_task_capsule",
                        help="ProxyScoreConfig registry name (score_task_* twins share it).")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint dir (or model.safetensors) to serve.")
    parser.add_argument("--prompt", default="",
                        help="Training prompt (tokenized for transform parity; the DINO "
                             "expert consumes images+state only).")
    parser.add_argument("--prediction_mode", default="config",
                        choices=("config", "score", "epsilon", "x0"),
                        help="Override the config's prediction_type for the served checkpoint "
                             "(default: keep the config's value). Use x0 for train-bc proxies.")
    parser.add_argument("--action_expert_variant", default=None,
                        help="Match a train-bc --action_expert_variant override (default: the "
                             "config's variant).")
    parser.add_argument("--action_norm_stats", default=None,
                        help="action_norm_stats.json to unnormalize with (default: the one "
                             "beside the checkpoint, else the config's pi05_droid quantiles).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fp16", action="store_true",
                        help="Run the chain/embed expert forwards under fp16 autocast "
                             "(score_batch already does). Off by default.")
    parser.add_argument("--kv_cache", action="store_true",
                        help="Cache the image prefix K/V once per replan so every DDIM level "
                             "and candidate is a 16-token forward. Off by default.")
    parser.add_argument("--score_chunk", type=int, default=0,
                        help="score_batch candidates per forward (0 = the 256 default).")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--make_random_checkpoint", default=None,
                        help="Write a random-init model.safetensors here and exit (dry-run gate).")
    args = parser.parse_args()
    if args.make_random_checkpoint:
        make_random_checkpoint(args)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required to serve.")
    serve(args)


if __name__ == "__main__":
    main()
