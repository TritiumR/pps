"""Capture observations, call the container proxy, and expose steering hooks."""
from __future__ import annotations

import base64
import io
import json
import subprocess
import time

import numpy as np
import torch

from .. import container, paths
paths.ensure_repo_on_path()

from sim_free_mpc.ddim import ddim_iteration_alphas
from sim_free_mpc.planner import task_tilt_weight

_CONTAINER_OPENPI = container.to_container(paths.REPO / "openpi")
_CONTAINER_PYTHONPATH = f"{_CONTAINER_OPENPI}/src:{container.to_container(paths.REPO)}"
_OPEN_APERTURE = 0.080
_RHO_SKIP = 1e-3


TASK_PROMPTS = {
    "stack": "stack the red block on the green block",
    "can": "put the can in the bin",
}


def host_to_container(path: str) -> str:
    """Map a host checkpoint path into the container."""
    return container.to_container(path)


def _array_to_b64(array: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(array))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_array(payload: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False)


class ProxyScoreClient:
    """Manage the persistent proxy score service over JSON lines."""

    def __init__(self, checkpoint, prompt, config="score_task_capsule", device="cpu",
                 threads=8, container=container.NAME, ready_timeout_s=900,
                 prediction_mode="config", kv_cache=False, fp16=False):
        self.checkpoint = host_to_container(checkpoint)
        self.prompt = prompt
        self.config = config
        self.prediction_mode = prediction_mode

        self.kv_cache = bool(kv_cache)
        self.fp16 = bool(fp16)
        self.device = device
        self.threads = threads
        self.container = container
        self.ready_timeout_s = ready_timeout_s
        self.ready_info = None
        self._proc = None

    def start(self):
        server_device = "cpu" if self.device == "cpu" else "cuda"
        inner = (
            f"cd {_CONTAINER_OPENPI} && "
            f"PYTHONPATH={_CONTAINER_PYTHONPATH} PYTHONUNBUFFERED=1 "
            f"exec {container.PYTHON} scripts/serve_mg_proxy_score.py "
            f"--config {self.config} --checkpoint {self.checkpoint} "
            f"--prediction_mode {self.prediction_mode} "
            f"{'--kv_cache ' if self.kv_cache else ''}"
            f"{'--fp16 ' if self.fp16 else ''}"
            f"--device {server_device} --threads {self.threads} "
            f"--prompt \"{self.prompt}\""
        )
        cuda = "" if self.device == "cpu" else self.device_visible()
        cmd = ["docker", "exec", "-i", "-e", f"CUDA_VISIBLE_DEVICES={cuda}",


               "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
               self.container, "bash", "-c", inner]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.ready_info = self._read_reply(timeout_s=self.ready_timeout_s)
        if self.ready_info.get("kind") != "ready":
            raise RuntimeError(f"Proxy server failed to start: {self.ready_info}")
        print(f"[proxy] server ready: {json.dumps(self.ready_info)}", flush=True)
        return self

    def device_visible(self):
        """Return the CUDA_VISIBLE_DEVICES value for the configured device."""
        if self.device.startswith("cuda:"):
            return self.device.split(":", 1)[1]
        return self.device

    def _read_reply(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(f"Proxy server exited (rc={self._proc.returncode}).")
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("Proxy server closed stdout.")
            if line.startswith("MGPX "):
                return json.loads(line[len("MGPX "):])
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for the proxy server reply.")

    def _rpc(self, req, timeout_s=300):
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        reply = self._read_reply(timeout_s)
        if reply.get("kind") == "error":
            raise RuntimeError(f"Proxy server error: {reply['error']}")
        return reply

    def chain(self, *, seed, num_iterations, joint_pos, gripper_pos, table, wrist):
        """Run one proxy reverse chain and return clean action predictions."""
        reply = self._rpc({
            "cmd": "chain", "seed": int(seed), "num_iterations": int(num_iterations),
            "joint_pos": [float(v) for v in np.asarray(joint_pos).reshape(-1)[:7]],
            "gripper_pos": float(gripper_pos),
            "table_b64": _array_to_b64(table), "wrist_b64": _array_to_b64(wrist)})
        return _b64_to_array(reply["x0_real_b64"]), float(reply.get("wall_s", np.nan))

    def embed(self, *, joint_pos, gripper_pos, table, wrist):
        """Cache the observation prefix and return its session token."""
        reply = self._rpc({
            "cmd": "embed",
            "joint_pos": [float(v) for v in np.asarray(joint_pos).reshape(-1)[:7]],
            "gripper_pos": float(gripper_pos),
            "table_b64": _array_to_b64(table), "wrist_b64": _array_to_b64(wrist)})
        return reply["obs"], float(reply.get("wall_s", np.nan))

    def score_batch(self, *, obs, iteration, num_iterations, x):
        """Score a batch of noisy proxy-space action chunks."""
        reply = self._rpc({
            "cmd": "score_batch", "obs": obs, "iteration": int(iteration),
            "num_iterations": int(num_iterations),
            "x_b64": _array_to_b64(np.asarray(x, dtype=np.float32))})
        return _b64_to_array(reply["score_b64"]), float(reply.get("wall_s", np.nan))

    def close(self):
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()
        self._proc = None


class ProxySteering:
    """Provide per-level proxy action injections for one rollout."""

    def __init__(self, mode, client, *, rho=0.15, schedule="flat", ddim_train_timesteps=100,
                 horizon=8, base_seed=0):
        if mode not in ("inject", "proxy_only", "expert", "select", "verify", "tilt"):
            raise ValueError(f"mode must be inject|proxy_only|expert|select, got {mode!r}")
        self.mode = mode
        self.client = client
        self.rho = float(rho)
        self.schedule = schedule
        self.ddim_train_timesteps = int(ddim_train_timesteps)
        self.horizon = int(horizon)
        self.base_seed = int(base_seed)
        self.replan_idx = 0
        self.last_wall_s = np.nan
        self.last_server_s = np.nan
        self._x0_real = None
        self._q0 = None

    def begin_replan(self, env, num_iterations):
        """Capture observations and fetch the proxy chain for one replan."""
        table = env.rgb("agentview", hw=224)
        wrist = env.rgb("robot0_eye_in_hand", hw=224)
        q0 = env.q0().numpy().astype(np.float32)
        gripper = float(np.clip(env.gripper_q() / _OPEN_APERTURE, 0.0, 1.0))
        t0 = time.perf_counter()
        self._x0_real, self.last_server_s = self.client.chain(
            seed=self.base_seed * 100003 + self.replan_idx,
            num_iterations=num_iterations,
            joint_pos=q0, gripper_pos=gripper, table=table, wrist=wrist)
        self.last_wall_s = time.perf_counter() - t0
        self._q0 = q0
        self.replan_idx += 1
        if self._x0_real.shape[0] != num_iterations or self._x0_real.shape[1] < self.horizon:
            raise ValueError(f"Proxy chain shape {self._x0_real.shape} does not cover "
                             f"{num_iterations} levels x horizon {self.horizon}.")

    def expert_chunk(self):
        """Return the proxy's final clean action chunk."""
        return np.array(self._x0_real[-1][: self.horizon], dtype=np.float32, copy=True)

    def inject_for(self, iteration, num_iterations, policy):
        """Build the planner injection for one denoising level."""
        rho = 1.0 if self.mode == "proxy_only" else self.rho
        if self.mode == "inject" and self.schedule == "frontload":
            alpha, _ = ddim_iteration_alphas(
                iteration=iteration, num_iterations=num_iterations,
                num_train_timesteps=self.ddim_train_timesteps)
            rho *= max(1.0 - float(alpha), 1e-6)
        if rho < _RHO_SKIP:
            return None
        x0 = self._to_model_space(self._x0_real[iteration][: self.horizon], policy)
        return {"x0": torch.as_tensor(x0, dtype=torch.float32), "rho": float(rho)}

    def tilt_for(self, iteration, num_iterations, policy, lam, temperature, noise,
                 dims=7, ess_cap=None, discrimination=False):
        """Gaussian tilt of the softmax toward the proxy's clean action, for one denoise level.

        Reweights the base's OWN candidates instead of adding foreign ones, so it never has to win
        a cost comparison -- the failure that gives injected candidates zero weight. The weight is
        SNR-tempered: muted at high noise where the implied clean target is unreliable.
        """
        alpha, _ = ddim_iteration_alphas(
            iteration=iteration, num_iterations=num_iterations,
            num_train_timesteps=self.ddim_train_timesteps)
        weight = task_tilt_weight(lam, temperature, noise, float(alpha))
        if weight <= 0.0:
            return None
        target = self._to_model_space(self._x0_real[iteration][: self.horizon], policy)
        out = {"target": torch.as_tensor(target, dtype=torch.float32),
               "weight": float(weight), "dims": dims}
        if ess_cap:
            out["ess_cap"] = float(ess_cap)
        if discrimination:
            out["discrimination"] = True
        return out

    def _to_model_space(self, x0_real, policy):
        """Convert proxy real actions into planner model space."""
        md = policy._metadata
        if md.get("use_quantile_norm", False):
            raise ValueError("Eval decode policy unexpectedly uses quantile norm.")
        stats = md["output_norm_stats"]["actions"]
        mean = np.asarray(stats.mean, dtype=np.float32)
        std = np.asarray(stats.std, dtype=np.float32)
        real = np.array(x0_real, dtype=np.float32, copy=True)
        real[:, :7] -= self._q0[None, :7]
        return (real - np.atleast_2d(mean)) / (np.atleast_2d(std) + 1e-6)


class AdditiveScoreSteering:
    """Blend the proxy score field into the planner score."""

    mode = "additive"

    def __init__(self, client, *, gamma, policy, horizon, mass_keep=0.9999, score_cap=0, seed=0,
                 at="candidates", ref="none"):
        self.client = client
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.mass_keep = float(mass_keep)
        self.score_cap = int(score_cap)
        self.at = str(at)
        self.ref = str(ref)
        self._rng = np.random.default_rng(seed)
        info = client.ready_info
        self.proxy_horizon = int(info["action_horizon"])
        dim = int(info["action_dim"])
        md = policy._metadata
        if md.get("use_quantile_norm", False):
            raise ValueError("Eval decode policy unexpectedly uses quantile norm.")
        stats = md["output_norm_stats"]["actions"]
        self._mean = np.asarray(stats.mean, dtype=np.float32)[..., :dim]
        self._std = np.asarray(stats.std, dtype=np.float32)[..., :dim] + 1e-6

        # Both bridges are affine, so the chain rule is one slope d(proxy)/d(planner): flat [D]
        # under droid_quantile, per-row [H, D] under demo_delta.
        self.action_norm = info.get("action_norm", "droid_quantile")
        if self.action_norm == "droid_quantile":
            q01 = np.asarray(info["action_q01"], dtype=np.float32)[:dim]
            q99 = np.asarray(info["action_q99"], dtype=np.float32)[:dim]
            self._off = q01
            self._scale = (q99 - q01 + 1e-6) / 2.0
            self._bias = -1.0
        elif self.action_norm == "demo_delta":
            self._off = np.asarray(info["action_mean_rows"], dtype=np.float32)[:, :dim]
            self._scale = np.asarray(info["action_std_rows"], dtype=np.float32)[:, :dim] + 1e-6
            self._bias = 0.0
        else:
            raise ValueError(f"Additive steering cannot bridge action_norm={self.action_norm!r}.")
        # Both may be per-row [H, D] with DIFFERENT H: the planner's is sliced to --horizon,
        # the proxy ships its full action_horizon. Align before dividing.
        if self._std.ndim == 2 and self._scale.ndim == 2:
            n = min(self._std.shape[0], self._scale.shape[0])
            self._slope = self._std[:n] / self._scale[:n]
        else:
            self._slope = self._std / self._scale
        self._obs = None
        self.last_embed_s = np.nan
        self.level_trace = []
        self.replan_idx = 0

    def begin_replan(self, env, num_iterations):
        """Capture observations and fetch the proxy chain for one replan."""
        del num_iterations
        table = env.rgb("agentview", hw=224)
        wrist = env.rgb("robot0_eye_in_hand", hw=224)
        q0 = env.q0().numpy().astype(np.float32)
        gripper = float(np.clip(env.gripper_q() / _OPEN_APERTURE, 0.0, 1.0))
        t0 = time.perf_counter()
        self._obs, _ = self.client.embed(
            joint_pos=q0, gripper_pos=gripper, table=table, wrist=wrist)
        self.last_embed_s = time.perf_counter() - t0
        self.level_trace = []
        self.replan_idx += 1

    def _rows(self, a, n):
        """Align a bridge array to n chunk rows; flat [D] arrays broadcast as they are."""
        if a.ndim == 1:
            return a
        if a.shape[0] < n:
            raise ValueError(f"Bridge covers {a.shape[0]} rows, need {n}.")
        return a[:n]

    def _to_proxy_space(self, x):
        """Convert planner model samples into proxy action space."""
        real_delta = x * self._std + self._mean
        n = x.shape[1]
        return (real_delta - self._rows(self._off, n)) / self._rows(self._scale, n) + self._bias

    def score_addend(self, *, samples, weights, x_t, base_score,
                     iteration, num_iterations, alpha_bar):
        """Return the proxy-derived score addend for one denoising level.

        at="xt" is the PPS form: score(x_t,t) = base(x_t,t) + gamma * proxy(x_t,t), the proxy read
        at the CURRENT iterate, one chunk, batch 1.

        at="candidates" (the original here) instead scores the planner's candidate population and
        weight-averages it. That is a different estimator and it costs ~1000x the batch -- the sole
        reason additive runs slower per env step than full Isaac Sim. It is also arguably fed
        off-distribution inputs: the candidates are clean x0 proposals, while a score model
        conditioned on level t expects x_t-distributed chunks.
        """
        del alpha_bar
        if self.at == "xt":
            x = x_t.detach().cpu().numpy().astype(np.float32)
            w = np.ones(x.shape[0], dtype=np.float32) / x.shape[0]
        else:
            x = samples.detach().cpu().numpy().astype(np.float32)
            w = weights.detach().cpu().numpy().astype(np.float32)
        rows = x.shape[1]
        x = x[:, : self.horizon]               # --kp appends waypoint rows; not proxy actions
        if self.at != "xt" and self.score_cap and x.shape[0] > self.score_cap:
            # The addend is sum_i w_i s_i, a weight-weighted mean. Drawing score_cap indices with
            # probability w makes the PLAIN mean over the draw an unbiased estimate of it, so the
            # cost of a level stops scaling with the candidate count. mass_keep does not bound
            # this: at low noise the weights flatten and it keeps essentially every candidate,
            # which is what makes additive cost 11 full forward passes a replan.
            idx = self._rng.choice(x.shape[0], size=self.score_cap, replace=True, p=w)
            x = x[idx]
            w = np.full(self.score_cap, 1.0 / self.score_cap, dtype=np.float32)
        if self.at != "xt" and self.mass_keep < 1.0:
            order = np.argsort(-w)
            keep = int(np.searchsorted(np.cumsum(w[order]), self.mass_keep) + 1)
            idx = order[: min(keep, w.shape[0])]
            x, w = x[idx], w[idx] / max(float(w[idx].sum()), 1e-12)
        y = self._to_proxy_space(x)
        if y.shape[1] < self.proxy_horizon:

            pad = np.repeat(y[:, -1:, :], self.proxy_horizon - y.shape[1], axis=1)
            y = np.concatenate([y, pad], axis=1)
        t0 = time.perf_counter()
        s_proxy, server_s = self.client.score_batch(
            obs=self._obs, iteration=iteration, num_iterations=num_iterations, x=y)
        wall_s = time.perf_counter() - t0
        slope = self._rows(self._slope, x.shape[1])
        if slope.ndim == 1:
            slope = slope[None, :]                 # one flat scale, broadcast over rows
        s_planner = s_proxy[:, : x.shape[1], :] * slope[None]
        addend = self.gamma * (w[:, None, None] * s_planner).sum(axis=0)
        if self.ref == "base":
            # PPS Eq (4) with v_ref := v_base, i.e. Eq (3): (1-gamma)*base + gamma*task.
            # A convex interpolation, bounded at every gamma -- unlike base + gamma*task,
            # whose magnitude grows with gamma and went NaN at the paper's own optimum.
            b = base_score.detach().cpu().numpy().astype(np.float32)[0]
            addend = addend - self.gamma * b[: addend.shape[0], : addend.shape[1]]
        if rows > addend.shape[0]:             # waypoint rows are the cost's, not the proxy's
            addend = np.pad(addend, ((0, rows - addend.shape[0]), (0, 0)))
        base_norm = float(torch.linalg.vector_norm(base_score.detach()).cpu())
        add_norm = float(np.linalg.norm(addend))
        self.level_trace.append({
            "it": int(iteration),
            "n_scored": int(x.shape[0]),
            "base_norm": round(base_norm, 4),
            "add_norm": round(add_norm, 4),
            "ratio": round(add_norm / max(base_norm, 1e-9), 4),
            "wall_s": round(wall_s, 3),
            "server_s": round(server_s, 3)})
        return torch.as_tensor(addend, dtype=torch.float32)
