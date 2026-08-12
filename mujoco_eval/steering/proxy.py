"""Capture observations, call the container proxy, and expose steering hooks."""
from __future__ import annotations

import base64
import io
import json
import math
import os
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
    # sort_can is deliberately ABSENT. Its destination is named by a colour that is redrawn every
    # episode, so any fixed string here would be a wrong instruction half the time, supplied
    # silently by a default. The caller must pass the episode's own instruction (from
    # SortCanTwoBin.goal_prompt) via --proxy_prompt; runner.py raises when it does not.
}


def host_to_container(path: str) -> str:
    """Map a host checkpoint path into the container."""
    return container.to_container(path)


def norm_stats_mismatch(task_info, ref_info, *, tol=1e-6, max_report=6):
    """Report where a task and reference proxy disagree on their action normalisation.

    C1b. The old fingerprint compared the NAME ("demo_delta" == "demo_delta") and passed two
    checkpoints whose stats differ -- ref_square_v2 centres the gripper at 0.0 where the task
    proxy centres it at 0.5, and its arm scales differ by up to 4%. The name is not the
    coordinate system; the numbers are. Returns a list of human-readable offences, empty when
    the two bridges are the same map.
    """
    out = []
    tn, rn = task_info.get("action_norm", "?"), ref_info.get("action_norm", "?")
    if tn != rn:
        return [f"action_norm {tn!r} vs {rn!r}"]
    keys = (("action_mean_rows", "mean"), ("action_std_rows", "scale")) if tn == "demo_delta" else (
        ("action_q01", "q01"), ("action_q99", "q99"))
    for key, label in keys:
        a, b = task_info.get(key), ref_info.get(key)
        if a is None or b is None:
            out.append(f"{label}: missing on {'task' if a is None else 'reference'}")
            continue
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        n = tuple(min(i, j) for i, j in zip(a.shape, b.shape))
        sl = tuple(slice(0, i) for i in n)
        if a.shape != b.shape:
            out.append(f"{label}: shape {a.shape} vs {b.shape}")
        d = np.abs(a[sl] - b[sl])
        bad = np.argwhere(d > tol)
        for idx in bad[:max_report]:
            idx = tuple(int(i) for i in idx)
            out.append(f"{label}{list(idx)}: {a[sl][idx]:.6g} vs {b[sl][idx]:.6g}")
        if len(bad) > max_report:
            out.append(f"{label}: ... and {len(bad) - max_report} more dims")
    return out


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
                 prediction_mode="config", kv_cache=False, fp16=False,
                 local_process=False, checkpoint_is_container=False):
        self.checkpoint = str(checkpoint) if checkpoint_is_container else host_to_container(checkpoint)
        self.prompt = prompt
        self.config = config
        self.prediction_mode = prediction_mode

        self.kv_cache = bool(kv_cache)
        self.fp16 = bool(fp16)
        self.device = device
        self.threads = threads
        self.container = container
        self.ready_timeout_s = ready_timeout_s
        self.local_process = bool(local_process)
        self.ready_info = None
        self._proc = None

    def start(self):
        server_device = "cpu" if self.device == "cpu" else "cuda"
        openpi_root = str(paths.REPO / "openpi") if self.local_process else _CONTAINER_OPENPI
        pythonpath = (f"{openpi_root}/src:{paths.REPO}" if self.local_process
                      else _CONTAINER_PYTHONPATH)
        inner = (
            f"cd {openpi_root} && "
            f"PYTHONPATH={pythonpath} PYTHONUNBUFFERED=1 "
            f"exec {container.PYTHON} scripts/serve_mg_proxy_score.py "
            f"--config {self.config} --checkpoint {self.checkpoint} "
            f"--prediction_mode {self.prediction_mode} "
            f"{'--kv_cache ' if self.kv_cache else ''}"
            f"{'--fp16 ' if self.fp16 else ''}"
            f"--device {server_device} --threads {self.threads} "
            f"--prompt \"{self.prompt}\""
        )
        cuda = "" if self.device == "cpu" else self.device_visible()
        if self.local_process:
            cmd = ["bash", "-c", inner]
            proc_env = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda,
                            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        else:
            cmd = ["docker", "exec", "-i", "-e", f"CUDA_VISIBLE_DEVICES={cuda}",
                   "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                   self.container, "bash", "-c", inner]
            proc_env = None
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None if self.local_process else subprocess.DEVNULL,
            text=True, bufsize=1, env=proc_env)
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

    def chain(self, *, seed, num_iterations, joint_pos, gripper_pos, table, wrist, goal=None):
        """Run one proxy reverse chain and return clean action predictions.

        `goal` is the already-normalised conditioning vector for a goal-conditioned checkpoint
        (goal_dim > 0). None -- the default -- leaves the request byte-identical to what every
        existing arm sends.
        """
        req = {
            "cmd": "chain", "seed": int(seed), "num_iterations": int(num_iterations),
            "joint_pos": [float(v) for v in np.asarray(joint_pos).reshape(-1)[:7]],
            "gripper_pos": float(gripper_pos),
            "table_b64": _array_to_b64(table), "wrist_b64": _array_to_b64(wrist)}
        if goal is not None:
            req["goal"] = [float(v) for v in np.asarray(goal).reshape(-1)]
        reply = self._rpc(req)
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
        if mode not in ("inject", "proxy_only", "expert", "select", "verify", "tilt", "fk"):
            raise ValueError(f"mode must be inject|proxy_only|expert|select, got {mode!r}")
        self.mode = mode
        self.client = client
        self.rho = float(rho)
        self.schedule = schedule
        self.ddim_train_timesteps = int(ddim_train_timesteps)
        self.horizon = int(horizon)
        self.base_seed = int(base_seed)
        self.replan_idx = 0
        # Offset into the proxy's own seed stream. 0 leaves every existing arm byte-identical;
        # the supervisor advances it to ask the SAME policy for a fresh draw from its own
        # distribution at the same observation (steering/supervisor.py).
        self.retry_salt = 0
        self.last_wall_s = np.nan
        self.last_server_s = np.nan
        self._x0_real = None
        self._q0 = None

    def begin_replan(self, env, num_iterations, goal=None):
        """Capture observations and fetch the proxy chain for one replan.

        `goal` is the already-normalised conditioning vector for a goal-conditioned proxy. None --
        the default -- reproduces every existing arm exactly.
        """
        table = env.rgb("agentview", hw=224)
        wrist = env.rgb("robot0_eye_in_hand", hw=224)
        q0 = env.q0().numpy().astype(np.float32)
        gripper = float(np.clip(env.gripper_q() / _OPEN_APERTURE, 0.0, 1.0))
        t0 = time.perf_counter()
        self._x0_real, self.last_server_s = self.client.chain(
            seed=self.base_seed * 100003 + self.replan_idx + self.retry_salt,
            num_iterations=num_iterations,
            joint_pos=q0, gripper_pos=gripper, table=table, wrist=wrist, goal=goal)
        self.last_wall_s = time.perf_counter() - t0
        self._q0 = q0
        self.replan_idx += 1
        if self._x0_real.shape[0] != num_iterations or self._x0_real.shape[1] < self.horizon:
            raise ValueError(f"Proxy chain shape {self._x0_real.shape} does not cover "
                             f"{num_iterations} levels x horizon {self.horizon}.")

    def expert_chunk(self):
        """Return the proxy's final clean action chunk."""
        return np.array(self._x0_real[-1][: self.horizon], dtype=np.float32, copy=True)

    def _pad_to_proxy(self, y):
        """Extend a short chunk up to the proxy's own horizon, in proxy space.

        See H1: the trailing rows of an AWE/keypose proxy are GOALS, not actions. Filling them by
        repeating the last action row is off-distribution; the proxy's own predicted goal rows are
        the in-distribution choice and are already computed by begin_replan.
        """
        need = self.proxy_horizon - y.shape[1]
        if need <= 0:
            return y
        goals = self.goal_rows()
        if goals is not None and len(goals) >= need:
            block = np.asarray(goals[-need:], dtype=y.dtype)
            pad = np.repeat(block[None], y.shape[0], axis=0)
        else:
            pad = np.repeat(y[:, -1:, :], need, axis=1)
        return np.concatenate([y, pad], axis=1)

    def goal_rows(self):
        """Return the trailing GOAL rows: AWE waypoints then the keypose, or None.

        Trained with --awe_waypoints K --keypose_tail, the chunk is [actions | W1..WK | keypose].
        Everything past `horizon` is a goal rather than an action, so it is sliced off before
        execution and was, until now, simply discarded.
        """
        if self._x0_real is None or self._x0_real.shape[1] <= self.horizon:
            return None
        return np.array(self._x0_real[-1][self.horizon:], dtype=np.float32, copy=True)

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

    def fk_target(self, iteration, num_iterations, policy):
        """Return this level's proxy target in model space and its alpha_bar.

        Same target as `tilt_for`; the caller applies the strength, because FK standardizes the
        potential across particles first and so needs to scale it afterwards.
        """
        alpha, _ = ddim_iteration_alphas(
            iteration=iteration, num_iterations=num_iterations,
            num_train_timesteps=self.ddim_train_timesteps)
        target = self._to_model_space(self._x0_real[iteration][: self.horizon], policy)
        return torch.as_tensor(target, dtype=torch.float32), float(alpha)

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
                 at="candidates", ref="none", ref_client=None, last_level=None, aux="pad"):
        self.client = client
        # How the proxy's trailing GOAL rows are supplied; see `_score_native`.
        self.aux = str(aux)
        self._shadow = None
        self._aux_seed = int(seed)
        self.aux_trace = []
        # PPS's third model. `ref="base"` subtracts the BASE score, which collapses Eq (4) to a
        # convex blend; only a reference PROXY -- same architecture, same score space -- makes
        # base + gamma*(task - ref) unbounded above max(base, task), which is the whole point.
        self.ref_client = ref_client
        self.ref_horizon = (int(ref_client.ready_info["action_horizon"])
                            if ref_client is not None else 0)
        self._ref_obs = None
        # Highest denoising level that still receives the addend; None applies it everywhere.
        self.last_level = None if last_level is None else int(last_level)
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
        (self.action_norm, self._off, self._scale,
         self._bias, self._slope) = self._bridge_of(info, dim)
        # C1b, CRITICAL. The reference proxy is a SEPARATE checkpoint with its own
        # action_norm_stats, so it needs its own bridge: fed y normalised by the TASK stats and
        # converted back with the TASK slope, its score is a field over a different variable and
        # s_task - s_ref is not a difference of anything. Identical stats make this bridge
        # identical to the task's, so the correction is a no-op exactly where the old code was
        # already right. See agent_tests/_vfy_ref_bridge.py.
        if ref_client is not None:
            (self.ref_action_norm, self._ref_off, self._ref_scale,
             self._ref_bias, self._ref_slope) = self._bridge_of(ref_client.ready_info, dim)
        else:
            self.ref_action_norm = None
            self._ref_off = self._ref_scale = self._ref_slope = None
            self._ref_bias = 0.0
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
        if self.ref_client is not None:
            # Same observation, its own server: the two score fields must differ only in weights.
            self._ref_obs, _ = self.ref_client.embed(
                joint_pos=q0, gripper_pos=gripper, table=table, wrist=wrist)
        self.last_embed_s = time.perf_counter() - t0
        self.level_trace = []
        self.aux_trace = []
        if self.aux == "native":
            self._init_shadow()
        self.replan_idx += 1

    def _init_shadow(self):
        """Draw the shadow latent's noise init exactly as the proxy's own sampler does.

        serve_mg_proxy_score.chain() seeds a CPU torch.Generator and draws
        randn([1, action_horizon, action_dim]); reproducing that draw here is what lets the
        shadow's goal rows be compared against a native run bit-for-bit.
        """
        gen = torch.Generator().manual_seed(self._aux_seed * 100003 + self.replan_idx)
        dim = self._std.shape[-1]
        self._shadow = torch.randn((1, self.proxy_horizon, dim),
                                   generator=gen).numpy().astype(np.float32)

    def _bridge_of(self, info, dim):
        """Build one proxy's affine bridge from its own ready payload.

        Returns (action_norm, off, scale, bias, slope) where y = (x*std + mean - off)/scale + bias
        maps the planner's model space into THAT proxy's action space, and slope = d y / d x is the
        chain-rule factor that pulls its score back into planner space.
        """
        action_norm = info.get("action_norm", "droid_quantile")
        if action_norm == "droid_quantile":
            q01 = np.asarray(info["action_q01"], dtype=np.float32)[:dim]
            q99 = np.asarray(info["action_q99"], dtype=np.float32)[:dim]
            off, scale, bias = q01, (q99 - q01 + 1e-6) / 2.0, -1.0
        elif action_norm == "demo_delta":
            off = np.asarray(info["action_mean_rows"], dtype=np.float32)[:, :dim]
            scale = np.asarray(info["action_std_rows"], dtype=np.float32)[:, :dim] + 1e-6
            bias = 0.0
        else:
            raise ValueError(f"Additive steering cannot bridge action_norm={action_norm!r}.")
        # Both may be per-row [H, D] with DIFFERENT H: the planner's is sliced to --horizon,
        # the proxy ships its full action_horizon. Align before dividing.
        if self._std.ndim == 2 and scale.ndim == 2:
            n = min(self._std.shape[0], scale.shape[0])
            slope = self._std[:n] / scale[:n]
        else:
            slope = self._std / scale
        return action_norm, off, scale, bias, slope

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

    def _to_ref_space(self, x):
        """Convert planner model samples into the REFERENCE proxy's action space."""
        if self._ref_scale is None:
            raise ValueError("No reference proxy is loaded.")
        real_delta = x * self._std + self._mean
        n = x.shape[1]
        return ((real_delta - self._rows(self._ref_off, n)) / self._rows(self._ref_scale, n)
                + self._ref_bias)

    def _to_planner_space(self, y):
        """Inverse of `_to_proxy_space`; needed to run the proxy's own chain in planner space."""
        n = y.shape[1]
        real_delta = (y - self._bias) * self._rows(self._scale, n) + self._rows(self._off, n)
        return (real_delta - self._mean) / self._std

    def _score_native(self, y_act, x_t, iteration, num_iterations):
        """Score on a NATIVE 21-row latent instead of padding the goal block.

        The proxy's chunk is [H actions | K AWE waypoints | keypose]. `_pad_to_proxy` fills those
        trailing GOAL rows by repeating the last ACTION row, which is unlike anything in training
        (H1) -- so the additive arms were reading the model off-distribution in exactly the block
        that encodes where the task is going.

        Here a SHADOW latent carries the goal rows on the proxy's own reverse chain: initialised
        from its own Gaussian noise (`_init_shadow`) and advanced every level by its own DDIM
        update (`_advance_shadow`), with the shadow's ACTION rows tracking the base's iterate x_t.
        The candidates are then scored against that same goal block, so every input the model sees
        is a legitimate 21-row state. Only the first `horizon` score rows leave this method; the
        goal rows steer nothing and can never be executed.
        """
        aux = self._shadow[:, self.horizon:]                     # this level's goal block
        xt_act = self._to_proxy_space(
            x_t.detach().cpu().numpy().astype(np.float32)[:, : self.horizon])
        head = np.concatenate([xt_act, aux], axis=1)             # [1, P, D] shadow input
        body = np.concatenate(
            [y_act, np.repeat(aux, y_act.shape[0], axis=0)], axis=1)      # [B, P, D]
        batch = np.concatenate([head, body], axis=0).astype(np.float32)
        # Requirement: a valid native-width state at every level, asserted rather than assumed.
        if batch.shape[1] != self.proxy_horizon or batch.shape[2] != self._std.shape[-1]:
            raise ValueError(f"native aux built a {list(batch.shape)} batch; "
                             f"expected [*, {self.proxy_horizon}, {self._std.shape[-1]}]")
        s_all, server_s = self.client.score_batch(
            obs=self._obs, iteration=iteration, num_iterations=num_iterations, x=batch)
        s_all = np.asarray(s_all, dtype=np.float32)
        self._advance_shadow(head, s_all[:1], iteration, num_iterations)
        return body, s_all[1:], server_s

    def _advance_shadow(self, y_t, score, iteration, num_iterations):
        """One DDIM x0-prediction step of the shadow chain -- the server's own chain update.

        Mirrors serve_mg_proxy_score.chain()'s prediction_mode="x0" branch term for term, so a
        shadow whose action rows equal a native run's reproduces that run's goal rows exactly.
        """
        alpha, alpha_prev = ddim_iteration_alphas(
            iteration=iteration, num_iterations=num_iterations,
            num_train_timesteps=int(self.client.ready_info.get("ddim_num_train_timesteps", 100)))
        beta = max(1.0 - float(alpha), 1e-6)
        x0 = (y_t + beta * score) / math.sqrt(max(float(alpha), 1e-6))
        eps = -math.sqrt(beta) * score
        self._shadow = (math.sqrt(max(float(alpha_prev), 0.0)) * x0
                        + math.sqrt(max(1.0 - float(alpha_prev), 0.0)) * eps).astype(np.float32)
        self.aux_trace.append({
            "it": int(iteration), "rows": int(y_t.shape[1]),
            "aux_rows": int(self.proxy_horizon - self.horizon),
            "aux_norm": round(float(np.linalg.norm(self._shadow[:, self.horizon:])), 4),
            "aux_x0_norm": round(float(np.linalg.norm(x0[:, self.horizon:])), 4)})

    def policy_pair_step(self, x_t, iteration, num_iterations):
        """One DDIM step under a CONVEX BLEND of two LEARNED score fields.

        The control for Cory's symmetry claim: `(1-g)*s_A + g*s_B` is a product of experts and is
        symmetric in A and B -- but only if both operands are score FUNCTIONS. Ours pairs a network
        against MBD's cost-weighted Monte-Carlo ESTIMATE, whose validity is level-dependent
        (measured: ||s_task - s_base||/||s_base|| runs 0.58 -> 12.20 across levels, against 0.45
        between two learned proxies) and whose implied distribution is near-degenerate (ESS ~ 1).

        This runs the SAME operator with both operands learned -- reference proxy as A, task proxy
        as B. If proxy x proxy composes while proxy x MBD degrades monotonically, the estimator is
        the cause and the symmetry argument holds only where both sides are functions.
        """
        # x_t arrives as a tensor from the planner path and as an ndarray from the pair path,
        # which never touches the MBD step.
        x = (x_t.detach().cpu().numpy() if hasattr(x_t, "detach")
             else np.asarray(x_t)).astype(np.float32)
        y = self._to_proxy_space(x)

        def _score(client, horizon, obs):
            yy = y[:, :horizon]
            if yy.shape[1] < horizon:
                yy = np.concatenate(
                    [yy, np.repeat(yy[:, -1:, :], horizon - yy.shape[1], axis=1)], axis=1)
            sc, _ = client.score_batch(obs=obs, iteration=iteration,
                                       num_iterations=num_iterations, x=yy)
            return np.asarray(sc)[:, : y.shape[1]]

        t0 = time.perf_counter()
        s_b = _score(self.client, self.proxy_horizon, self._obs)          # task proxy
        s_a = _score(self.ref_client, self.ref_horizon, self._ref_obs)    # reference proxy
        rows = min(s_a.shape[1], s_b.shape[1])
        g = self.gamma
        score = np.zeros_like(s_b)
        score[:, :rows] = (1.0 - g) * s_a[:, :rows] + g * s_b[:, :rows]
        score[:, rows:] = s_b[:, rows:]        # rows only the task proxy has

        alpha, alpha_prev = ddim_iteration_alphas(
            iteration=iteration, num_iterations=num_iterations,
            num_train_timesteps=int(self.client.ready_info.get("ddim_num_train_timesteps", 100)))
        beta = max(1.0 - float(alpha), 1e-6)
        yy = y[:, : score.shape[1]]
        x0 = (yy + beta * score) / math.sqrt(max(float(alpha), 1e-6))
        eps = -math.sqrt(beta) * score
        y_next = (math.sqrt(max(float(alpha_prev), 0.0)) * x0
                  + math.sqrt(max(1.0 - float(alpha_prev), 0.0)) * eps)
        na, nb = float(np.linalg.norm(s_a[:, :rows])), float(np.linalg.norm(s_b[:, :rows]))
        self.level_trace.append({
            "it": int(iteration), "wall_s": round(time.perf_counter() - t0, 3),
            "ref_norm": round(na, 4), "task_norm": round(nb, 4),
            "pair_ratio": round(float(np.linalg.norm(s_b[:, :rows] - s_a[:, :rows]))
                                / max(nb, 1e-9), 4)})
        out = self._to_planner_space(y_next[:, : x.shape[1]])
        # The decode path wants a tensor, as policy_step returns.
        return torch.as_tensor(out, dtype=torch.float32)

    def policy_step(self, x_t, iteration, num_iterations):
        """One DDIM step of the PROXY's own reverse chain, returned in planner space.

        Replicates serve_mg_proxy_score.py's `chain` update exactly, but driven from the harness
        so an MBD step can be blended in at every level -- the inverted direction, where the
        trained policy is the base and the cost is the steering signal.
        """
        x = x_t.detach().cpu().numpy().astype(np.float32)
        y = self._to_proxy_space(x)
        if y.shape[1] < self.proxy_horizon:      # the model wants its full chunk
            y = self._pad_to_proxy(y)
        t0 = time.perf_counter()
        score, server_s = self.client.score_batch(
            obs=self._obs, iteration=iteration, num_iterations=num_iterations, x=y)
        alpha, alpha_prev = ddim_iteration_alphas(
            iteration=iteration, num_iterations=num_iterations,
            num_train_timesteps=int(self.client.ready_info.get("ddim_num_train_timesteps", 100)))
        beta = max(1.0 - float(alpha), 1e-6)
        x0 = (y + beta * score) / math.sqrt(max(float(alpha), 1e-6))
        eps = -math.sqrt(beta) * score
        y_next = (math.sqrt(max(float(alpha_prev), 0.0)) * x0
                  + math.sqrt(max(1.0 - float(alpha_prev), 0.0)) * eps)
        self.level_trace.append({
            "it": int(iteration), "wall_s": round(time.perf_counter() - t0, 3),
            "server_s": round(float(server_s), 3)})
        out = self._to_planner_space(y_next[:, : x.shape[1]])
        # C4. x0 is the model's CLEAN prediction; y_next is the noisy next iterate. Ranking
        # key-pose geometry needs the former -- FK of a noisy chunk is not a reachable pose --
        # while the chain needs the latter. Both are already computed here; only y_next used to
        # escape, so keypose_fk was costing x_{t-1} as though it were an action chunk.
        self._last_step = {
            "y0": x0,                      # proxy-space clean chunk, incl. real goal rows
            "x0_hat": torch.as_tensor(self._to_planner_space(x0[:, : x.shape[1]]),
                                      dtype=x_t.dtype, device=x_t.device),
            "eps": eps, "alpha": float(alpha), "alpha_prev": float(alpha_prev),
            "rows": int(x.shape[1]),
            "y_t": y,                      # noisy proxy-space iterate this level started from
            "x_prev": torch.as_tensor(out, dtype=x_t.dtype, device=x_t.device)}
        return torch.as_tensor(out, dtype=x_t.dtype, device=x_t.device)

    def _pad_to_proxy(self, y):
        """Extend a short chunk up to the proxy's own horizon, in proxy space.

        H1. The trailing rows of an AWE/keypose proxy are GOALS, not actions, so filling them by
        repeating the last action row is off-distribution. When a proxy chain has run this level
        (policy_base, keypose_fk) its own clean prediction supplies real goal rows. The plain
        additive path runs no chain, so there is nothing in-distribution to use -- say so once
        rather than let a fabricated goal block look like a measurement.
        """
        need = self.proxy_horizon - y.shape[1]
        if need <= 0:
            return y
        st = getattr(self, "_last_step", None)
        if st is not None and st.get("y0") is not None and st["y0"].shape[1] >= self.proxy_horizon:
            block = st["y0"][:, -need:, :].astype(y.dtype)
            pad = np.repeat(block[:1], y.shape[0], axis=0)
            return np.concatenate([y, pad], axis=1)
        if not getattr(self, "_warned_pad", False):
            self._warned_pad = True
            print(f"[proxy] H1: padding {need} goal row(s) by repeating the last ACTION row -- "
                  f"the proxy expects waypoints/keypose there, so its score is read "
                  f"off-distribution. Match --horizon to the proxy's chunk to avoid this.",
                  flush=True)
        return np.concatenate([y, np.repeat(y[:, -1:, :], need, axis=1)], axis=1)

    def policy_step_full(self, x_t, iteration, num_iterations):
        """policy_step, but returning {x0_hat, eps, x_prev, alpha, alpha_prev}.

        Callers that RANK or MODIFY the chunk want x0_hat; callers that only advance the chain
        want x_prev. Keeping policy_step's return type unchanged leaves every existing caller
        (policy_base, the handoff paths) behaving exactly as before.
        """
        self.policy_step(x_t, iteration, num_iterations)
        return dict(self._last_step)

    def redo_ddim(self, x0_planner, iteration=None, recompute_eps=False):
        """Re-run the DDIM update from a MODIFIED clean chunk, in planner space.

        The guided x0 has to re-enter the chain through the same operator the model's own step
        used, or the guidance is applied in one parameterisation and integrated in another.

        `recompute_eps` re-derives eps from the GUIDED clean chunk,
        eps' = (y_t - sqrt(abar_t) y0') / sqrt(1 - abar_t), which is the DDIM update proper: a
        displacement of y0 then moves the iterate by sqrt(abar_prev) - sqrt(abar_t (1 - abar_prev)
        / (1 - abar_t)) instead of sqrt(abar_prev). Freezing eps applies 2-5x that, front-loaded
        on the noisiest levels; the recomputed profile tracks his rectified-flow 1/(N-it) to ~0.05
        over the first two thirds of the chain. It is exactly the identity when y0 is unmodified,
        so it can never change the pure-policy branch.
        """
        st = self._last_step
        y0 = self._to_proxy_space(
            x0_planner.detach().cpu().numpy().astype(np.float32))
        if recompute_eps:
            alpha = max(st["alpha"], 1e-12)
            eps = ((st["y_t"][:, : y0.shape[1]] - math.sqrt(alpha) * y0)
                   / math.sqrt(max(1.0 - alpha, 1e-12)))
        else:
            eps = st["eps"][:, : y0.shape[1]]
        y_next = (math.sqrt(max(st["alpha_prev"], 0.0)) * y0
                  + math.sqrt(max(1.0 - st["alpha_prev"], 0.0)) * eps)
        out = self._to_planner_space(y_next[:, : st["rows"]])
        return torch.as_tensor(out, dtype=x0_planner.dtype, device=x0_planner.device)

    def support_distance(self, samples, x0_proxy):
        """How far the proxy's preferred chunk sits outside the base's proposal cloud, in sigmas.

        The acceptance test for any composition. Two situations produce the same nonzero
        ||s_task - s_base|| but demand opposite treatment:
          d small -- the proxy prefers something the base already proposes. The product of experts
                     REWEIGHTS shared support; composition is valid and selects.
          d large -- the proxy's target lies in the base's tails. The product has no mass there, so
                     composition INTERPOLATES into a region neither model endorses, and no gamma
                     repairs it.
        Cory measured 4.25 sigma on the gripper channel in the Isaac run; we measured a base that
        converts 0 of 82 grasps at the place phase, which is the same condition by another route.
        Returned per-dimension-pooled and per-row so a phase-structured gap is visible.
        """
        s = np.asarray(samples, dtype=np.float64)
        s = s.reshape(s.shape[0], -1) if s.ndim > 2 else s
        x = np.asarray(x0_proxy, dtype=np.float64).reshape(-1)[: s.shape[1]]
        mu, sd = s.mean(axis=0), s.std(axis=0)
        sd = np.where(sd < 1e-9, 1e-9, sd)
        z = (x - mu[: x.shape[0]]) / sd[: x.shape[0]]
        return {"d_mean": float(np.abs(z).mean()), "d_max": float(np.abs(z).max()),
                "d_rms": float(np.sqrt((z ** 2).mean())),
                "frac_outside_3sig": float((np.abs(z) > 3.0).mean())}

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
        # PPS applies the addend only for denoise_time >= steer_step; we applied it at every
        # level, and the measured addend/base ratio climbs 0.09 -> 9.6 across the chain, so the
        # last levels were effectively "follow the task proxy". `last_level` drops them.
        if self.last_level is not None and iteration > self.last_level:
            return None
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
        t0 = time.perf_counter()
        if self.aux == "native":
            y, s_proxy, server_s = self._score_native(y, x_t, iteration, num_iterations)
        else:
            if y.shape[1] < self.proxy_horizon:
                # H1. An AWE proxy expects [actions | W1..WK | keypose]. Repeating the last ACTION
                # row into those goal slots hands the model an input unlike anything in training,
                # and the score it returns is measured off-distribution. Prefer the proxy's OWN
                # predicted goal rows when a chain has produced them; fall back to repetition.
                y = self._pad_to_proxy(y)
            s_proxy, server_s = self.client.score_batch(
                obs=self._obs, iteration=iteration, num_iterations=num_iterations, x=y)
        wall_s = time.perf_counter() - t0
        slope = self._rows(self._slope, x.shape[1])
        if slope.ndim == 1:
            slope = slope[None, :]                 # one flat scale, broadcast over rows
        s_planner = s_proxy[:, : x.shape[1], :] * slope[None]
        task_field = (w[:, None, None] * s_planner).sum(axis=0)
        addend = self.gamma * task_field
        ref_field = None
        if self.ref == "proxy":
            # PPS Eq (4) proper: the reference is a proxy distilled from the base, so the
            # difference isolates the task increment while the shared prior cancels. Scored on
            # the SAME y and level as the task proxy, so only the weights differ.
            # Scored at the SAME planner point x, but normalised with the REFERENCE's own stats
            # and converted back with the REFERENCE's own slope (C1b): both fields then live over
            # the planner variable, which is what makes their difference meaningful.
            # The task proxy may carry a keypose tail the reference does not, so pad or trim to
            # the reference's own horizon. The extra row has no reference to cancel against and
            # contributes nothing to the difference.
            hr = self.ref_horizon
            yr = self._to_ref_space(x)
            yr = (yr[:, :hr] if yr.shape[1] >= hr
                  else np.concatenate([yr, np.zeros((yr.shape[0], hr - yr.shape[1], yr.shape[2]),
                                                    np.float32)], axis=1))
            s_ref, _ = self.ref_client.score_batch(
                obs=self._ref_obs, iteration=iteration, num_iterations=num_iterations, x=yr)
            rows_ref = min(x.shape[1], s_ref.shape[1])
            ref_slope = self._rows(self._ref_slope, rows_ref)
            if ref_slope.ndim == 1:
                ref_slope = ref_slope[None, :]
            r_planner = s_ref[:, :rows_ref, :] * ref_slope[None]
            if rows_ref < addend.shape[0]:
                r_planner = np.pad(r_planner, ((0, 0), (0, addend.shape[0] - rows_ref), (0, 0)))
            ref_field = (w[:, None, None] * r_planner).sum(axis=0)
            addend = addend - self.gamma * ref_field
        elif self.ref == "base":
            # PPS Eq (4) with v_ref := v_base, i.e. Eq (3): (1-gamma)*base + gamma*task.
            # A convex interpolation, bounded at every gamma -- unlike base + gamma*task,
            # whose magnitude grows with gamma and went NaN at the paper's own optimum.
            b = base_score.detach().cpu().numpy().astype(np.float32)[0]
            addend = addend - self.gamma * b[: addend.shape[0], : addend.shape[1]]
        if rows > addend.shape[0]:             # waypoint rows are the cost's, not the proxy's
            addend = np.pad(addend, ((0, rows - addend.shape[0]), (0, 0)))
        base_norm = float(torch.linalg.vector_norm(base_score.detach()).cpu())
        add_norm = float(np.linalg.norm(addend))
        # Support test (see support_distance): reconstruct the proxy's OWN preferred clean chunk
        # from the score it returned, x0 = (y + beta*s)/sqrt(alpha), and measure how far that sits
        # from the base's candidate cloud in units of the cloud's own spread.
        _supp = {}
        if samples is not None:
            # score_addend does `del alpha_bar` at the top, so recover the level's alpha from the
            # same DDIM schedule the proxy server reports.
            _ab, _ = ddim_iteration_alphas(
                iteration=iteration, num_iterations=num_iterations,
                num_train_timesteps=int(self.client.ready_info.get("ddim_num_train_timesteps", 100)))
            _ab = float(_ab)
            _beta = max(1.0 - _ab, 1e-6)
            # C3a. `samples` is the PLANNER-space cloud and s_planner is the score w.r.t. the
            # planner variable, so the iterate in x0 = (x_t + beta*s)/sqrt(alpha) must be the
            # planner-space x too. Using proxy-space y here silently mixed the two and was only
            # right when the bridge happened to be the identity (--align_proxy_norm on AND
            # matching gripper offsets).
            _x0 = (x[:, : s_planner.shape[1]] + _beta * s_planner) / math.sqrt(max(_ab, 1e-6))
            _cl = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
            _supp = {("supp_" + k): round(v, 3)
                     for k, v in self.support_distance(_cl, _x0[0]).items()}
        # Direction vs magnitude. ||task - base|| / ||base|| alone conflates the two: a field that
        # points the SAME way but is 10x larger looks identical to one pointing elsewhere. The
        # first is a scaling bug (fixable by normalisation), the second is genuine disagreement
        # (not fixable). Every claim about "they want different motions" rests on telling them
        # apart, so log the cosine and the task field's own norm.
        b_np = base_score.detach().cpu().numpy().astype(np.float64)[0]
        b_rows = b_np[: addend.shape[0], : addend.shape[1]].reshape(-1)
        # C3b. The addend is gamma*(task - ref) under ref=proxy, so dividing it by gamma gives the
        # DIFFERENCE, not the task field; labelling that `cos_task_base` was wrong. Use the task
        # field the proxy actually returned, and log the composed direction under its own name.
        _tf = np.asarray(task_field, dtype=np.float64)
        if addend.shape[0] > _tf.shape[0]:
            _tf = np.pad(_tf, ((0, addend.shape[0] - _tf.shape[0]), (0, 0)))
        task_dir = _tf[: addend.shape[0], : addend.shape[1]].reshape(-1)
        denom = np.linalg.norm(task_dir) * np.linalg.norm(b_rows)
        cos = float(task_dir @ b_rows / denom) if denom > 1e-12 else float("nan")
        a_rows = (np.asarray(addend, dtype=np.float64).reshape(-1)
                  / max(float(self.gamma), 1e-9))
        _den_a = np.linalg.norm(a_rows) * np.linalg.norm(b_rows)
        cos_addend = float(a_rows @ b_rows / _den_a) if _den_a > 1e-12 else float("nan")
        # PPS Eq (4) assumes the reference is an on-policy distillation of the base, so that
        # s_base - s_ref cancels and the residual is a pure task increment. That assumption is
        # never checked; if s_ref points elsewhere the residual carries a spurious -s_ref drag.
        _ref = {}
        if ref_field is not None:
            r_rows = np.asarray(ref_field, dtype=np.float64).reshape(-1)[: b_rows.size]
            bb = b_rows[: r_rows.size]
            dn = np.linalg.norm(r_rows) * np.linalg.norm(bb)
            _ref = {"cos_ref_base": round(float(r_rows @ bb / dn), 4) if dn > 1e-12 else None,
                    "ref_over_base": round(float(np.linalg.norm(r_rows)
                                                 / max(np.linalg.norm(bb), 1e-9)), 4)}
        self.level_trace.append({
            "it": int(iteration),
            "n_scored": int(x.shape[0]),
            # Width the model actually saw, and the width that steered: the guarantee that the
            # goal rows never reach the planner has to be visible in the trace, not just the code.
            "rows_in": int(y.shape[1]),
            "rows_steered": int(addend.shape[0]),
            "aux_mode": self.aux,
            "base_norm": round(base_norm, 4),
            "add_norm": round(add_norm, 4),
            "task_norm": round(float(np.linalg.norm(task_dir)), 4),
            "cos_task_base": round(cos, 4),
            # The composed field that is actually added (task, task-ref or task-base over gamma).
            "cos_addend_base": round(cos_addend, 4),
            "ratio": round(add_norm / max(base_norm, 1e-9), 4),
            **_ref,
            **_supp,
            "wall_s": round(wall_s, 3),
            "server_s": round(server_s, 3)})
        return torch.as_tensor(addend, dtype=torch.float32)
