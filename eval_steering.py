import argparse
import atexit
import dataclasses
import json
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Any

from tqdm import tqdm


_DEFAULT_WORKERS = 1
_DEFAULT_GPUS = "0"
_WORKER_PROGRESS_HANDLE = None
_WORKER_PROGRESS_HANDLE_PATH = None
_INIT_PROCESS_START = time.perf_counter()
_INIT_LAST_STAGE_TIME = _INIT_PROCESS_START
_INIT_LAST_STAGE = "process start"
# Per-episode wall accumulators for --profile. Timing is unconditional so the profiled and
# unprofiled step paths are identical; only the report and the sim wrappers are opt-in.
_PROFILE_TOTALS: dict[str, float] = {}


# ================================================================== per-bucket runtime profiling

def _prof_toc(bucket: str, start: float) -> None:
    """Accumulate elapsed wall since `start` into a named profiling bucket."""
    _PROFILE_TOTALS[bucket] = _PROFILE_TOTALS.get(bucket, 0.0) + (
        time.perf_counter() - start
    )


def _prof_wrap(owner: Any, name: str, bucket: str) -> None:
    """Wrap `owner.name` in place so its wall time lands in `bucket`."""
    original = getattr(owner, name)

    def timed(*call_args: Any, **call_kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original(*call_args, **call_kwargs)
        finally:
            _prof_toc(bucket, start)

    setattr(owner, name, timed)


def _prof_install(env) -> None:
    """Install --profile wrappers on the sim/scene calls inside env.step.

    Sensors are deliberately not wrapped: the closure holds a bound method, and the
    resulting reference cycle defers Camera.__del__ past the replicator teardown, which
    aborts the process at env.close(). scene.update covers the sensor fetch in aggregate.
    """
    _prof_wrap(env.sim, "step", "sim.physx")
    _prof_wrap(env.sim, "render", "sim.render")
    _prof_wrap(env.scene, "update", "scene.update")
    _prof_wrap(env.observation_manager, "compute", "obs.compute")


# Buckets that partition the control loop; the sim.* / scene.* / sensor.* wrappers are
# nested inside env.step and are reported separately as its internal split.
_PROFILE_LOOP_BUCKETS = (
    "planner.replan",
    "env.step",
    "vlm.observe_step",
    "debug.step_log",
    "obs.to_policy",
    "success_term",
    "video.frame_build",
)


def _build_profile_report(
    *, reset_s: float, loop_s: float, encode_s: float, steps: int
) -> dict[str, Any]:
    """Per-episode wall budget: top-level buckets plus the env.step internal split."""
    totals = dict(_PROFILE_TOTALS)
    loop = {name: totals.get(name, 0.0) for name in _PROFILE_LOOP_BUCKETS}
    inside_step = {
        name: value
        for name, value in totals.items()
        if name not in _PROFILE_LOOP_BUCKETS
    }
    episode_s = reset_s + loop_s + encode_s
    return {
        "steps": steps,
        "episode_s": episode_s,
        "reset_s": reset_s,
        "loop_s": loop_s,
        "video_encode_s": encode_s,
        "loop_buckets_s": loop,
        "loop_unaccounted_s": loop_s - sum(loop.values()),
        "env_step_internal_s": inside_step,
    }


def _print_profile_report(seed: int, report: dict[str, Any]) -> None:
    """Print the per-episode budget as seconds and percent of episode wall."""
    episode_s = report["episode_s"] or 1.0
    steps = max(report["steps"], 1)
    print(f"[PROFILE] seed={seed} episode_s={report['episode_s']:.1f} steps={steps}")
    rows = [
        ("reset (env + grounding)", report["reset_s"]),
        ("video encode + transcode", report["video_encode_s"]),
        *report["loop_buckets_s"].items(),
        ("loop unaccounted", report["loop_unaccounted_s"]),
    ]
    for name, value in sorted(rows, key=lambda item: -item[1]):
        print(
            f"[PROFILE]   {name:<26} {value:8.2f}s  {100.0 * value / episode_s:5.1f}%  "
            f"{1000.0 * value / steps:7.2f} ms/step"
        )
    print("[PROFILE]   -- inside env.step --")
    for name, value in sorted(report["env_step_internal_s"].items(), key=lambda i: -i[1]):
        print(
            f"[PROFILE]   {name:<26} {value:8.2f}s  {100.0 * value / episode_s:5.1f}%  "
            f"{1000.0 * value / steps:7.2f} ms/step"
        )


# ========================================================================= multi-worker launcher

def _parse_gpu_ids(value: str) -> list[int]:
    tokens = value.replace(",", " ").split()
    if not tokens:
        raise ValueError(
            "--gpus must contain at least one GPU index, for example --gpus 0,1."
        )
    gpu_ids = []
    for token in tokens:
        if token.startswith("cuda:"):
            token = token[5:]
        try:
            gpu_id = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid GPU index {token!r} in --gpus {value!r}.") from exc
        if gpu_id < 0:
            raise ValueError(f"GPU indices must be non-negative, got {gpu_id}.")
        gpu_ids.append(gpu_id)
    return gpu_ids


def _multi_worker_preparse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task")
    parser.add_argument("--exp_name", default="eval")
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=51)
    parser.add_argument("--task_num_steps", type=int, default=225)
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    parser.add_argument("--gpus", default=_DEFAULT_GPUS)
    parser.add_argument("--worker-id", type=int, default=-1)
    parser.add_argument("--worker-progress-path", default=None)
    return parser.parse_known_args(argv)[0]


def _remove_value_options(argv: list[str], option_names: set[str]) -> list[str]:
    result = []
    skip_value = False
    for arg in argv:
        if skip_value:
            skip_value = False
            continue
        if arg in option_names:
            skip_value = True
            continue
        if any(arg.startswith(f"{name}=") for name in option_names):
            continue
        result.append(arg)
    return result


def _split_seed_ranges(
    seed_start: int, seed_end: int, workers: int
) -> list[tuple[int, int]]:
    num_seeds = seed_end - seed_start
    if num_seeds <= 0:
        raise ValueError("--seed_end must be greater than --seed_start.")
    workers = min(workers, num_seeds)
    base, remainder = divmod(num_seeds, workers)
    ranges = []
    start = seed_start
    for worker_id in range(workers):
        count = base + (1 if worker_id < remainder else 0)
        ranges.append((start, start + count))
        start += count
    return ranges


def _emit_worker_progress(args: argparse.Namespace, event: str, **payload: Any) -> None:
    global _WORKER_PROGRESS_HANDLE, _WORKER_PROGRESS_HANDLE_PATH

    progress_path = getattr(args, "worker_progress_path", None)
    if progress_path is None:
        return
    if _WORKER_PROGRESS_HANDLE_PATH != progress_path:
        if _WORKER_PROGRESS_HANDLE is not None:
            _WORKER_PROGRESS_HANDLE.close()
        _WORKER_PROGRESS_HANDLE = open(
            progress_path, "a", encoding="utf-8", buffering=1
        )
        _WORKER_PROGRESS_HANDLE_PATH = progress_path
    record = {
        "event": event,
        "time": time.time(),
        "worker_id": args.worker_id,
        **payload,
    }
    _WORKER_PROGRESS_HANDLE.write(json.dumps(record, sort_keys=True) + "\n")
    if event == "run_end":
        _WORKER_PROGRESS_HANDLE.close()
        _WORKER_PROGRESS_HANDLE = None
        _WORKER_PROGRESS_HANDLE_PATH = None


def _report_initialization_stage(
    args: argparse.Namespace, stage: str, **details: Any
) -> None:
    """Report initialization progress to stdout or the multi-worker progress UI."""
    global _INIT_LAST_STAGE, _INIT_LAST_STAGE_TIME

    now = time.perf_counter()
    total_seconds = now - _INIT_PROCESS_START
    previous_seconds = now - _INIT_LAST_STAGE_TIME
    payload = {
        "stage": stage,
        "elapsed_s": round(total_seconds, 3),
        "previous_stage": _INIT_LAST_STAGE,
        "previous_stage_s": round(previous_seconds, 3),
        **details,
    }
    _emit_worker_progress(args, "init_stage", **payload)

    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    message = (
        f"[INIT +{total_seconds:7.2f}s] {stage} "
        f"(previous: {_INIT_LAST_STAGE}, {previous_seconds:.2f}s)"
    )
    if detail_text:
        message += f" | {detail_text}"
    # Worker stdout is retained in its log file, while the parent renders this
    # same event on the worker's progress bar.
    print(message, flush=True)

    _INIT_LAST_STAGE = stage
    _INIT_LAST_STAGE_TIME = now


def _wait_for_worker_start(args: argparse.Namespace) -> None:
    _emit_worker_progress(args, "ready", device=args.device)
    barrier_path = args.worker_start_barrier
    if barrier_path is None:
        return
    while not os.path.exists(barrier_path):
        if args.worker_parent_pid > 0:
            try:
                os.kill(args.worker_parent_pid, 0)
            except ProcessLookupError as exc:
                raise RuntimeError("Multi-worker parent exited before releasing workers.") from exc
        time.sleep(0.1)


def _read_worker_progress(worker: dict[str, Any]) -> None:
    progress_path = worker["progress_path"]
    if not os.path.exists(progress_path):
        return
    with open(progress_path, encoding="utf-8") as handle:
        handle.seek(worker["progress_offset"])
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_name = event.get("event")
            if event_name == "init_stage":
                stage = event.get("stage", "initializing")
                elapsed = event.get("elapsed_s", 0.0)
                worker["bar"].set_description_str(
                    f"worker {worker['id'] + 1} gpu {worker['gpu']} init: {stage}"
                )
                worker["bar"].set_postfix_str(f"init {elapsed:.1f}s", refresh=True)
            elif event_name == "ready":
                worker["ready"] = True
                worker["bar"].set_description_str(
                    f"worker {worker['id'] + 1} gpu {worker['gpu']} ready"
                )
            elif event_name == "rollout_start":
                worker["bar"].set_description_str(
                    f"worker {worker['id'] + 1} gpu {worker['gpu']} seed {event['seed']}"
                )
            elif event_name == "step":
                completed = event["rollout_index"] * event["task_num_steps"] + event["step"]
                worker["bar"].n = min(completed, worker["bar"].total)
                worker["bar"].refresh()
            elif event_name == "rollout_end":
                completed = (event["rollout_index"] + 1) * event["task_num_steps"]
                worker["bar"].n = min(completed, worker["bar"].total)
                worker["bar"].set_postfix_str(
                    "success" if event.get("success") else "failed", refresh=True
                )
            elif event_name == "run_end":
                worker["results_path"] = event.get("results_path")
                worker["output_path"] = event.get("output_path")
                worker["bar"].n = worker["bar"].total
                worker["bar"].refresh()
        worker["progress_offset"] = handle.tell()


def _run_multi_worker_launcher(pre_args: argparse.Namespace, argv: list[str]) -> int:
    if pre_args.task is None:
        print("error: --task is required", file=sys.stderr)
        return 2
    if pre_args.workers <= 0:
        print("error: --workers must be positive", file=sys.stderr)
        return 2
    try:
        gpu_ids = _parse_gpu_ids(pre_args.gpus)
        seed_ranges = _split_seed_ranges(
            pre_args.seed_start, pre_args.seed_end, pre_args.workers
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if len(seed_ranges) < pre_args.workers:
        print(
            f"Using {len(seed_ranges)} workers because only {len(seed_ranges)} seeds were requested.",
            file=sys.stderr,
        )

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
    launcher_dir = os.path.join(
        repo_dir,
        "results",
        pre_args.task,
        pre_args.exp_name,
        "_workers",
        run_id,
    )
    os.makedirs(launcher_dir, exist_ok=False)
    barrier_path = os.path.join(launcher_dir, "start.barrier")
    stripped_argv = _remove_value_options(
        argv,
        {
            "--workers",
            "--gpus",
            "--worker-id",
            "--worker-progress-path",
            "--worker-start-barrier",
            "--worker-rollout-offset",
            "--worker-parent-pid",
            "--seed_start",
            "--seed_end",
            "--exp_name",
            "--device",
        },
    )

    workers = []
    rollout_offset = 0
    for worker_id, (worker_seed_start, worker_seed_end) in enumerate(seed_ranges):
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]
        progress_path = os.path.join(launcher_dir, f"worker_{worker_id:02d}.jsonl")
        log_path = os.path.join(launcher_dir, f"worker_{worker_id:02d}.log")
        open(progress_path, "w", encoding="utf-8").close()
        worker_exp_name = os.path.join(
            pre_args.exp_name, f"worker_{worker_id:02d}_gpu_{gpu_id}"
        )
        command = [
            sys.executable,
            os.path.abspath(__file__),
            *stripped_argv,
            "--workers",
            "1",
            "--gpus",
            str(gpu_id),
            "--worker-id",
            str(worker_id),
            "--worker-progress-path",
            progress_path,
            "--worker-start-barrier",
            barrier_path,
            "--worker-rollout-offset",
            str(rollout_offset),
            "--worker-parent-pid",
            str(os.getpid()),
            "--seed_start",
            str(worker_seed_start),
            "--seed_end",
            str(worker_seed_end),
            "--exp_name",
            worker_exp_name,
            "--device",
            "cuda:0",
        ]
        bar = tqdm(
            total=(worker_seed_end - worker_seed_start) * pre_args.task_num_steps,
            desc=f"worker {worker_id + 1} gpu {gpu_id} initializing",
            position=worker_id,
            leave=True,
            dynamic_ncols=True,
        )
        workers.append(
            {
                "id": worker_id,
                "gpu": gpu_id,
                "seed_start": worker_seed_start,
                "seed_end": worker_seed_end,
                "rollout_offset": rollout_offset,
                "progress_path": progress_path,
                "progress_offset": 0,
                "log_path": log_path,
                "log_handle": None,
                "command": command,
                "process": None,
                "ready": False,
                "results_path": None,
                "output_path": None,
                "bar": bar,
            }
        )
        rollout_offset += worker_seed_end - worker_seed_start

    def terminate_workers() -> None:
        # A worker that is blocked waiting on the start barrier is inside Isaac Sim, which
        # installs its own SIGTERM handling and does not necessarily exit on it. Escalate to
        # SIGKILL, otherwise the "wait for every worker to exit" loop below never returns and
        # the whole job burns its wall clock with nothing running.
        live = []
        for worker in workers:
            process = worker["process"]
            if process is not None and process.poll() is None:
                process.terminate()
                live.append(process)
        deadline = time.time() + 30.0
        for process in live:
            try:
                process.wait(timeout=max(deadline - time.time(), 0.1))
            except subprocess.TimeoutExpired:
                process.kill()

    def report_worker_failure(worker, reason: str) -> None:
        """Surface a dead worker's own last words; they are only in its log file."""
        if worker["log_handle"] is not None:
            worker["log_handle"].flush()
        try:
            with open(worker["log_path"], encoding="utf-8", errors="replace") as handle:
                tail = [line.rstrip() for line in handle.readlines()[-40:]]
        except OSError:
            tail = []
        print(
            f"worker {worker['id']} (gpu {worker['gpu']}, seeds "
            f"{worker['seed_start']}-{worker['seed_end'] - 1}) failed to initialize: {reason}. "
            f"Tail of {worker['log_path']}:",
            file=sys.stderr,
        )
        for line in tail:
            print(f"  | {line}", file=sys.stderr)

    atexit.register(terminate_workers)
    init_stall_s = float(os.environ.get("VLMDP_WORKER_INIT_STALL_S", 300.0))
    failed_to_initialize = False
    for wave_start in range(0, len(workers), len(gpu_ids)):
        wave = workers[wave_start : wave_start + len(gpu_ids)]
        for worker in wave:
            worker["log_handle"] = open(worker["log_path"], "w", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # Isolate each Isaac Sim worker to one physical GPU. Without this,
            # every process enumerates all GPUs and concurrently runs the
            # expensive IOMMU/P2P validation during startup.
            env["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
            worker["process"] = subprocess.Popen(
                worker["command"],
                cwd=repo_dir,
                env=env,
                stdout=worker["log_handle"],
                stderr=subprocess.STDOUT,
            )
        for worker in wave:
            worker["progress_size"] = 0
            worker["progress_time"] = time.time()
        while not all(worker["ready"] for worker in wave):
            for worker in workers:
                _read_worker_progress(worker)
            now = time.time()
            for worker in wave:
                if worker["ready"]:
                    continue
                reason = None
                if worker["process"].poll() is not None:
                    reason = f"process exited with code {worker['process'].returncode}"
                else:
                    # A worker that aborts during grounding (e.g. the ReKep preflight refusing an
                    # unusable mask) raises SystemExit, and Isaac Sim then wedges on shutdown --
                    # the process never exits, so polling alone never notices. Treat a worker that
                    # has stopped emitting init progress as failed, or the launcher waits forever.
                    try:
                        size = os.path.getsize(worker["progress_path"])
                    except OSError:
                        size = worker["progress_size"]
                    if size != worker["progress_size"]:
                        worker["progress_size"] = size
                        worker["progress_time"] = now
                    elif now - worker["progress_time"] > init_stall_s:
                        reason = (f"no initialization progress for {init_stall_s:.0f}s while still "
                                  f"running (hung, most likely after a fatal error)")
                if reason is None:
                    continue
                if not worker.get("reported"):
                    worker["reported"] = True
                    report_worker_failure(worker, reason)
                failed_to_initialize = True
            if failed_to_initialize:
                break
            time.sleep(0.1)
        if failed_to_initialize:
            break

    if failed_to_initialize:
        terminate_workers()
    else:
        with open(barrier_path, "w", encoding="utf-8") as handle:
            handle.write("start\n")

    while any(
        worker["process"] is not None and worker["process"].poll() is None
        for worker in workers
    ):
        for worker in workers:
            _read_worker_progress(worker)
        time.sleep(0.1)
    for worker in workers:
        _read_worker_progress(worker)

    failed_workers = []
    manifest_workers = []
    for worker in workers:
        process = worker["process"]
        return_code = process.wait() if process is not None else 1
        if return_code != 0:
            failed_workers.append(worker)
            worker["bar"].set_description_str(
                f"worker {worker['id'] + 1} gpu {worker['gpu']} ERROR"
            )
        else:
            worker["bar"].n = worker["bar"].total
            worker["bar"].refresh()
            worker["bar"].set_description_str(
                f"worker {worker['id'] + 1} gpu {worker['gpu']} complete"
            )
        worker["bar"].close()
        if worker["log_handle"] is not None:
            worker["log_handle"].close()
        manifest_workers.append(
            {
                "worker_id": worker["id"],
                "gpu": worker["gpu"],
                "seed_start": worker["seed_start"],
                "seed_end": worker["seed_end"],
                "return_code": return_code,
                "log_path": worker["log_path"],
                "results_path": worker["results_path"],
                "output_path": worker["output_path"],
            }
        )

    manifest_path = os.path.join(launcher_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "workers": len(workers),
                "gpus": gpu_ids,
                "seed_start": pre_args.seed_start,
                "seed_end": pre_args.seed_end,
                "worker_runs": manifest_workers,
            },
            handle,
            indent=2,
        )

    atexit.unregister(terminate_workers)
    if failed_workers:
        print(
            f"{len(failed_workers)} worker(s) failed. Logs: {launcher_dir}",
            file=sys.stderr,
        )
        for worker in failed_workers:
            print(f"  worker {worker['id']}: {worker['log_path']}", file=sys.stderr)
        return 1
    print(f"All workers completed. Manifest: {manifest_path}")
    return 0


_MULTI_WORKER_PRE_ARGS = _multi_worker_preparse(sys.argv[1:])
if (
    _MULTI_WORKER_PRE_ARGS.worker_id < 0
    and _MULTI_WORKER_PRE_ARGS.workers > 1
    and "--help" not in sys.argv
    and "-h" not in sys.argv
):
    raise SystemExit(_run_multi_worker_launcher(_MULTI_WORKER_PRE_ARGS, sys.argv[1:]))

_report_initialization_stage(
    _MULTI_WORKER_PRE_ARGS,
    "loading Python/model dependencies",
    task=_MULTI_WORKER_PRE_ARGS.task,
)

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENPI_SRC_DIR = os.path.join(_REPO_DIR, "openpi", "src")
if _OPENPI_SRC_DIR not in sys.path:
    sys.path.insert(0, _OPENPI_SRC_DIR)
_ISAACLAB_DIR = os.path.join(_REPO_DIR, "IsaacLab")
for _isaaclab_pkg in (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaaclab_rl",
    "isaaclab_mimic",
):
    _isaaclab_pkg_src = os.path.join(_ISAACLAB_DIR, "source", _isaaclab_pkg)
    if _isaaclab_pkg_src not in sys.path:
        sys.path.insert(0, _isaaclab_pkg_src)
from isaaclab.app import AppLauncher
import pinocchio  # noqa: F401  -- must import before Isaac Sim (load order)

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.policies import policy_config


import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

import copy
import re

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from sim_free_mpc import AccelActionMPC, AccelMPCConfig, SimFreeMPC, SimFreeMPCConfig
from sim_free_mpc.action_space import (
    DemoDeltaDecodePolicy,
    clamp_real_action_chunk,
    decode_model_action_chunks,
    load_action_norm_stats_json,
)
from sim_free_mpc.ddim import ddim_iteration_alphas
from sim_free_mpc.planner import task_tilt_weight
from sim_free_mpc.score_steering import combine_scores, steer_scale_for_stage


DEFAULT_BASE_CHECKPOINT_DIR = "openpi/checkpoints/pytorch/pi05_droid_jointpos"
TASK_PROMPTS_PATH = os.path.join(_REPO_DIR, "task_prompts.json")

_SOUND_VIDEO_SCALE = None
_SOUND_VIDEO_MAX_DISTANCE_M = 0.05
_SOUND_AUDIO_CACHE = None
_SOUND_AUDIO_SAMPLE_RATE = 48_000
_SOUND_AUDIO_ATTENUATION_POWER = 2.0
_SOUND_AUDIO_REFERENCE_DISTANCE = 1.0
_SOUND_AUDIO_MIN_DISTANCE = 1e-3
_LAST_INFERENCE_RUNTIME = {}
_WEIGHT_SCALE_CENTER_OFFSET_DEBUG = (-0.0470425, 0.0, 0.0272255)
_WEIGHT_SCALE_TOP_OFFSET_Z_DEBUG = 0.0523800


# =========================================================================== runtime determinism

def _seed_runtime(seed: int) -> None:
    """Seed policy, MPC, and environment-facing random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _enable_deterministic_runtime(seed: int) -> None:
    """Enable strict deterministic PyTorch/CUDA execution before app startup."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    _seed_runtime(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _score_update_mode_for_mpc_update(update_mode: str) -> str:
    if update_mode in ("mbd_score_action_prox", "mbd_score_action_warm"):
        return "mbd_score"
    return update_mode


def _phone_ringtone_path():
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/phone/ringtone.wav",
        )
    )


# ===================================================== policy inputs and norm-stat compatibility

def _to_numpy_unbatched(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def _find_first_present(data, keys):
    for key in keys:
        if key in data:
            return key
    return None


def _policy_uses_thermal_inputs(policy) -> bool:
    transforms = getattr(policy._input_transform, "transforms", ())
    return any(type(transform).__name__ == "_OverlayThermalDroidImages" for transform in transforms)


def _validate_policy_environment_inputs(policy, raw_obs: dict, role: str) -> None:
    if (getattr(policy, "_metadata", {}) or {}).get("decode_only"):
        # A decode-only policy has no network, so it consumes no sensor stream to validate.
        return

    model_type = policy._model.config.model_type

    has_rgb = (
        "observation/exterior_image_1_left" in raw_obs
        and "observation/wrist_image_left" in raw_obs
    )
    has_sound = "observation/sound" in raw_obs or (
        "observation/mic1_log_mel" in raw_obs
        and "observation/mic2_log_mel" in raw_obs
    )
    has_thermal = (
        "observation/thermal_exterior_image_1_left" in raw_obs
        and "observation/thermal_wrist_image_left" in raw_obs
    )
    has_pointcloud = "observation/pointcloud" in raw_obs or (
        "observation/pointcloud_coord" in raw_obs
        and "observation/pointcloud_color" in raw_obs
    )

    if model_type in (
        _model.ModelType.PI0,
        _model.ModelType.PI05,
        _model.ModelType.PROXY,
        _model.ModelType.PROXY_SCORE,
        _model.ModelType.PROXY_SOUND,
        _model.ModelType.RESIDUAL,
    ) and not has_rgb:
        raise ValueError(
            f"{role} model ({model_type.value}) requires RGB observations, but the environment does not provide "
            "'observation/exterior_image_1_left' and 'observation/wrist_image_left'."
        )

    if model_type == _model.ModelType.PROXY_SOUND and not has_sound:
        raise ValueError(
            f"{role} model ({model_type.value}) requires sound observations, but the environment does not provide "
            "'observation/sound' or both 'observation/mic1_log_mel' and 'observation/mic2_log_mel'."
        )

    if _policy_uses_thermal_inputs(policy) and not has_thermal:
        raise ValueError(
            f"{role} model ({model_type.value}) requires thermal observations, but the environment does not provide "
            "'observation/thermal_exterior_image_1_left' and 'observation/thermal_wrist_image_left'."
        )

    if model_type in (
        _model.ModelType.PROXY_POINTCLOUD,
        _model.ModelType.PROXY_DP3,
    ) and not has_pointcloud:
        raise ValueError(
            f"{role} model ({model_type.value}) requires pointcloud observations, but the environment does not provide "
            "'observation/pointcloud' or both 'observation/pointcloud_coord' and 'observation/pointcloud_color'."
        )


def _obs_to_input_checked(policy, raw_obs: dict, role: str):
    _validate_policy_environment_inputs(policy, raw_obs, role)
    try:
        return policy.obs_to_input(raw_obs)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Failed to build inputs for {role} model ({policy._model.config.model_type.value}): {exc}"
        ) from exc


def _policy_input_norm_stats(policy):
    transforms = getattr(policy._input_transform, "transforms", ())
    for transform in transforms:
        if type(transform).__name__ == "Normalize":
            return transform.norm_stats, transform.use_quantiles
    return None, None


def _stat_values_for_compare(stats, use_quantiles: bool):
    if stats is None:
        return ()
    if use_quantiles and stats.q01 is not None and stats.q99 is not None:
        return (np.asarray(stats.q01), np.asarray(stats.q99))
    return (np.asarray(stats.mean), np.asarray(stats.std))


def _warn_if_norm_mismatch(
    base_policy,
    task_policy,
    ref_policy,
    *,
    action_dim: int,
):
    policies = {
        "base": base_policy,
        "task": task_policy,
        "ref": ref_policy,
    }
    policies = {role: policy for role, policy in policies.items() if policy is not None}
    norm_info = {
        role: _policy_input_norm_stats(policy)
        for role, policy in policies.items()
    }

    for role, (norm_stats, _) in norm_info.items():
        missing = [
            key
            for key in ("state", "actions")
            if norm_stats is None or key not in norm_stats
        ]
        if missing:
            print(f"WARNING: {role} policy is missing normalization stats for {missing}.")

    if "base" not in norm_info:
        return
    base_stats, base_use_quantiles = norm_info["base"]
    if base_stats is None:
        return

    for role in ("task", "ref"):
        if role not in norm_info:
            continue
        role_stats, role_use_quantiles = norm_info[role]
        if role_stats is None:
            continue
        if base_use_quantiles != role_use_quantiles:
            print(
                "WARNING: normalization mode differs between base and "
                f"{role}: base use_quantiles={base_use_quantiles}, "
                f"{role} use_quantiles={role_use_quantiles}."
            )
        for key in ("state", "actions"):
            if key not in base_stats or key not in role_stats:
                continue
            compare_dim = action_dim if key == "actions" else min(action_dim, 8)
            base_values = _stat_values_for_compare(base_stats[key], base_use_quantiles)
            role_values = _stat_values_for_compare(role_stats[key], role_use_quantiles)
            for base_value, role_value in zip(base_values, role_values, strict=False):
                dims = min(compare_dim, base_value.shape[-1], role_value.shape[-1])
                if not np.allclose(
                    base_value[..., :dims],
                    role_value[..., :dims],
                    rtol=1e-4,
                    atol=1e-5,
                ):
                    print(
                        "WARNING: normalized steering space may be mismatched: "
                        f"base and {role} {key} stats differ in the first {dims} dims."
                    )
                    break


def _assert_score_space_compatibility(base_policy, task_policy, ref_policy, args) -> None:
    score_mode = _score_steering_mode(args)
    if score_mode not in ("full", "task"):
        return

    if base_policy is None or task_policy is None:
        raise ValueError(f"{score_mode} score steering requires base and task policies.")
    score_policies = {"task": task_policy}
    if score_mode == "full":
        if ref_policy is None:
            raise ValueError("Full score steering requires a reference policy.")
        score_policies["ref"] = ref_policy

    for role, policy in score_policies.items():
        if not _is_score_proxy(policy._model):
            raise ValueError(
                f"{score_mode} score steering requires {role} to be a ProxyScore checkpoint; "
                f"got {policy._model.config.model_type.value!r}."
            )

    expected_timesteps = int(args.mpc_ddim_train_timesteps)
    for role, policy in score_policies.items():
        model = policy._model
        actual = int(getattr(model.config, "ddim_num_train_timesteps", -1))
        if actual != expected_timesteps:
            raise ValueError(
                f"{role} ProxyScore DDIM train timesteps ({actual}) do not match "
                f"--mpc_ddim_train_timesteps ({expected_timesteps})."
            )

    policies = {"base": base_policy, **score_policies}
    norm_info = {role: _policy_input_norm_stats(policy) for role, policy in policies.items()}
    base_stats, base_use_quantiles = norm_info["base"]
    if base_stats is None:
        raise ValueError("Base policy has no input norm_stats; score-space MPC cannot decode a shared action space.")

    action_dim = min(
        int(getattr(policy._model.config, "action_dim", 8))
        for policy in policies.values()
    )
    for role in score_policies:
        role_stats, role_use_quantiles = norm_info[role]
        if role_stats is None:
            raise ValueError(f"{role} ProxyScore policy has no input norm_stats.")
        if base_use_quantiles != role_use_quantiles:
            raise ValueError(
                "Score-space policy normalization mode mismatch: "
                f"base use_quantiles={base_use_quantiles}, {role} use_quantiles={role_use_quantiles}."
            )
        for key in ("state", "actions"):
            if key not in base_stats or key not in role_stats:
                raise ValueError(f"Missing {key!r} norm_stats in base or {role} policy.")
            compare_dim = action_dim if key == "actions" else min(action_dim, 8)
            base_values = _stat_values_for_compare(base_stats[key], base_use_quantiles)
            role_values = _stat_values_for_compare(role_stats[key], role_use_quantiles)
            for base_value, role_value in zip(base_values, role_values, strict=False):
                dims = min(compare_dim, base_value.shape[-1], role_value.shape[-1])
                if not np.allclose(
                    base_value[..., :dims],
                    role_value[..., :dims],
                    rtol=1e-4,
                    atol=1e-5,
                ):
                    raise ValueError(
                        "Score-space normalized steering space mismatch: "
                        f"base and {role} {key} norm_stats differ in the first {dims} dims."
                    )


# ======================================================== proxy prediction: flow and score heads

def _run_sequence_proxy_expert(
    model,
    prefix_embs,
    prefix_pad_masks,
    suffix_embs,
    suffix_pad_masks,
    adarms_cond,
):
    if hasattr(model, "_run_action_expert"):
        return model._run_action_expert(
            prefix_embs,
            prefix_pad_masks,
            suffix_embs,
            suffix_pad_masks,
            adarms_cond,
        )

    embs = torch.cat([prefix_embs, suffix_embs], dim=1)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    position_ids = position_ids.to(dtype=torch.long)

    hidden_states, _ = model.expert_model.forward(
        attention_mask=pad_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=embs,
        use_cache=False,
        adarms_cond=adarms_cond,
    )
    suffix_out = hidden_states[:, -model.config.action_horizon :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def _prepare_proxy_steering(model, observation):
    model_type = model.config.model_type

    if model_type in (_model.ModelType.PROXY, _model.ModelType.PROXY_SCORE):
        images, img_masks, state = model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks
        )
        return {
            "kind": "sequence",
            "state": state,
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_att_masks": prefix_att_masks,
        }

    if model_type == _model.ModelType.PROXY_SOUND:
        images, img_masks, sound, state = model._preprocess_observation(
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, _ = model.embed_prefix(images, img_masks, sound)
        return {
            "kind": "sequence",
            "state": state,
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
        }

    if model_type == _model.ModelType.PROXY_POINTCLOUD:
        pointcloud, point_mask, state = model._preprocess_observation(
            observation, train=False
        )
        if hasattr(model, "_log_ptv3_token_counts"):
            prefix_embs, prefix_pad_masks, _, point_token_counts = model.embed_prefix(
                pointcloud,
                point_mask,
                train=False,
                return_token_counts=True,
            )
            model._log_ptv3_token_counts(
                point_token_counts,
                train=False,
                loaded_point_count=pointcloud.shape[1],
            )
        else:
            prefix_embs, prefix_pad_masks, _ = model.embed_prefix(pointcloud, point_mask)
        return {
            "kind": "sequence",
            "state": state,
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
        }

    if model_type == _model.ModelType.PROXY_DP3:
        pointcloud, point_mask, state = model._preprocess_observation(
            observation, train=False
        )
        obs_features = model.encode_observation(pointcloud, point_mask, state, train=False)
        return {
            "kind": "dp3",
            "obs_features": obs_features,
        }

    raise ValueError(f"Unsupported steer model type: {model_type}")


def _predict_proxy_flow(prepared_proxy, model, x_t_path, time_cond):
    action_dim = model.config.action_dim
    x_t_model = x_t_path[:, :, :action_dim]

    if prepared_proxy["kind"] == "dp3":
        return model._run_dp3(x_t_model, time_cond, prepared_proxy["obs_features"])

    suffix_embs, suffix_pad_masks, _, adarms_cond = model.embed_suffix(
        prepared_proxy["state"],
        x_t_model,
        time_cond,
    )
    return _run_sequence_proxy_expert(
        model,
        prepared_proxy["prefix_embs"],
        prepared_proxy["prefix_pad_masks"],
        suffix_embs,
        suffix_pad_masks,
        adarms_cond,
    )


def _is_score_proxy(model) -> bool:
    return model.config.model_type == _model.ModelType.PROXY_SCORE


def _proxy_score_time_cond(args, iteration: int, device, dtype) -> torch.Tensor:
    ddim_iteration_alphas(
        iteration=iteration,
        num_iterations=args.num_steps + 1,
        num_train_timesteps=args.mpc_ddim_train_timesteps,
    )
    step_ratio = int(args.mpc_ddim_train_timesteps) // int(args.num_steps + 1)
    timestep = int((int(args.num_steps + 1) - 1 - int(iteration)) * step_ratio)
    value = timestep / max(float(args.mpc_ddim_train_timesteps - 1), 1.0)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _score_cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            lhs.detach().reshape(1, -1),
            rhs.detach().reshape(1, -1),
            dim=-1,
            eps=1e-8,
        )[0].cpu()
    )


def _relative_score_error(target: torch.Tensor, prediction: torch.Tensor) -> float:
    error_norm = torch.linalg.vector_norm((prediction - target).detach())
    target_norm = torch.linalg.vector_norm(target.detach()).clamp_min(1e-8)
    return float((error_norm / target_norm).cpu())


def _predict_proxy_score(prepared_proxy, model, x_t_path, time_cond):
    if not _is_score_proxy(model):
        raise ValueError(
            "Score-space PPS steering requires task/ref checkpoints with "
            f"model_type={_model.ModelType.PROXY_SCORE.value!r}; got "
            f"{model.config.model_type.value!r}."
        )
    action_dim = model.config.action_dim
    x_t_model = x_t_path[:, :, :action_dim]
    if prepared_proxy["kind"] != "sequence":
        raise ValueError("ProxyScorePytorch currently supports sequence image proxies only.")
    return model.predict_score_from_prefix(
        prepared_proxy["state"],
        prepared_proxy["prefix_embs"],
        prepared_proxy["prefix_pad_masks"],
        x_t_model,
        time_cond,
        prefix_att_masks=prepared_proxy.get("prefix_att_masks"),
    )


def _sequence_proxy_flow_from_prefix(
    model,
    state,
    prefix_embs,
    prefix_pad_masks,
    x_t_path,
    time_cond,
    action_dim: int,
):
    x_t_model = x_t_path[:, :, :action_dim]
    suffix_embs, suffix_pad_masks, _, adarms_cond = model.embed_suffix(
        state,
        x_t_model,
        time_cond,
    )
    embs = torch.cat([prefix_embs, suffix_embs], dim=1)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    position_ids = (torch.cumsum(pad_masks, dim=1) - 1).to(dtype=torch.long)

    hidden_states, _ = model.expert_model.forward(
        attention_mask=pad_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=embs,
        use_cache=False,
        adarms_cond=adarms_cond,
    )
    suffix_out = hidden_states[:, -model.config.action_horizon :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


# ===================================================================== compiled steering forward

def _eval_steer_forward_all(
    base_model,
    task_model,
    ref_model,
    base_images,
    base_img_masks,
    lang_tokens,
    lang_masks,
    base_state,
    task_images,
    task_img_masks,
    task_state,
    ref_images,
    ref_img_masks,
    ref_state,
    x_t,
    num_steps: int,
    proxy_action_dim: int,
    steer_step: torch.Tensor,
    steer_scale: torch.Tensor,
    share_proxy_dino: bool,
    only_steer: bool,
    no_steer: bool,
):
    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(base_images, base_img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(
        base_prefix_pad_masks, base_prefix_att_masks
    )
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1
    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        base_prefix_att_2d_masks
    )

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    task_prefix_embs, task_prefix_pad_masks, _ = task_model.embed_prefix(
        task_images, task_img_masks
    )
    if share_proxy_dino:
        ref_prefix_embs = task_prefix_embs
        ref_prefix_pad_masks = task_prefix_pad_masks
    else:
        ref_prefix_embs, ref_prefix_pad_masks, _ = ref_model.embed_prefix(
            ref_images, ref_img_masks
        )

    bsize = x_t.shape[0]
    device = x_t.device
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    for _ in range(num_steps):
        expanded_time = denoise_time.expand(bsize)
        base_v_t = base_model.denoise_step(
            base_state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_t,
            expanded_time,
        )
        task_v_t = _sequence_proxy_flow_from_prefix(
            task_model,
            task_state,
            task_prefix_embs,
            task_prefix_pad_masks,
            x_t,
            expanded_time,
            proxy_action_dim,
        )
        ref_v_t = _sequence_proxy_flow_from_prefix(
            ref_model,
            ref_state,
            ref_prefix_embs,
            ref_prefix_pad_masks,
            x_t,
            expanded_time,
            proxy_action_dim,
        )

        steer_mask = (denoise_time >= steer_step).to(dtype=base_v_t.dtype)
        if no_steer:
            v_t = base_v_t
        elif only_steer:
            steered_v_t = base_v_t.clone()
            steered_v_t[:, :, :proxy_action_dim] = task_v_t
            v_t = torch.where(steer_mask.to(dtype=torch.bool), steered_v_t, base_v_t)
        else:
            v_t = base_v_t.clone()
            v_t[:, :, :proxy_action_dim] += (
                steer_mask * steer_scale * (task_v_t - ref_v_t)
            )

        x_t = x_t + dt * v_t
        denoise_time = denoise_time + dt

    return x_t


_compiled_eval_steer_forward_all = None
_compiled_eval_steer_failed = False


def _get_compiled_eval_steer_forward():
    global _compiled_eval_steer_forward_all
    if _compiled_eval_steer_forward_all is None:
        if os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            _compiled_eval_steer_forward_all = _eval_steer_forward_all
            print("eval steer forward: using eager (torch.compile disabled)")
        else:
            _compiled_eval_steer_forward_all = torch.compile(
                _eval_steer_forward_all,
                mode="max-autotune",
            )
            print("eval steer forward: compiled with max-autotune")
    return _compiled_eval_steer_forward_all


_STAGE_CTX_KEYS = ("grasp_obj", "payload", "place_target", "constraint", "path_fns",
                   "held_idx", "held_offset", "keypoints", "place_point", "carry_z",
                   "destination", "task_tilt", "steer_events")


def _stage_stripped_context(ctx):
    """The unconditioned twin of a stage context: scene intact, stage structure neutralized.

    Stage-gated terms self-gate off (no grasp_obj/payload), the reach target becomes the hand
    itself (no pull), and only the stage-agnostic prior (smoothness, floor, keepouts) remains --
    the cost-space analog of CFG's unconditional branch."""
    out = {k: v for k, v in ctx.items() if k not in _STAGE_CTX_KEYS}
    out["grasp_obj"] = None
    out["payload"] = None
    out["place_target"] = None
    out["gripper_intent"] = None
    out["subtasks"] = {}
    out["placed"] = frozenset()
    eef = out.get("eef_pos")
    if eef is not None:
        out["target"] = np.asarray(eef, dtype=np.float32).reshape(-1)[:3]
    return out


# ====================================================== which arms are on, and what each forbids

def _score_steering_mode(args) -> str | None:
    if getattr(args, "full_steer", False):
        return "full"
    if getattr(args, "task_steer", False):
        return "task"
    if getattr(args, "vlm_base", False):
        return "base"
    return None


def _uses_vlm_mpc_base(args) -> bool:
    return _score_steering_mode(args) is not None


def _standalone_policy_role(args) -> str | None:
    if getattr(args, "ref_only", False):
        return "ref"
    if getattr(args, "task_only", False):
        return "task"
    return None


def _required_policy_roles(args) -> set[str]:
    standalone_role = _standalone_policy_role(args)
    if standalone_role is not None:
        return {standalone_role}

    score_mode = _score_steering_mode(args)
    # Proposal injection needs the task proxy loaded even in base-only mode: it supplies candidates,
    # not a score residual, so no reference is required.
    inject = float(getattr(args, "inject_proxy", 0.0)) > 0.0
    if score_mode == "base" or getattr(args, "no_steer", False):
        return {"base", "task"} if inject else {"base"}
    if score_mode == "task":
        return {"base", "task"}
    return {"base", "task", "ref"}


def _steering_mode_name(args) -> str:
    score_mode = _score_steering_mode(args)
    if score_mode == "base" or getattr(args, "no_steer", False):
        return "base_only"
    if score_mode == "full":
        return "base_plus_task_minus_ref"
    if score_mode == "task":
        return "base_to_task"
    if getattr(args, "only_steer", False):
        return "only_steer"
    return "task_minus_ref"


def _uses_accel_action_mpc(args) -> bool:
    return _uses_vlm_mpc_base(args) and getattr(args, "mpc_optimize_space", "action") == "accel"


def _base_source_name(args) -> str:
    standalone_role = _standalone_policy_role(args)
    if standalone_role is not None:
        return f"{standalone_role}_only"
    score_mode = _score_steering_mode(args)
    if score_mode is None:
        return "pi_checkpoint"
    if _uses_accel_action_mpc(args):
        return "accel_action_mppi"
    if score_mode == "full":
        return "mbd_full_steer"
    if score_mode == "task":
        return "mbd_task_steer"
    return "mbd_base"


def _base_decode_only_blockers(args) -> list[str]:
    """Reasons --base_decode_only cannot be honored; empty when the base network is unused.

    Every MBD mode drives the chunk from the planner: the sole base forward pass is gated on
    `use_vlm_mpc_base`, which is true for base, task and full score steering alike. Score
    steering adds a proxy score, not a base forward pass, so it decodes without base weights
    too -- only `base_model.config.*` and `sample_noise` are read, both of which the
    decode-only stub provides.
    """
    blockers = []
    if not _uses_vlm_mpc_base(args):
        blockers.append(
            f"base_source={_base_source_name(args)} forward-passes the base network "
            "(only the MBD planner bases decode without it)"
        )
    if getattr(args, "compare_difference", False):
        blockers.append("--compare_difference runs the base velocity field")
    return blockers


def _base_action_space_blockers(args) -> list[str]:
    """Reasons --base_action_space demo_delta cannot be honored; empty when the flag applies.

    Only the MBD planner bases decode through `decode_model_action_chunks`; every other base
    hands its chunk to the checkpoint's own output transforms, which cannot be re-scaled here.
    """
    blockers = []
    if not _uses_vlm_mpc_base(args):
        blockers.append(
            f"base_source={_base_source_name(args)} decodes through the checkpoint's own "
            "output transforms (demo-delta decoding is MBD-planner only)"
        )
    if _uses_accel_action_mpc(args):
        blockers.append(
            "--mpc_optimize_space accel plans in joint/acceleration space and never decodes "
            "through the action stats"
        )
    if not getattr(args, "base_action_stats", ""):
        blockers.append("--base_action_stats <action_norm_stats.json> is required")
    return blockers


def _disable_out_of_reach_colliders(env, env_ids, prim_path_regex, center, radius):
    """Prestartup: disable colliders in a static asset that the robot can never reach.

    Contact needs proximity, so a collider outside the robot's reach box cannot make one. Only
    `physics:collisionEnabled` is written, so prims stay visible and the rendered image -- the
    perception front-end's input -- is unchanged.
    """
    del env_ids
    import isaaclab.sim as sim_utils
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = get_current_stage()
    # Bounds are world-space, `center` is env-relative: the box must follow each env origin.
    origins = getattr(getattr(env, "scene", None), "env_origins", None)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    kept, disabled, unbounded = 0, 0, 0
    for env_index, root_path in enumerate(
        sorted(sim_utils.find_matching_prim_paths(prim_path_regex, stage))
    ):
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            continue
        origin = (
            [float(v) for v in origins[env_index]]
            if origins is not None and env_index < len(origins)
            else [0.0, 0.0, 0.0]
        )
        lo = [origin[i] + center[i] - radius for i in range(3)]
        hi = [origin[i] + center[i] + radius for i in range(3)]
        for prim in Usd.PrimRange(root):
            # Joints carry the same attribute, where it filters jointed-body collision instead.
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdPhysics.Joint):
                continue
            attr = prim.GetAttribute("physics:collisionEnabled")
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                # No usable bound: keep the collider. A diet must never guess a prim away.
                unbounded += 1
                kept += 1
                continue
            bmin, bmax = box.GetMin(), box.GetMax()
            if all(bmin[i] <= hi[i] and bmax[i] >= lo[i] for i in range(3)):
                kept += 1
                continue
            if not (attr and attr.IsValid()):
                attr = UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr()
            attr.Set(False)
            disabled += 1
    print(
        f"[scene] collider diet: disabled {disabled}, kept {kept} "
        f"({unbounded} kept for having no bound); reach box center={tuple(center)} "
        f"radius={radius} m; visuals untouched",
        flush=True,
    )


def _fast_gt_blockers(args) -> list[str]:
    """Reasons --fast_gt is unsafe here; empty only for a fully ground-truth-grounded run.

    It drops every camera and thins physics, which is free of decision content only when nothing
    in the loop reads pixels.
    """
    blockers = []
    if args.vlm_state != "gt":
        blockers.append(f"--vlm_state {args.vlm_state} (needs gt)")
    if args.vlm_track != "fk":
        blockers.append(f"--vlm_track {args.vlm_track} (needs fk)")
    if args.vlm_cost.startswith("rekep"):
        blockers.append(f"--vlm_cost {args.vlm_cost} proposes keypoints from images (needs gt)")
    if not getattr(args, "vlm_base", False):
        blockers.append("--fast_gt only applies to the geometric MBD base (--vlm_base)")
    # Catch-all: anything that reads pixels for any other reason.
    seen = " ".join(blockers)
    for consumer in _pixel_consumers(args):
        if consumer.split(" (")[0] not in seen:
            blockers.append(consumer)
    return blockers


def _pixel_consumers(args) -> list[str]:
    """Everything in this configuration that reads camera pixels."""
    consumers = []
    if not getattr(args, "base_decode_only", False) and _base_source_name(args) != "mbd_base":
        consumers.append("the base policy network")
    if _required_policy_roles(args) & {"task", "ref"}:
        consumers.append("a proxy policy (task/ref)")
    if args.vlm_state == "real":
        consumers.append("--vlm_state real (perception front-end)")
    if args.vlm_track != "fk":
        consumers.append(f"--vlm_track {args.vlm_track}")
    if args.vlm_cost.startswith("rekep"):
        consumers.append(f"--vlm_cost {args.vlm_cost} (ReKep keypoint proposal reads table_cam)")
    if getattr(args, "mpc_debug_video_overlay", False):
        consumers.append("--mpc_debug_video_overlay")
    return consumers


# ================================================================= action inference entry points

def _can_use_compiled_infer(base_policy, task_policy, ref_policy, args) -> bool:
    if _uses_vlm_mpc_base(args):
        return False
    if args.no_steer:
        return False
    if args.compare_difference:
        return False
    if base_policy._model.config.model_type not in (
        _model.ModelType.PI0,
        _model.ModelType.PI05,
    ):
        return False
    if task_policy._model.config.model_type != _model.ModelType.PROXY:
        return False
    if ref_policy._model.config.model_type != _model.ModelType.PROXY:
        return False
    return True


def _infer_actions_compiled(base_policy, task_policy, ref_policy, raw_obs, args):
    global _LAST_INFERENCE_RUNTIME
    base_obs, base_inputs = _obs_to_input_checked(base_policy, raw_obs, "base")
    task_obs, _ = _obs_to_input_checked(task_policy, raw_obs, "task")
    ref_obs, _ = _obs_to_input_checked(ref_policy, raw_obs, "ref")

    bsize = base_obs.state.shape[0]
    device = base_obs.state.device

    base_model = base_policy._model
    task_model = task_policy._model
    ref_model = ref_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = task_model.config.action_dim
    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    noise = base_model.sample_noise(actions_shape, device)

    base_images, base_img_masks, lang_tokens, lang_masks, base_state = (
        base_model._preprocess_observation(base_obs, train=False)
    )
    task_images, task_img_masks, task_state = task_model._preprocess_observation(
        task_obs, train=False
    )
    ref_images, ref_img_masks, ref_state = ref_model._preprocess_observation(
        ref_obs, train=False
    )

    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )
    share_proxy_dino = (
        getattr(task_model.config, "freeze_dino_encoder", False)
        and getattr(ref_model.config, "freeze_dino_encoder", False)
        and getattr(task_model.config, "dino_model_name", None)
        == getattr(ref_model.config, "dino_model_name", None)
    )

    forward_fn = _get_compiled_eval_steer_forward()
    x_t = forward_fn(
        base_model,
        task_model,
        ref_model,
        base_images,
        base_img_masks,
        lang_tokens,
        lang_masks,
        base_state,
        task_images,
        task_img_masks,
        task_state,
        ref_images,
        ref_img_masks,
        ref_state,
        noise,
        args.num_steps,
        proxy_action_dim,
        torch.as_tensor(0.0, dtype=torch.float32, device=device),
        torch.as_tensor(args.steer_scale, dtype=torch.float32, device=device),
        share_proxy_dino,
        args.only_steer,
        args.no_steer,
    )

    actions = base_policy.output_to_actions(base_inputs, x_t)
    _LAST_INFERENCE_RUNTIME = {
        "base_source": "pi_checkpoint",
        "used_base_model_velocity": True,
        "steering_mode": _steering_mode_name(args),
        "checked_vlm_task_ref_shapes": False,
        "x_t_shape": tuple(x_t.shape),
        "v_vlm_shape": None,
        "score_shape": None,
        "v_task_shape": None,
        "v_ref_shape": None,
        "proxy_task_shape": None,
        "proxy_ref_shape": None,
        "mpc_last": None,
    }
    return actions, {}


def infer_actions(base_policy, task_policy, ref_policy, raw_obs, args):
    global _compiled_eval_steer_failed
    if (
        not _compiled_eval_steer_failed
        and _can_use_compiled_infer(base_policy, task_policy, ref_policy, args)
    ):
        try:
            return _infer_actions_compiled(
                base_policy, task_policy, ref_policy, raw_obs, args
            )
        except Exception as exc:
            _compiled_eval_steer_failed = True
            print(
                f"Compiled eval inference failed once; falling back to eager infer_actions. Error: {exc}"
            )
    return _infer_actions_eager(base_policy, task_policy, ref_policy, raw_obs, args)


def infer_actions_with_mpc(
    base_policy,
    task_policy,
    ref_policy,
    raw_obs,
    args,
    *,
    mpc_planner=None,
    mpc_context=None,
    warm_shift_steps=0,
    base_decode_policy=None,
):
    global _LAST_INFERENCE_RUNTIME
    standalone_role = _standalone_policy_role(args)
    if standalone_role is not None:
        standalone_policy = ref_policy if standalone_role == "ref" else task_policy
        if standalone_policy is None:
            raise ValueError(f"--{standalone_role}_only requires a {standalone_role} checkpoint.")
        _validate_policy_environment_inputs(standalone_policy, raw_obs, standalone_role)
        outputs = standalone_policy.infer(raw_obs)
        actions = np.asarray(outputs["actions"], dtype=np.float32)
        # Clamp in executable joint space: the first target is relative to the current robot
        # state, each later one to the preceding clamped target.
        max_joint_delta = (
            args.mpc_joint_delta_clip if args.mpc_joint_delta_clip > 0.0 else None
        )
        actions = (
            clamp_real_action_chunk(
                torch.as_tensor(actions, dtype=torch.float32),
                current_joint_pos=raw_obs.get("observation/joint_position"),
                max_joint_delta=max_joint_delta,
            )
            .cpu()
            .numpy()
        )
        _LAST_INFERENCE_RUNTIME = {
            "base_source": f"{standalone_role}_only",
            "used_base_model_velocity": False,
            "steering_mode": f"{standalone_role}_only",
            "x_t_shape": None,
            "v_vlm_shape": None,
            "score_shape": None,
            "v_task_shape": None,
            "v_ref_shape": None,
            "proxy_task_shape": None,
            "proxy_ref_shape": None,
            "output_action_shape": tuple(actions.shape),
            "mpc_last": None,
            "mpc_trace": [],
        }
        return actions, {}
    if _uses_vlm_mpc_base(args) and mpc_planner is None:
        raise ValueError("VLM/MPC base mode requires a SimFreeMPC planner.")
    if not _uses_vlm_mpc_base(args):
        return infer_actions(base_policy, task_policy, ref_policy, raw_obs, args)
    if mpc_planner is None:
        return infer_actions(base_policy, task_policy, ref_policy, raw_obs, args)
    return _infer_actions_eager(
        base_policy,
        task_policy,
        ref_policy,
        raw_obs,
        args,
        mpc_planner=mpc_planner,
        mpc_context=mpc_context,
        warm_shift_steps=warm_shift_steps,
        base_decode_policy=base_decode_policy,
    )


def _infer_actions_eager(
    base_policy,
    task_policy,
    ref_policy,
    raw_obs,
    args,
    *,
    mpc_planner=None,
    mpc_context=None,
    warm_shift_steps=0,
    base_decode_policy=None,
):
    global _LAST_INFERENCE_RUNTIME
    base_obs, base_inputs = _obs_to_input_checked(base_policy, raw_obs, "base")

    bsize = base_obs.state.shape[0]
    device = base_obs.state.device
    need_compare = args.compare_difference
    use_vlm_mpc_base = _uses_vlm_mpc_base(args)
    score_steering_mode = _score_steering_mode(args)
    disable_steering = bool(getattr(args, "no_steer", False)) or score_steering_mode == "base"
    if score_steering_mode == "task":
        need_task = True
        need_ref = False
    elif score_steering_mode == "full":
        need_task = True
        need_ref = True
    else:
        need_task = (not disable_steering) or need_compare
        need_ref = need_task
    # Proposal injection needs the task proxy to propose candidates, but no reference: it extends the
    # candidate support and the geometric cost still weights every candidate.
    if float(getattr(args, "inject_proxy", 0.0)) > 0.0:
        need_task = True
    need_task_and_ref = need_task and need_ref
    if need_compare and use_vlm_mpc_base:
        raise ValueError(
            "--compare_difference compares against the pi checkpoint base path and is "
            "not compatible with --vlm_base."
        )
    if use_vlm_mpc_base and (mpc_planner is None or mpc_context is None):
        raise ValueError("VLM/MPC base mode requires mpc_planner and mpc_context.")

    base_model = base_policy._model
    task_model = task_policy._model if task_policy is not None else None
    ref_model = ref_policy._model if ref_policy is not None else None

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = task_model.config.action_dim if need_task else None
    compare_action_dim = proxy_action_dim - 1 if need_compare else None

    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    noise = base_model.sample_noise(actions_shape, device)

    if use_vlm_mpc_base:
        state = None
        base_prefix_pad_masks = None
        base_past_key_values = None
    else:
        images, img_masks, lang_tokens, lang_masks, state = (
            base_model._preprocess_observation(base_obs, train=False)
        )

        base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
            base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        )
        base_prefix_att_2d_masks = make_att_2d_masks(
            base_prefix_pad_masks, base_prefix_att_masks
        )
        base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1

        base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
            base_prefix_att_2d_masks
        )
        base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        _, base_past_key_values = base_model.paligemma_with_expert.forward(
            attention_mask=base_prefix_att_2d_masks_4d,
            position_ids=base_prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[base_prefix_embs, None],
            use_cache=True,
        )

    if need_task:
        if task_policy is None or task_model is None:
            raise ValueError("The selected steering mode requires a task policy.")
        task_obs, _ = _obs_to_input_checked(task_policy, raw_obs, "task")
        prepared_task = _prepare_proxy_steering(task_model, task_obs)
        if need_ref:
            if ref_policy is None or ref_model is None:
                raise ValueError("The selected steering mode requires a reference policy.")
            ref_obs, _ = _obs_to_input_checked(ref_policy, raw_obs, "ref")
            prepared_ref = _prepare_proxy_steering(ref_model, ref_obs)
        else:
            prepared_ref = None
        if (not use_vlm_mpc_base) and _is_score_proxy(task_model):
            raise ValueError(
                "ProxyScore checkpoints require --full_steer or --task_steer. "
                "The default pi-checkpoint path operates in flow/velocity space."
            )
        if (not use_vlm_mpc_base) and need_ref and _is_score_proxy(ref_model):
            raise ValueError(
                "ProxyScore checkpoints require --full_steer or --task_steer. "
                "The default pi-checkpoint path operates in flow/velocity space."
            )
    else:
        prepared_task = None
        prepared_ref = None

    # if the DINO encoder is frozen and the model names are the same, use the same prefix embeddings for mimic
    if (
        need_task
        and need_ref
        and
        getattr(task_model.config, "freeze_dino_encoder", False)
        and getattr(ref_model.config, "freeze_dino_encoder", False)
        and getattr(task_model.config, "dino_model_name", None)
        == getattr(ref_model.config, "dino_model_name", None)
        and prepared_task["kind"] == "sequence"
        and prepared_ref["kind"] == "sequence"
    ):
        prepared_ref["prefix_embs"] = prepared_task["prefix_embs"]
        prepared_ref["prefix_pad_masks"] = prepared_task["prefix_pad_masks"]

    dt = -1.0 / args.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    action_warm_started = False
    if use_vlm_mpc_base and args.mpc_update == "mbd_score_action_warm":
        x_t, action_warm_started = mpc_planner.warm_start_noise(
            noise,
            shift_steps=warm_shift_steps,
            current_state=base_inputs["state"],
        )
    else:
        x_t = noise
    teacher_path_x_t = noise.clone() if need_compare else None
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)
    runtime_stats = {
        "base_source": _base_source_name(args),
        "used_base_model_velocity": False,
        "steering_mode": _steering_mode_name(args),
        "checked_vlm_task_ref_shapes": False,
        "x_t_shape": tuple(x_t.shape),
        "v_vlm_shape": None,
        "score_shape": None,
        "v_task_shape": None,
        "v_ref_shape": None,
        "proxy_task_shape": None,
        "proxy_ref_shape": None,
        "mpc_last": None,
        "mpc_trace": [],
        "action_warm_started": bool(action_warm_started),
        "action_warm_shift_steps": int(warm_shift_steps) if action_warm_started else 0,
    }

    def record_mpc_stats(stats):
        if args.mpc_update == "mbd_score_action_warm":
            stats = dict(stats)
            stats.update(
                {
                    "action_warm_started": bool(action_warm_started),
                    "action_warm_shift_steps": int(warm_shift_steps) if action_warm_started else 0,
                }
            )
        runtime_stats["mpc_last"] = stats
        if args.mpc_debug:
            runtime_stats["mpc_trace"].append(_mpc_debug_stats(stats))

    if _uses_accel_action_mpc(args):
        if not disable_steering:
            raise ValueError("--mpc_optimize_space accel currently supports --vlm_base base-only mode.")
        if args.mpc_update not in ("legacy_score", "mbd_score"):
            raise ValueError(
                "Acceleration action-space MPC supports --mpc_update legacy_score or mbd_score. "
                "DDIM/action-prox variants are score-space only for this direct-action planner."
            )
        base_outputs = base_policy.infer(raw_obs)
        base_actions = np.asarray(base_outputs["actions"], dtype=np.float32)
        if base_actions.ndim != 2:
            raise ValueError(f"Expected base policy actions [H,D], got {tuple(base_actions.shape)}")
        if base_actions.shape[-1] < 8:
            current_gripper = raw_obs.get("observation/gripper_position")
            if current_gripper is None:
                raise ValueError(
                    "Acceleration action-space MPC needs a gripper trajectory, but base actions have "
                    f"{base_actions.shape[-1]} dims and no observation/gripper_position was found."
                )
            gripper_value = np.asarray(current_gripper, dtype=np.float32).reshape(-1)[0]
            gripper_traj = np.full((base_actions.shape[0], 1), gripper_value, dtype=np.float32)
        else:
            gripper_traj = base_actions[:, 7:8]
        gripper_tensor = torch.as_tensor(gripper_traj, device=device, dtype=torch.float32)
        if args.mpc_update == "mbd_score":
            planned_actions, geom_stats = mpc_planner.plan_mbd_score(
                context=mpc_context,
                gripper_traj=gripper_tensor,
                num_iterations=args.num_steps + 1,
                score_scale=args.gamma_base,
                device=device,
                dtype=torch.float32,
            )
        else:
            planned_actions, geom_stats = mpc_planner.plan(
                context=mpc_context,
                gripper_traj=gripper_tensor,
                device=device,
                dtype=torch.float32,
            )
        current_joint_pos = raw_obs.get("observation/joint_position")
        max_joint_delta = args.mpc_joint_delta_clip if args.mpc_joint_delta_clip > 0.0 else None
        actions = (
            clamp_real_action_chunk(
                planned_actions,
                current_joint_pos=current_joint_pos,
                max_joint_delta=max_joint_delta,
            )
            .detach()
            .cpu()
            .numpy()
        )
        runtime_stats["score_shape"] = tuple(planned_actions.shape)
        record_mpc_stats(geom_stats)
        if args.mpc_debug_stdout:
            print(
                f"vlm_mpc_{geom_stats['update_mode']} "
                f"cost_min={geom_stats['cost_min']:.4f} "
                f"cost_mean={geom_stats['cost_mean']:.4f} "
                f"cost_weighted={geom_stats['cost_weighted']:.4f} "
                f"target_delta_norm={geom_stats['target_delta_norm']:.4f} "
                f"accel_norm={geom_stats['accel_norm']:.4f} "
                f"score_norm={geom_stats.get('score_norm', 0.0):.4f}"
                f"{_format_mpc_term_debug(geom_stats)}",
                flush=True,
            )
        _LAST_INFERENCE_RUNTIME = runtime_stats
        return actions, {}

    if need_compare:
        shared_compare_stats = {
            "ref_minus_base": None,
            "task_minus_ref": None,
        }
        teacher_compare_stats = {
            "ref_minus_base": None,
            "task_minus_ref": None,
        }

    mpc_denoise_iteration = 0
    mpc_denoise_iterations = args.num_steps + 1
    if mpc_planner is not None:
        # Per-level planner traces (inject_weight_share) accumulate across the denoise chain; the
        # diagnostics for this inference are only emitted from the LAST level, so the trace has to
        # start empty here or it would carry over from the previous inference.
        mpc_planner.begin_inference()
    # --ddim_final_level: run the DDIM/score paths' last reverse transition.
    # ddim_iteration_alphas accepts iteration in [0, num_iterations), and at the last one
    # (iteration == num_steps) timestep is 0 and prev_timestep < 0, which is the only way to
    # reach its set_alpha_to_one branch -- i.e. the final refinement level. The default bound
    # (-dt/2) stops at iteration num_steps-1, so that level never runs and both the parameter
    # and that branch are dead. Opt-in, because running it changes every MPC inference and so
    # is not comparable to runs without it. The flow path integrates x_t from t=1 to t=0 in
    # exactly num_steps steps and must NOT take an extra one, so the bound stays path-local.
    _ddim_levels = bool(getattr(args, "ddim_final_level", False)) and use_vlm_mpc_base and (
        disable_steering or score_steering_mode in ("full", "task")
    )
    denoise_stop = (dt / 2) if _ddim_levels else (-dt / 2)
    while denoise_time >= denoise_stop:
        expanded_time = denoise_time.expand(bsize)

        if use_vlm_mpc_base and float(getattr(args, "inject_proxy", 0.0)) > 0.0:
            # Proposal injection: the task proxy's Tweedie clean action becomes the centre for a
            # fraction of the MBD candidates, so the expert extends the candidate SUPPORT while the
            # geometric cost still decides which candidate wins.
            _inj_alpha, _ = ddim_iteration_alphas(
                iteration=mpc_denoise_iteration,
                num_iterations=mpc_denoise_iterations,
                num_train_timesteps=args.mpc_ddim_train_timesteps,
            )
            # Schedule BEFORE the model call: under frontload the late levels carry rho ~1e-4, so
            # paying the proxy forward there cost 6.2 s/replan against the base's 3.5 for
            # candidates the schedule then wiped out.
            _inj_beta = max(1.0 - float(_inj_alpha), 1e-6)
            _inj_rho = float(args.inject_proxy)
            if args.inject_schedule == "frontload":
                _inj_rho *= _inj_beta
            if args.gated_inject:
                _inj_rho *= float(mpc_context.get("steer_authority", 0.0))
            if _inj_rho >= 1e-3:
                _inj_time = _proxy_score_time_cond(
                    args, mpc_denoise_iteration, x_t.device, x_t.dtype
                ).expand(bsize)
                _inj_score = _predict_proxy_score(
                    prepared_task, task_model, x_t, _inj_time
                )
                _inj_x0 = (
                    x_t[:, :, : _inj_score.shape[-1]] + _inj_beta * _inj_score
                ) / max(float(_inj_alpha), 1e-6) ** 0.5
                mpc_context["inject"] = {"x0": _inj_x0.detach()[0], "rho": _inj_rho}
            else:
                mpc_context.pop("inject", None)
        elif isinstance(mpc_context, dict):
            mpc_context.pop("inject", None)

        if use_vlm_mpc_base and disable_steering:
            if args.mpc_update == "legacy_score":
                x_t, geom_stats = mpc_planner.step_score_space(
                    x_t,
                    base_inputs,
                    mpc_context,
                    step_scale=args.gamma_base,
                )
            elif args.mpc_update == "mbd_score":
                x_t, geom_stats = mpc_planner.step_mbd_score(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    score_scale=args.gamma_base,
                )
            elif args.mpc_update == "mbd_score_action_prox" and args.exact_cfg > 0.0:
                # Exact cost-space CFG, no proxies: conditioned is the base's own estimate,
                # unconditioned the SAME estimator on a stage-stripped context. One measure and one
                # schedule, so measure mismatch and calibration drift cannot occur.
                base_score, base_numerator, geom_stats = (
                    mpc_planner.estimate_mbd_score_action_prox_terms(
                        x_t,
                        base_inputs,
                        mpc_context,
                        iteration=mpc_denoise_iteration,
                        num_iterations=mpc_denoise_iterations,
                    )
                )
                empty_score, _, _ = mpc_planner.estimate_mbd_score_action_prox_terms(
                    x_t,
                    base_inputs,
                    _stage_stripped_context(mpc_context),
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                )
                cfg_residual = base_score - empty_score
                cfg_gamma = float(args.exact_cfg)
                if args.gated_tilt:
                    cfg_gamma *= float(mpc_context.get("steer_authority", 0.0))
                x_t = mpc_planner.step_from_mbd_residual(
                    x_t,
                    base_numerator,
                    cfg_residual,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    base_scale=args.gamma_base,
                    residual_scale=cfg_gamma,
                    active_dims=int(geom_stats.get("active_dims", x_t.shape[-1])),
                )
                geom_stats = dict(geom_stats)
                geom_stats["exact_cfg_gamma"] = cfg_gamma
                geom_stats["exact_cfg_residual_norm"] = float(
                    torch.linalg.vector_norm(cfg_residual.detach()).cpu()
                )
            elif args.mpc_update == "mbd_score_action_prox":
                x_t, geom_stats = mpc_planner.step_mbd_score_action_prox(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    score_scale=args.gamma_base,
                )
            elif args.mpc_update == "mbd_score_action_warm":
                x_t, geom_stats = mpc_planner.step_mbd_score_action_warm(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    score_scale=args.gamma_base,
                )
            else:
                x_t, geom_stats = mpc_planner.step_ddim(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    step_scale=args.gamma_base,
                )
            runtime_stats["score_shape"] = tuple(x_t.shape)
            record_mpc_stats(geom_stats)
            if args.mpc_debug_stdout:
                print(
                    f"vlm_mpc_{geom_stats['update_mode']} "
                    f"cost_min={geom_stats['cost_min']:.4f} "
                    f"cost_mean={geom_stats['cost_mean']:.4f} "
                    f"cost_weighted={geom_stats['cost_weighted']:.4f} "
                    f"target_delta_norm={geom_stats['target_delta_norm']:.4f} "
                    f"score_norm={geom_stats['score_norm']:.4f}"
                    f"{_format_mpc_term_debug(geom_stats)}",
                    flush=True,
                )
            mpc_denoise_iteration += 1
            denoise_time += dt
            continue

        if use_vlm_mpc_base and score_steering_mode in ("full", "task"):
            if task_model is None or not _is_score_proxy(task_model):
                raise ValueError("Score steering requires a ProxyScore task checkpoint.")
            if score_steering_mode == "full" and (
                ref_model is None or not _is_score_proxy(ref_model)
            ):
                raise ValueError("Full score steering requires a ProxyScore reference checkpoint.")
            if args.mpc_update == "legacy_score":
                raise ValueError(
                    "Score steering does not support --mpc_update legacy_score."
                )

            score_time = _proxy_score_time_cond(
                args,
                mpc_denoise_iteration,
                device,
                x_t.dtype,
            ).expand(bsize)
            task_score = None
            if args.task_tilt > 0.0:
                # Tilted selection: the proxy enters as a per-candidate Gaussian
                # tilt inside the MBD softmax; the additive combine below then
                # runs with steer_scale forced to 0.
                task_score = _predict_proxy_score(
                    prepared_task,
                    task_model,
                    x_t,
                    score_time,
                )
                tilt_dir = task_score
                if score_steering_mode == "full":
                    # Residual-direction tilt (PPS Eq. 2 in cost space): the expert direction is
                    # task - ref, so what the pair AGREES on cancels and the tilt carries only the
                    # conditioning-induced change. Requires --ref_checkpoint_dir.
                    ref_score_tilt = _predict_proxy_score(
                        prepared_ref,
                        ref_model,
                        x_t,
                        score_time,
                    )
                    tilt_dir = task_score - ref_score_tilt
                tilt_abar, _ = ddim_iteration_alphas(
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    num_train_timesteps=args.mpc_ddim_train_timesteps,
                )
                tilt_beta = max(1.0 - float(tilt_abar), 1e-6)
                tilt_dims = tilt_dir.shape[-1]
                tilt_target = (
                    x_t[0, :, :tilt_dims].detach() + tilt_beta * tilt_dir[0].detach()
                ) / max(float(tilt_abar), 1e-6) ** 0.5
                mpc_context["task_tilt"] = {
                    "target": tilt_target,
                    "weight": task_tilt_weight(
                        args.task_tilt,
                        args.mpc_temperature,
                        args.mpc_noise,
                        float(tilt_abar),
                    ),
                    "dims": args.task_tilt_dims if args.task_tilt_dims > 0 else None,
                    "dim_weights": (
                        [1.0] * 7 + [args.task_tilt_gripper_weight]
                        if args.task_tilt_gripper_weight >= 0.0 else None
                    ),
                }
                if args.gated_tilt:
                    # Track-2 gated expert: authority from the bridge's plan-authored failure
                    # gate; discrimination + ESS cap are the implicit factors in the planner.
                    mpc_context["task_tilt"]["authority"] = float(
                        mpc_context.get("steer_authority", 0.0)
                    )
                    mpc_context["task_tilt"]["discrimination"] = True
                    mpc_context["task_tilt"]["ess_cap"] = args.tilt_ess_cap
            else:
                mpc_context.pop("task_tilt", None)

            if args.mpc_update == "mbd_score_action_prox":
                base_score, base_numerator, geom_stats = (
                    mpc_planner.estimate_mbd_score_action_prox_terms(
                        x_t,
                        base_inputs,
                        mpc_context,
                        iteration=mpc_denoise_iteration,
                        num_iterations=mpc_denoise_iterations,
                    )
                )
            elif args.mpc_update == "mbd_score_action_warm":
                base_score, base_numerator, geom_stats = (
                    mpc_planner.estimate_mbd_score_action_warm_terms(
                        x_t,
                        base_inputs,
                        mpc_context,
                        iteration=mpc_denoise_iteration,
                        num_iterations=mpc_denoise_iterations,
                    )
                )
            else:
                base_score, base_numerator, geom_stats = (
                    mpc_planner.estimate_mbd_score_terms(
                        x_t,
                        base_inputs,
                        mpc_context,
                        iteration=mpc_denoise_iteration,
                        num_iterations=mpc_denoise_iterations,
                    )
                )
            if task_score is None:
                task_score = _predict_proxy_score(
                    prepared_task,
                    task_model,
                    x_t,
                    score_time,
                )
            if task_score.shape[:2] != x_t.shape[:2] or task_score.shape[-1] > x_t.shape[-1]:
                raise ValueError(
                    "Task score shape is incompatible with x_t: "
                    f"task={tuple(task_score.shape)}, x_t={tuple(x_t.shape)}."
                )

            task_full_score = torch.zeros_like(x_t)
            task_full_score[:, :, : task_score.shape[-1]] = task_score
            ref_score = None
            ref_full_score = None
            if score_steering_mode == "full":
                ref_score = _predict_proxy_score(
                    prepared_ref,
                    ref_model,
                    x_t,
                    score_time,
                )
                if ref_score.shape != task_score.shape:
                    raise ValueError(
                        "task/ref score shapes must match: "
                        f"task={tuple(task_score.shape)}, ref={tuple(ref_score.shape)}."
                    )
                ref_full_score = torch.zeros_like(x_t)
                ref_full_score[:, :, : ref_score.shape[-1]] = ref_score

            step_steer_scale = steer_scale_for_stage(
                geom_stats.get("cost_stage"),
                default=args.steer_scale,
                grasp=args.grasp_steer_scale,
                lift=args.lift_steer_scale,
                place=args.place_steer_scale,
            )
            if args.steer_anneal:
                # Ramp lambda across the denoise trajectory instead of holding it constant.
                frac = mpc_denoise_iteration / max(mpc_denoise_iterations - 1, 1)
                step_steer_scale = float(
                    args.steer_anneal_start
                    + (args.steer_anneal_end - args.steer_anneal_start) * frac
                )
            if args.task_tilt > 0.0:
                # Steering already happened inside the softmax; no additive term.
                step_steer_scale = 0.0
            if args.steer_gamma_gripper is not None:
                # Per-channel gamma: arm dims keep the stage scale, the gripper channel
                # (action dim 7) gets its own gain. Dims beyond the action layout keep
                # the arm scale (the residual is zero there anyway).
                per_dim = torch.full(
                    (task_full_score.shape[-1],),
                    float(step_steer_scale),
                    dtype=task_full_score.dtype,
                    device=task_full_score.device,
                )
                if per_dim.shape[0] > 7:
                    per_dim[7] = float(args.steer_gamma_gripper)
                step_steer_scale = per_dim
            combined_score = combine_scores(
                base_score,
                task_full_score,
                mode=score_steering_mode,
                steer_scale=step_steer_scale,
                ref_score=ref_full_score,
                base_scale=args.gamma_base,
            )
            scaled_base_score = args.gamma_base * base_score
            residual_score = (
                task_full_score - ref_full_score
                if ref_full_score is not None
                else task_full_score - scaled_base_score
            )
            proxy_dims = task_score.shape[-1]
            base_proxy_score = base_score[..., :proxy_dims]
            task_proxy_score = task_full_score[..., :proxy_dims]
            residual_proxy_score = residual_score[..., :proxy_dims]
            combined_proxy_score = combined_score[..., :proxy_dims]
            score_state = x_t[..., :proxy_dims].detach()
            base_proxy_norm = torch.linalg.vector_norm(base_proxy_score.detach())
            task_proxy_norm = torch.linalg.vector_norm(task_proxy_score.detach())
            residual_proxy_norm = torch.linalg.vector_norm(residual_proxy_score.detach())
            if isinstance(step_steer_scale, torch.Tensor):
                applied_residual_norm = torch.linalg.vector_norm(
                    (step_steer_scale[:proxy_dims] * residual_proxy_score).detach()
                )
            else:
                applied_residual_norm = abs(step_steer_scale) * residual_proxy_norm

            active_dims = int(geom_stats.get("active_dims", task_score.shape[-1]))
            score_update_mode = _score_update_mode_for_mpc_update(args.mpc_update)
            if score_update_mode == "mbd_score":
                x_t = mpc_planner.step_from_mbd_residual(
                    x_t,
                    base_numerator,
                    residual_score,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    base_scale=args.gamma_base,
                    residual_scale=step_steer_scale,
                    active_dims=active_dims,
                )
            else:
                x_t = mpc_planner.step_from_score(
                    x_t,
                    combined_score,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    update_mode=score_update_mode,
                    score_scale=1.0,
                    active_dims=active_dims,
                )

            runtime_stats["score_shape"] = tuple(combined_score.shape)
            runtime_stats["proxy_task_shape"] = tuple(task_score.shape)
            runtime_stats["proxy_ref_shape"] = (
                tuple(ref_score.shape) if ref_score is not None else None
            )
            runtime_stats["v_task_shape"] = None
            runtime_stats["v_ref_shape"] = None
            runtime_stats["checked_vlm_task_ref_shapes"] = True
            geom_stats = dict(geom_stats)
            geom_stats.update(
                {
                    "update_mode": f"{args.mpc_update}_score_steer",
                    "score_steering_mode": score_steering_mode,
                    "score_base_norm": float(torch.linalg.vector_norm(base_score.detach()).cpu()),
                    "score_base_proxy_norm": float(
                        torch.linalg.vector_norm(base_proxy_score.detach()).cpu()
                    ),
                    "score_task_norm": float(torch.linalg.vector_norm(task_full_score.detach()).cpu()),
                    "score_task_base_ratio": float(
                        (task_proxy_norm / base_proxy_norm.clamp_min(1e-8)).cpu()
                    ),
                    "score_residual_norm": float(torch.linalg.vector_norm(residual_score.detach()).cpu()),
                    "score_residual_proxy_norm": float(residual_proxy_norm.cpu()),
                    "score_applied_residual_norm": float(applied_residual_norm.cpu()),
                    "score_applied_residual_ratio": float(
                        (applied_residual_norm / base_proxy_norm.clamp_min(1e-8)).cpu()
                    ),
                    "score_steer_scale": (
                        step_steer_scale.detach().cpu().tolist()
                        if isinstance(step_steer_scale, torch.Tensor)
                        else step_steer_scale
                    ),
                    "score_combined_norm": float(torch.linalg.vector_norm(combined_score.detach()).cpu()),
                    "score_base_task_cosine": _score_cosine(
                        base_proxy_score, task_proxy_score
                    ),
                    "score_base_combined_cosine": _score_cosine(
                        base_proxy_score, combined_proxy_score
                    ),
                    "proxy_score_time": float(score_time[0].detach().cpu()),
                }
            )
            if args.mpc_debug:
                geom_stats.update(
                    {
                        "score_state_values": score_state,
                        "score_base_values": base_proxy_score.detach(),
                        "score_task_values": task_proxy_score.detach(),
                        "score_combined_values": combined_proxy_score.detach(),
                    }
                )
            if ref_full_score is not None:
                geom_stats["score_ref_norm"] = float(
                    torch.linalg.vector_norm(ref_full_score.detach()).cpu()
                )
                geom_stats["score_ref_base_cosine"] = _score_cosine(
                    base_proxy_score, ref_score
                )
                geom_stats["score_ref_base_relative_error"] = _relative_score_error(
                    base_proxy_score, ref_score
                )
                geom_stats["score_task_ref_cosine"] = _score_cosine(task_score, ref_score)
            # Direction, not just magnitude: norms alone cannot tell a quiet proxy from an opposed one.
            if base_proxy_score.shape[-1] > 7:
                geom_stats["score_base_task_cosine_gripper"] = _score_cosine(
                    base_proxy_score[..., 7:8], task_proxy_score[..., 7:8]
                )
                geom_stats["score_base_task_cosine_arm"] = _score_cosine(
                    base_proxy_score[..., :7], task_proxy_score[..., :7]
                )
            if combined_score.shape[-1] > 7:
                score_components = {
                    "base": base_score,
                    "task": task_full_score,
                    "residual": residual_score,
                    "combined": combined_score,
                }
                if ref_full_score is not None:
                    score_components["ref"] = ref_full_score
                for component_name, component_score in score_components.items():
                    gripper_score = component_score[..., 7].detach()
                    geom_stats[f"score_{component_name}_gripper_first"] = float(
                        gripper_score.reshape(-1)[0].cpu()
                    )
                    geom_stats[f"score_{component_name}_gripper_mean"] = float(
                        gripper_score.mean().cpu()
                    )
                    geom_stats[f"score_{component_name}_gripper_norm"] = float(
                        torch.linalg.vector_norm(gripper_score).cpu()
                    )
            record_mpc_stats(geom_stats)
            if args.mpc_debug_stdout:
                print(
                    f"score_step={mpc_denoise_iteration} "
                    f"t={geom_stats['proxy_score_time']:.4f} "
                    f"base={geom_stats['score_base_proxy_norm']:.4f} "
                    f"task={geom_stats['score_task_norm']:.4f} "
                    f"task/base={geom_stats['score_task_base_ratio']:.4f} "
                    f"lambda={geom_stats['score_steer_scale']:.4f} "
                    f"lambda_residual/base={geom_stats['score_applied_residual_ratio']:.4f} "
                    f"cos_bt={geom_stats.get('score_base_task_cosine', float('nan')):+.3f} "
                    f"cos_arm={geom_stats.get('score_base_task_cosine_arm', float('nan')):+.3f} "
                    f"cos_grip={geom_stats.get('score_base_task_cosine_gripper', float('nan')):+.3f} "
                    f"base_grip0={geom_stats.get('score_base_gripper_first', float('nan')):.4f} "
                    f"task_grip0={geom_stats.get('score_task_gripper_first', float('nan')):.4f} "
                    f"base_grip_mean={geom_stats.get('score_base_gripper_mean', float('nan')):.4f} "
                    f"task_grip_mean={geom_stats.get('score_task_gripper_mean', float('nan')):.4f} "
                    f"base_grip_norm={geom_stats.get('score_base_gripper_norm', float('nan')):.4f} "
                    f"task_grip_norm={geom_stats.get('score_task_gripper_norm', float('nan')):.4f} "
                    f"update={geom_stats['update_mode']} "
                    f"cost_min={geom_stats['cost_min']:.4f} "
                    f"cost_mean={geom_stats['cost_mean']:.4f} "
                    f"cost_weighted={geom_stats['cost_weighted']:.4f} "
                    f"combined_score_norm={geom_stats['score_combined_norm']:.4f}"
                    f"{_format_mpc_term_debug(geom_stats)}",
                    flush=True,
                )
            mpc_denoise_iteration += 1
            denoise_time += dt
            continue

        if use_vlm_mpc_base:
            base_v_t, geom_stats = mpc_planner.step(
                x_t,
                base_inputs,
                mpc_context,
                dt=dt,
            )
            if base_v_t.shape != x_t.shape:
                raise ValueError(
                    "VLM/MPC base velocity shape must match x_t: "
                    f"got {tuple(base_v_t.shape)} vs {tuple(x_t.shape)}."
                )
            runtime_stats["v_vlm_shape"] = tuple(base_v_t.shape)
            record_mpc_stats(geom_stats)
            if args.mpc_debug_stdout:
                print(
                    "vlm_mpc_base "
                    f"cost_min={geom_stats['cost_min']:.4f} "
                    f"cost_mean={geom_stats['cost_mean']:.4f} "
                    f"cost_weighted={geom_stats['cost_weighted']:.4f} "
                    f"target_delta_norm={geom_stats['target_delta_norm']:.4f}"
                    f"{_format_mpc_term_debug(geom_stats)}",
                    flush=True,
                )
        else:
            base_v_t = base_model.denoise_step(
                state,
                base_prefix_pad_masks,
                base_past_key_values,
                x_t,
                expanded_time,
            )
            runtime_stats["used_base_model_velocity"] = True
        if need_compare:
            teacher_base_v_t = base_model.denoise_step(
                state,
                base_prefix_pad_masks,
                base_past_key_values,
                teacher_path_x_t,
                expanded_time,
            )

        if denoise_time >= 0.0:
            if need_task_and_ref:
                task_v_t = _predict_proxy_flow(
                    prepared_task, task_model, x_t, expanded_time
                )
                ref_v_t = _predict_proxy_flow(
                    prepared_ref, ref_model, x_t, expanded_time
                )
            else:
                task_v_t = None
                ref_v_t = None
            if need_task_and_ref:
                if task_v_t.shape != ref_v_t.shape:
                    raise ValueError(
                        "task/ref velocity shapes must match: "
                        f"task={tuple(task_v_t.shape)}, ref={tuple(ref_v_t.shape)}."
                    )
                if task_v_t.shape[:2] != x_t.shape[:2] or task_v_t.shape[-1] > x_t.shape[-1]:
                    raise ValueError(
                        "task/ref velocity shape is incompatible with x_t: "
                        f"task={tuple(task_v_t.shape)}, x_t={tuple(x_t.shape)}."
                    )
                runtime_stats["proxy_task_shape"] = tuple(task_v_t.shape)
                runtime_stats["proxy_ref_shape"] = tuple(ref_v_t.shape)
                runtime_stats["v_task_shape"] = tuple(task_v_t.shape)
                runtime_stats["v_ref_shape"] = tuple(ref_v_t.shape)
                if need_compare:
                    teacher_task_v_t = _predict_proxy_flow(
                        prepared_task, task_model, teacher_path_x_t, expanded_time
                    )
                    teacher_ref_v_t = _predict_proxy_flow(
                        prepared_ref, ref_model, teacher_path_x_t, expanded_time
                    )

            if need_compare:
                shared_base = base_v_t[:, :, :compare_action_dim]
                shared_task = task_v_t[:, :, :compare_action_dim]
                shared_ref = ref_v_t[:, :, :compare_action_dim]
                teacher_base = teacher_base_v_t[:, :, :compare_action_dim]
                teacher_task = teacher_task_v_t[:, :, :compare_action_dim]
                teacher_ref = teacher_ref_v_t[:, :, :compare_action_dim]

                shared_compare_stats["ref_minus_base"] = accumulate_stats(
                    shared_compare_stats["ref_minus_base"],
                    compute_batch_metrics(shared_base, shared_ref),
                )
                shared_compare_stats["task_minus_ref"] = accumulate_stats(
                    shared_compare_stats["task_minus_ref"],
                    compute_batch_metrics(shared_ref, shared_task),
                )
                teacher_compare_stats["ref_minus_base"] = accumulate_stats(
                    teacher_compare_stats["ref_minus_base"],
                    compute_batch_metrics(teacher_base, teacher_ref),
                )
                teacher_compare_stats["task_minus_ref"] = accumulate_stats(
                    teacher_compare_stats["task_minus_ref"],
                    compute_batch_metrics(teacher_ref, teacher_task),
                )

            if use_vlm_mpc_base:
                if disable_steering:
                    v_t = args.gamma_base * base_v_t
                    runtime_stats["checked_vlm_task_ref_shapes"] = False
                else:
                    task_full_v_t = torch.zeros_like(x_t)
                    ref_full_v_t = torch.zeros_like(x_t)
                    task_full_v_t[:, :, :proxy_action_dim] = task_v_t
                    ref_full_v_t[:, :, :proxy_action_dim] = ref_v_t
                    runtime_stats["v_task_shape"] = tuple(task_full_v_t.shape)
                    runtime_stats["v_ref_shape"] = tuple(ref_full_v_t.shape)
                    v_t = (
                        args.gamma_base * base_v_t
                        + args.steer_scale * (task_full_v_t - ref_full_v_t)
                    )
                    runtime_stats["checked_vlm_task_ref_shapes"] = True
                if args.only_steer and not disable_steering:
                    v_t = task_full_v_t
            else:
                if disable_steering:
                    v_t = base_v_t
                else:
                    v_t = base_v_t.clone()
                    v_t[:, :, :proxy_action_dim] += args.steer_scale * (task_v_t - ref_v_t)
                    if args.only_steer:  # testing the code for steering only
                        v_t[:, :, :proxy_action_dim] = task_v_t
        else:
            v_t = base_v_t

        x_t = x_t + dt * v_t
        if need_compare:
            teacher_path_x_t = teacher_path_x_t + dt * teacher_base_v_t
        denoise_time += dt

    if use_vlm_mpc_base and args.mpc_update == "mbd_score_action_warm":
        mpc_planner.set_warm_action(x_t, state=base_inputs["state"])

    if base_decode_policy is None:
        actions = base_policy.output_to_actions(base_inputs, x_t)
    else:
        # The executed chunk must leave the same space the planner scored it in, so it cannot go
        # through the checkpoint's output transforms.
        actions = (
            decode_model_action_chunks(base_decode_policy, base_inputs, x_t, apply_clamp=False)
            .real_actions[0]
            .detach()
            .cpu()
            .numpy()
        )
    if use_vlm_mpc_base:
        current_joint_pos = raw_obs.get("observation/joint_position")
        max_joint_delta = (
            None
            if args.sampler == "truncated"
            else args.mpc_joint_delta_clip if args.mpc_joint_delta_clip > 0.0 else None
        )
        action_tensor = torch.as_tensor(actions, device=device, dtype=torch.float32)
        actions = (
            clamp_real_action_chunk(
                action_tensor,
                current_joint_pos=current_joint_pos,
                max_joint_delta=max_joint_delta,
            )
            .detach()
            .cpu()
            .numpy()
        )
    compare_stats = {}
    if need_compare:
        compare_stats["shared_flow_path"] = {
            key: value
            for key, value in shared_compare_stats.items()
            if value is not None
        }
        compare_stats["teacher_denoise_path"] = {
            key: value
            for key, value in teacher_compare_stats.items()
            if value is not None
        }
    _LAST_INFERENCE_RUNTIME = runtime_stats
    return actions, compare_stats


# ============================================================ environment observation extraction

def get_pi_observation(env_obs_dict):
    obs = dict()
    joint_pos = _to_numpy_unbatched(env_obs_dict["joint_pos"])
    obs["observation/joint_position"] = joint_pos[:7]
    if joint_pos.shape[0] > 7:
        obs["observation/gripper_position"] = joint_pos[7:8]
    else:
        gripper_key = _find_first_present(
            env_obs_dict, ("gripper_pos", "gripper_position")
        )
        if gripper_key is None:
            raise KeyError("Missing gripper position in IsaacLab observation.")
        obs["observation/gripper_position"] = _to_numpy_unbatched(
            env_obs_dict[gripper_key]
        )[:1]

    table_cam_key = _find_first_present(env_obs_dict, ("table_cam",))
    if table_cam_key is not None:
        obs["observation/exterior_image_1_left"] = _to_numpy_unbatched(
            env_obs_dict[table_cam_key]
        )

    wrist_cam_key = _find_first_present(env_obs_dict, ("wrist_cam",))
    if wrist_cam_key is not None:
        obs["observation/wrist_image_left"] = _to_numpy_unbatched(
            env_obs_dict[wrist_cam_key]
        )

    thermal_table_cam_key = _find_first_present(
        env_obs_dict,
        ("thermal_table_cam", "thermal_exterior_image_1_left"),
    )
    if thermal_table_cam_key is not None:
        obs["observation/thermal_exterior_image_1_left"] = _to_numpy_unbatched(
            env_obs_dict[thermal_table_cam_key]
        )

    thermal_wrist_cam_key = _find_first_present(
        env_obs_dict,
        ("thermal_wrist_cam", "thermal_wrist_image_left"),
    )
    if thermal_wrist_cam_key is not None:
        obs["observation/thermal_wrist_image_left"] = _to_numpy_unbatched(
            env_obs_dict[thermal_wrist_cam_key]
        )

    point_coord_key = _find_first_present(
        env_obs_dict,
        ("pointcloud_coord", "point_positions", "point_position"),
    )
    point_color_key = _find_first_present(
        env_obs_dict,
        ("pointcloud_color", "point_color"),
    )
    if point_coord_key is not None and point_color_key is not None:
        obs["observation/pointcloud_coord"] = _to_numpy_unbatched(
            env_obs_dict[point_coord_key]
        )
        obs["observation/pointcloud_color"] = _to_numpy_unbatched(
            env_obs_dict[point_color_key]
        )
    else:
        pointcloud_key = _find_first_present(env_obs_dict, ("pointcloud", "point_cloud"))
        if pointcloud_key is not None:
            obs["observation/pointcloud"] = _to_numpy_unbatched(env_obs_dict[pointcloud_key])

    mic1_key = _find_first_present(env_obs_dict, ("mic1_log_mel",))
    mic2_key = _find_first_present(env_obs_dict, ("mic2_log_mel",))
    if mic1_key is not None and mic2_key is not None:
        obs["observation/mic1_log_mel"] = _to_numpy_unbatched(env_obs_dict[mic1_key])
        obs["observation/mic2_log_mel"] = _to_numpy_unbatched(env_obs_dict[mic2_key])
    else:
        sound_key = _find_first_present(env_obs_dict, ("sound",))
        if sound_key is not None:
            obs["observation/sound"] = _to_numpy_unbatched(env_obs_dict[sound_key])

    return obs


def _task_name_for_mpc(task_name: str) -> str:
    task = task_name.lower()
    for name in ("pot", "weight", "tea", "capsule"):
        if name in task:
            return name
    return task_name


def _context_tensor(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach()
        if value.ndim > 0 and value.shape[0] == 1:
            value = value[0]
        return value
    value = torch.as_tensor(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def _extract_subtasks(env_obs_dict):
    subtasks = {}
    raw = env_obs_dict.get("subtask_terms")
    if raw is None:
        return subtasks
    for key, value in raw.items():
        tensor = _context_tensor(value)
        if tensor is not None:
            subtasks[key] = tensor
    return subtasks


def _extract_scene_objects(env, names, env_origin=None):
    objects = {}
    for name in names:
        try:
            asset = env.scene[name]
        except Exception:
            continue
        data = getattr(asset, "data", None)
        if data is None:
            continue
        item = {}
        if hasattr(data, "root_pos_w"):
            pos = _context_tensor(data.root_pos_w)
            if pos is not None and env_origin is not None:
                pos = pos - env_origin.to(device=pos.device, dtype=pos.dtype)
            item["pos"] = pos
        if hasattr(data, "root_quat_w"):
            item["quat"] = _context_tensor(data.root_quat_w)
        if item:
            objects[name] = item
    return objects


def _extract_capsule_mpc_state(env, env_origin=None):
    """Extract the live lid link pose and joint position used by capsule_flow."""
    try:
        capsule = env.scene["capsule"]
    except Exception:
        return {}, None

    lid_joint_pos = None
    try:
        lid_joint_ids, _ = capsule.find_joints(["RevoluteJoint_capsule_coffee_maker_3_up"])
        if len(lid_joint_ids) == 1:
            lid_joint_pos = _context_tensor(capsule.data.joint_pos[:, lid_joint_ids[0]])
    except Exception:
        pass

    lid_objects = {}
    try:
        lid_body_ids, _ = capsule.find_bodies(["E_shell_8"])
        if len(lid_body_ids) == 1:
            lid_pos = _context_tensor(capsule.data.body_pos_w[:, lid_body_ids[0], :])
            lid_quat = _context_tensor(capsule.data.body_quat_w[:, lid_body_ids[0], :])
            if lid_pos is not None and env_origin is not None:
                lid_pos = lid_pos - env_origin.to(device=lid_pos.device, dtype=lid_pos.dtype)
            if lid_pos is not None and lid_quat is not None:
                lid_objects["capsule_lid"] = {"pos": lid_pos, "quat": lid_quat}
    except Exception:
        pass
    return lid_objects, lid_joint_pos


# ========================================================================== MPC context assembly

def build_mpc_context(env, env_obs_dict, args):
    policy_obs = env_obs_dict["policy"]
    env_origin = None
    if hasattr(env.scene, "env_origins"):
        env_origin = _context_tensor(env.scene.env_origins)
    robot_root_pos = None
    robot_root_quat = None
    try:
        robot = env.scene["robot"]
        robot_root_pos = _context_tensor(robot.data.root_pos_w)
        robot_root_quat = _context_tensor(robot.data.root_quat_w)
        if robot_root_pos is not None and env_origin is not None:
            robot_root_pos = robot_root_pos - env_origin.to(
                device=robot_root_pos.device,
                dtype=robot_root_pos.dtype,
            )
    except Exception:
        pass
    try:
        ee_frame = env.scene["ee_frame"]
        ee_frame.update(0.0, force_recompute=True)
        fk_source_pos = _context_tensor(ee_frame.data.source_pos_w)
        fk_source_quat = _context_tensor(ee_frame.data.source_quat_w)
        if fk_source_pos is not None:
            if env_origin is not None:
                fk_source_pos = fk_source_pos - env_origin.to(
                    device=fk_source_pos.device,
                    dtype=fk_source_pos.dtype,
                )
            robot_root_pos = fk_source_pos
        if fk_source_quat is not None:
            robot_root_quat = fk_source_quat
    except Exception:
        pass
    context = {
        "task": _task_name_for_mpc(args.task),
        "subtasks": _extract_subtasks(env_obs_dict),
        "joint_pos": _context_tensor(policy_obs.get("joint_pos")),
        "joint_vel": _context_tensor(policy_obs.get("joint_vel")),
        "eef_pos": _context_tensor(policy_obs.get("eef_pos")),
        "eef_quat": _context_tensor(policy_obs.get("eef_quat")),
        "gripper_pos": _context_tensor(policy_obs.get("gripper_pos")),
        "env_origin": env_origin,
        "robot_root_pos": robot_root_pos,
        "robot_root_quat": robot_root_quat,
    }
    context["objects"] = _extract_scene_objects(
        env,
        (
            "pot",
            "cover",
            "egg",
            "pear",
            "apple",
            "mango",
            "cabbage",
            "scale",
            "teapot",
            "teacup",
            "capsule",
            "can",
        ),
        env_origin=env_origin,
    )
    capsule_objects, capsule_lid_joint_pos = _extract_capsule_mpc_state(
        env,
        env_origin=env_origin,
    )
    context["objects"].update(capsule_objects)
    if capsule_lid_joint_pos is not None:
        context["capsule_lid_joint_pos"] = capsule_lid_joint_pos
    return context


# ================================================================ MPC debug logging and overlays

def _format_mpc_term_debug(stats: dict[str, Any], *, limit: int = 8) -> str:
    terms = []
    suffix = "_weighted"
    for key, value in stats.items():
        if not (key.startswith("term_") and key.endswith(suffix)):
            continue
        name = key[len("term_") : -len(suffix)]
        try:
            terms.append((name, float(value)))
        except (TypeError, ValueError):
            continue
    if not terms:
        return ""
    terms.sort(key=lambda item: abs(item[1]), reverse=True)
    text = ",".join(f"{name}:{value:.4g}" for name, value in terms[:limit])
    return f" terms={text}"


def _jsonable_debug_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            scalar = value.reshape(-1)[0].item()
            if isinstance(scalar, (bool, np.bool_)):
                return bool(scalar)
            if isinstance(scalar, (int, np.integer)):
                return int(scalar)
            return float(scalar)
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _jsonable_debug_value(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable_debug_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_debug_value(item) for item in value]
    return value


def _debug_subtasks(env_obs_dict) -> dict[str, bool]:
    return {
        key: bool(_jsonable_debug_value(value))
        for key, value in _extract_subtasks(env_obs_dict).items()
    }


def _observe_mpc_subtasks(mpc_planner, subtasks: dict[str, bool]) -> None:
    cost = getattr(mpc_planner, "cost", None)
    observer = getattr(cost, "observe_subtasks", None)
    if callable(observer):
        observer(subtasks)


def _debug_phase_from_subtasks(task_name: str, subtasks: dict[str, bool]) -> str:
    task = task_name.lower()
    if "weight" in task:
        if subtasks.get("grasp_apple", False):
            return "place_apple"
        if subtasks.get("grasp_pear", False):
            return "place_pear"
        if subtasks.get("pear_on_scale", False):
            return "grasp_apple"
        return "grasp_pear"
    if "capsule" in task:
        if subtasks.get("grasp_pod", False):
            return "place_pod"
        if subtasks.get("open_coffee_lid", False):
            return "grasp_pod"
        return "open_lid"
    return "unknown"


def _mpc_debug_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {}
    keys = (
        "update_mode",
        "cost_style",
        "cost_stage",
        "optimize_space",
        "cost_min",
        "cost_mean",
        "cost_std",
        "cost_weighted",
        "weight_max",
        "weight_ess",
        "weight_entropy",
        "task_tilt_weight",
        "task_tilt_base_cost_mean",
        "task_tilt_authority",
        "task_tilt_lambda_eff",
        "exact_cfg_gamma",
        "exact_cfg_residual_norm",
        "steer_authority",
        "steer_events_closed_empty",
        "steer_events_backtracks",
        "stage_env_steps",
        "fk_fork_m",
        "fk_best_cost",
        "fk_cost_spread",
        "inject_rho",
        "inject_weight_share",
        "inject_share_first",
        "inject_share_max",
        "inject_share_mean",
        "inject_share_levels",
        "cost_feasibility_best",
        "cost_feasibility_min",
        "cost_feasibility_weighted",
        "cost_task_best",
        "cost_task_min",
        "cost_task_weighted",
        "cost_prior_best",
        "cost_prior_min",
        "cost_prior_weighted",
        # Under --rank_mode roles, cost_* is the RANKING scalar; cost_true_total_* is the real cost,
        # logged so an arm that ranks differently stays comparable to one that ranked on the total.
        "rank_mode",
        "prior_weight",
        "prior_weight_schedule",
        "cost_true_total_best",
        "cost_true_total_min",
        "cost_true_total_weighted",
        "target_delta_norm",
        "accel_norm",
        "score_norm",
        "score_base_norm",
        "score_base_proxy_norm",
        "score_task_norm",
        "score_task_base_ratio",
        "score_ref_norm",
        "score_residual_norm",
        "score_residual_proxy_norm",
        "score_applied_residual_norm",
        "score_applied_residual_ratio",
        "score_steer_scale",
        "score_combined_norm",
        "score_base_task_cosine",
        "score_base_combined_cosine",
        "score_ref_base_cosine",
        "score_ref_base_relative_error",
        "score_task_ref_cosine",
        "proxy_score_time",
        "score_base_gripper_first",
        "score_base_gripper_mean",
        "score_base_gripper_norm",
        "score_task_gripper_first",
        "score_task_gripper_mean",
        "score_task_gripper_norm",
        "score_ref_gripper_first",
        "score_ref_gripper_mean",
        "score_ref_gripper_norm",
        "score_residual_gripper_first",
        "score_residual_gripper_mean",
        "score_residual_gripper_norm",
        "score_combined_gripper_first",
        "score_combined_gripper_mean",
        "score_combined_gripper_norm",
        "gripper_mean",
        "proposal_center",
        "proposal_noise_scale",
        "action_warm_started",
        "action_warm_shift_steps",
        "score_state_values",
        "score_base_values",
        "score_task_values",
        "score_combined_values",
    )
    payload = {key: _jsonable_debug_value(stats[key]) for key in keys if key in stats}
    # --mpc_eval_mean_plan: the cost of the plan that actually executes, which no whitelisted key
    # carries. Grouped rather than listed one by one because the per-term rows are named after
    # whichever cost terms the config selected.
    mean_plan = {k[len("mean_plan_"):]: _jsonable_debug_value(v)
                 for k, v in stats.items() if k.startswith("mean_plan_")}
    if mean_plan:
        payload["mean_plan"] = dict(sorted(mean_plan.items()))
    best_terms = {}
    weighted_terms = {}
    best_debug = {}
    weighted_debug = {}
    for key, value in stats.items():
        if key.startswith("term_") and key.endswith("_best"):
            best_terms[key[len("term_") : -len("_best")]] = _jsonable_debug_value(value)
        elif key.startswith("term_") and key.endswith("_weighted"):
            weighted_terms[key[len("term_") : -len("_weighted")]] = _jsonable_debug_value(value)
        elif key.startswith("debug_") and key.endswith("_best"):
            best_debug[key[len("debug_") : -len("_best")]] = _jsonable_debug_value(value)
        elif key.startswith("debug_") and key.endswith("_weighted"):
            weighted_debug[key[len("debug_") : -len("_weighted")]] = _jsonable_debug_value(value)
    if weighted_terms:
        payload["terms_weighted"] = dict(
            sorted(weighted_terms.items(), key=lambda item: abs(float(item[1])), reverse=True)
        )
    if best_terms:
        payload["terms_best"] = dict(
            sorted(best_terms.items(), key=lambda item: abs(float(item[1])), reverse=True)
        )
    if weighted_debug:
        payload["debug_weighted"] = dict(sorted(weighted_debug.items()))
    if best_debug:
        payload["debug_best"] = dict(sorted(best_debug.items()))
    return payload


def _write_mpc_debug_log(handle, event: str, **payload) -> None:
    if handle is None:
        return
    record = {
        "time": time.time(),
        "event": event,
        **payload,
    }
    handle.write(json.dumps(_jsonable_debug_value(record), sort_keys=True) + "\n")
    handle.flush()


def _debug_action_gripper(action_step) -> float | None:
    action = np.asarray(action_step).reshape(-1)
    if action.shape[0] <= 7:
        return None
    return float(action[7])


def _collect_mpc_debug_frames(env, *, axis_length: float = 0.08):
    try:
        ee_frame = env.scene["ee_frame"]
        ee_frame.update(0.0, force_recompute=True)

        ee_pos = ee_frame.data.target_pos_w[:1]
        ee_quat = ee_frame.data.target_quat_w[:1]

        frames = []
        is_capsule_task = False
        try:
            coffee_maker = env.scene["capsule"]
            pod = env.scene["can"]
            is_capsule_task = True
            frames.append(("gripper", ee_pos[0, 0], ee_quat[0, 0]))

            pod_pos = pod.data.root_pos_w[:1]
            pod_quat = pod.data.root_quat_w[:1]
            frames.append(("capsule", pod_pos[0], pod_quat[0]))

            lid_body_ids, _ = coffee_maker.find_bodies(["E_shell_8"])
            if len(lid_body_ids) == 1:
                lid_body_id = lid_body_ids[0]
                lid_pos = coffee_maker.data.body_pos_w[:1, lid_body_id, :]
                lid_quat = coffee_maker.data.body_quat_w[:1, lid_body_id, :]
                frames.append(("lid", lid_pos[0], lid_quat[0]))
        except Exception:
            pass

        if not is_capsule_task:
            frames.append(("ee", ee_pos[0, 0], ee_quat[0, 0]))
            try:
                pear = env.scene["pear"]
                pear_pos = pear.data.root_pos_w[:1]
                pear_quat = pear.data.root_quat_w[:1]
                frames.append(("pear", pear_pos[0], pear_quat[0]))
            except Exception:
                pass

        try:
            scale = env.scene["scale"]
            scale_pos = scale.data.root_pos_w[:1]
            scale_quat = scale.data.root_quat_w[:1]
            scale_top_offset = torch.tensor(
                [
                    _WEIGHT_SCALE_CENTER_OFFSET_DEBUG[0],
                    _WEIGHT_SCALE_CENTER_OFFSET_DEBUG[1],
                    _WEIGHT_SCALE_CENTER_OFFSET_DEBUG[2] + _WEIGHT_SCALE_TOP_OFFSET_Z_DEBUG,
                ],
                device=scale_pos.device,
                dtype=scale_pos.dtype,
            ).view(1, 3)
            scale_top_pos = scale_pos + _quat_apply_wxyz(scale_quat, scale_top_offset)
            frames.append(("scale_top", scale_top_pos[0], scale_quat[0]))
        except Exception:
            pass
        if ee_pos.shape[1] >= 3:
            frames.extend(
                [
                    ("rf", ee_pos[0, 1], ee_quat[0, 1]),
                    ("lf", ee_pos[0, 2], ee_quat[0, 2]),
                ]
            )

        axes = {}
        local_axes = torch.eye(3, device=ee_pos.device, dtype=ee_pos.dtype)
        for name, pos, quat in frames:
            endpoints = [pos]
            for axis in local_axes:
                endpoints.append(pos + axis_length * _quat_apply_wxyz(quat, axis))
            axes[name] = torch.stack(endpoints, dim=0).detach()
        return axes
    except Exception as exc:
        print(f"mpc_debug_video_overlay collect failed: {exc}", flush=True)
        return {}


def _project_world_points_to_camera(points_w: torch.Tensor, camera) -> np.ndarray:
    data = camera.data
    cam_pos = data.pos_w[:1].to(device=points_w.device, dtype=points_w.dtype)
    cam_quat = data.quat_w_ros[:1].to(device=points_w.device, dtype=points_w.dtype)
    intr = data.intrinsic_matrices[0].to(device=points_w.device, dtype=points_w.dtype)
    points_cam = _quat_apply_inverse_wxyz(cam_quat, points_w - cam_pos).reshape(-1, 3)
    z = points_cam[:, 2]
    pixels = torch.full((points_cam.shape[0], 2), float("nan"), device=points_w.device, dtype=points_w.dtype)
    valid = z > 1e-4
    if valid.any():
        pixels[valid, 0] = intr[0, 0] * points_cam[valid, 0] / z[valid] + intr[0, 2]
        pixels[valid, 1] = intr[1, 1] * points_cam[valid, 1] / z[valid] + intr[1, 2]
    return pixels.detach().cpu().numpy()


def _draw_projected_debug_axes(image: np.ndarray, env, camera_name: str, axes: dict[str, torch.Tensor]) -> np.ndarray:
    if not axes:
        return image
    try:
        camera = env.scene[camera_name]
    except Exception:
        return image
    out = image.copy()
    height, width = out.shape[:2]
    axis_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    label_colors = {
        "pear": (255, 255, 255),
        "scale_top": (128, 255, 128),
        "ee": (255, 255, 0),
        "gripper": (255, 255, 0),
        "lid": (0, 165, 255),
        "capsule": (255, 128, 255),
        "rf": (255, 0, 255),
        "lf": (0, 255, 255),
    }
    for name, points in axes.items():
        pixels = _project_world_points_to_camera(points, camera)
        if not np.isfinite(pixels[0]).all():
            continue
        origin = tuple(np.round(pixels[0]).astype(int))
        if not (0 <= origin[0] < width and 0 <= origin[1] < height):
            continue
        cv2.circle(out, origin, 3, label_colors.get(name, (255, 255, 255)), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            name,
            (origin[0] + 4, origin[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            label_colors.get(name, (255, 255, 255)),
            1,
            cv2.LINE_AA,
        )
        for axis_idx, color in enumerate(axis_colors, start=1):
            if not np.isfinite(pixels[axis_idx]).all():
                continue
            end = tuple(np.round(pixels[axis_idx]).astype(int))
            cv2.line(out, origin, end, color, 2, lineType=cv2.LINE_AA)
    return out


def _to_uint8_image(image):
    image = _to_numpy_unbatched(image)
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    if image.size > 0 and np.nanmax(image) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _has_thermal_observation(obs):
    return (
        "observation/thermal_exterior_image_1_left" in obs
        and "observation/thermal_wrist_image_left" in obs
    )


def _has_sound_observation(obs):
    return "observation/sound" in obs or (
        "observation/mic1_log_mel" in obs and "observation/mic2_log_mel" in obs
    )


def _get_sound_spectrograms(obs):
    if "observation/mic1_log_mel" in obs and "observation/mic2_log_mel" in obs:
        return obs["observation/mic1_log_mel"], obs["observation/mic2_log_mel"]

    sound = _to_numpy_unbatched(obs["observation/sound"])
    sound = np.asarray(sound)
    if sound.ndim == 3 and sound.shape[0] == 2:
        return sound[0], sound[1]
    if sound.ndim == 3 and sound.shape[-1] == 2:
        return sound[..., 0], sound[..., 1]
    raise ValueError(f"Expected sound shape [2, F, T] or [F, T, 2], got {sound.shape}.")


def _to_numpy_spectrogram(spectrogram):
    spectrogram = _to_numpy_unbatched(spectrogram)
    spectrogram = np.asarray(spectrogram)
    while spectrogram.ndim > 2 and spectrogram.shape[0] == 1:
        spectrogram = spectrogram[0]
    if spectrogram.ndim != 2:
        raise ValueError(f"Expected spectrogram shape [F, T], got {spectrogram.shape}.")
    return spectrogram.astype(np.float32)


def _build_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max):
    hz_to_mel = lambda freq_hz: 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft // 2)

    mel_fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for mel_idx in range(n_mels):
        left = bin_indices[mel_idx]
        center = bin_indices[mel_idx + 1]
        right = bin_indices[mel_idx + 2]
        if center > left:
            mel_fb[mel_idx, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            mel_fb[mel_idx, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return mel_fb


def _get_sound_video_scale():
    global _SOUND_VIDEO_SCALE
    if _SOUND_VIDEO_SCALE is not None:
        return _SOUND_VIDEO_SCALE

    sample_rate = 48_000
    n_fft = 2048
    n_mels = 80
    f_min = 50.0
    eps = 1e-8
    reference_distance = 1.0
    attenuation_power = _SOUND_AUDIO_ATTENUATION_POWER

    try:
        import soundfile as sf
        from scipy import signal

        audio, sr = sf.read(_phone_ringtone_path(), always_2d=True)
        audio = audio.mean(axis=1).astype(np.float64)
        if sr != sample_rate:
            gcd = np.gcd(sr, sample_rate)
            audio = signal.resample_poly(audio, up=sample_rate // gcd, down=sr // gcd)

        peak = np.max(np.abs(audio))
        if peak > 0.0:
            audio = audio / peak

        win_length = int(round(25.0 / 1000.0 * sample_rate))
        hop_length = int(round(10.0 / 1000.0 * sample_rate))
        _, _, zxx = signal.stft(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        mel_fb = _build_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=sample_rate / 2,
        )
        base_log_mel = np.log(mel_fb @ (np.abs(zxx) ** 2) + eps)
        gain_at_max = 2.0 * attenuation_power * np.log(reference_distance / _SOUND_VIDEO_MAX_DISTANCE_M)
        vmax = float(np.percentile(base_log_mel, 99) + gain_at_max)
    except Exception as exc:
        print(f"[WARN] Failed to compute global sound video scale: {exc}")
        vmax = 2.0

    vmin = float(np.log(eps))
    if vmax <= vmin:
        vmax = vmin + 1.0
    _SOUND_VIDEO_SCALE = (vmin, vmax)
    return _SOUND_VIDEO_SCALE


def _load_phone_ringtone_audio(sample_rate=_SOUND_AUDIO_SAMPLE_RATE):
    global _SOUND_AUDIO_CACHE
    if _SOUND_AUDIO_CACHE is not None and _SOUND_AUDIO_CACHE["sample_rate"] == sample_rate:
        return _SOUND_AUDIO_CACHE["audio"]

    import soundfile as sf
    from scipy import signal

    audio, sr = sf.read(_phone_ringtone_path(), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float64)
    if sr != sample_rate:
        gcd = np.gcd(sr, sample_rate)
        audio = signal.resample_poly(audio, up=sample_rate // gcd, down=sr // gcd)

    peak = np.max(np.abs(audio))
    if peak > 0.0:
        audio = audio / peak
    if len(audio) == 0:
        raise ValueError("Phone ringtone WAV is empty.")

    _SOUND_AUDIO_CACHE = {"sample_rate": sample_rate, "audio": audio.astype(np.float32)}
    return _SOUND_AUDIO_CACHE["audio"]


def _quat_apply_wxyz(quat, vec):
    quat_xyz = quat[..., 1:]
    quat_w = quat[..., :1]
    t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat_w * t + torch.cross(quat_xyz, t, dim=-1)


def _quat_apply_inverse_wxyz(quat, vec):
    quat_inv = quat.clone()
    quat_inv[..., 1:] = -quat_inv[..., 1:]
    return _quat_apply_wxyz(quat_inv, vec)


def _get_gripper_mic_distances(env, mic_spacing=0.25, mic_axis=(0.0, 1.0, 0.0)):
    try:
        ee_frame = env.scene["ee_frame"]
        phone = env.scene["phone_1"]
    except KeyError:
        return None

    ee_pos_w = ee_frame.data.target_pos_w[:1, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:1, 0, :]
    axis_local = torch.tensor(mic_axis, dtype=ee_pos_w.dtype, device=ee_pos_w.device).view(1, 3)
    axis_local = axis_local / torch.linalg.vector_norm(axis_local, dim=1, keepdim=True).clamp_min(1e-6)
    axis_w = _quat_apply_wxyz(ee_quat_w, axis_local)

    mic_offset_w = 0.5 * mic_spacing * axis_w
    mic1_pos_w = ee_pos_w + mic_offset_w
    mic2_pos_w = ee_pos_w - mic_offset_w
    phone_pos_w = phone.data.root_pos_w[:1]

    distance_mic1 = torch.linalg.vector_norm(mic1_pos_w - phone_pos_w, dim=1).clamp_min(
        _SOUND_AUDIO_MIN_DISTANCE
    )
    distance_mic2 = torch.linalg.vector_norm(mic2_pos_w - phone_pos_w, dim=1).clamp_min(
        _SOUND_AUDIO_MIN_DISTANCE
    )
    return float(distance_mic1.item()), float(distance_mic2.item())


def _build_stereo_sound_audio_frame(
    env,
    frame_index,
    fps=15,
    sample_rate=_SOUND_AUDIO_SAMPLE_RATE,
):
    distances = _get_gripper_mic_distances(env)
    if distances is None:
        return None

    source_audio = _load_phone_ringtone_audio(sample_rate=sample_rate)
    start_sample = int(round(frame_index * sample_rate / fps))
    end_sample = int(round((frame_index + 1) * sample_rate / fps))
    sample_indices = np.arange(start_sample, end_sample, dtype=np.int64)
    mono = source_audio[np.mod(sample_indices, len(source_audio))]

    distance_mic1, distance_mic2 = distances
    gain_mic1 = (_SOUND_AUDIO_REFERENCE_DISTANCE / distance_mic1) ** _SOUND_AUDIO_ATTENUATION_POWER
    gain_mic2 = (_SOUND_AUDIO_REFERENCE_DISTANCE / distance_mic2) ** _SOUND_AUDIO_ATTENUATION_POWER
    return np.stack((mono * gain_mic1, mono * gain_mic2), axis=1).astype(np.float32)


def _write_stereo_sound_audio(audio_path, audio_frames, sample_rate=_SOUND_AUDIO_SAMPLE_RATE):
    if not audio_frames:
        return None

    import soundfile as sf

    audio = np.concatenate(audio_frames, axis=0)
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.98
    sf.write(audio_path, audio, sample_rate)
    return audio_path


def _visualize_log_mel_spectrogram(
    spectrogram,
    target_height,
    target_width,
    label,
    vmin=None,
    vmax=None,
):
    spectrogram = _to_numpy_spectrogram(spectrogram)
    spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)

    if vmin is None:
        vmin = float(np.percentile(spectrogram, 1))
    if vmax is None:
        vmax = float(np.percentile(spectrogram, 99))
    if vmax <= vmin:
        normalized = np.zeros_like(spectrogram, dtype=np.float32)
    else:
        normalized = (spectrogram - vmin) / (vmax - vmin)

    image = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    image = np.flipud(image)
    image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return image


def _ffmpeg_exe():
    """ffmpeg binary: PATH if present, else the imageio-ffmpeg bundled build (if installed)."""
    exe = shutil.which("ffmpeg")
    if exe is not None:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_h264(video_path, label="video"):
    """Re-encode an OpenCV 'mp4v' file to H.264/yuv420p in place so browsers/VSCode can play it.

    cv2.VideoWriter's mp4v (MPEG-4 Part 2) does not decode in Chromium-based players. On any
    failure (no ffmpeg, transcode error) the original mp4v file is kept.
    """
    exe = _ffmpeg_exe()
    if exe is None:
        print(f"[WARN] no ffmpeg available; {label} left as mp4v (not browser-playable)", flush=True)
        return
    tmp = video_path + ".h264.mp4"
    cmd = [exe, "-y", "-loglevel", "error", "-i", video_path,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp, video_path)
    except Exception as exc:
        print(f"[WARN] H.264 transcode failed for {label}: {exc}", flush=True)
        try:
            os.remove(tmp)
        except OSError:
            pass


def _mux_audio_into_video(tmp_video_path, final_video_path, audio_path, label):
    if audio_path is None or not os.path.exists(audio_path):
        os.replace(tmp_video_path, final_video_path)
        return

    mux_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        tmp_video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        final_video_path,
    ]
    try:
        subprocess.run(mux_cmd, check=True)
        os.remove(tmp_video_path)
        os.remove(audio_path)
        print(f"{label} video saved with stereo sound")
    except Exception as exc:
        print(f"[WARN] Failed to mux {label} audio into video: {exc}")
        try:
            os.remove(audio_path)
        except OSError:
            pass
        os.replace(tmp_video_path, final_video_path)


def _overlay_thermal_on_rgb(rgb_image, thermal_image, alpha=0.45):
    rgb_image = _to_uint8_image(rgb_image)
    thermal_image = _to_uint8_image(thermal_image)

    if rgb_image.shape[:2] != thermal_image.shape[:2]:
        thermal_image = cv2.resize(
            thermal_image,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return cv2.addWeighted(rgb_image, 1.0 - alpha, thermal_image, alpha, 0.0)


def _draw_vlm_overlay(image: np.ndarray, env, camera_name: str, vlm: dict) -> np.ndarray:
    """Draw the VLM bridge's live state: numbered keypoints, the stage's target/seat geometry,
    and the payload->seat line (the active place constraint)."""
    try:
        camera = env.scene[camera_name]
    except Exception:
        return image
    out = image.copy()
    height, width = out.shape[:2]

    def px(p):
        pts = torch.as_tensor(np.asarray(p, dtype=np.float32).reshape(-1, 3))
        pix = _project_world_points_to_camera(pts, camera)
        good = []
        for u, v in pix:
            if np.isfinite((u, v)).all() and 0 <= u < width and 0 <= v < height:
                good.append((int(round(u)), int(round(v))))
            else:
                good.append(None)
        return good

    kps = vlm.get("keypoints")
    if kps is not None and len(kps):
        for i, at in enumerate(px(kps)):
            if at is None:
                continue
            cv2.circle(out, at, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(out, str(i), (at[0] + 3, at[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    payload = vlm.get("payload")
    seat = vlm.get("seat")
    if payload is not None and seat is not None:
        a, b = px(np.stack([payload, seat]))
        if a is not None and b is not None:
            cv2.line(out, a, b, (255, 0, 255), 1, cv2.LINE_AA)
    for key, color, marker in (("target", (0, 255, 0), cv2.MARKER_CROSS),
                               ("seat", (255, 0, 255), cv2.MARKER_TILTED_CROSS),
                               ("payload", (0, 165, 255), cv2.MARKER_DIAMOND)):
        p = vlm.get(key)
        if p is None:
            continue
        at = px(p)[0]
        if at is not None:
            cv2.drawMarker(out, at, color, marker, 10, 1, cv2.LINE_AA)
    cv2.putText(out, f"{vlm.get('stage', '')}  {'HOLDING' if vlm.get('holding') else 'open'}",
                (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _build_rollout_frame(obs, use_thermal_overlay=False, debug_overlay=None):
    table_image = _to_uint8_image(obs["observation/exterior_image_1_left"])
    # Under a render diet the wrist camera may be gone; the frame is then table-only.
    wrist_raw = obs.get("observation/wrist_image_left")
    wrist_image = None if wrist_raw is None else _to_uint8_image(wrist_raw)

    if use_thermal_overlay and _has_thermal_observation(obs):
        table_image = _overlay_thermal_on_rgb(
            table_image, obs["observation/thermal_exterior_image_1_left"]
        )
        if wrist_image is not None:
            wrist_image = _overlay_thermal_on_rgb(
                wrist_image, obs["observation/thermal_wrist_image_left"]
            )

    if debug_overlay is not None:
        env = debug_overlay.get("env")
        axes = debug_overlay.get("axes", {})
        table_image = _draw_projected_debug_axes(table_image, env, "table_cam", axes)
        if wrist_image is not None:
            wrist_image = _draw_projected_debug_axes(wrist_image, env, "wrist_cam", axes)
        vlm = debug_overlay.get("vlm")
        if vlm is not None:
            try:
                table_image = _draw_vlm_overlay(table_image, env, "table_cam", vlm)
            except Exception as exc:
                print(f"[vlm_dp] overlay draw failed: {exc}", flush=True)

    frame = (
        table_image
        if wrist_image is None
        else np.concatenate((table_image, wrist_image), axis=1)
    )
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if _has_sound_observation(obs):
        table_only = cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB)
        spectrogram_width = table_only.shape[1]
        mic1_spectrogram, mic2_spectrogram = _get_sound_spectrograms(obs)
        mic_vmin, mic_vmax = _get_sound_video_scale()
        mic1_image = _visualize_log_mel_spectrogram(
            mic1_spectrogram,
            target_height=table_only.shape[0],
            target_width=spectrogram_width,
            label="mic1_log_mel",
            vmin=mic_vmin,
            vmax=mic_vmax,
        )
        mic2_image = _visualize_log_mel_spectrogram(
            mic2_spectrogram,
            target_height=table_only.shape[0],
            target_width=spectrogram_width,
            label="mic2_log_mel",
            vmin=mic_vmin,
            vmax=mic_vmax,
        )
        return np.concatenate((mic2_image, table_only, mic1_image), axis=1)

    return frame


def _slugify(text: str) -> str:
    safe_chars = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("-")
    return "".join(safe_chars).strip("-") or "eval"


def _video_config_slug(args) -> str:
    parts = [
        f"sampler-{getattr(args, 'sampler', 'base')}",
        f"grad-{getattr(args, 'grad_calc', 'mbd')}",
        _base_source_name(args),
        f"update-{args.mpc_update}" if _uses_vlm_mpc_base(args) else "update-flow",
        f"cost-{getattr(args, 'mpc_cost', 'na')}",
        f"space-{getattr(args, 'mpc_optimize_space', 'na')}",
        f"g{float(args.gamma_base):g}",
        f"n{float(args.mpc_noise):g}",
        f"t{float(args.mpc_temperature):g}",
        f"clip{float(args.mpc_joint_delta_clip):g}",
    ]
    if args.determine:
        parts.append("det")
    for prefix, name in (
        ("sg", "grasp_steer_scale"),
        ("sl", "lift_steer_scale"),
        ("sp", "place_steer_scale"),
    ):
        value = getattr(args, name, None)
        if value is not None:
            parts.append(f"{prefix}{float(value):g}")
    return _slugify("_".join(parts))


def _experiment_output_name(args, *, run_id: str) -> str:
    return f"{run_id}_{_video_config_slug(args)}"


def _episode_video_name(seed, success, *, suffix=""):
    status = "success" if success else "fail"
    suffix = _slugify(str(suffix).strip("_")) if suffix else ""
    suffix_part = f"_{suffix}" if suffix else ""
    return f"{seed}_{status}{suffix_part}.mp4"


def _code_provenance(config_paths=()):
    """Identify the CODE this run executed, not just its flags.

    The working tree is normally dirty (this is a research branch), so a git hash alone is not
    enough: two runs an hour apart can share a HEAD and execute materially different code. That is
    exactly what happened on 2026-07-30 -- a 4/14 arm was labelled "old code" in a handoff and the
    claim could not be checked afterwards, because results.json recorded flags only. So record HEAD,
    the dirty flag, and a content digest of the files that decide behaviour.
    """
    import hashlib
    import pathlib
    import subprocess

    def _git(*a):
        """stdout, or None if git could not answer. None means UNKNOWN, never 'clean'."""
        try:
            r = subprocess.run(("git", *a), cwd=_REPO_DIR, capture_output=True, text=True,
                               timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    watched = (
        "eval_steering.py", "sim_free_mpc/planner.py", "sim_free_mpc/dial_sampler.py",
        "sim_free_mpc/action_space.py", "vlm_dp/bridge.py", "vlm_dp/world.py",
        "vlm_dp/grasp_sensor.py", "vlm_dp/context.py", "vlm_dp/stage.py",
        "vlm_dp/cost/base_cost.py", "vlm_dp/cost/terms.py", "vlm_dp/perception.py",
        "vlm_dp/visual_tracker.py", "rekep/isaaclab_helpers.py",
    )
    # The active cost YAML decides as much behaviour as any source file and is edited far more
    # often -- one was modified between arms that were then compared as if only code had changed.
    files = {}
    combined = hashlib.sha256()
    for rel in (*watched, *(str(p) for p in config_paths if p)):
        try:
            path = pathlib.Path(rel)
            raw = (path if path.is_absolute() else pathlib.Path(_REPO_DIR, rel)).read_bytes()
        except OSError:
            continue
        files[rel] = hashlib.sha256(raw).hexdigest()[:12]
        combined.update(rel.encode())
        combined.update(raw)
    # git is often absent inside the eval container. Report None (unknown) rather than False, so a
    # missing git can never be read as a clean tree. code_digest is the load-bearing field either way:
    # it identifies the executed source regardless of git.
    dirty = _git("status", "--porcelain")
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "git_dirty_files": None if dirty is None else len([l for l in dirty.splitlines() if l.strip()]),
        "code_digest": combined.hexdigest()[:16],
        "file_digests": files,
    }


def _write_experiment_results(path: str, payload: dict[str, Any]) -> None:
    episodes = payload.get("episodes", [])
    # Errored episodes (e.g. an ungroundable scene) are not policy failures, so they are excluded
    # from the denominator rather than silently counted against the success rate.
    scored = [episode for episode in episodes if not episode.get("errored")]
    successes = sum(bool(episode.get("success")) for episode in scored)
    payload["summary"] = {
        "num_episodes": len(scored),
        "num_successes": successes,
        "success_rate": successes / len(scored) if scored else 0.0,
        "num_errored": len(episodes) - len(scored),
        # Both readings, so neither has to be recomputed and the excluded-denominator choice
        # cannot be mistaken for a hidden one: an ungroundable scene is not a policy failure,
        # but a run with many of them did not attempt as many episodes as it requested.
        "num_requested": len(episodes),
        "success_rate_including_errored": (
            successes / len(episodes) if episodes else 0.0
        ),
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable_debug_value(payload), handle, indent=2, default=str)
    os.replace(tmp_path, path)


def _video_config_lines(args, *, seed: int) -> list[str]:
    return [
        f"seed={seed} task={args.task} prompt={args.prompt}",
        (
            f"base={_base_source_name(args)} update={args.mpc_update} "
            f"cost={getattr(args, 'mpc_cost', 'na')} "
            f"space={getattr(args, 'mpc_optimize_space', 'na')} "
            f"gamma={args.gamma_base:g} steps={args.num_steps} "
            f"rollout_steps={args.task_num_steps} spi={args.steps_per_inference}"
        ),
        (
            f"sampler={args.sampler} grad={args.grad_calc} "
            f"samples={args.mpc_num_samples} iters={args.mpc_iterations} "
            f"noise={args.mpc_noise:g} temp={args.mpc_temperature:g} "
            f"joint_clip={args.mpc_joint_delta_clip:g} determine={int(args.determine)}"
        ),
    ]


def _add_video_header(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if frame.size == 0:
        return frame
    height = 26 * len(lines) + 12
    header = np.zeros((height, frame.shape[1], frame.shape[2]), dtype=frame.dtype)
    header[:] = 12
    y = 24
    for line in lines:
        cv2.putText(
            header,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 26
    return np.concatenate((header, frame), axis=0)


def _first_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().flatten()[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    return bool(value)


def _first_list(value):
    if torch.is_tensor(value):
        return value.detach().flatten().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _debug_scene_root_pos(env, asset_name: str):
    try:
        asset = env.scene[asset_name]
        return asset.data.root_pos_w[0].detach().cpu().tolist()
    except Exception:
        return None


def _capture_debug_object_xy_guard(env, asset_name: str, *, margin: float):
    try:
        asset = env.scene[asset_name]
        pos = asset.data.root_pos_w.detach()
        return {
            "asset_name": asset_name,
            "xy_min": pos[:, :2].clone() - float(margin),
            "xy_max": pos[:, :2].clone() + float(margin),
        }
    except Exception as exc:
        print(f"debug_object_xy_guard unavailable for {asset_name}: {exc}", flush=True)
        return None


def _subtask_flag_from_obs(env_obs_dict, key: str) -> bool:
    raw = env_obs_dict.get("subtask_terms", {})
    if key not in raw:
        return False
    return _first_bool(raw[key])


def _apply_debug_object_xy_guard(env, guard_state, *, release: bool) -> bool:
    if guard_state is None or release:
        return False
    asset_name = guard_state["asset_name"]
    try:
        asset = env.scene[asset_name]
        pos = asset.data.root_pos_w.detach().clone()
        quat = asset.data.root_quat_w.detach().clone()
        xy_min = guard_state["xy_min"].to(device=pos.device, dtype=pos.dtype)
        xy_max = guard_state["xy_max"].to(device=pos.device, dtype=pos.dtype)
        below = pos[:, :2] < xy_min
        above = pos[:, :2] > xy_max

        pose_changed = bool((below | above).any().item())
        if pose_changed:
            pos[:, :2] = torch.maximum(torch.minimum(pos[:, :2], xy_max), xy_min)
            asset.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1))

        vel = asset.data.root_vel_w.detach().clone()
        outward_x = (below[:, 0] & (vel[:, 0] < 0.0)) | (above[:, 0] & (vel[:, 0] > 0.0))
        outward_y = (below[:, 1] & (vel[:, 1] < 0.0)) | (above[:, 1] & (vel[:, 1] > 0.0))
        if bool((outward_x | outward_y).any().item()):
            vel[outward_x, 0] = 0.0
            vel[outward_y, 1] = 0.0
            asset.write_root_velocity_to_sim(vel)
        return pose_changed
    except Exception as exc:
        print(f"debug_object_xy_guard failed for {asset_name}: {exc}", flush=True)
        return False


def _print_task_debug_done(
    *,
    env,
    step_idx: int,
    terminated,
    truncated,
    task_success: bool,
):
    print(
        "task_debug_done "
        f"step={step_idx} "
        f"terminated={_first_bool(terminated)} "
        f"truncated={_first_bool(truncated)} "
        f"task_success={task_success}",
        flush=True,
    )
    try:
        terms = env.termination_manager.get_active_iterable_terms(0)
        term_text = ", ".join(f"{name}={_first_list(values)[0]}" for name, values in terms)
        print(f"task_debug_done terms: {term_text}", flush=True)
    except Exception as exc:
        print(f"task_debug_done terms unavailable: {exc}", flush=True)

    scene_positions = {
        name: _debug_scene_root_pos(env, name)
        for name in ("apple", "pear", "pot", "egg", "cover")
    }
    scene_positions = {name: pos for name, pos in scene_positions.items() if pos is not None}
    if scene_positions:
        print(
            "task_debug_done scene_root_pos_after_possible_reset: "
            + json.dumps(scene_positions),
            flush=True,
        )


def _episode_sort_key(name):
    match = re.search(r"(\d+)$", name)
    if match is None:
        return name
    return int(match.group(1))


def _load_hdf5_state(group, device):
    state = dict()
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            state[key] = torch.from_numpy(value[:]).to(device=device)
        else:
            state[key] = _load_hdf5_state(value, device)
    return state


def compute_batch_metrics(
    teacher_flows: torch.Tensor,
    student_flows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    diff = student_flows - teacher_flows

    teacher_vec = teacher_flows.reshape(teacher_flows.shape[0], teacher_flows.shape[1], -1)
    student_vec = student_flows.reshape(student_flows.shape[0], student_flows.shape[1], -1)
    cosine = torch.nn.functional.cosine_similarity(student_vec, teacher_vec, dim=-1)

    return {
        "sq_error_sum": diff.pow(2).sum(dtype=torch.float64),
        "abs_error_sum": diff.abs().sum(dtype=torch.float64),
        "teacher_sq_sum": teacher_flows.pow(2).sum(dtype=torch.float64),
        "student_sq_sum": student_flows.pow(2).sum(dtype=torch.float64),
        "cosine_sum": cosine.sum(dtype=torch.float64),
        "element_count": torch.tensor(diff.numel(), dtype=torch.float64, device=diff.device),
        "vector_count": torch.tensor(cosine.numel(), dtype=torch.float64, device=diff.device),
        "batch_count": torch.tensor(teacher_flows.shape[0], dtype=torch.float64, device=diff.device),
    }


def accumulate_stats(
    stats: dict[str, torch.Tensor] | None,
    batch_stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if stats is None:
        return {key: value.clone() for key, value in batch_stats.items()}
    for key, value in batch_stats.items():
        stats[key] += value
    return stats


def stats_to_serializable(stats: dict[str, torch.Tensor] | None) -> dict[str, float]:
    if stats is None:
        return {}
    return {key: float(value.item()) for key, value in stats.items()}


def average_stats_per_step(
    stats: dict[str, torch.Tensor] | None,
    n_steps: int,
) -> dict[str, float]:
    if stats is None or n_steps <= 0:
        return {}
    return {key: float((value / n_steps).item()) for key, value in stats.items()}


def summarize_metrics(stats: dict[str, torch.Tensor] | None) -> dict[str, float]:
    if stats is None:
        return {}

    eps = torch.finfo(torch.float64).eps
    element_count = torch.clamp(stats["element_count"], min=eps)
    vector_count = torch.clamp(stats["vector_count"], min=eps)
    teacher_sq_sum = torch.clamp(stats["teacher_sq_sum"], min=eps)

    return {
        "mse": float((stats["sq_error_sum"] / element_count).item()),
        "mae": float((stats["abs_error_sum"] / element_count).item()),
        "cosine": float((stats["cosine_sum"] / vector_count).item()),
        "teacher_rms": float(torch.sqrt(stats["teacher_sq_sum"] / element_count).item()),
        "student_rms": float(torch.sqrt(stats["student_sq_sum"] / element_count).item()),
        "rel_l2": float(torch.sqrt(stats["sq_error_sum"] / teacher_sq_sum).item()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the model on the real droid robot."
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Task prompt. Defaults to the matching task_prompts.json entry.",
    )
    parser.add_argument("--exp_name", type=str, default="eval")
    parser.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help=(
            "Number of independent evaluation worker processes. Defaults to 1. "
            "Seeds are divided into contiguous, balanced ranges."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=_DEFAULT_GPUS,
        help=(
            "Comma-separated logical GPU indices assigned round-robin to workers, "
            "for example --gpus 0,1. Defaults to GPU 0."
        ),
    )
    parser.add_argument(
        "--worker-id", "--worker_id", type=int, default=-1, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-progress-path", type=str, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-start-barrier", type=str, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-rollout-offset", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-parent-pid", type=int, default=-1, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to the output directory."
    )
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=51)
    parser.add_argument(
        "--determine",
        "--deterministic-eval",
        dest="determine",
        action="store_true",
        help=(
            "Enable strict reproducibility settings for PyTorch/CUDA, RTX rendering, "
            "environment seeding, and PhysX enhanced determinism."
        ),
    )
    parser.add_argument(
        "--load_init_from_dataset",
        type=str,
        default=None,
        help="Optional HDF5 dataset path. If provided, rollout i loads initial_state from the i-th episode instead of using randomized reset state.",
    )
    parser.add_argument(
        "--initial_action_after_reset",
        action="store_true",
        help=(
            "Optionally step one hold action immediately after reset. Disabled by default "
            "so the first policy observation matches the reset or dataset initial state."
        ),
    )
    parser.add_argument(
        "--base_checkpoint_dir",
        type=str,
        default=DEFAULT_BASE_CHECKPOINT_DIR,
        help="Base policy checkpoint directory. Defaults to the weight-task pi05 Droid joint-position checkpoint.",
    )
    parser.add_argument(
        "--task_checkpoint_dir",
        type=str,
        default=None,
        help="Task proxy checkpoint directory. Defaults to the matching task_prompts.json entry.",
    )
    parser.add_argument(
        "--task_attention",
        "--task-attention",
        choices=("config", "causal", "bidirectional"),
        default="config",
        help=(
            "Attention mask used by the task score proxy. 'config' uses the model config; "
            "the explicit modes are checkpoint-compatibility/ablation overrides."
        ),
    )
    parser.add_argument(
        "--ref_checkpoint_dir",
        type=str,
        default=None,
        help="Reference proxy checkpoint directory. Defaults to the matching task_prompts.json entry.",
    )
    standalone_group = parser.add_mutually_exclusive_group()
    standalone_group.add_argument(
        "--ref_only",
        "--ref-only",
        dest="ref_only",
        action="store_true",
        help="Evaluate the reference proxy directly from noise, without base/MPC/task steering.",
    )
    standalone_group.add_argument(
        "--task_only",
        "--task-only",
        dest="task_only",
        action="store_true",
        help="Evaluate the task proxy directly from noise, without base/MPC/ref steering.",
    )
    parser.add_argument(
        "--steer_scale",
        type=float,
        default=0.4,
        help="Lambda multiplying the task residual in full/task score steering.",
    )
    parser.add_argument(
        "--grasp_steer_scale",
        type=float,
        default=None,
        help="Optional steering scale for MPC grasp stages; defaults to --steer_scale.",
    )
    parser.add_argument(
        "--lift_steer_scale",
        type=float,
        default=None,
        help="Optional steering scale for MPC lift stages; defaults to --steer_scale.",
    )
    parser.add_argument(
        "--place_steer_scale",
        type=float,
        default=None,
        help="Optional steering scale for MPC place stages; defaults to --steer_scale.",
    )
    parser.add_argument(
        "--steer_gamma_gripper",
        type=float,
        default=None,
        help="Optional distinct steering scale for the gripper action channel (dim 7). "
        "Arm channels keep the stage steering scale. Per-channel gamma: the A3 mechanism "
        "data localizes steering damage to gripper anti-tracking, so gamma_g=0 tests "
        "arm-only additive steering.",
    )
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument(
        "--abort_no_subtask_by",
        type=int,
        default=0,
        help="End a rollout early (as a failure) if NO subtask flag has fired by this env step. "
        "0 disables. A screens-only speed lever: decided-stuck episodes stop burning the step cap.",
    )
    parser.add_argument(
        "--task_num_steps",
        type=int,
        default=225,
        help="Maximum rollout control steps. Default 225 is 15 seconds at 15 Hz.",
    )
    parser.add_argument(
        "--steps_per_inference",
        type=int,
        default=8,
        help=(
            "Number of environment control steps executed from each inferred action chunk. "
            "Smaller values replan more often. Default 8 preserves the previous behavior."
        ),
    )
    parser.add_argument("--only_steer", action="store_true")
    parser.add_argument(
        "--no_steer",
        action="store_true",
        help="Disable task/ref steering and use only the base path.",
    )
    parser.add_argument("--compare_difference", action="store_true")
    score_mode_group = parser.add_mutually_exclusive_group()
    score_mode_group.add_argument(
        "--vlm_base",
        "--vlm-base",
        dest="vlm_base",
        action="store_true",
        help="Evaluate only the FK/cost MBD base score, without task/ref steering.",
    )
    score_mode_group.add_argument(
        "--full_steer",
        "--full-steer",
        dest="full_steer",
        action="store_true",
        help="Run score-space MBD base + steer_scale * (task - ref).",
    )
    parser.add_argument(
        "--steer_anneal",
        action="store_true",
        help=(
            "Ramp the steering scale linearly across the denoise trajectory instead of holding it "
            "constant, from --steer_anneal_start (highest noise) to --steer_anneal_end (lowest)."
        ),
    )
    parser.add_argument("--steer_anneal_start", type=float, default=0.0,
                        help="Steering scale at the highest-noise denoise step.")
    parser.add_argument("--steer_anneal_end", type=float, default=1.0,
                        help="Steering scale at the lowest-noise denoise step.")
    parser.add_argument(
        "--task_tilt",
        type=float,
        default=0.0,
        help=(
            "FK/SVDD-style tilted selection: weight lambda for a per-candidate Gaussian tilt "
            "toward the task proxy's implied clean action inside the MBD softmax (SNR-tempered "
            "per denoise step). Replaces the additive combine: steer_scale is forced to 0. "
            "0 disables. Use with --task_steer to load the task proxy."
        ),
    )
    parser.add_argument(
        "--task_tilt_dims",
        type=int,
        default=0,
        help=(
            "Restrict the tilt penalty to the first N action dims (7 = arm only, plan_ref "
            "convention; keeps the near-binary gripper dim from dominating at high lambda). "
            "0 = all dims."
        ),
    )
    parser.add_argument(
        "--mpc_mode_window",
        type=float,
        default=float("inf"),
        help=(
            "Restrict the DIAL softmax to candidates within this many temperature units of the "
            "best before the weighted mean. Averaging then happens inside one mode instead of "
            "across two, where the mean is a plan neither mode supports. inf disables."
        ),
    )
    parser.add_argument(
        "--mpc_eval_mean_plan",
        action="store_true",
        help=(
            "Also score the plan the sampler RETURNS, not only the population it drew. Every logged "
            "term is a statistic of the candidates (value at the argmin, or the softmax-weighted "
            "mean of the values); the plan that executes is the weighted mean of the candidates, "
            "whose cost is different for any non-convex term and is never otherwise computed. Logs "
            "mean_plan_cost, mean_plan_cost_excess and per-term mean_plan_term_*."
        ),
    )
    parser.add_argument(
        "--mpc_logit_norm",
        choices=("raw", "std"),
        default="raw",
        help=(
            "Softmax logit normalization for the MBD sampler. 'std' divides costs by the batch "
            "std before the temperature (DIAL-style): temperature becomes sharpness in std "
            "units, constant across denoise levels. 'raw' is the historical behavior."
        ),
    )
    parser.add_argument(
        "--mpc_ancestral_eta",
        type=float,
        default=0.0,
        help=(
            "Marginal re-noising between denoise levels: x_{k-1} += eta*sqrt(1-abar_prev)*z "
            "(never at the final level). 0 = deterministic chain (historical); 1 = full "
            "ancestral sampling."
        ),
    )
    parser.add_argument(
        "--rank_mode",
        choices=("total", "roles"),
        default="total",
        help=(
            "What the MBD softmax ranks candidates by. 'total' is the plain cost sum (deployed "
            "behaviour). 'roles' uses the TERM_ROLES split, feasibility + task + "
            "prior_weight*prior, so the execution/search prior shapes rather than decides. "
            "Measured motivation: near contact the task signal separating a good chunk from doing "
            "nothing is ~1.5 units against a +12..22 prior charge, so the prior decides alone."
        ),
    )
    parser.add_argument(
        "--prior_weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the TERM_ROLES 'prior' bucket under --rank_mode roles. 1.0 reproduces "
            "--rank_mode total exactly; 0.0 removes the prior from selection. A scalar rather than "
            "a switch because the prior exists to protect a weak sampler and the base IS that "
            "sampler -- removing it outright is a hypothesis, not a fix."
        ),
    )
    parser.add_argument(
        "--prior_weight_high",
        type=float,
        default=1.0,
        help=(
            "Prior weight at HIGH noise under --prior_weight_schedule alpha. At the first denoise "
            "levels the base's candidates are decoded noise whose cost is almost entirely the prior "
            "(measured: ~53,632 of ~53,676), so the prior is the garbage filter there and must be "
            "kept. --prior_weight then applies at LOW noise, where candidates are plausible and the "
            "prior mostly blocks demo-shaped motion."
        ),
    )
    parser.add_argument(
        "--prior_weight_schedule",
        choices=("flat", "alpha"),
        default="flat",
        help=(
            "flat: --prior_weight at every level (default; 1.0 reproduces --rank_mode total). "
            "alpha: interpolate prior_weight_high -> prior_weight linearly in alpha_bar, i.e. keep "
            "the prior while the candidates are noise and stand it down as they become plausible. A "
            "constant is wrong at both ends in opposite directions -- see --prior_weight_high."
        ),
    )
    parser.add_argument(
        "--feasibility_gate",
        type=float,
        default=0.0,
        help=(
            "Under --rank_mode roles, exclude candidates whose feasibility cost exceeds the best "
            "feasibility by more than this margin. 0 disables the hard gate (feasibility still "
            "competes on magnitude). If nothing passes, the whole population is kept."
        ),
    )
    parser.add_argument(
        "--task_tilt_gripper_weight",
        type=float,
        default=-1.0,
        help=(
            "Calibrated per-dim tilt: arm dims at weight 1, gripper dim at this weight "
            "(e.g. 0.01 = variance-matched for a near-binary channel). Negative = disabled; "
            "overrides --task_tilt_dims when set."
        ),
    )
    parser.add_argument(
        "--fk_fork",
        type=int,
        default=1,
        help="Window-scoped best-of-M Feynman-Kac fork: inside open failure windows, run M "
        "independent denoise chains and commit to the lowest-cost plan. 1 disables. Requires "
        "the failure gate (steer_authority in context).",
    )
    parser.add_argument(
        "--fk_always",
        action="store_true",
        help="Ungated fork: run the best-of-M fork on EVERY inference, ignoring the failure "
        "gate. Isolates the selection mechanism from the gate (M x inference cost).",
    )
    parser.add_argument(
        "--inject_proxy",
        type=float,
        default=0.0,
        help="Proposal injection: draw this fraction of MBD candidates from the task proxy's "
        "implied clean action instead of the noise prior, extending the candidate support. The "
        "cost still weights every candidate. 0 disables. Needs the task proxy, not a reference.",
    )
    parser.add_argument(
        "--inject_schedule",
        choices=["flat", "frontload"],
        default="flat",
        help="How the injection fraction varies over denoise levels: flat holds it constant; "
        "frontload scales it by (1 - alpha_bar), concentrating the expert where the sampler is "
        "still choosing global structure.",
    )
    parser.add_argument(
        "--gated_inject",
        action="store_true",
        help="Scope proposal injection to open failure windows (multiply the fraction by the "
        "bridge's steer_authority). Off isolates the mechanism from the gate.",
    )
    parser.add_argument(
        "--exact_cfg",
        type=float,
        default=0.0,
        help="Exact cost-space CFG gamma: steer the base by (conditioned - unconditioned) scores "
        "computed by the base's own estimator (second call on a stage-stripped context). No "
        "proxies. 0 disables. Combine with --gated_tilt to scope to failure windows.",
    )
    parser.add_argument(
        "--gated_tilt",
        action="store_true",
        help="Track-2 gated expert: multiply the tilt weight by the bridge's plan-authored "
        "failure-gate authority, enable the candidate-discrimination implicit gate and the "
        "ESS cap. Requires --task_tilt > 0 and a config with a steering: block.",
    )
    parser.add_argument(
        "--tilt_ess_cap",
        type=float,
        default=3.0,
        help="Gated tilt: bound the tilt's logit dispersion to this many temperature units "
        "(closed-form guard against collapsing the candidate population).",
    )
    score_mode_group.add_argument(
        "--task_steer",
        "--task-steer",
        dest="task_steer",
        action="store_true",
        help=(
            "Run score-space base + steer_scale * (task - base), without loading ref. "
            "gamma_base scales the effective base before interpolation."
        ),
    )
    parser.add_argument(
        "--gamma_base",
        type=float,
        default=1.0,
        help="Scale multiplying the MBD base score before score composition.",
    )
    parser.add_argument("--mpc_num_samples", type=int, default=4096)
    parser.add_argument("--mpc_iterations", type=int, default=1)
    parser.add_argument("--mpc_noise", type=float, default=1.0)
    parser.add_argument("--mpc_temperature", type=float, default=0.15)
    parser.add_argument(
        "--grad_calc",
        choices=("mbd", "backprop"),
        default="mbd",
        help=(
            "Cost-score estimator. 'mbd' uses weighted sample displacement; "
            "'backprop' uses the softmax-weighted cost gradient through action decoding and FK."
        ),
    )
    parser.add_argument(
        "--mpc_update",
        choices=(
            "ddim",
            "mbd_score",
            "mbd_score_action_prox",
            "mbd_score_action_warm",
            "legacy_score",
        ),
        default="mbd_score_action_prox",
        help=(
            "Reverse update used by MBD base/full/task score modes. Defaults to "
            "mbd_score_action_prox to match the current Weight ref distillation teacher."
        ),
    )
    parser.add_argument(
        "--sampler",
        choices=("base", "truncated"),
        default="base",
        help=(
            "MPC proposal sampler. 'base' uses unconstrained Gaussians and the "
            "final joint-delta clamp; 'truncated' autoregressively samples only "
            "valid decoded joint targets, automatically enables low-frequency "
            "interpolation (10 Hz -> 40 Hz by default), and disables that final "
            "delta clamp."
        ),
    )
    parser.add_argument(
        "--mpc_cost",
        choices=(
            "priority",
            "ref_style",
            "explore",
            "grasp_flow",
            "grasp_flow_ex",
            "grasp_flow_fake",
            "grasp_flow_loose",
            "capsule_flow",
        ),
        default="priority",
        help="Cost function used by sim-free MPC.",
    )
    parser.add_argument(
        "--vlm_cost",
        choices=("none", "gt", "rekep_fake", "rekep_real", "rekep_fake_vlm", "rekep_real_vlm"),
        default="none",
        help=(
            "Attach the ReKep CompositeCost (via vlm_dp.VlmDpBridge) to the sim-free MPC planner, "
            "with the named grounding source providing stages/targets. Requires a score-steering "
            "mode (--vlm_base/--task_steer/--full_steer) and --mpc_cost priority. Default 'none' "
            "leaves the planner untouched."
        ),
    )
    parser.add_argument(
        "--vlm_cost_config",
        type=str,
        default=os.path.join("vlm_dp", "configs", "base.yaml"),
        help="Cost config YAML for --vlm_cost (term weights + gripper geometry).",
    )
    parser.add_argument(
        "--vlm_state",
        choices=("gt", "real"),
        default="gt",
        help=(
            "Object-state source for --vlm_cost: 'gt' reads poses from the sim; 'real' builds a "
            "SensedWorld (GroundedSAM perception + FK-while-held from the aperture sensor)."
        ),
    )
    parser.add_argument(
        "--vlm_track",
        choices=("fk", "reperceive", "visual"),
        default="fk",
        help=(
            "Between-look tracking for --vlm_state real: 'fk' dead-reckons the held object only; "
            "'reperceive' re-segments periodically; 'visual' corrects per step from CoTracker."
        ),
    )
    parser.add_argument(
        "--vlm_segment",
        choices=("groundedsam", "sam_vlm"),
        default="groundedsam",
        help="Segmenter for --vlm_state real: text-grounded boxes, or SAM regions named by a VLM.",
    )
    parser.add_argument(
        "--vlm_vocab",
        default=None,
        help="Comma-separated object names to restrict the perception vocabulary to (detector text kept "
             "from the task_prompts.json objects entry). Unnamed objects become anonymous obstacle blobs "
             "instead of tracked entities. Default None keeps the full declared vocabulary.",
    )
    parser.add_argument(
        "--vlm_derive_vocab",
        action="store_true",
        help="Derive the object vocabulary and roles from the instruction (GPT-4o), ignoring any declared "
             "objects entry. Only referents are named; distractors and fixtures become anonymous obstacle "
             "blobs and the support surface is measured geometrically. The generalizable, zero-config path.",
    )
    parser.add_argument(
        "--mpc_optimize_space",
        choices=("action", "accel"),
        default="action",
        help=(
            "Parameterization used by sim-free MPC sampling. 'accel' runs pure "
            "MPPI over real joint accelerations and integrates directly to action chunks."
        ),
    )
    parser.add_argument(
        "--mpc_ddim_train_timesteps",
        type=int,
        default=100,
        help="Number of training timesteps used to discretize the DDIM cosine schedule.",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help=(
            "Optimize sim-free MPC at a lower knot rate and "
            "interpolate back to the action horizon."
        ),
    )
    parser.add_argument(
        "--interpolation_method",
        choices=("bspline", "linear"),
        default="bspline",
        help="Interpolation used to expand low-frequency control points.",
    )
    parser.add_argument(
        "--interpolate_low_frequency",
        type=float,
        default=10.0,
        help="Low-frequency knot rate used by --interpolate.",
    )
    parser.add_argument(
        "--interpolate_high_frequency",
        type=float,
        default=40.0,
        help=(
            "High-frequency rate used by --interpolate to compute the knot ratio. "
            "This does not change the Isaac env control rate."
        ),
    )
    parser.add_argument(
        "--mpc_joint_delta_clip",
        type=float,
        default=0.15,
        help=(
            "Clamp decoded MPC, --ref-only, or --task-only joint-position targets to this many "
            "radians per control step. "
            "Set to 0 to disable the per-step delta clamp. Joint limits are still enforced."
        ),
    )
    parser.add_argument(
        "--ddim_final_level",
        action="store_true",
        help=(
            "Run the final DDIM reverse transition on the score/MPC paths. num_iterations is "
            "num_steps+1, but the loop stops one short, so iteration num_steps (timestep 0, "
            "alpha_prev=1.0 -- the only level reaching ddim_iteration_alphas' set_alpha_to_one "
            "branch) never executes. Changes every MPC inference, so it is not comparable to "
            "runs without it. Does not affect the flow path, which correctly takes num_steps."
        ),
    )
    parser.add_argument(
        "--cost_executable_actions",
        action="store_true",
        help=(
            "Score MPC candidates after the execution clamp (joint limits + per-step "
            "joint delta) instead of before it. Without this the planner can select a "
            "plan whose low cost depends on motion that is truncated before env.step. "
            "Changes the optimisation landscape, so it is not comparable to runs without it."
        ),
    )
    parser.add_argument("--mpc_debug", action="store_true")
    parser.add_argument(
        "--mpc_debug_stdout",
        action="store_true",
        help="Also print detailed MPC debug iterations to stdout. By default --mpc_debug writes them to a jsonl log.",
    )
    parser.add_argument("--task_debug", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Report a per-episode wall-time budget (physics, render, sensors, planner, "
            "video, reset) to stdout and into the --mpc_debug rollout_end record."
        ),
    )
    parser.add_argument(
        "--debug_hold_pear",
        action="store_true",
        help=(
            "Debug-only: before pear is detected as grasped, keep it inside a "
            "small horizontal XY guard window so it does not slide off the table edge."
        ),
    )
    parser.add_argument(
        "--debug_hold_pear_xy_margin",
        type=float,
        default=0.10,
        help="Half-width in meters of the debug pear XY guard window around the reset position.",
    )
    parser.add_argument(
        "--mpc_debug_axis_length",
        type=float,
        default=0.08,
        help="Axis length in meters for --mpc_debug_video_overlay.",
    )
    parser.add_argument(
        "--mpc_debug_video_overlay",
        action="store_true",
        help=(
            "Debug-only: draw projected task-object/gripper/finger axes directly "
            "on saved rollout videos. Capsule adds lid/gripper/capsule axes. "
            "Works in headless mode."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run one warmup inference, print base-source and shape diagnostics, then exit.",
    )
    parser.add_argument(
        "--base_decode_only",
        action="store_true",
        help=(
            "Build the base policy's decode surface (transforms + norm stats) from the "
            "checkpoint WITHOUT loading model weights. Only valid where the base network is "
            "never forward-passed (the geometric MBD base, --vlm_base). Saves the pi0.5 load "
            "at startup and its GPU residency."
        ),
    )
    parser.add_argument(
        "--base_action_space",
        type=str,
        default="policy",
        choices=["policy", "demo_delta"],
        help=(
            "Action representation the MBD planner optimizes and decodes in. 'policy' (default) "
            "keeps the base checkpoint's own normalization -- for pi0.5 that is the DROID "
            "quantile band, which every validated Isaac result is co-tuned with. 'demo_delta' "
            "switches to the demonstration joint-delta scale of --base_action_stats, the space "
            "mujoco_eval/robolab_eval already plan in. Only the ACTION half changes; the state "
            "is still unnormalized with the checkpoint's stats."
        ),
    )
    parser.add_argument(
        "--base_action_stats",
        type=str,
        default="",
        help=(
            "action_norm_stats-style JSON ('mean' and 'std' arrays) used by "
            "--base_action_space demo_delta. Required by that mode, ignored otherwise."
        ),
    )
    parser.add_argument(
        "--static_collision_off",
        type=str,
        default="",
        help=(
            "Comma-separated static scene assets whose colliders are disabled. Decorative geometry "
            "a task never contacts still costs PhysX broadphase, and that cost is superlinear in "
            "num_envs -- the measured blocker for vectorized multi-env eval. Visuals are untouched. "
            "Empty (default) leaves the scene exactly as configured. MEASURED WARNING: on "
            "Isaac-Weight-Droid-*, 'interactive_kitchen' is LOAD-BEARING -- it is the support "
            "surface, and disabling it drops the pear through the floor and terminates every "
            "episode at step 1. Gate any asset before using it here."
        ),
    )
    parser.add_argument(
        "--fast_gt",
        action="store_true",
        help=(
            "Named fast configuration for GROUND-TRUTH-GROUNDED runs only. Expands to "
            "--render none plus the collider diet on --fast_gt_asset. Refuses to run unless the "
            "grounding, world state and tracking are all ground truth (--vlm_cost gt, "
            "--vlm_state gt, --vlm_track fk) and nothing else reads pixels, so it cannot be "
            "applied silently to a perception run."
        ),
    )
    parser.add_argument(
        "--fast_gt_asset",
        type=str,
        default="interactive_kitchen",
        help="Scene asset --fast_gt runs the collider diet on. Empty disables that half.",
    )
    parser.add_argument(
        "--collider_diet",
        type=str,
        default="",
        help=(
            "Scene asset to run a PER-PRIM collider diet on (e.g. 'interactive_kitchen'). "
            "Colliders whose world bound lies outside --collider_diet_radius of the robot base "
            "are disabled at prestartup; everything within reach, including the support surface, "
            "is kept. Prims stay active and visible, so the rendered image is unchanged. Unlike "
            "--static_collision_off this works on assets that are decorative AND load-bearing."
        ),
    )
    parser.add_argument(
        "--collider_diet_radius",
        type=float,
        default=1.5,
        help=(
            "Half-extent in meters of the reach box around the robot base used by "
            "--collider_diet. Must exceed the arm's maximum reach plus the largest distance a "
            "manipulated object can travel."
        ),
    )
    parser.add_argument(
        "--render",
        choices=("policy", "video", "none"),
        default="policy",
        help=(
            "Camera budget. 'policy' (default) keeps today's behaviour: both 720p cameras "
            "rendered every control step. 'video' keeps only the table camera, for the artifact "
            "video. 'none' removes all cameras -- valid only when nothing consumes pixels."
        ),
    )
    parser.add_argument(
        "--render_stride",
        type=int,
        default=1,
        help=(
            "Render (and record) one frame every N control steps under --render video. "
            "1 keeps the control-rate render."
        ),
    )
    parser.add_argument(
        "--video_stride",
        type=int,
        default=1,
        help=(
            "Keep one artifact-video frame every N control steps. Unlike --render_stride this "
            "does NOT touch sim.render_interval, so every pixel consumer still sees a freshly "
            "rendered frame on every control step; only the mp4 is thinned. Playback stays "
            "real-time (fps is divided to match)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Explicit seed list (e.g. '42,43,44') run back-to-back inside one booted app, "
            "instead of the --seed_start/--seed_end range. Amortizes app boot across arms."
        ),
    )
    return parser


def _task_prompt_entry(task_name: str) -> dict[str, str] | None:
    try:
        with open(TASK_PROMPTS_PATH, encoding="utf-8") as handle:
            entries = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load task prompt config {TASK_PROMPTS_PATH}: {exc}") from exc

    if not isinstance(entries, dict):
        raise ValueError(f"Expected an object at the top level of {TASK_PROMPTS_PATH}.")

    for entry in entries.values():
        if isinstance(entry, dict) and entry.get("task_id") == task_name:
            return entry

    task_lower = task_name.lower()
    key_matches = [
        entry
        for key, entry in entries.items()
        if isinstance(entry, dict) and (key.lower() == task_lower or key.lower() in task_lower)
    ]
    if len(key_matches) == 1:
        return key_matches[0]
    return None


def _repo_relative_path(value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(_REPO_DIR, value)


def _apply_task_prompt_defaults(args, parser: argparse.ArgumentParser) -> None:
    required_fields = ["prompt"]
    required_roles = _required_policy_roles(args)
    if "task" in required_roles:
        required_fields.append("task_checkpoint_dir")
    if "ref" in required_roles:
        required_fields.append("ref_checkpoint_dir")
    missing_fields = [
        field
        for field in required_fields
        if getattr(args, field) is None
    ]
    if not missing_fields:
        return

    try:
        entry = _task_prompt_entry(args.task)
    except ValueError as exc:
        parser.error(str(exc))
    if entry is None:
        parser.error(
            f"Task {args.task!r} has no matching entry in {TASK_PROMPTS_PATH}. "
            f"Provide {', '.join('--' + field for field in required_fields)} explicitly."
        )

    for field in missing_fields:
        entry_field = field
        if field.endswith("_checkpoint_dir") and (
            _score_steering_mode(args) in ("full", "task")
            or _standalone_policy_role(args) is not None
            # Proposal injection reads the proxy through _predict_proxy_score, so it needs a
            # PROXY_SCORE checkpoint; defaulting to the flow proxy either fails at load or silently
            # loads the wrong model and raises at the first denoise step.
            or float(getattr(args, "inject_proxy", 0.0)) > 0.0
        ):
            entry_field = f"score_{field}"
        value = entry.get(entry_field)
        if not isinstance(value, str) or not value:
            parser.error(
                f"Task {args.task!r} is missing a valid {entry_field!r} in {TASK_PROMPTS_PATH}."
            )
        if field.endswith("_checkpoint_dir"):
            value = _repo_relative_path(value)
        setattr(args, field, value)


parser = parse_args()
AppLauncher.add_app_launcher_args(parser)
# Default the IsaacLab app to headless rendering with cameras enabled so these
# flags are not needed on the inference command line.
parser.set_defaults(enable_cameras=True, headless=True)

args = parser.parse_args()
if args.workers <= 0:
    parser.error("--workers must be positive.")
try:
    configured_gpu_ids = _parse_gpu_ids(args.gpus)
except ValueError as exc:
    parser.error(str(exc))
if args.worker_id < 0 and args.workers == 1 and not any(
    value == "--device" or value.startswith("--device=") for value in sys.argv[1:]
):
    args.device = f"cuda:{configured_gpu_ids[0]}"
_apply_task_prompt_defaults(args, parser)
if args.determine:
    _enable_deterministic_runtime(args.seed_start)
    deterministic_render_arg = "--/isaaclab/render/deterministic=true"
    kit_args = (args.kit_args or "").split()
    if deterministic_render_arg not in kit_args:
        kit_args.append(deterministic_render_arg)
    args.kit_args = " ".join(kit_args)
standalone_role = _standalone_policy_role(args)
if standalone_role is not None:
    incompatible_flags = [
        flag
        for flag, enabled in (
            ("--vlm_base", args.vlm_base),
            ("--full_steer", args.full_steer),
            ("--task_steer", args.task_steer),
            ("--no_steer", args.no_steer),
            ("--only_steer", args.only_steer),
            ("--compare_difference", args.compare_difference),
        )
        if enabled
    ]
    if incompatible_flags:
        parser.error(
            f"--{standalone_role}_only directly evaluates one proxy and cannot be combined with "
            + ", ".join(incompatible_flags)
            + "."
        )
score_steering_mode = _score_steering_mode(args)
if score_steering_mode in ("full", "task"):
    incompatible_flags = [
        flag
        for flag, enabled in (
            ("--no_steer", args.no_steer),
            ("--only_steer", args.only_steer),
            ("--compare_difference", args.compare_difference),
        )
        if enabled
    ]
    if incompatible_flags:
        parser.error(
            f"--{score_steering_mode}_steer already defines the score composition and cannot "
            "be combined with " + ", ".join(incompatible_flags) + "."
        )
if score_steering_mode == "base" and (args.only_steer or args.compare_difference):
    parser.error("--vlm_base is base-only and cannot be combined with --only_steer or --compare_difference.")
if args.sampler == "truncated":
    if not _uses_vlm_mpc_base(args):
        parser.error("--sampler truncated requires a VLM/MPC base mode.")
    if args.mpc_optimize_space != "action":
        parser.error("--sampler truncated requires --mpc_optimize_space action.")
    if args.mpc_joint_delta_clip <= 0.0:
        parser.error(
            "--sampler truncated requires a positive --mpc_joint_delta_clip."
        )
    # Truncated proposals are optimized as low-frequency control points and
    # interpolated back to the policy action horizon. The CLI defaults define
    # the standard 10 Hz -> 40 Hz path and remain user-overridable.
    args.interpolate = True
if args.interpolate and (
    args.interpolate_low_frequency <= 0.0
    or args.interpolate_high_frequency <= 0.0
):
    parser.error("Interpolation frequencies must both be positive.")
if args.grad_calc == "backprop":
    if not _uses_vlm_mpc_base(args):
        parser.error("--grad_calc backprop requires a VLM/MPC base mode.")
    if args.mpc_update not in ("ddim", "mbd_score"):
        parser.error(
            "--grad_calc backprop currently supports --mpc_update ddim or mbd_score."
        )
    if args.mpc_optimize_space != "action":
        parser.error("--grad_calc backprop requires --mpc_optimize_space action.")
if args.vlm_state == "real" and args.vlm_cost == "none":
    parser.error("--vlm_state real only affects the --vlm_cost bridge; set --vlm_cost as well.")
if args.vlm_track != "fk" and args.vlm_state != "real":
    parser.error(f"--vlm_track {args.vlm_track} corrects a sensed world and needs --vlm_state real; "
                 "--vlm_state gt reads object poses from the simulator and ignores tracking.")
if args.mpc_cost == "capsule_flow" and "capsule" not in args.task.lower():
    parser.error(
        "--mpc_cost capsule_flow requires a capsule task, for example "
        "--task Isaac-Capsule-Droid-Visuomotor-v0."
    )
if _uses_accel_action_mpc(args):
    if score_steering_mode != "base":
        raise ValueError("--mpc_optimize_space accel currently supports --vlm_base base-only mode.")
    if args.mpc_update not in ("legacy_score", "mbd_score"):
        raise ValueError(
            "Acceleration action-space MPC supports --mpc_update legacy_score or mbd_score. "
            "DDIM/action-prox variants are score-space only for this direct-action planner."
        )
if args.steps_per_inference <= 0:
    raise ValueError("--steps_per_inference must be positive.")
if args.base_decode_only:
    _decode_only_blockers = _base_decode_only_blockers(args)
    if _decode_only_blockers:
        parser.error(
            "--base_decode_only cannot be used here: " + "; ".join(_decode_only_blockers)
        )
if args.base_action_space != "policy":
    _action_space_blockers = _base_action_space_blockers(args)
    if _action_space_blockers:
        parser.error(
            f"--base_action_space {args.base_action_space} cannot be used here: "
            + "; ".join(_action_space_blockers)
        )
if args.fast_gt:
    _fast_gt_reasons = _fast_gt_blockers(args)
    if _fast_gt_reasons:
        parser.error(
            "--fast_gt is only valid for a ground-truth-grounded run: "
            + "; ".join(_fast_gt_reasons)
        )
    # An explicit --render wins: dropping every camera also drops the artifact video, and a
    # rollout nobody can watch is its own kind of cost.
    if not any(a == "--render" or a.startswith("--render=") for a in sys.argv[1:]):
        args.render = "none"
    if args.fast_gt_asset and not args.collider_diet:
        args.collider_diet = args.fast_gt_asset
    print(
        f"[fast_gt] render={args.render} collider_diet={args.collider_diet or 'off'} "
        f"(grounding/state/tracking are all ground truth)",
        flush=True,
    )
if args.render == "none":
    _render_none_blockers = _pixel_consumers(args)
    if _render_none_blockers:
        parser.error(
            "--render none removes all cameras, but this run consumes pixels: "
            + "; ".join(_render_none_blockers)
        )
if args.render_stride < 1:
    parser.error("--render_stride must be >= 1.")
if args.render_stride > 1 and args.render != "video":
    parser.error("--render_stride only applies to --render video.")
if args.render_stride > 1:
    # Widening sim.render_interval makes a per-step pixel consumer re-read a held frame.
    _render_stride_blockers = _pixel_consumers(args)
    if _render_stride_blockers:
        parser.error(
            "--render_stride > 1 starves the pixel consumers in this run "
            f"({'; '.join(_render_stride_blockers)}); use --video_stride to thin the video."
        )
if args.video_stride < 1:
    parser.error("--video_stride must be >= 1.")
# Explicit seed lists are a single-process convenience; the multi-worker launcher splits ranges.
if args.seeds is not None:
    if args.workers > 1:
        parser.error("--seeds is single-process; use --seed_start/--seed_end with --workers.")
    if args.load_init_from_dataset is not None:
        parser.error("--seeds and --load_init_from_dataset both index episodes; use one.")

_report_initialization_stage(
    args,
    "arguments parsed",
    task=args.task,
    device=args.device,
    worker=args.worker_id if args.worker_id >= 0 else "standalone",
)

# output path
output_path = os.path.join("results", f"{args.task}/{args.exp_name}")

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Make the robot env
# No cameras -> AppLauncher picks the lighter camera-free headless kit experience.
if args.render == "none":
    args.enable_cameras = False
_report_initialization_stage(args, "starting Isaac Sim")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
_report_initialization_stage(args, "Isaac Sim ready")

_report_initialization_stage(args, "importing Isaac extensions and task registry")
import gymnasium as gym

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Side-effect imports: these register the gym task ids the rollout looks up.
import isaaclab_mimic.envs                 # noqa: F401
import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
import isaaclab_tasks                      # noqa: F401
_report_initialization_stage(args, "Isaac extensions and task registry ready")


# Setup output paths and get env name
output_dir = os.path.join("results", f"{args.task}/{args.exp_name}")
output_file_name = "eval.hdf5"
task_name = args.task
if task_name:
    task_name = args.task.split(":")[-1]
env_name = task_name

print(f"Environment name: {env_name}", flush=True)

# Configure environment
_report_initialization_stage(args, "parsing environment config", environment=env_name)
env_cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)
_report_initialization_stage(args, "environment config ready", environment=env_name)

if args.vlm_cost.startswith("rekep") or args.vlm_state == "real":
    # The ReKep keypoint proposal and the SensedWorld perception read depth (+ instance seg for
    # fixture calibration) from table_cam; the annotators must be added to the cfg before
    # gym.make. RGB-only otherwise (byte-identical). startswith("rekep") covers the _vlm variants too.
    from rekep import isaaclab_helpers as _rekep_helpers

    _rekep_helpers.augment_table_cam_with_depth_and_seg(env_cfg)

# Control steps between recorded video frames; render_stride already thins the render itself.
_video_record_period = args.render_stride * args.video_stride

if args.render != "policy":
    # Render diet: drop the cameras nothing in this configuration reads. Sensors AND their
    # observation terms have to go together, or the obs manager asks a missing sensor for pixels.
    _dropped_cams = ("table_cam", "wrist_cam") if args.render == "none" else ("wrist_cam",)
    for _cam in _dropped_cams:
        setattr(env_cfg.scene, _cam, None)
        setattr(env_cfg.observations.policy, _cam, None)
    if args.render == "none":
        env_cfg.rerender_on_reset = False
    if args.render_stride > 1:
        # render_interval counts PHYSICS substeps; decimation of them is one control step.
        env_cfg.sim.render_interval = env_cfg.decimation * args.render_stride
    print(
        f"[render] mode={args.render} dropped={list(_dropped_cams)} "
        f"stride={args.render_stride} enable_cameras={args.enable_cameras}",
        flush=True,
    )

if args.static_collision_off:
    # Physics diet, visuals untouched: a static asset the task never contacts still pays PhysX
    # broadphase, superlinearly in num_envs. Fails loudly on a bad name, since a silent no-op would
    # look like a speedup that never happened.
    _diet_names = [n.strip() for n in args.static_collision_off.split(",") if n.strip()]
    for _name in _diet_names:
        _asset_cfg = getattr(env_cfg.scene, _name, None)
        if _asset_cfg is None:
            raise SystemExit(f"--static_collision_off: no scene asset named {_name!r}")
        _collision = getattr(getattr(_asset_cfg, "spawn", None), "collision_props", None)
        if _collision is None:
            raise SystemExit(
                f"--static_collision_off: {_name!r} has no spawn.collision_props to disable"
            )
        _collision.collision_enabled = False
    print(f"[scene] colliders disabled on {_diet_names} (visuals unchanged)", flush=True)

if args.collider_diet:
    # Per-prim, because --static_collision_off is whole-asset and this asset is both
    # decorative and load-bearing.
    from isaaclab.managers import EventTermCfg as _EventTerm

    _diet_root = args.collider_diet
    if getattr(env_cfg.scene, _diet_root, None) is None:
        raise SystemExit(f"--collider_diet: no scene asset named {_diet_root!r}")
    _reach_center = tuple(float(v) for v in env_cfg.scene.robot.init_state.pos)
    env_cfg.events.collider_diet = _EventTerm(
        func=_disable_out_of_reach_colliders,
        mode="prestartup",
        params={
            "prim_path_regex": f"/World/envs/env_.*/{_diet_root}",
            "center": _reach_center,
            "radius": args.collider_diet_radius,
        },
    )

env_cfg.env_name = env_name
if args.determine:
    env_cfg.seed = args.seed_start
    env_cfg.sim.physx.enable_enhanced_determinism = True
    print(
        "Deterministic eval enabled: "
        f"startup_seed={env_cfg.seed}, "
        "physx_enhanced_determinism=True, strict_torch=True, deterministic_rtx=requested",
        flush=True,
    )

# Extract success checking function
success_term = None
if hasattr(env_cfg.terminations, "success"):
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
else:
    raise NotImplementedError(
        "No success termination term was found in the environment."
    )

# Configure for data generation
# env_cfg.terminations = None
# env_cfg.observations.policy.concatenate_terms = False

# Create environment
_report_initialization_stage(args, "creating simulation environment")
env = gym.make(env_name, cfg=env_cfg).unwrapped
if args.profile:
    _prof_install(env)
_report_initialization_stage(args, "simulation environment ready")

# Derive the training-config name from the checkpoint dir, whose layout is
# ".../checkpoints/<config_name>/<exp_name>/<step>" for proxies and
# ".../checkpoints/pytorch/<config_name>" for the base.
def _config_name_from_checkpoint_dir(checkpoint_dir):
    if checkpoint_dir is None:
        raise ValueError(
            "A checkpoint dir is required so the training config name can be derived from it."
        )
    parts = [p for p in os.path.normpath(checkpoint_dir).split(os.sep) if p]
    if "checkpoints" in parts:
        idx = parts.index("checkpoints") + 1
        # skip an optional framework wrapper segment (e.g. "pytorch")
        if idx < len(parts) and parts[idx] == "pytorch":
            idx += 1
        if idx < len(parts):
            return parts[idx]
    raise ValueError(
        f"Could not derive a training config name from checkpoint dir: {checkpoint_dir!r}. "
        "Expected '.../checkpoints/<config_name>/<exp_name>/<step>' or "
        "'.../checkpoints/pytorch/<config_name>'."
    )


# load checkpoint
base_checkpoint_dir = args.base_checkpoint_dir
task_checkpoint_dir = args.task_checkpoint_dir
ref_checkpoint_dir = args.ref_checkpoint_dir
_report_initialization_stage(args, "resolving checkpoint configs")
base_policy = None
task_policy = None
ref_policy = None
required_policy_roles = _required_policy_roles(args)
if "base" in required_policy_roles:
    base_config_name = _config_name_from_checkpoint_dir(base_checkpoint_dir)
    base_config = _config.get_config(base_config_name)
    _report_initialization_stage(
        args,
        "loading base policy",
        config=base_config_name,
        decode_only=args.base_decode_only,
    )
    base_policy = policy_config.create_trained_policy(
        base_config,
        base_checkpoint_dir,
        pytorch_device=args.device,
        load_weights=not args.base_decode_only,
    )
    _report_initialization_stage(args, "base policy ready", config=base_config_name)
if "task" in required_policy_roles:
    task_config_name = _config_name_from_checkpoint_dir(task_checkpoint_dir)
    task_config = _config.get_config(task_config_name)
    if args.task_attention != "config":
        if not hasattr(task_config.model, "bidirectional_attention"):
            raise ValueError(
                "--task_attention is only supported for ProxyScore task models; "
                f"config {task_config_name!r} uses {type(task_config.model).__name__}."
            )
        task_config = dataclasses.replace(
            task_config,
            model=dataclasses.replace(
                task_config.model,
                bidirectional_attention=args.task_attention == "bidirectional",
            ),
        )
    task_attention = (
        "bidirectional"
        if getattr(task_config.model, "bidirectional_attention", False)
        else "causal"
    )
    _report_initialization_stage(
        args,
        "loading task policy",
        config=task_config_name,
        attention=task_attention,
    )
    task_policy = policy_config.create_trained_policy(
        task_config,
        task_checkpoint_dir,
        sample_kwargs={"num_steps": args.num_steps} if standalone_role == "task" else None,
        pytorch_device=args.device,
    )
    _report_initialization_stage(args, "task policy ready", config=task_config_name)
if "ref" in required_policy_roles:
    ref_config_name = _config_name_from_checkpoint_dir(ref_checkpoint_dir)
    ref_config = _config.get_config(ref_config_name)
    _report_initialization_stage(args, "loading ref policy", config=ref_config_name)
    ref_policy = policy_config.create_trained_policy(
        ref_config,
        ref_checkpoint_dir,
        sample_kwargs={"num_steps": args.num_steps} if standalone_role == "ref" else None,
        pytorch_device=args.device,
    )
    _report_initialization_stage(args, "ref policy ready", config=ref_config_name)

base_source = _base_source_name(args)
print(f"Base source: {base_source}", flush=True)
loaded_model_types = {
    role: policy._model.config.model_type.value
    for role, policy in (
        ("base", base_policy),
        ("task", task_policy),
        ("ref", ref_policy),
    )
    if policy is not None
}
print(
    "Loaded checkpoints: "
    + ", ".join(f"{role}={model_type}" for role, model_type in loaded_model_types.items()),
    flush=True,
)
if args.mpc_debug and base_policy is not None:
    base_policy._metadata = {
        **(getattr(base_policy, "_metadata", {}) or {}),
        "debug_torch_output_to_actions_norm_stats": True,
    }

# Opt-in action representation for the MBD base. None keeps the checkpoint's own decode surface,
# so every existing run is untouched.
base_decode_policy = None
if args.base_action_space == "demo_delta":
    _delta_mean, _delta_std = load_action_norm_stats_json(args.base_action_stats)
    base_decode_policy = DemoDeltaDecodePolicy(
        base_policy,
        _delta_mean,
        _delta_std,
        source=f"demo_delta:{args.base_action_stats}",
    )
    print(
        "Base action space: demo_delta "
        f"(stats={args.base_action_stats}, mean_shape={tuple(_delta_mean.shape)}, "
        f"std_shape={tuple(_delta_std.shape)})",
        flush=True,
    )
    print(
        f"Base demo delta std (model-space scale): {np.round(np.asarray(_delta_std), 4)}",
        flush=True,
    )

CONTROL_FREQUENCY = 15

mpc_planner = None
if _uses_vlm_mpc_base(args):
    inferred_task = _task_name_for_mpc(args.task)
    if _uses_accel_action_mpc(args):
        mpc_planner = AccelActionMPC(
            AccelMPCConfig(
                task_name=inferred_task,
                num_samples=args.mpc_num_samples,
                iterations=args.mpc_iterations,
                noise=args.mpc_noise,
                temperature=args.mpc_temperature,
                cost_style=args.mpc_cost,
                control_frequency=CONTROL_FREQUENCY,
                ddim_num_train_timesteps=args.mpc_ddim_train_timesteps,
            )
        )
    else:
        mpc_planner = SimFreeMPC(
            base_decode_policy or base_policy,
            SimFreeMPCConfig(
                task_name=inferred_task,
                num_samples=args.mpc_num_samples,
                iterations=args.mpc_iterations,
                noise=args.mpc_noise,
                temperature=args.mpc_temperature,
                action_dims=8,
                joint_delta_clip=args.mpc_joint_delta_clip,
                ddim_num_train_timesteps=args.mpc_ddim_train_timesteps,
                interpolate=args.interpolate,
                control_frequency=args.interpolate_high_frequency,
                sampler=args.sampler,
                interpolate_frequency=args.interpolate_low_frequency,
                interpolation_method=args.interpolation_method,
                cost_style=args.mpc_cost,
                optimize_space=args.mpc_optimize_space,
                grad_calc=args.grad_calc,
                logit_norm=args.mpc_logit_norm,
                mode_window=args.mpc_mode_window,
                eval_mean_plan=args.mpc_eval_mean_plan,
                ancestral_eta=args.mpc_ancestral_eta,
                rank_mode=args.rank_mode,
                prior_weight=args.prior_weight,
                prior_weight_high=args.prior_weight_high,
                prior_weight_schedule=args.prior_weight_schedule,
                feasibility_gate=args.feasibility_gate,
                cost_executable_actions=args.cost_executable_actions,
            ),
        )
    print(
        "Sim-free MPC planner enabled: "
        f"base_source={base_source}, task={inferred_task}, samples={args.mpc_num_samples}, "
        f"iterations={args.mpc_iterations}, update={args.mpc_update}, cost={args.mpc_cost}, "
        f"optimize_space={args.mpc_optimize_space}, sampler={args.sampler}, "
        f"grad_calc={args.grad_calc}, "
        f"ddim_train_timesteps={args.mpc_ddim_train_timesteps}, gamma_base={args.gamma_base}, "
        f"joint_delta_clip={args.mpc_joint_delta_clip}, interpolate={args.interpolate}, "
        f"interpolation_method={args.interpolation_method}, "
        f"interpolate_low_frequency={args.interpolate_low_frequency}, "
        f"interpolate_high_frequency={args.interpolate_high_frequency}",
        flush=True,
    )

vlm_bridge = None
if args.vlm_cost != "none":
    import yaml

    from vlm_dp import config_paths as vlm_config_paths
    from vlm_dp.bridge import VlmDpBridge

    if mpc_planner is None:
        raise SystemExit(
            "--vlm_cost needs a score-steering mode (--vlm_base/--task_steer/--full_steer): "
            "the sim-free MPC planner is only built under those modes."
        )
    _vlm_entry = _task_prompt_entry(args.task) or {}
    _vlm_roles = {k: _vlm_entry[k]
                  for k in ("grasp_obj", "place_obj", "grasp_objs", "support") if k in _vlm_entry}
    _vlm_vocab = _vlm_entry.get("objects")
    if args.vlm_derive_vocab:
        # Force the generalizable path: ignore any declared objects and roles, derive both from the
        # instruction. Support is not named here, so it falls to the geometric derivation (A.2).
        _vlm_vocab = None
        _vlm_roles.pop("support", None)
        _vlm_entry = {**_vlm_entry, "objects": None}
    if args.vlm_vocab and _vlm_vocab is not None:
        # Restrict to the named subset, keeping each name's detector text. Dropped objects are not
        # detected or tracked; they enter the cost as anonymous obstacle blobs instead. Isolates the
        # effect of naming distractors (they can capture a keypoint's identity when tracked).
        _keep = [n.strip() for n in args.vlm_vocab.split(",") if n.strip()]
        _missing = [n for n in _keep if n not in _vlm_vocab]
        if _missing:
            raise SystemExit(f"--vlm_vocab names {_missing} not in the task's objects {list(_vlm_vocab)}")
        _vlm_vocab = {n: _vlm_vocab[n] for n in _keep}
        print(f"[vocabulary] restricted to {list(_vlm_vocab)} (dropped "
              f"{[n for n in _vlm_entry['objects'] if n not in _vlm_vocab]})", flush=True)
    if _vlm_vocab is None and args.vlm_cost != "none":
        # No hand-authored vocabulary -> read it off the instruction, so a new task needs no config.
        # A declared entry still wins, so existing tasks are untouched until each is A/B'd across.
        from vlm_dp.grounding.vocabulary import derive as _derive_vocab

        _derived = _derive_vocab(_vlm_entry.get("prompt") or args.prompt or "")
        _vlm_vocab = _derived["objects"]
        for _k in ("grasp_objs", "place_obj"):
            if _k not in _vlm_roles and _derived.get(_k):
                _vlm_roles[_k] = _derived[_k]
        if "grasp_obj" not in _vlm_roles and _derived["grasp_objs"]:
            _vlm_roles["grasp_obj"] = _derived["grasp_objs"][0]
        print(f"[vocabulary] derived from the instruction: objects={_vlm_vocab} "
              f"grasp_objs={_derived['grasp_objs']} place_obj={_derived['place_obj']}", flush=True)
    with open(vlm_config_paths.resolve(args.vlm_cost_config, _REPO_DIR)) as _f:
        _vlm_cost_cfg = yaml.safe_load(_f)
    vlm_bridge = VlmDpBridge(
        args.vlm_cost,
        _vlm_roles,
        _vlm_cost_cfg,
        task_key=_task_name_for_mpc(args.task),
        device=args.device,
        state=args.vlm_state,
        track=args.vlm_track,
        segment=args.vlm_segment,
        vocab=_vlm_vocab,
        fixtures=_vlm_entry.get("fixtures", ()),
    )
    vlm_bridge.attach_cost(mpc_planner)
    print(
        f"[vlm_dp] CompositeCost attached: ground={args.vlm_cost}, state={args.vlm_state}, "
        f"roles={_vlm_roles}, terms={list(_vlm_cost_cfg['cost']['terms'])}",
        flush=True,
    )

dataset_file = None
dataset_demo_names = None
if args.load_init_from_dataset is not None:
    dataset_file = h5py.File(args.load_init_from_dataset, "r")
    dataset_demo_names = sorted(dataset_file["data"].keys(), key=_episode_sort_key)
    num_rollouts = args.seed_end - args.seed_start
    required_episodes = args.worker_rollout_offset + num_rollouts
    if required_episodes > len(dataset_demo_names):
        raise ValueError(
            f"Dataset {args.load_init_from_dataset} only has {len(dataset_demo_names)} episodes, "
            f"but episodes through index {required_episodes - 1} were requested."
        )

if standalone_role is None:
    loaded_policies = {
        role: policy
        for role, policy in (
            ("base", base_policy),
            ("task", task_policy),
            ("ref", ref_policy),
        )
        if policy is not None
    }
    horizons = {
        role: int(policy._model.config.action_horizon)
        for role, policy in loaded_policies.items()
    }
    if len(set(horizons.values())) > 1:
        raise ValueError(f"Action horizon mismatch across loaded policies: {horizons}.")
    if task_policy is not None and ref_policy is not None:
        task_action_dim = int(task_policy._model.config.action_dim)
        ref_action_dim = int(ref_policy._model.config.action_dim)
        if task_action_dim != ref_action_dim:
            raise ValueError(
                "Action dimension mismatch between task/ref policies: "
                f"task={task_action_dim}, ref={ref_action_dim}."
            )

    proxy_action_dims = [
        int(policy._model.config.action_dim)
        for policy in (task_policy, ref_policy)
        if policy is not None
    ]
    _warn_if_norm_mismatch(
        base_policy,
        task_policy,
        ref_policy,
        action_dim=min(proxy_action_dims) if proxy_action_dims else 8,
    )
    _assert_score_space_compatibility(base_policy, task_policy, ref_policy, args)

steps_per_inference = int(args.steps_per_inference)
print(
    f"steps_per_inference={steps_per_inference} "
    f"({steps_per_inference / CONTROL_FREQUENCY:.3f}s between replans at {CONTROL_FREQUENCY}Hz)",
    flush=True,
)

_report_initialization_stage(args, "resetting environment for warmup")
env_obs_dict, _ = env.reset(seed=args.seed_start if args.determine else None)
if args.determine:
    # Decouple policy/MPC randomness from random draws consumed by env.reset().
    _seed_runtime(args.seed_start)
if args.vlm_cost != "none":
    vlm_bridge.reset(env)
_report_initialization_stage(args, "building warmup observation")
obs = get_pi_observation(env_obs_dict["policy"])
obs["prompt"] = args.prompt
_report_initialization_stage(args, "compiling/warming policy inference")
with torch.no_grad():
    warmup_actions, _ = infer_actions_with_mpc(
        base_policy,
        task_policy,
        ref_policy,
        copy.deepcopy(obs),
        args,
        mpc_planner=mpc_planner,
        mpc_context=(
            None
            if standalone_role is not None
            else (
                vlm_bridge.context(env, env_obs_dict)
                if args.vlm_cost != "none"
                else build_mpc_context(env, env_obs_dict, args)
            )
        ),
        base_decode_policy=base_decode_policy,
    )
_report_initialization_stage(args, "initialization complete")
_wait_for_worker_start(args)

if args.dry_run:
    print("Dry-run inference check passed", flush=True)
    print(
        "  checkpoints_loaded: "
        + ", ".join(
            f"{role}={model_type}" for role, model_type in loaded_model_types.items()
        ),
        flush=True,
    )
    print(f"  base_source: {_LAST_INFERENCE_RUNTIME.get('base_source')}", flush=True)
    print(
        "  used_base_model_velocity: "
        f"{_LAST_INFERENCE_RUNTIME.get('used_base_model_velocity')}",
        flush=True,
    )
    print(f"  steering_mode: {_LAST_INFERENCE_RUNTIME.get('steering_mode')}", flush=True)
    print(f"  x_t_shape: {_LAST_INFERENCE_RUNTIME.get('x_t_shape')}", flush=True)
    print(f"  v_vlm_shape: {_LAST_INFERENCE_RUNTIME.get('v_vlm_shape')}", flush=True)
    print(f"  score_shape: {_LAST_INFERENCE_RUNTIME.get('score_shape')}", flush=True)
    print(f"  v_task_shape: {_LAST_INFERENCE_RUNTIME.get('v_task_shape')}", flush=True)
    print(f"  v_ref_shape: {_LAST_INFERENCE_RUNTIME.get('v_ref_shape')}", flush=True)
    print(f"  proxy_task_shape: {_LAST_INFERENCE_RUNTIME.get('proxy_task_shape')}", flush=True)
    print(f"  proxy_ref_shape: {_LAST_INFERENCE_RUNTIME.get('proxy_ref_shape')}", flush=True)
    print(f"  output_action_shape: {np.asarray(warmup_actions).shape}", flush=True)
    if _LAST_INFERENCE_RUNTIME.get("mpc_last") is not None:
        print(f"  mpc_last: {_LAST_INFERENCE_RUNTIME['mpc_last']}", flush=True)
    env.close()
    if dataset_file is not None:
        dataset_file.close()
    simulation_app.close()
    sys.exit(0)

print("Ready!", flush=True)
comparison_metadata = {
    "shared_flow_path": {
        "description": "Compare on the rollout's shared denoise path.",
        "comparisons": {
            "ref_minus_base": {
                "teacher_model": "base_model",
                "student_model": "ref_model",
            },
            "task_minus_ref": {
                "teacher_model": "ref_model",
                "student_model": "task_model",
            },
        },
    },
    "teacher_denoise_path": {
        "description": "Compare on the base model's own denoise path.",
        "comparisons": {
            "ref_minus_base": {
                "teacher_model": "base_model",
                "student_model": "ref_model",
            },
            "task_minus_ref": {
                "teacher_model": "ref_model",
                "student_model": "task_model",
            },
        },
    },
}
overall_comparison_stats = {
    path_name: {
        comparison_name: None
        for comparison_name in path_metadata["comparisons"]
    }
    for path_name, path_metadata in comparison_metadata.items()
}
episode_comparison_summaries = []
total_comparison_observation_steps = 0
total_inference_time_s = 0.0
total_inference_calls = 0
video_run_id = time.strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
experiment_output_path = os.path.join(
    output_path,
    _experiment_output_name(args, run_id=video_run_id),
)
os.makedirs(experiment_output_path, exist_ok=False)
experiment_results_path = os.path.join(experiment_output_path, "results.json")
experiment_results = {
    "run_id": video_run_id,
    "config_slug": _video_config_slug(args),
    "command": sys.argv,
    "config": vars(args),
    # Which CODE ran, so an arm's identity survives an uncommitted working tree (see _code_provenance).
    # The active cost YAML is hashed alongside it: editing weights changes behaviour as much as
    # editing a term, and it is the change most likely to go unrecorded between two compared arms.
    "code": _code_provenance((getattr(args, "vlm_cost_config", None),)),
    "episodes": [],
}
_write_experiment_results(experiment_results_path, experiment_results)
print(
    f"Experiment output: {experiment_output_path}; max rollout duration "
    f"{args.task_num_steps / CONTROL_FREQUENCY:.1f}s",
    flush=True,
)
mpc_debug_log_file = None
mpc_debug_log_path = None
if args.mpc_debug:
    mpc_debug_log_path = os.path.join(experiment_output_path, "mpc_debug.jsonl")
    mpc_debug_log_file = open(mpc_debug_log_path, "w", encoding="utf-8")
    print(f"MPC debug log: {mpc_debug_log_path}", flush=True)
    _write_mpc_debug_log(
        mpc_debug_log_file,
        "run_start",
        run_id=video_run_id,
        task=args.task,
        prompt=args.prompt,
        sampler=args.sampler,
        grad_calc=args.grad_calc,
        mpc_cost=args.mpc_cost,
        mpc_update=args.mpc_update,
        mpc_optimize_space=args.mpc_optimize_space,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        determine=args.determine,
        steps_per_inference=steps_per_inference,
        task_num_steps=args.task_num_steps,
    )
# One booted app serves every seed in the queue; boot is paid once per process, not per episode.
eval_seeds = (
    [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.seeds
    else list(range(args.seed_start, args.seed_end))
)
print(f"[queue] {len(eval_seeds)} episode(s) in one app: seeds={eval_seeds}", flush=True)
for rollout_idx, seed in enumerate(eval_seeds):
    success = None
    episode_comparison_stats = {
        path_name: {
            comparison_name: None
            for comparison_name in path_metadata["comparisons"]
        }
        for path_name, path_metadata in comparison_metadata.items()
    }
    episode_comparison_observation_steps = 0
    episode_inference_time_s = 0.0
    episode_inference_calls = 0

    # Set seed for generation. Preserve the original path unless strict
    # determinism was explicitly requested.
    if args.determine:
        _seed_runtime(seed)
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if mpc_planner is not None:
        mpc_planner.reset_episode()

    _profile_reset_start = time.perf_counter()

    # Reset before starting
    if dataset_file is not None:
        if args.determine:
            env.seed(seed)
        initial_state = _load_hdf5_state(
            dataset_file["data"][
                dataset_demo_names[rollout_idx + args.worker_rollout_offset]
            ]["initial_state"],
            env.device,
        )
        env_obs_dict, _ = env.reset_to(initial_state, env_ids=None, is_relative=True)
    else:
        env_obs_dict, _ = env.reset(seed=seed if args.determine else None)
    if args.determine:
        # Make the policy/MPC random stream independent of reset implementation details.
        _seed_runtime(seed)

    if args.initial_action_after_reset:
        # Optional hold step for environments that need an action-buffer flush. Keep
        # the current gripper command too, otherwise dataset-initialized eval is shifted.
        initial_joint_pos = _to_numpy_unbatched(env_obs_dict["policy"]["joint_pos"])
        initial_action = initial_joint_pos[:8]
        env_obs_dict, _, _, _, _ = env.step(
            torch.as_tensor(
                initial_action[None],
                dtype=torch.float32,
                device=env.device,
            )
        )

    if args.vlm_cost != "none":
        try:
            vlm_bridge.reset(env)
        except (ValueError, AssertionError, IndexError) as exc:
            # An ungroundable scene must cost one episode, not the rest of the run. Recorded as
            # errored and excluded from the success denominator: it is not a policy failure.
            print(f"[eval] seed {seed}: grounding failed ({exc}); episode skipped", flush=True)
            experiment_results["episodes"].append({
                "rollout_index": rollout_idx,
                "seed": seed,
                "errored": True,
                "error": f"grounding: {exc}",
            })
            _write_experiment_results(experiment_results_path, experiment_results)
            continue

    current_subtasks = _debug_subtasks(env_obs_dict)
    _observe_mpc_subtasks(mpc_planner, current_subtasks)
    current_phase = _debug_phase_from_subtasks(args.task, current_subtasks)
    # Ever-fired set for the futility abort: flags are INSTANTANEOUS (grasp_pear drops on release,
    # pear_on_scale drops if the pear slides off), so an episode mid-recovery reads all-false.
    subtasks_ever_fired = {k for k, v in (current_subtasks or {}).items() if v}
    print(f"phase seed={seed} step=0 {current_phase} subtasks={current_subtasks}", flush=True)
    _write_mpc_debug_log(
        mpc_debug_log_file,
        "rollout_start",
        seed=seed,
        rollout_idx=rollout_idx,
        step=0,
        phase=current_phase,
        subtasks=current_subtasks,
        # The {placeholder} values THIS episode's plan was rendered with. With the per-step
        # grounded keypoints below, the log carries everything an offline re-evaluation of any
        # completion predicate against this exact episode needs.
        **({"plan_fields": _pf} if (_pf := (getattr(getattr(vlm_bridge, "grounding", None),
                                                    "plan_fields", None)
                                            if vlm_bridge is not None else None)) else {}),
    )
    _emit_worker_progress(
        args,
        "rollout_start",
        seed=seed,
        rollout_index=rollout_idx,
        task_num_steps=args.task_num_steps,
    )

    pear_guard_state = (
        _capture_debug_object_xy_guard(
            env,
            "pear",
            margin=args.debug_hold_pear_xy_margin,
        )
        if args.debug_hold_pear
        else None
    )
    if pear_guard_state is not None:
        print(
            "debug_hold_pear enabled "
            f"xy_min={pear_guard_state['xy_min'].detach().cpu().tolist()} "
            f"xy_max={pear_guard_state['xy_max'].detach().cpu().tolist()}",
            flush=True,
        )

    excute_frames = []
    thermal_overlay_frames = []
    sound_audio_frames = []
    obs = get_pi_observation(env_obs_dict["policy"])
    obs["prompt"] = args.prompt
    video_header_lines = _video_config_lines(args, seed=seed)

    # Reset (env + grounding) is charged whole; the loop buckets start clean so the sim
    # wrappers only report per-step cost.
    _profile_reset_s = time.perf_counter() - _profile_reset_start
    _PROFILE_TOTALS.clear()
    _profile_loop_start = time.perf_counter()

    # ========== policy control loop ==============
    step_idx = 0
    success = False
    force_replan = False
    action_start_step = -steps_per_inference
    actions = None
    latch_raw = {}
    for step_idx in tqdm(
        range(args.task_num_steps),
        desc="Policy Control Loop",
        disable=args.worker_progress_path is not None,
    ):
        try:
            if (
                actions is None
                or force_replan
                or step_idx - action_start_step >= steps_per_inference
            ):
                # print('predict_action')
                # run inference
                with torch.no_grad():
                    infer_start = time.perf_counter()
                    warm_shift_steps = (
                        0 if actions is None else max(step_idx - action_start_step, 0)
                    )
                    if args.vlm_cost != "none":
                        vlm_bridge.advance(_debug_subtasks(env_obs_dict))
                    mpc_context = (
                        None
                        if standalone_role is not None
                        else (
                            vlm_bridge.context(env, env_obs_dict, warm_shift_steps)
                            if args.vlm_cost != "none"
                            else build_mpc_context(env, env_obs_dict, args)
                        )
                    )

                    fk_m = int(getattr(args, "fk_fork", 1))
                    fk_active = fk_m > 1 and (
                        bool(getattr(args, "fk_always", False))
                        or (isinstance(mpc_context, dict)
                            and float(mpc_context.get("steer_authority", 0.0)) > 0.0)
                    )
                    if fk_active:
                        # Window-scoped best-of-M fork: inside an open failure window the base's
                        # basin has just been shown wrong, so run M independent chains from fresh
                        # noise and commit to the cheapest -- selection, not in-basin tilting.
                        fk_best = None
                        fk_costs = []
                        for _fk in range(fk_m):
                            a_i, s_i = infer_actions_with_mpc(
                                base_policy,
                                task_policy,
                                ref_policy,
                                copy.deepcopy(obs),
                                args,
                                mpc_planner=mpc_planner,
                                mpc_context=mpc_context,
                                warm_shift_steps=warm_shift_steps,
                                base_decode_policy=base_decode_policy,
                            )
                            # Converged cost lives in the module-level runtime record, not in the
                            # returned stats (those stay empty unless --compare_difference).
                            last = _LAST_INFERENCE_RUNTIME.get("mpc_last") or {}
                            j_i = float(last.get("cost_min", float("inf")))
                            fk_costs.append(j_i)
                            if fk_best is None or j_i < fk_best[0]:
                                fk_best = (j_i, a_i, s_i)
                        _, actions, compare_stats = fk_best
                        compare_stats = dict(compare_stats or {})
                        compare_stats["fk_fork_m"] = fk_m
                        compare_stats["fk_best_cost"] = fk_best[0]
                        finite = [c for c in fk_costs if c != float("inf")]
                        if len(finite) > 1:
                            compare_stats["fk_cost_spread"] = max(finite) - min(finite)
                    else:
                        actions, compare_stats = infer_actions_with_mpc(
                            base_policy,
                            task_policy,
                            ref_policy,
                            copy.deepcopy(obs),
                            args,
                            mpc_planner=mpc_planner,
                            mpc_context=mpc_context,
                            warm_shift_steps=warm_shift_steps,
                            base_decode_policy=base_decode_policy,
                        )
                    # Gate observability: stamp authority + evidence every inference (active or
                    # not), so window behavior is auditable from mpc_debug alone.
                    if isinstance(mpc_context, dict) and "steer_authority" in mpc_context:
                        compare_stats = dict(compare_stats or {})
                        compare_stats["steer_authority"] = float(mpc_context["steer_authority"])
                        ev = mpc_context.get("steer_events") or {}
                        compare_stats["steer_events_closed_empty"] = int(ev.get("closed_empty", 0))
                        compare_stats["steer_events_backtracks"] = int(ev.get("backtracks", 0))
                        if "stage_env_steps" in mpc_context:
                            compare_stats["stage_env_steps"] = int(mpc_context["stage_env_steps"])
                    if args.vlm_cost != "none":
                        # Filter before observe_plan: the hold sensor must see what will actually
                        # be commanded. Costs, scores and plan_ref (arm-only) are untouched.
                        actions, latch_suppressed = vlm_bridge.filter_plan(
                            actions, steps_per_inference
                        )
                        latch_raw = dict(latch_suppressed)
                        vlm_bridge.observe_plan(actions)
                    infer_elapsed = time.perf_counter() - infer_start
                    _prof_toc("planner.replan", infer_start)
                    if args.mpc_debug:
                        # Gate/fork observability rides the eval loop (compare_stats), not the
                        # planner's mpc_last — merge those keys in explicitly or they are lost.
                        _gate_keys = ("steer_authority", "steer_events_closed_empty",
                                      "steer_events_backtracks", "stage_env_steps",
                                      "fk_fork_m", "fk_best_cost", "fk_cost_spread")
                        _mpc_logged = dict(_mpc_debug_stats(_LAST_INFERENCE_RUNTIME.get("mpc_last")) or {})
                        _mpc_logged.update({k: compare_stats[k] for k in _gate_keys
                                            if isinstance(compare_stats, dict) and k in compare_stats})
                        # Shadow completion-predicate decision beside the scalar sub-goal one.
                        # Read off the bridge, never off mpc_context: the planner never saw it.
                        _pred = (getattr(vlm_bridge, "pred_shadow", None)
                                 if args.vlm_cost != "none" else None)
                        _write_mpc_debug_log(
                            mpc_debug_log_file,
                            "inference",
                            seed=seed,
                            step=step_idx,
                            phase=current_phase,
                            subtasks=current_subtasks,
                            elapsed_s=infer_elapsed,
                            mpc=_mpc_logged,
                            **({"pred_shadow": _pred} if _pred is not None else {}),
                            mpc_trace=_LAST_INFERENCE_RUNTIME.get("mpc_trace", []),
                        )
                    episode_inference_time_s += infer_elapsed
                    episode_inference_calls += 1
                    total_inference_time_s += infer_elapsed
                    total_inference_calls += 1
                    force_replan = False
                    if args.compare_difference and compare_stats:
                        episode_comparison_observation_steps += 1
                        for path_name, path_stats in compare_stats.items():
                            for comparison_name, batch_stats in path_stats.items():
                                episode_comparison_stats[path_name][
                                    comparison_name
                                ] = accumulate_stats(
                                    episode_comparison_stats[path_name][
                                        comparison_name
                                    ],
                                    batch_stats,
                                )

                # execute actions
                start_idx = 0
                end_idx = start_idx + steps_per_inference
                actions = actions[start_idx:end_idx]
                action_start_step = step_idx

            action_step = actions[step_idx - action_start_step]
            if args.vlm_cost != "none":
                # One sensor sample per APPLIED action, before the step that applies it: the
                # aperture reflects the previous command and pairs with the one now being sent.
                _prof_start = time.perf_counter()
                vlm_bridge.observe_step(action_step)
                _prof_toc("vlm.observe_step", _prof_start)

            # perform step
            _prof_start = time.perf_counter()
            env_obs_dict, rewards, terminated, truncated, extras = env.step(
                torch.as_tensor(
                    action_step[None],
                    dtype=torch.float32,
                    device=env.device,
                )
            )
            _prof_toc("env.step", _prof_start)
            if args.debug_hold_pear:
                pear_grasped = _subtask_flag_from_obs(env_obs_dict, "grasp_pear")
                pose_changed = _apply_debug_object_xy_guard(
                    env,
                    pear_guard_state,
                    release=pear_grasped,
                )
                if pose_changed:
                    env_obs_dict = env.observation_manager.compute(update_history=True)

            next_subtasks = _debug_subtasks(env_obs_dict)
            _observe_mpc_subtasks(mpc_planner, next_subtasks)
            next_phase = _debug_phase_from_subtasks(args.task, next_subtasks)
            if next_phase != current_phase:
                print(
                    f"phase_transition seed={seed} step={step_idx + 1} "
                    f"{current_phase}->{next_phase} subtasks={next_subtasks}",
                    flush=True,
                )
                _write_mpc_debug_log(
                    mpc_debug_log_file,
                    "phase_transition",
                    seed=seed,
                    step=step_idx + 1,
                    from_phase=current_phase,
                    to_phase=next_phase,
                    subtasks=next_subtasks,
                )
                force_replan = True
            current_phase = next_phase
            current_subtasks = next_subtasks
            # Futility abort (screens only): a rollout with no subtask EVER fired by the deadline
            # is already decided. Ever-fired, not currently-true -- flags are instantaneous and all
            # drop during a recovery, exactly when the episode must not be culled.
            subtasks_ever_fired |= {k for k, v in (next_subtasks or {}).items() if v}
            if (getattr(args, "abort_no_subtask_by", 0) > 0
                    and step_idx + 1 >= int(args.abort_no_subtask_by)
                    and next_subtasks
                    and not subtasks_ever_fired):
                print(f"futility_abort seed={seed} step={step_idx + 1}: no subtask ever fired", flush=True)
                _write_mpc_debug_log(mpc_debug_log_file, "futility_abort", seed=seed, step=step_idx + 1)
                break
            _prof_start = time.perf_counter()
            policy_step_obs = env_obs_dict["policy"]
            step_trace = {
                "action": np.asarray(action_step),
                "joint_pos": _to_numpy_unbatched(policy_step_obs["joint_pos"]),
            }
            # Keep the raw command visible when the gripper latch suppressed it, so log mining
            # still sees the glitch at the decision level.
            _latch_g = latch_raw.get(step_idx - action_start_step)
            if _latch_g is not None:
                step_trace["action_gripper_raw"] = _latch_g
            if "eef_pos" in policy_step_obs:
                step_trace["eef_pos"] = _to_numpy_unbatched(policy_step_obs["eef_pos"])
            # Object poses per step: without these a rollout log cannot rebuild the cost context
            # offline, so executed plans cannot be re-scored against demonstrations.
            _rigid = getattr(env.scene, "rigid_objects", None) or {}
            if _rigid:
                step_trace["object_poses"] = {
                    _n: _to_numpy_unbatched(_o.data.root_state_w[:, :7])
                    for _n, _o in _rigid.items()
                }
            # The controller's BELIEF, beside the ground truth above. Under --vlm_state real the
            # arm descends toward the belief, so an upward-biased one stops the gripper short and
            # reads as a cost fault; logging both makes estimate quality checkable per step.
            _bel = getattr(vlm_bridge, "world", None) if "vlm_bridge" in dir() else None
            if _bel is not None and getattr(_bel, "_pos", None):
                try:
                    step_trace["object_beliefs"] = {
                        _n: np.asarray(_p, dtype=np.float64) for _n, _p in _bel._pos.items()
                    }
                except Exception:      # diagnostics must never take down a rollout
                    pass
            # The GROUNDED KEYPOINT ARRAY, plus the TCP and aperture pushed alongside it. This is
            # the exact frame the completion predicates saw this step, so a log alone is enough to
            # re-evaluate any predicate offline against the run that produced it.
            _ph = getattr(vlm_bridge, "_pred_hist", None) if "vlm_bridge" in dir() else None
            if _ph is not None and len(_ph):
                try:
                    step_trace["grounding_kp"] = np.round(_ph.kp[-1], 4)
                    step_trace["grounding_tcp"] = np.round(_ph.eef[-1], 4)
                    step_trace["grounding_aperture"] = round(float(_ph.gripper_aperture[-1]), 4)
                except Exception:      # diagnostics must never take down a rollout
                    pass
            _write_mpc_debug_log(
                mpc_debug_log_file,
                "step",
                seed=seed,
                step=step_idx + 1,
                phase=current_phase,
                subtasks=current_subtasks,
                action_gripper=_debug_action_gripper(action_step),
                **step_trace,
            )
            _prof_toc("debug.step_log", _prof_start)

            _prof_start = time.perf_counter()
            obs = get_pi_observation(env_obs_dict["policy"])
            obs["prompt"] = args.prompt
            _prof_toc("obs.to_policy", _prof_start)

            # Check for task success using success_term
            _prof_start = time.perf_counter()
            task_success = bool(success_term.func(env, **success_term.params)[0])
            _prof_toc("success_term", _prof_start)

            # save visualization
            _prof_start = time.perf_counter()
            record_frame = args.render != "none" and (step_idx % _video_record_period == 0)
            debug_overlay = None
            if record_frame and args.mpc_debug_video_overlay:
                debug_overlay = {
                    "env": env,
                    "axes": _collect_mpc_debug_frames(
                        env,
                        axis_length=args.mpc_debug_axis_length,
                    ),
                }
            if record_frame and vlm_bridge is not None:
                debug_overlay = debug_overlay or {"env": env, "axes": {}}
                try:
                    debug_overlay["vlm"] = vlm_bridge.viz()
                except Exception as exc:
                    print(f"[vlm_dp] viz overlay failed: {exc}", flush=True)
            if record_frame:
                vis_image = _build_rollout_frame(
                    obs,
                    use_thermal_overlay=False,
                    debug_overlay=debug_overlay,
                )
                excute_frames.append(_add_video_header(vis_image, video_header_lines))
            if record_frame and _has_sound_observation(obs):
                sound_audio_frame = _build_stereo_sound_audio_frame(
                    env,
                    frame_index=len(sound_audio_frames),
                    fps=CONTROL_FREQUENCY,
                )
                if sound_audio_frame is not None:
                    sound_audio_frames.append(sound_audio_frame)
            if _has_thermal_observation(obs):
                thermal_overlay_frames.append(
                    _add_video_header(
                        _build_rollout_frame(
                            obs,
                            use_thermal_overlay=True,
                            debug_overlay=debug_overlay,
                        ),
                        video_header_lines,
                    )
                )
            _prof_toc("video.frame_build", _prof_start)

            step_idx += 1
            _emit_worker_progress(
                args,
                "step",
                seed=seed,
                rollout_index=rollout_idx,
                step=step_idx,
                task_num_steps=args.task_num_steps,
            )

            if terminated or truncated or task_success:
                if args.task_debug:
                    _print_task_debug_done(
                        env=env,
                        step_idx=step_idx,
                        terminated=terminated,
                        truncated=truncated,
                        task_success=task_success,
                    )
                print("terminated or truncated or task completed")
                success = task_success
                _write_mpc_debug_log(
                    mpc_debug_log_file,
                    "done",
                    seed=seed,
                    step=step_idx,
                    phase=current_phase,
                    subtasks=current_subtasks,
                    terminated=terminated,
                    truncated=truncated,
                    task_success=task_success,
                )
                break

        except KeyboardInterrupt:
            print("Interrupted!")
            break

    _profile_loop_s = time.perf_counter() - _profile_loop_start
    _profile_encode_start = time.perf_counter()

    # save excute_frames as video
    video_name = _episode_video_name(seed, success)
    if success:
        print("success")
    else:
        print("fail")

    video_path = os.path.join(experiment_output_path, video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if not excute_frames:
        # --render none: no camera, so no artifact video. Logs and results.json still land.
        print(f"[render] no frames recorded for seed {seed}; skipping video", flush=True)
        video_name = None
    else:
        audio_path = _write_stereo_sound_audio(
            os.path.join(experiment_output_path, f"{seed}_recording.wav"),
            sound_audio_frames,
        )
        video_write_path = (
            os.path.join(experiment_output_path, f"{seed}_recording.mp4")
            if audio_path is not None
            else video_path
        )
        # A strided recording holds fewer frames per second of rollout; keep playback real-time.
        video_fps = CONTROL_FREQUENCY / _video_record_period
        out = cv2.VideoWriter(
            video_write_path,
            fourcc,
            video_fps,
            (excute_frames[0].shape[1], excute_frames[0].shape[0]),
        )
        for frame in excute_frames:
            out.write(frame)
        out.release()
        _transcode_h264(video_write_path, "rollout video")
        if audio_path is not None:
            _mux_audio_into_video(video_write_path, video_path, audio_path, "rollout")

    thermal_video_name = None
    if thermal_overlay_frames:
        thermal_video_name = _episode_video_name(
            seed,
            success,
            suffix="thermal_overlay",
        )
        thermal_video_path = os.path.join(experiment_output_path, thermal_video_name)
        thermal_out = cv2.VideoWriter(
            thermal_video_path,
            fourcc,
            15,
            (
                thermal_overlay_frames[0].shape[1],
                thermal_overlay_frames[0].shape[0],
            ),
        )
        for frame in thermal_overlay_frames:
            thermal_out.write(frame)
        thermal_out.release()
        _transcode_h264(thermal_video_path, "thermal video")

    print("video saved")
    _profile_extra = {}
    if args.profile:
        _profile_extra["profile"] = _build_profile_report(
            reset_s=_profile_reset_s,
            loop_s=_profile_loop_s,
            encode_s=time.perf_counter() - _profile_encode_start,
            steps=step_idx,
        )
        _print_profile_report(seed, _profile_extra["profile"])
    _write_mpc_debug_log(
        mpc_debug_log_file,
        "rollout_end",
        seed=seed,
        rollout_idx=rollout_idx,
        steps=step_idx,
        phase=current_phase,
        subtasks=current_subtasks,
        success=success,
        video_path=video_path,
        **_profile_extra,
    )
    _emit_worker_progress(
        args,
        "rollout_end",
        seed=seed,
        rollout_index=rollout_idx,
        steps=step_idx,
        task_num_steps=args.task_num_steps,
        success=bool(success),
    )
    if episode_inference_calls:
        avg_infer_ms = 1000.0 * episode_inference_time_s / episode_inference_calls
        print(
            f"seed {seed} infer_actions avg latency: {avg_infer_ms:.2f} ms "
            f"over {episode_inference_calls} calls"
        )
    else:
        avg_infer_ms = None

    experiment_results["episodes"].append(
        {
            "rollout_index": rollout_idx,
            "seed": seed,
            "success": bool(success),
            "steps": step_idx,
            "video": video_name,
            "thermal_video": thermal_video_name,
            "inference_calls": episode_inference_calls,
            "average_inference_ms": avg_infer_ms,
        }
    )
    _write_experiment_results(experiment_results_path, experiment_results)

    if args.compare_difference:
        episode_summary = {
            "rollout_index": rollout_idx,
            "seed": seed,
            "success": bool(success),
            "n_observation_steps": episode_comparison_observation_steps,
            "paths": {},
        }
        for path_name, path_metadata in comparison_metadata.items():
            path_summary = {
                "description": path_metadata["description"],
                "comparisons": {},
            }
            for comparison_name, metadata in path_metadata["comparisons"].items():
                episode_stats = episode_comparison_stats[path_name][comparison_name]
                if episode_stats is None:
                    continue
                overall_comparison_stats[path_name][comparison_name] = accumulate_stats(
                    overall_comparison_stats[path_name][comparison_name],
                    episode_stats,
                )
                path_summary["comparisons"][comparison_name] = {
                    **metadata,
                    "total_stats": stats_to_serializable(episode_stats),
                    "average_stats_per_observation_step": average_stats_per_step(
                        episode_stats, episode_comparison_observation_steps
                    ),
                    "metrics": summarize_metrics(episode_stats),
                }
            if path_summary["comparisons"]:
                episode_summary["paths"][path_name] = path_summary

        total_comparison_observation_steps += episode_comparison_observation_steps
        episode_comparison_summaries.append(episode_summary)

env.close()
if dataset_file is not None:
    dataset_file.close()

if args.compare_difference:
    comparison_summary = {
        "compare_difference": True,
        "output_path": experiment_output_path,
        "episodes": episode_comparison_summaries,
        "overall": {
            "n_episodes": len(episode_comparison_summaries),
            "n_observation_steps": total_comparison_observation_steps,
            "paths": {},
        },
    }
    for path_name, path_metadata in comparison_metadata.items():
        path_summary = {
            "description": path_metadata["description"],
            "comparisons": {},
        }
        for comparison_name, metadata in path_metadata["comparisons"].items():
            overall_stats = overall_comparison_stats[path_name][comparison_name]
            if overall_stats is None:
                continue
            path_summary["comparisons"][comparison_name] = {
                **metadata,
                "total_stats": stats_to_serializable(overall_stats),
                "average_stats_per_observation_step": average_stats_per_step(
                    overall_stats, total_comparison_observation_steps
                ),
                "metrics": summarize_metrics(overall_stats),
            }
        if path_summary["comparisons"]:
            comparison_summary["overall"]["paths"][path_name] = path_summary

    comparison_path = os.path.join(experiment_output_path, "compare_difference.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"compare difference statistics saved to {comparison_path}")
    experiment_results["compare_difference"] = os.path.basename(comparison_path)

if total_inference_calls:
    avg_infer_ms = 1000.0 * total_inference_time_s / total_inference_calls
    print(
        f"overall infer_actions avg latency: {avg_infer_ms:.2f} ms "
        f"over {total_inference_calls} calls"
    )

experiment_results["total_inference_calls"] = total_inference_calls
experiment_results["total_inference_time_s"] = total_inference_time_s
_write_experiment_results(experiment_results_path, experiment_results)

_write_mpc_debug_log(
    mpc_debug_log_file,
    "run_end",
    total_inference_calls=total_inference_calls,
    total_inference_time_s=total_inference_time_s,
)
if mpc_debug_log_file is not None:
    mpc_debug_log_file.close()
_emit_worker_progress(
    args,
    "run_end",
    results_path=os.path.abspath(experiment_results_path),
    output_path=os.path.abspath(experiment_output_path),
)

# Close the simulation app after environment is closed
simulation_app.close()
