from __future__ import annotations

import math


def squaredcos_cap_v2_alpha_bar(t: float) -> float:
    """Continuous alpha_bar used by diffusers' squared-cosine schedule."""
    return math.cos((float(t) + 0.008) / 1.008 * math.pi / 2.0) ** 2


def ddim_alphas_cumprod(
    num_train_timesteps: int,
    *,
    beta_schedule: str = "squaredcos_cap_v2",
) -> list[float]:
    if beta_schedule != "squaredcos_cap_v2":
        raise ValueError(f"unsupported beta_schedule={beta_schedule!r}")
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")

    alphas = []
    alpha_cumprod = 1.0
    for idx in range(int(num_train_timesteps)):
        t1 = idx / float(num_train_timesteps)
        t2 = (idx + 1) / float(num_train_timesteps)
        beta = min(1.0 - squaredcos_cap_v2_alpha_bar(t2) / squaredcos_cap_v2_alpha_bar(t1), 0.999)
        alpha_cumprod *= 1.0 - beta
        alphas.append(alpha_cumprod)
    return alphas


def ddim_iteration_alphas(
    *,
    iteration: int,
    num_iterations: int,
    num_train_timesteps: int = 100,
    beta_schedule: str = "squaredcos_cap_v2",
    set_alpha_to_one: bool = True,
) -> tuple[float, float]:
    """Return (alpha_bar_t, alpha_bar_prev) for one reverse DDIM iteration."""
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if iteration < 0 or iteration >= num_iterations:
        raise ValueError(
            f"iteration must be in [0, {num_iterations}), got {iteration}"
        )
    if num_iterations > num_train_timesteps:
        raise ValueError("num_iterations must be <= num_train_timesteps")

    alphas_cumprod = ddim_alphas_cumprod(
        num_train_timesteps,
        beta_schedule=beta_schedule,
    )
    step_ratio = int(num_train_timesteps) // int(num_iterations)
    if step_ratio <= 0:
        raise ValueError("num_iterations must be <= num_train_timesteps")

    timestep = int((int(num_iterations) - 1 - int(iteration)) * step_ratio)
    prev_timestep = timestep - step_ratio
    alpha_t = float(alphas_cumprod[timestep])
    if prev_timestep >= 0:
        alpha_prev = float(alphas_cumprod[prev_timestep])
    else:
        alpha_prev = 1.0 if bool(set_alpha_to_one) else float(alphas_cumprod[0])
    return alpha_t, alpha_prev


def ddim_clean_sample_std_scale(alpha_bar: float) -> float:
    """Std multiplier for clean candidates centered at y_t / sqrt(alpha_bar_t)."""
    alpha_bar = max(float(alpha_bar), 1e-12)
    return math.sqrt(max(1.0 - alpha_bar, 0.0) / alpha_bar)
