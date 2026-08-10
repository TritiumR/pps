from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .action_space import decode_model_action_chunks, rebase_model_action_chunk
from .costs import PriorityStateCost
from .costs_capsule_flow import CapsuleFlowStateCost
from .costs_explore import ExploreStateCost
from .costs_grasp_flow import GraspFlowStateCost
from .costs_grasp_flow_ex import GraspFlowStateCost as ExtendedGraspFlowStateCost
from .costs_grasp_flow_fake import GraspFlowStateCost as FakeGraspFlowStateCost
from .costs_grasp_flow_loose import GraspFlowStateCost as LooseGraspFlowStateCost
from .costs_ref_style import RefStyleStateCost
from .ddim import ddim_clean_sample_std_scale, ddim_iteration_alphas
from .dial_sampler import DIALSampler, DIALSamplerConfig
from .fk import PandaFK, quat_mul_wxyz, transform_points_wxyz
from .truncated_sampler import sample_truncated_model_action_chunks


def task_tilt_penalty(
    samples: torch.Tensor,
    target: torch.Tensor,
    weight: float,
    dims: int | None = None,
    dim_weights=None,
) -> torch.Tensor:
    """Per-candidate Gaussian tilt toward a clean-action target chunk.

    Adds weight * sum(err^2) to the sampler cost, so the softmax becomes an FK/SVDD-style
    reweighting of the base's own proposals rather than a score-space addend. The SUM over chunk
    coordinates, not the mean, is the true Gaussian log-density scale.

    dims restricts the tilt to the first N coordinates: the near-binary gripper coordinate
    otherwise dominates the distance at high lam. dim_weights scales each coordinate instead,
    and overrides dims when given.
    """
    horizon = min(samples.shape[1], target.shape[0])   # --kp appends rows the target lacks
    if dim_weights is not None:
        w = torch.as_tensor(dim_weights, device=samples.device, dtype=samples.dtype)
        n_dims = min(w.shape[0], samples.shape[2])
        aligned = target.to(device=samples.device, dtype=samples.dtype)[:horizon, :n_dims]
        err = samples[:, :horizon, :n_dims] - aligned.unsqueeze(0)
        return weight * (err.pow(2) * w[:n_dims].view(1, 1, -1)).sum(dim=(1, 2))
    n_dims = samples.shape[2] if dims is None else min(int(dims), samples.shape[2])
    aligned = target.to(device=samples.device, dtype=samples.dtype)[:horizon, :n_dims]
    err = samples[:, :horizon, :n_dims] - aligned.unsqueeze(0)
    return weight * err.pow(2).sum(dim=(1, 2))


def task_tilt_weight(
    lam: float,
    temperature: float,
    noise: float,
    alpha_bar: float,
    eps: float = 1e-6,
) -> float:
    """SNR-tempered tilt weight: lam * T * abar / (2 * noise^2 * beta).

    The abar factor mutes the tilt at high noise, where the proxy's implied
    clean target carries 1/sqrt(abar)-amplified error; near the end of the
    denoise trajectory the weight grows like 1/beta (DAS-style late ramp).
    """
    beta = max(1.0 - float(alpha_bar), eps)
    sigma_sq = float(noise) ** 2 * beta
    return float(lam) * float(temperature) * float(alpha_bar) / max(2.0 * sigma_sq, eps)


@dataclass(frozen=True)
class SimFreeMPCConfig:
    task_name: str = "auto"
    num_samples: int = 64
    iterations: int = 2
    noise: float = 0.35
    temperature: float = 0.15
    beta_opt_iter: float = 1.0
    beta_horizon: float = 1.0
    action_dims: int = 8
    flow_eps: float = 1e-3
    joint_delta_clip: float = 0.25
    # Cost candidates as they would actually execute (joint limits + per-step delta
    # clamp) instead of pre-clamp. Off by default: it changes the optimisation
    # landscape, so arms with and without it are not comparable.
    cost_executable_actions: bool = False
    ddim_num_train_timesteps: int = 100
    interpolate: bool = False
    control_frequency: float = 40.0
    interpolate_frequency: float = 5.0
    interpolation_method: str = "bspline"
    cost_style: str = "priority"
    optimize_space: str = "action"
    sampler: str = "base"
    grad_calc: str = "mbd"
    logit_norm: str = "raw"  # "raw" | "std" (DIAL-style scale-free softmax logits)
    # Restrict the softmax to candidates within this many temperature units of the best before the
    # weighted mean, so averaging happens INSIDE a mode rather than across two. inf = old behaviour.
    mode_window: float = float("inf")
    # "draw" makes the sampler return a categorical draw from the candidate weights instead of
    # their mean -- see DIALSamplerConfig.estimator. Default "mean" reproduces MBD exactly.
    estimator: str = "mean"
    draw_below: float = float("inf")
    # Score the plan the sampler RETURNS, not the population it drew: sum_i w_i * term(sample_i)
    # equals term(sum_i w_i * sample_i) only for a linear cost, and ours is not. Measured gap on a
    # departure: 2.5e-6 across the population against 0.06 on the executed path. Off by default.
    eval_mean_plan: bool = False
    ancestral_eta: float = 0.0  # >0: marginal re-noising between denoise levels
    # What the softmax ranks on. "total" is the plain sum (deployed); "roles" splits it into
    # feasibility + task + prior_weight * prior. The prior exists to keep a weak sampler out of
    # trouble, not to judge a trajectory, yet near contact it charges a demonstration +12..22
    # against a ~1.5 task separation, so it decides selection alone. prior_weight=1.0 == "total".
    rank_mode: str = "total"          # "total" | "roles"
    # prior_weight applies at LOW noise, where candidates are plausible and the prior mostly blocks
    # demo-shaped motion; prior_weight_high at HIGH noise, where candidates are decoded noise and the
    # prior is the only thing rejecting them. A constant is wrong at both ends in opposite
    # directions, and alpha is the natural interpolant: the fraction of clean signal in x_t.
    prior_weight: float = 1.0
    prior_weight_high: float = 1.0
    prior_weight_schedule: str = "flat"   # "flat" (constant prior_weight) | "alpha"
    # Optional hard gate: candidates whose feasibility exceeds (best feasibility + this margin) are
    # excluded from the softmax. 0 disables it, leaving feasibility to compete on magnitude as today.
    # If nothing passes the gate the whole population is kept, so the sampler can never be starved.
    feasibility_gate: float = 0.0


class SimFreeMPC:
    """FK/cost-only MPC used as a geometric steering term for PPS."""

    def __init__(self, policy: Any, config: SimFreeMPCConfig):
        self.policy = policy
        self.config = config
        self.fk = PandaFK()
        self._inject_share_trace: list[float] = []   # per-denoise-level; see begin_inference
        self.sampler = DIALSampler(
            DIALSamplerConfig(
                num_samples=config.num_samples,
                iterations=config.iterations,
                noise=config.noise,
                temperature=config.temperature,
                beta_opt_iter=config.beta_opt_iter,
                beta_horizon=config.beta_horizon,
                action_dims=config.action_dims,
                logit_norm=config.logit_norm,
                mode_window=config.mode_window,
                estimator=config.estimator,
                draw_below=config.draw_below,
            )
        )
        self._warm_action: torch.Tensor | None = None
        self._warm_state: torch.Tensor | None = None
        if config.cost_style == "ref_style":
            self.cost = RefStyleStateCost(config.task_name)
        elif config.cost_style == "explore":
            self.cost = ExploreStateCost(config.task_name)
        elif config.cost_style == "grasp_flow":
            self.cost = GraspFlowStateCost(config.task_name)
        elif config.cost_style == "grasp_flow_ex":
            self.cost = ExtendedGraspFlowStateCost(config.task_name)
        elif config.cost_style == "grasp_flow_fake":
            self.cost = FakeGraspFlowStateCost(config.task_name)
        elif config.cost_style == "grasp_flow_loose":
            self.cost = LooseGraspFlowStateCost(config.task_name)
        elif config.cost_style == "capsule_flow":
            self.cost = CapsuleFlowStateCost(config.task_name)
        elif config.cost_style == "priority":
            self.cost = PriorityStateCost(config.task_name)
        else:
            raise ValueError(f"Unknown sim-free MPC cost_style: {config.cost_style!r}")
        if config.optimize_space not in ("action", "accel"):
            raise ValueError(f"Unknown sim-free MPC optimize_space: {config.optimize_space!r}")
        if config.sampler not in ("base", "truncated"):
            raise ValueError(f"Unknown MPC sampler: {config.sampler!r}")
        if config.grad_calc not in ("mbd", "backprop"):
            raise ValueError(f"Unknown MPC gradient calculation method: {config.grad_calc!r}")
        if config.grad_calc == "backprop" and config.optimize_space != "action":
            raise ValueError("Backprop cost gradients only support action optimize_space.")
        if config.sampler == "truncated":
            if config.optimize_space != "action":
                raise ValueError(
                    "The truncated sampler only supports action optimize_space."
                )
            if config.joint_delta_clip <= 0.0:
                raise ValueError(
                    "The truncated sampler requires a positive joint_delta_clip."
                )
        if config.interpolation_method not in ("bspline", "linear"):
            raise ValueError(
                "Unknown sim-free MPC interpolation_method: "
                f"{config.interpolation_method!r}"
            )
        if config.interpolate and (
            config.control_frequency <= 0.0
            or config.interpolate_frequency <= 0.0
        ):
            raise ValueError("Interpolation frequencies must both be positive.")

    def reset_action_warm(self) -> None:
        self._warm_action = None
        self._warm_state = None

    def _add_inject_trace_diagnostics(self, diagnostics: dict) -> None:
        """Summarise the per-level injection shares of this denoise chain.

        `inject_weight_share` alone is the LAST level's value, which the frontload schedule and the
        Tweedie collapse (x0 -> x_t as alpha -> 1) both drive to ~0 no matter what injection did.
        The max and the first-level value are the honest reads: the first three levels carry 78% of
        the injection authority.
        """
        trace = getattr(self, "_inject_share_trace", None)
        if not trace:
            return
        diagnostics["inject_share_first"] = float(trace[0])
        diagnostics["inject_share_max"] = float(max(trace))
        diagnostics["inject_share_mean"] = float(sum(trace) / len(trace))
        diagnostics["inject_share_levels"] = int(len(trace))

    def begin_inference(self) -> None:
        """Start a fresh denoise chain: clear per-level traces the diagnostics summarise."""
        self._inject_share_trace = []

    def reset_episode(self) -> None:
        """Reset planner state that must not leak across environment episodes."""
        self.begin_inference()
        self.reset_action_warm()
        reset_cost = getattr(self.cost, "reset", None)
        if callable(reset_cost):
            reset_cost()

    def set_warm_action(
        self,
        action: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> None:
        if action.ndim != 3:
            raise ValueError(f"Expected warm action [B,H,D], got {tuple(action.shape)}")
        self._warm_action = action.detach().clone()
        self._warm_state = None if state is None else torch.as_tensor(state).detach().clone()

    def warm_start_noise(
        self,
        fallback_noise: torch.Tensor,
        *,
        shift_steps: int,
        current_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool]:
        """Shift and rebase the previous action as the next initial noise."""
        warm_action = self._warm_action
        if warm_action is None or warm_action.shape != fallback_noise.shape:
            return fallback_noise, False

        warm_action = warm_action.to(device=fallback_noise.device, dtype=fallback_noise.dtype)
        horizon = warm_action.shape[1]
        shift = min(max(int(shift_steps), 0), horizon)
        if shift == 0:
            shifted_action = warm_action.clone()
        elif shift == horizon:
            shifted_action = warm_action[:, -1:, :].expand_as(warm_action).clone()
        else:
            tail = warm_action[:, -1:, :].expand(-1, shift, -1)
            shifted_action = torch.cat((warm_action[:, shift:, :], tail), dim=1)

        warm_state = self._warm_state
        if warm_state is not None and current_state is not None:
            shifted_action = rebase_model_action_chunk(
                self.policy,
                shifted_action,
                previous_state=warm_state,
                current_state=current_state,
            )
        return shifted_action, True

    def _interpolation_knot_count(self, horizon: int) -> int:
        if not self.config.interpolate or horizon <= 1:
            return horizon
        ratio = self.config.interpolate_frequency / max(self.config.control_frequency, self.config.flow_eps)
        knots = int(math.ceil(horizon * ratio))
        return max(2, min(horizon, knots))

    def _truncated_max_joint_deltas(
        self,
        proposal_horizon: int,
        output_horizon: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return knot bounds that preserve the 40 Hz per-step delta limit."""
        base_delta = float(self.config.joint_delta_clip)
        deltas = torch.full(
            (proposal_horizon,),
            base_delta,
            device=device,
            dtype=dtype,
        )
        if (
            not self.config.interpolate
            or proposal_horizon <= 1
            or proposal_horizon == output_horizon
        ):
            return deltas

        # y = mapping @ control_points; rewriting each control point as p_0 plus cumulative knot
        # deltas gives an exact gain from bounded knot deltas to adjacent output steps, for linear
        # and clamped-cubic alike, with no frequency-ratio heuristic.
        control_basis = torch.eye(proposal_horizon, device=device, dtype=dtype)
        mapping = self._interpolate_control_points(control_basis, output_horizon)
        cumulative_delta_coeffs = mapping[:, 1:].flip(1).cumsum(1).flip(1)
        step_gain = (
            cumulative_delta_coeffs[1:] - cumulative_delta_coeffs[:-1]
        ).abs().sum(dim=1).max()
        if not torch.isfinite(step_gain) or step_gain <= 0.0:
            raise ValueError("Interpolation produced an invalid truncated-sampler step gain.")

        # The first control point is the first executed action and must remain
        # within one high-frequency step of the current joint state. Later knot
        # deltas may be wider while the interpolated 40 Hz trajectory stays safe.
        deltas[1:] = base_delta / step_gain
        return deltas

    def _configure_sampler_proposal(
        self,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        output_horizon: int,
    ) -> None:
        if self.config.sampler == "base":
            inject = (context or {}).get("inject")
            rho = float((inject or {}).get("rho", 0.0))
            if inject is not None and rho > 0.0 and inject.get("x0") is not None:
                self.sampler.proposal_fn = self._make_inject_proposal(inject["x0"], rho)
            else:
                self.sampler.proposal_fn = None
                self._last_inject_slice = None
            return

        current_joint_pos = context.get("joint_pos")
        if current_joint_pos is None:
            raise ValueError(
                "The truncated sampler requires context['joint_pos']."
            )

        def proposal_fn(mean, noise_scale, num_samples, generator):
            max_joint_delta = self._truncated_max_joint_deltas(
                mean.shape[0],
                output_horizon,
                device=mean.device,
                dtype=mean.dtype,
            )
            return sample_truncated_model_action_chunks(
                self.policy,
                policy_inputs,
                mean,
                noise_scale,
                num_samples,
                current_joint_pos=current_joint_pos,
                max_joint_delta=max_joint_delta,
                generator=generator,
            )

        self.sampler.proposal_fn = proposal_fn

    def _make_inject_proposal(self, x0_expert: torch.Tensor, rho: float):
        """Mixture proposal: a fraction rho of candidates centred on the expert's clean action.

        The expert extends the candidate SUPPORT; the cost still weights every candidate, so an
        implausible proposal simply loses the softmax. Row 0 stays the current mean.

        Returns the per-sample log importance ratio alongside the samples. Without it the estimator's
        target silently becomes exp(-J)*q_mix, i.e. rho would grant the expert region prior mass the base
        never assigned it. Both components share the scale, so the ratio collapses to
        -log[(1-rho) + rho*exp(d_b - d_c)] with no normalizing constants.
        """
        expert = torch.as_tensor(x0_expert).detach()
        if expert.ndim == 3:
            expert = expert[0]

        def proposal_fn(mean, noise_scale, num_samples, generator):
            scale = torch.as_tensor(noise_scale, device=mean.device, dtype=mean.dtype)
            if scale.ndim == 0:
                scale_view = scale.view(1, 1, 1)
            elif scale.ndim == 1:
                scale_view = scale.view(1, mean.shape[0], 1)
            else:
                scale_view = scale.unsqueeze(0)
            noise = torch.randn(
                (num_samples, *mean.shape),
                device=mean.device,
                dtype=mean.dtype,
                generator=generator,
            )
            samples = mean.unsqueeze(0) + noise * scale_view
            samples[0] = mean
            n_inj = min(max(int(round(rho * num_samples)), 0), num_samples - 1)
            if n_inj == 0:
                self._last_inject_slice = None
                return samples, None
            centre = self._control_point_resample(
                expert.to(device=mean.device, dtype=mean.dtype)[:, : mean.shape[-1]],
                mean.shape[0],
            )
            lo = num_samples - n_inj
            samples[lo:] = centre.unsqueeze(0) + noise[lo:] * scale_view
            self._last_inject_slice = (lo, num_samples)

            # Importance correction, evaluated for EVERY row: a Gaussian-branch sample can also land
            # near the expert centre, so the ratio is a property of the point, not of which branch
            # drew it. rho_eff is the realized share after integer rounding.
            rho_eff = n_inj / float(num_samples)
            inv_two_var = 1.0 / (2.0 * torch.clamp(scale_view, min=1e-6) ** 2)
            d_base = ((samples - mean.unsqueeze(0)) ** 2 * inv_two_var).flatten(1).sum(-1)
            d_expert = ((samples - centre.unsqueeze(0)) ** 2 * inv_two_var).flatten(1).sum(-1)
            log_importance = -torch.logaddexp(
                torch.log(torch.as_tensor(1.0 - rho_eff, device=mean.device, dtype=mean.dtype)
                          .clamp_min(1e-30)),
                torch.log(torch.as_tensor(rho_eff, device=mean.device, dtype=mean.dtype)) + d_base - d_expert,
            )
            return samples, log_importance

        return proposal_fn

    @staticmethod
    def _linear_resample(sequence: torch.Tensor, output_horizon: int) -> torch.Tensor:
        if sequence.shape[-2] == output_horizon:
            return sequence
        if sequence.shape[-2] == 1:
            return sequence.expand(*sequence.shape[:-2], output_horizon, sequence.shape[-1])

        original_ndim = sequence.ndim
        if original_ndim == 2:
            sequence = sequence.unsqueeze(0)
        if sequence.ndim != 3:
            raise ValueError(f"Expected sequence [H,D] or [B,H,D], got {tuple(sequence.shape)}")

        resampled = F.interpolate(
            sequence.transpose(1, 2),
            size=output_horizon,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        if original_ndim == 2:
            return resampled[0]
        return resampled

    @staticmethod
    def _bspline_basis(
        num_control_points: int,
        output_horizon: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        degree: int = 3,
    ) -> torch.Tensor:
        if num_control_points <= 0 or output_horizon <= 0:
            raise ValueError("B-spline interpolation expects positive horizon sizes.")
        if num_control_points == 1:
            return torch.ones(output_horizon, 1, device=device, dtype=dtype)

        degree = min(degree, num_control_points - 1)
        interior_count = num_control_points - degree - 1
        start = torch.zeros(degree + 1, device=device, dtype=dtype)
        end = torch.ones(degree + 1, device=device, dtype=dtype)
        if interior_count > 0:
            interior = torch.linspace(0.0, 1.0, interior_count + 2, device=device, dtype=dtype)[1:-1]
            knots = torch.cat([start, interior, end])
        else:
            knots = torch.cat([start, end])

        u = torch.linspace(0.0, 1.0, output_horizon, device=device, dtype=dtype)
        basis = torch.stack(
            [((u >= knots[i]) & (u < knots[i + 1])).to(dtype) for i in range(knots.numel() - 1)],
            dim=1,
        )
        for level in range(1, degree + 1):
            cols = basis.shape[1] - 1
            next_basis = []
            for i in range(cols):
                left_den = knots[i + level] - knots[i]
                if torch.abs(left_den) > 0:
                    left = (u - knots[i]) / left_den * basis[:, i]
                else:
                    left = torch.zeros_like(u)

                right_den = knots[i + level + 1] - knots[i + 1]
                if torch.abs(right_den) > 0:
                    right = (knots[i + level + 1] - u) / right_den * basis[:, i + 1]
                else:
                    right = torch.zeros_like(u)
                next_basis.append(left + right)
            basis = torch.stack(next_basis, dim=1)

        terminal = u == 1.0
        if terminal.any():
            basis[terminal] = 0.0
            basis[terminal, -1] = 1.0
        row_sum = torch.clamp(basis.sum(dim=1, keepdim=True), min=torch.finfo(dtype).eps)
        return basis / row_sum

    @classmethod
    def _bspline_resample(cls, sequence: torch.Tensor, output_horizon: int) -> torch.Tensor:
        if sequence.shape[-2] == output_horizon:
            return sequence
        if sequence.shape[-2] == 1:
            return sequence.expand(*sequence.shape[:-2], output_horizon, sequence.shape[-1])

        original_ndim = sequence.ndim
        if original_ndim == 2:
            sequence = sequence.unsqueeze(0)
        if sequence.ndim != 3:
            raise ValueError(f"Expected sequence [H,D] or [B,H,D], got {tuple(sequence.shape)}")

        basis = cls._bspline_basis(
            sequence.shape[1],
            output_horizon,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        resampled = torch.einsum("oh,bhd->bod", basis, sequence)
        if original_ndim == 2:
            return resampled[0]
        return resampled

    def _linear_control_point_resample(
        self,
        sequence: torch.Tensor,
        output_horizon: int,
    ) -> torch.Tensor:
        if sequence.shape[-2] == output_horizon:
            return sequence
        if sequence.shape[-2] == 1:
            return sequence.expand(*sequence.shape[:-2], output_horizon, sequence.shape[-1])

        knot_steps = max(
            1,
            int(round(self.config.control_frequency / self.config.interpolate_frequency)),
        )
        knot_position = torch.arange(
            output_horizon,
            device=sequence.device,
            dtype=sequence.dtype,
        ) / float(knot_steps)
        lower = torch.floor(knot_position).to(torch.long)
        lower = lower.clamp(0, sequence.shape[-2] - 1)
        upper = (lower + 1).clamp(0, sequence.shape[-2] - 1)
        alpha = knot_position - lower.to(sequence.dtype)
        return (
            (1.0 - alpha)[..., None] * sequence[..., lower, :]
            + alpha[..., None] * sequence[..., upper, :]
        )

    def _control_point_resample(self, sequence: torch.Tensor, output_horizon: int) -> torch.Tensor:
        if (
            self.config.interpolate
            and self.config.interpolation_method == "linear"
            and sequence.shape[-2] != output_horizon
        ):
            knot_steps = max(
                1,
                int(round(self.config.control_frequency / self.config.interpolate_frequency)),
            )
            indices = torch.arange(output_horizon, device=sequence.device) * knot_steps
            indices = indices.clamp(max=sequence.shape[-2] - 1)
            return sequence[..., indices, :]
        return self._linear_resample(sequence, output_horizon)

    def _interpolate_control_points(self, sequence: torch.Tensor, output_horizon: int) -> torch.Tensor:
        if not self.config.interpolate:
            return self._linear_resample(sequence, output_horizon)
        if self.config.interpolation_method == "linear":
            return self._linear_control_point_resample(sequence, output_horizon)
        if self.config.interpolation_method == "bspline":
            return self._bspline_resample(sequence, output_horizon)
        raise ValueError(
            "Unknown sim-free MPC interpolation_method: "
            f"{self.config.interpolation_method!r}"
        )

    @staticmethod
    def _trajectory_to_accel_code(sequence: torch.Tensor) -> torch.Tensor:
        """Encode a trajectory as a same-shaped acceleration-space code.

        The code is a linear, invertible coordinate transform:
        code[0] is the first position, code[1] is the first velocity, and
        code[2:] are second differences. MBD/DDIM updates must happen in this
        code space if sampling is done in acceleration space; updating the
        original action trajectory with an acceleration-space weighted mean
        mixes coordinates and gives inconsistent reverse steps.
        """
        if sequence.ndim != 2:
            raise ValueError(f"Expected sequence [H,D], got {tuple(sequence.shape)}")
        code = torch.zeros_like(sequence)
        code[0] = sequence[0]
        if sequence.shape[0] > 1:
            code[1] = sequence[1] - sequence[0]
        if sequence.shape[0] > 2:
            code[2:] = sequence[2:] - 2.0 * sequence[1:-1] + sequence[:-2]
        return code

    @staticmethod
    def _accel_code_to_trajectory(code: torch.Tensor) -> torch.Tensor:
        if code.ndim != 3:
            raise ValueError(f"Expected accel code [K,H,D], got {tuple(code.shape)}")
        horizon = code.shape[1]
        if horizon == 0:
            return code
        traj = [code[:, 0, :]]
        if horizon > 1:
            traj.append(traj[0] + code[:, 1, :])
        for step in range(2, horizon):
            traj.append(2.0 * traj[-1] - traj[-2] + code[:, step, :])
        return torch.stack(traj, dim=1)

    def _optimize_chunk(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
    ):
        if x_t.shape[0] != 1:
            raise ValueError("SimFreeMPC MVP currently expects batch size 1.")

        active_dims = min(self.config.action_dims, x_t.shape[-1])
        horizon = x_t.shape[1]
        opt_horizon = self._interpolation_knot_count(horizon)
        mean0 = self._control_point_resample(x_t[0, :, :active_dims].detach(), opt_horizon)
        self._configure_sampler_proposal(
            policy_inputs,
            context,
            output_horizon=horizon,
        )

        def cost_from_positions(samples: torch.Tensor) -> torch.Tensor:
            full_horizon_samples = self._interpolate_control_points(samples, horizon)
            return self._cost_active_samples(full_horizon_samples, x_t, active_dims, policy_inputs, context)

        if self.config.optimize_space == "accel":
            mean_accel = self._trajectory_to_accel_code(mean0)

            def cost_fn(samples: torch.Tensor) -> torch.Tensor:
                positions = self._accel_code_to_trajectory(samples)
                return cost_from_positions(positions)

            result = self.sampler.optimize(mean_accel, cost_fn)
            result_mean = self._accel_code_to_trajectory(result.mean.unsqueeze(0))[0]
        else:
            result = self.sampler.optimize(mean0, cost_from_positions)
            result_mean = result.mean
        target = x_t.detach().clone()
        target[:, :, :active_dims] = self._interpolate_control_points(result_mean, horizon).unsqueeze(0)
        return target, result, active_dims

    def _cost_active_samples(
        self,
        samples: torch.Tensor,
        x_template: torch.Tensor,
        active_dims: int,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
    ) -> torch.Tensor:
        full = x_template.detach().repeat(samples.shape[0], 1, 1)
        full[:, :, :active_dims] = samples
        # Candidates are costed WITHOUT the execution clamp by default, so a plan can be scored
        # on motion that is truncated before env.step -- measured: executed within-chunk joint
        # deltas sit at the clamp value on the median step, so the clamp is saturating and the
        # scored trajectory routinely is not the executed one. cost_executable_actions=True
        # scores exactly what would be executed. Opt-in: it changes the optimisation landscape,
        # so runs with and without it are not comparable.
        if self.config.cost_executable_actions and self.config.joint_delta_clip > 0.0:
            decoded = decode_model_action_chunks(
                self.policy,
                policy_inputs,
                full,
                apply_clamp=True,
                current_joint_pos=context.get("joint_pos"),
                max_joint_delta=self.config.joint_delta_clip,
            )
        else:
            decoded = decode_model_action_chunks(
                self.policy,
                policy_inputs,
                full,
                apply_clamp=False,
            )
        real = decoded.real_actions
        joints = real[..., :7]
        fk = self.fk.forward(joints)
        ee_pos = fk.ee_pos
        ee_quat = fk.ee_quat
        root_pos = context.get("robot_root_pos")
        root_quat = context.get("robot_root_quat")
        if root_pos is not None and root_quat is not None:
            if not torch.is_tensor(root_pos):
                root_pos = torch.as_tensor(root_pos)
            if not torch.is_tensor(root_quat):
                root_quat = torch.as_tensor(root_quat)
            ee_pos = transform_points_wxyz(
                root_pos.to(device=ee_pos.device, dtype=ee_pos.dtype),
                root_quat.to(device=ee_pos.device, dtype=ee_pos.dtype),
                ee_pos,
            )
            ee_quat = quat_mul_wxyz(
                root_quat.to(device=ee_quat.device, dtype=ee_quat.dtype),
                ee_quat,
            )
        if self.config.cost_style in (
            "ref_style",
            "explore",
            "grasp_flow",
            "grasp_flow_ex",
            "grasp_flow_fake",
            "grasp_flow_loose",
            "capsule_flow",
        ):
            total = self.cost(real_actions=real, tcp_pos=ee_pos, tcp_quat=ee_quat, context=context)
        else:
            total = self.cost(real_actions=real, ee_pos=ee_pos, ee_quat=ee_quat, context=context)
        return self._ranking_scalar(total)

    def _prior_weight_now(self) -> float:
        """Prior weight for the level being ranked.

        "flat" is the constant prior_weight (and reproduces rank_mode="total" at 1.0). "alpha"
        interpolates prior_weight_high (high noise, candidates are garbage, keep the filter) to
        prior_weight (low noise, candidates are plausible, the prior only blocks) linearly in
        alpha_bar. Without a recorded alpha it falls back to flat rather than guessing.
        """
        low = float(self.config.prior_weight)
        if self.config.prior_weight_schedule != "alpha":
            return low
        alpha = getattr(self, "_rank_alpha", None)
        if alpha is None:
            return low
        high = float(self.config.prior_weight_high)
        a = min(max(float(alpha), 0.0), 1.0)
        return high * (1.0 - a) + low * a

    def _ranking_scalar(self, total: torch.Tensor) -> torch.Tensor:
        """What the softmax orders candidates by. `total` (the true cost) is preserved for logging.

        Under rank_mode="roles" the execution/search prior is down-weighted so it shapes rather than
        decides, and an optional hard gate drops candidates that are physically far worse than the
        best available. Only a cost exposing the TERM_ROLES split can do this; anything else falls
        back to the plain total rather than silently ranking on a partial sum.
        """
        self._last_cost_total = total.detach()
        if self.config.rank_mode == "total":
            return total
        parts = [getattr(self.cost, f"last_cost_{r}", None) for r in ("feasibility", "task", "prior")]
        if any(p is None or p.shape != total.shape for p in parts):
            return total
        feasibility, task, prior = (p.to(device=total.device, dtype=total.dtype) for p in parts)
        ranked = feasibility + task + self._prior_weight_now() * prior
        gate = float(self.config.feasibility_gate)
        if gate > 0.0:
            keep = feasibility <= feasibility.min() + gate
            if bool(keep.any()):        # never starve the sampler: an all-fail gate keeps everyone
                # A large FINITE penalty, not inf: the diagnostics form sum(cost * weight), and
                # inf * 0 is NaN, so an excluded candidate would poison cost_weighted.
                excluded = ranked.max().detach() + 1.0e4
                ranked = torch.where(keep, ranked, excluded.expand_as(ranked))
        return ranked

    def _backprop_clean_score(
        self,
        result,
        x_t: torch.Tensor,
        active_dims: int,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        alpha_bar: float,
    ) -> torch.Tensor:
        """Estimate the annealed score from weighted clean-sample cost gradients."""
        if self.config.optimize_space != "action":
            raise ValueError("Backprop cost gradients only support action optimize_space.")
        if self.config.temperature <= 0.0:
            raise ValueError("Backprop cost gradients require a positive temperature.")

        horizon = x_t.shape[1]
        with torch.enable_grad():
            clean_samples = self._interpolate_control_points(
                result.samples.detach(),
                horizon,
            )
            clean_samples = clean_samples.detach().requires_grad_(True)
            costs = self._cost_active_samples(
                clean_samples,
                x_t,
                active_dims,
                policy_inputs,
                context,
            )
            weights = result.weights.detach().to(device=costs.device, dtype=costs.dtype)
            weighted_cost = torch.sum(weights * costs)
            (sample_cost_grads,) = torch.autograd.grad(
                weighted_cost,
                clean_samples,
                create_graph=False,
                retain_graph=False,
            )

        mean_clean_cost_grad = sample_cost_grads.sum(dim=0, keepdim=True)
        sqrt_alpha = torch.as_tensor(
            alpha_bar,
            device=x_t.device,
            dtype=x_t.dtype,
        ).sqrt()
        score = torch.zeros_like(x_t)
        score[:, :, :active_dims] = (
            -mean_clean_cost_grad
            / (
                float(self.config.temperature)
                * torch.clamp(sqrt_alpha, min=self.config.flow_eps)
            )
        ).detach()
        return score

    def _score_returned_plan(self, result, cost_fn) -> dict[str, float]:
        """Cost of the plan the sampler RETURNS, beside the population statistics already logged.

        terms_best and terms_weighted are both statistics of the sampled population -- the value at
        the argmin candidate and the softmax-weighted mean of the values. Neither is the cost of
        sum_i w_i * sample_i, which is what actually executes, and for a non-convex term the two can
        disagree without limit: candidates passing either side of a seated object each score zero on
        a footprint keepout while their mean drives straight through it. Reading best-vs-weighted
        cannot see that, because it never evaluates the mean.

        Returned keys are prefixed ``mean_plan_``; ``mean_plan_cost_excess`` is the headline --
        cost(mean) - sum_i w_i cost(sample_i), positive whenever the executed plan is worse than the
        population it was averaged from.
        """
        out: dict[str, float] = {}
        j = cost_fn(result.mean.unsqueeze(0))
        out["mean_plan_cost"] = float(j.reshape(-1)[0])
        weighted = float(torch.sum(result.costs * result.weights).detach().cpu())
        out["mean_plan_cost_population_weighted"] = weighted
        out["mean_plan_cost_min"] = float(result.costs.min().detach().cpu())
        out["mean_plan_cost_excess"] = out["mean_plan_cost"] - weighted
        terms = getattr(self.cost, "last_terms", None)
        if terms:
            for name, values in terms.items():
                flat = values.reshape(-1)
                if flat.numel() >= 1:
                    out[f"mean_plan_term_{name}"] = float(flat[0].detach().cpu())
        return out

    def _last_cost_term_diagnostics(self, result) -> dict[str, Any]:
        cost = getattr(self, "cost", None)
        terms = getattr(cost, "last_terms", None)
        if not terms:
            return {}
        best_idx = int(torch.argmin(result.costs).detach().cpu())
        diagnostics: dict[str, Any] = {}
        stage = getattr(cost, "last_stage", None)
        if stage is not None:
            diagnostics["cost_stage"] = stage
        for name, values in terms.items():
            if values.ndim != 1 or values.shape[0] != result.costs.shape[0]:
                continue
            values = values.to(device=result.weights.device, dtype=result.weights.dtype)
            diagnostics[f"term_{name}_best"] = float(values[best_idx].detach().cpu())
            diagnostics[f"term_{name}_weighted"] = float(torch.sum(values * result.weights).detach().cpu())
        # The same total, split so selection can gate on feasibility, rank on task progress and
        # only tie-break on the prior. Under rank_mode="roles" result.costs is the ranking scalar,
        # so the true total is logged beside it and stays comparable across arms.
        diagnostics["rank_mode"] = self.config.rank_mode
        if self.config.rank_mode != "total":
            diagnostics["prior_weight"] = self._prior_weight_now()
            diagnostics["prior_weight_schedule"] = self.config.prior_weight_schedule
            true_total = getattr(self, "_last_cost_total", None)
            if true_total is not None and true_total.shape == result.costs.shape:
                true_total = true_total.to(device=result.weights.device, dtype=result.weights.dtype)
                diagnostics["cost_true_total_best"] = float(true_total[best_idx].detach().cpu())
                diagnostics["cost_true_total_min"] = float(true_total.min().detach().cpu())
                diagnostics["cost_true_total_weighted"] = float(
                    torch.sum(true_total * result.weights).detach().cpu())
        for role in ("feasibility", "task", "prior"):
            values = getattr(cost, f"last_cost_{role}", None)
            if values is None or values.ndim != 1 or values.shape[0] != result.costs.shape[0]:
                continue
            values = values.to(device=result.weights.device, dtype=result.weights.dtype)
            diagnostics[f"cost_{role}_best"] = float(values[best_idx].detach().cpu())
            diagnostics[f"cost_{role}_min"] = float(values.min().detach().cpu())
            diagnostics[f"cost_{role}_weighted"] = float(
                torch.sum(values * result.weights).detach().cpu()
            )
        debug_values = getattr(cost, "last_debug", None)
        if debug_values:
            for name, values in debug_values.items():
                if values.ndim != 1 or values.shape[0] != result.costs.shape[0]:
                    continue
                values = values.to(device=result.weights.device, dtype=result.weights.dtype)
                diagnostics[f"debug_{name}_best"] = float(values[best_idx].detach().cpu())
                diagnostics[f"debug_{name}_weighted"] = float(torch.sum(values * result.weights).detach().cpu())
        return diagnostics

    def _optimize_ddim_clean_chunk(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        alpha_bar: float,
    ):
        if x_t.shape[0] != 1:
            raise ValueError("SimFreeMPC MVP currently expects batch size 1.")

        active_dims = min(self.config.action_dims, x_t.shape[-1])
        horizon = x_t.shape[1]
        opt_horizon = self._interpolation_knot_count(horizon)
        sqrt_alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype).sqrt()
        active_x_opt = self._control_point_resample(x_t.detach()[0, :, :active_dims], opt_horizon)
        self._configure_sampler_proposal(
            policy_inputs,
            context,
            output_horizon=horizon,
        )
        clean_std = self.config.noise * ddim_clean_sample_std_scale(alpha_bar)

        def cost_from_positions(samples: torch.Tensor) -> torch.Tensor:
            full_horizon_samples = self._interpolate_control_points(samples, horizon)
            return self._cost_active_samples(full_horizon_samples, x_t, active_dims, policy_inputs, context)

        if self.config.optimize_space == "accel":
            active_code = self._trajectory_to_accel_code(active_x_opt)
            mean_accel = active_code / torch.clamp(sqrt_alpha, min=self.config.flow_eps)

            def cost_fn(samples: torch.Tensor) -> torch.Tensor:
                positions = self._accel_code_to_trajectory(samples)
                return cost_from_positions(positions)

            result = self.sampler.optimize_with_noise_scale(mean_accel, cost_fn, noise_scale=clean_std)
            result_mean = self._accel_code_to_trajectory(result.mean.unsqueeze(0))[0]
        else:
            clean_center = active_x_opt / torch.clamp(sqrt_alpha, min=self.config.flow_eps)
            result = self.sampler.optimize_with_noise_scale(
                clean_center,
                cost_from_positions,
                noise_scale=clean_std,
            )
            result_mean = result.mean
        x0_hat = x_t.detach().clone()
        x0_hat[:, :, :active_dims] = self._interpolate_control_points(result_mean, horizon).unsqueeze(0)
        return x0_hat, result, active_dims, clean_std

    def _diagnostics(
        self,
        *,
        result,
        target: torch.Tensor,
        x_t: torch.Tensor,
        score: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        weights = result.weights.detach()
        weight_ess = torch.reciprocal(torch.clamp(torch.sum(weights.square()), min=1e-12))
        weight_entropy = -torch.sum(weights * torch.log(torch.clamp(weights, min=1e-12)))
        diagnostics = {
            "cost_min": float(result.costs.min().detach().cpu()),
            "cost_mean": float(result.costs.mean().detach().cpu()),
            "cost_std": float(result.costs.std(unbiased=False).detach().cpu()),
            "cost_weighted": float(torch.sum(result.costs * result.weights).detach().cpu()),
            "weight_max": float(weights.max().cpu()),
            "weight_ess": float(weight_ess.cpu()),
            "weight_entropy": float(weight_entropy.cpu()),
            "target_delta_norm": float(torch.linalg.vector_norm((target - x_t).detach()).cpu()),
            "interpolate": bool(self.config.interpolate),
            "cost_style": self.config.cost_style,
            "optimize_space": self.config.optimize_space,
            "sampler": self.config.sampler,
            "grad_calc": self.config.grad_calc,
        }
        if self.config.interpolate:
            diagnostics.update(
                {
                    "interpolate_horizon": int(x_t.shape[1]),
                    "interpolate_knot_count": int(self._interpolation_knot_count(x_t.shape[1])),
                    "interpolate_frequency": float(self.config.interpolate_frequency),
                    "control_frequency": float(self.config.control_frequency),
                    "interpolate_method": (
                        "clamped_cubic_bspline"
                        if self.config.interpolation_method == "bspline"
                        else "linear"
                    ),
                }
            )
        if score is not None:
            diagnostics.update(
                {
                    "score_norm": float(torch.linalg.vector_norm(score.detach()).cpu()),
                    "score_abs_mean": float(score.detach().abs().mean().cpu()),
                    "noise_scale_min": float(result.noise_scale.min().detach().cpu()),
                    "noise_scale_max": float(result.noise_scale.max().detach().cpu()),
                }
            )
        diagnostics.update(self._last_cost_term_diagnostics(result))
        # Here rather than per update-mode branch, which already missed mbd_score_action_prox once.
        # Consumed, not just read, so another mode cannot report a stale plan's cost as its own.
        mean_plan = getattr(self, "_last_mean_plan", None)
        self._last_mean_plan = None
        if mean_plan:
            diagnostics.update(mean_plan)
        return diagnostics

    def _optimize_action_prox_chunk(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        alpha_bar: float,
        final: bool = False,
    ):
        """Optimize clean-action candidates around the scaled noisy action."""
        if x_t.shape[0] != 1:
            raise ValueError("MBD action-space sampler currently expects batch size 1.")
        if self.config.optimize_space != "action":
            raise ValueError("MBD action-prox/warm paths only support action optimize_space.")

        active_dims = min(self.config.action_dims, x_t.shape[-1])
        horizon = x_t.shape[1]
        opt_horizon = self._interpolation_knot_count(horizon)
        sqrt_alpha = torch.as_tensor(
            alpha_bar,
            device=x_t.device,
            dtype=x_t.dtype,
        ).sqrt()
        resampled = self._control_point_resample(
            x_t.detach()[0, :, :active_dims],
            opt_horizon,
        )
        # Base sampler has no per-step truncation: the priority (vlm_dp) cost needs the unscaled
        # center, since scaled (z_t / sqrt(alpha_bar)) blows up at high noise. The truncated sampler
        # clamps decoded joint deltas, so it keeps the scaled center everywhere.
        if self.config.cost_style == "priority" and self.config.sampler == "base":
            proposal_center = resampled
        else:
            proposal_center = resampled / torch.clamp(sqrt_alpha, min=self.config.flow_eps)
        self._configure_sampler_proposal(
            policy_inputs,
            context,
            output_horizon=horizon,
        )
        proposal_std = float(self.config.noise) * math.sqrt(
            max(1.0 - float(alpha_bar), 0.0)
        )

        def cost_from_positions(
            samples: torch.Tensor,
            include_tilt: bool = True,
        ) -> torch.Tensor:
            full_horizon_samples = self._interpolate_control_points(samples, horizon)
            cost = self._cost_active_samples(
                full_horizon_samples,
                x_t,
                active_dims,
                policy_inputs,
                context,
            )
            tilt = context.get("task_tilt")
            if tilt is not None and include_tilt:
                lam = float(tilt["weight"]) * float(tilt.get("authority", 1.0))
                if lam > 0.0:
                    pen = task_tilt_penalty(
                        full_horizon_samples,
                        tilt["target"],
                        1.0,
                        dims=tilt.get("dims"),
                        dim_weights=tilt.get("dim_weights"),
                    )
                    flat = pen.reshape(-1)
                    if tilt.get("discrimination"):
                        # Implicit gate: expert weight only where the expert actually separates
                        # the candidate population (no information -> no authority).
                        qs = torch.quantile(flat, torch.tensor(
                            [0.1, 0.5, 0.9], device=flat.device, dtype=flat.dtype))
                        rel = float((qs[2] - qs[0]) / (qs[1].abs() + 1e-8))
                        lam = lam * (rel / (1.0 + rel))
                    ess_cap = tilt.get("ess_cap")
                    if ess_cap:
                        # Target EFFECTIVE SAMPLE SIZE, in candidates -- not a raw logit bound.
                        # For softmax weights whose logits have std sigma, ESS/N ~= exp(-sigma^2),
                        # so holding the tilt's OWN logit dispersion at sqrt(log(N/ess_cap)) stops
                        # it from pushing the population below ess_cap by itself. The previous form
                        # fed ess_cap in as the dispersion directly, so --tilt_ess_cap 64 asked for
                        # 64 nats: measured ESS 1.0 of 512 at both 64 and 128, i.e. the guard
                        # always collapsed the very population it was meant to protect.
                        std = float(flat.std())
                        n = int(flat.numel())
                        if std > 1e-9 and n > float(ess_cap) > 0.0:
                            sigma = math.sqrt(math.log(n / float(ess_cap)))
                            lam = min(lam, sigma * float(self.config.temperature) / std)
                    cost = cost + lam * pen
                    self._last_tilt_lambda = lam
            return cost

        self._last_tilt_lambda = None
        self._last_inject_slice = None
        self._last_inject_share = None
        result = self.sampler.optimize_with_noise_scale(
            proposal_center,
            cost_from_positions,
            noise_scale=proposal_std,
        )
        if self._last_inject_slice is not None and result.weights is not None:
            # Share of softmax weight won by expert candidates: the direct read on whether injection
            # contributes at this noise level (and hence on the rho schedule).
            lo, hi = self._last_inject_slice
            self._last_inject_share = float(result.weights[lo:hi].sum())
            # Runs per denoise level, but diagnostics emit once per inference, so a bare value
            # reports only the last level -- where rho is ~1e-4 and reads ~0 whatever injection did.
            # Keep the trace: the early levels carry 78% of the injection authority.
            self._inject_share_trace.append(self._last_inject_share)
        self._last_tilt_base_cost = None
        self._last_mean_plan = None
        if context.get("task_tilt") is not None or self.config.eval_mean_plan:
            # Both re-enter the cost on a batch of one, overwriting cost.last_terms, after which
            # the per-term diagnostics silently drop every row. Snapshot first.
            population_terms = getattr(self.cost, "last_terms", None)
            with torch.no_grad():
                if context.get("task_tilt") is not None:
                    # Thin-product diagnostic: the UNTILTED cost of the tilted winner.
                    base_j = cost_from_positions(result.mean.unsqueeze(0), include_tilt=False)
                    self._last_tilt_base_cost = float(base_j.reshape(-1)[0])
                if self.config.eval_mean_plan:
                    self._last_mean_plan = self._score_returned_plan(result, cost_from_positions)
            if population_terms is not None:
                self.cost.last_terms = population_terms
        # DDRS: tilt the FINAL clean cloud's own weights by a learned log-ratio, then re-reduce.
        # Only here, and only on the last level: this is the one population whose distribution
        # matches the negatives the ratio was fit against. `self.ddrs` is None unless a caller
        # attaches one, so every existing run takes the untouched path above.
        self._last_ddrs = None
        if final and getattr(self, "ddrs", None) is not None and result.weights is not None:
            from .ddrs import ess as _ess, tilt_weights
            with torch.no_grad():
                real = self._decode_samples_for_ddrs(
                    result.samples, x_t, active_dims, horizon, policy_inputs, context)
                r = self.ddrs.log_ratio(real, context, context.get("joint_pos"))
                w_new = tilt_weights(result.weights, r, self.ddrs.gamma)
                mean_new = (w_new.view(-1, *([1] * (result.samples.ndim - 1)))
                            * result.samples).sum(dim=0)
                self._last_ddrs = {
                    "gamma": float(self.ddrs.gamma),
                    "r_mean": float(r.mean()), "r_std": float(r.std()),
                    "ess_before": _ess(result.weights), "ess_after": _ess(w_new),
                    "argmax_before": int(torch.argmax(result.weights)),
                    "argmax_after": int(torch.argmax(w_new)),
                    "mean_shift": float(torch.linalg.vector_norm(mean_new - result.mean)),
                    "rank_of_base_choice": int(
                        (r > r[int(torch.argmax(result.weights))]).sum()),
                }
                result = dataclasses.replace(result, mean=mean_new, weights=w_new)
        x0_hat = x_t.detach().clone()
        x0_hat[:, :, :active_dims] = self._interpolate_control_points(
            result.mean,
            horizon,
        ).unsqueeze(0)
        return x0_hat, result, active_dims, proposal_std

    def _decode_samples_for_ddrs(self, samples, x_template, active_dims, horizon,
                                 policy_inputs, context):
        """Knot samples -> decoded REAL action chunks [K, horizon, D].

        The same two steps `_cost_active_samples` takes before it costs a candidate -- expand the
        control points, write them into the template, decode -- so the ratio scores exactly the
        trajectories the cost ranked, not a differently-decoded copy of them.
        """
        full = self._interpolate_control_points(samples, horizon)
        chunk = x_template.detach().repeat(full.shape[0], 1, 1)
        chunk[:, :, :active_dims] = full
        decoded = decode_model_action_chunks(self.policy, policy_inputs, chunk, apply_clamp=False)
        return decoded.real_actions.detach().cpu().numpy()

    def step(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        dt: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        target, result, _ = self._optimize_chunk(x_t, policy_inputs, context)

        dt_abs = torch.abs(torch.as_tensor(dt, device=x_t.device, dtype=x_t.dtype))
        flow = -(target - x_t.detach()) / torch.clamp(dt_abs, min=self.config.flow_eps)

        return flow, self._diagnostics(result=result, target=target, x_t=x_t)

    def step_score_space(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        step_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        target, result, active_dims = self._optimize_chunk(x_t, policy_inputs, context)
        variance = torch.clamp(
            self._interpolate_control_points(
                result.noise_scale.to(device=x_t.device, dtype=x_t.dtype).view(-1, 1),
                x_t.shape[1],
            )
            .squeeze(-1)
            .pow(2),
            min=self.config.flow_eps,
        )
        score = torch.zeros_like(x_t)
        score[:, :, :active_dims] = (
            target[:, :, :active_dims] - x_t.detach()[:, :, :active_dims]
        ) / variance.view(1, -1, 1)
        next_x = x_t.detach().clone()
        next_x[:, :, :active_dims] = (
            x_t.detach()[:, :, :active_dims]
            + step_scale * variance.view(1, -1, 1) * score[:, :, :active_dims]
        )
        diagnostics = self._diagnostics(result=result, target=target, x_t=x_t, score=score)
        diagnostics["update_mode"] = "score_space"
        return next_x, diagnostics

    def step_ddim(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
        step_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        self._rank_alpha = float(alpha_bar)      # read by _ranking_scalar for the prior schedule
        x0_hat, result, active_dims, clean_std = self._optimize_ddim_clean_chunk(
            x_t,
            policy_inputs,
            context,
            alpha_bar=alpha_bar,
        )

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))
        sqrt_beta = torch.sqrt(beta)

        score = torch.zeros_like(x_t)
        active_x = x_t.detach()[:, :, :active_dims]
        active_x0 = x0_hat[:, :, :active_dims]
        if self.config.grad_calc == "backprop":
            score = self._backprop_clean_score(
                result,
                x_t,
                active_dims,
                policy_inputs,
                context,
                alpha_bar=alpha_bar,
            )
            active_score = score[:, :, :active_dims]
            score_x0 = (active_x + beta * active_score) / sqrt_alpha
            pred_epsilon = -sqrt_beta * active_score
            active_prev = (
                torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * score_x0
                + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * pred_epsilon
            )
        elif self.config.optimize_space == "accel":
            horizon = x_t.shape[1]
            opt_horizon = self._interpolation_knot_count(horizon)
            active_x_opt = self._control_point_resample(active_x[0], opt_horizon)
            active_code = self._trajectory_to_accel_code(active_x_opt)
            clean_code = result.mean.to(device=x_t.device, dtype=x_t.dtype)
            code_score = (-active_code + sqrt_alpha * clean_code) / beta
            score[:, :, :active_dims] = self._interpolate_control_points(
                self._accel_code_to_trajectory(code_score.unsqueeze(0))[0],
                horizon,
            ).unsqueeze(0)
            pred_epsilon = (active_code - sqrt_alpha * clean_code) / sqrt_beta
            prev_code = (
                torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * clean_code
                + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * pred_epsilon
            )
            active_prev = self._interpolate_control_points(
                self._accel_code_to_trajectory(prev_code.unsqueeze(0))[0],
                horizon,
            ).unsqueeze(0)
        else:
            score[:, :, :active_dims] = (-active_x + sqrt_alpha * active_x0) / beta

            pred_epsilon = (active_x - sqrt_alpha * active_x0) / sqrt_beta
            active_prev = (
                torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * active_x0
                + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * pred_epsilon
            )

        next_x = x_t.detach().clone()
        next_x[:, :, :active_dims] = active_x + float(step_scale) * (active_prev - active_x)

        diagnostics = self._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
        diagnostics.update(
            {
                "update_mode": "ddim",
                "ddim_iteration": int(iteration),
                "ddim_num_iterations": int(num_iterations),
                "alpha_bar": float(alpha_bar),
                "alpha_bar_prev": float(alpha_bar_prev),
                "clean_sample_std": float(clean_std),
                "ddim_step_delta_norm": float(torch.linalg.vector_norm((next_x - x_t).detach()).cpu()),
            }
        )
        return next_x, diagnostics

    def step_mbd_score(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
        score_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        self._rank_alpha = float(alpha_bar)      # read by _ranking_scalar for the prior schedule
        x0_hat, result, active_dims, clean_std = self._optimize_ddim_clean_chunk(
            x_t,
            policy_inputs,
            context,
            alpha_bar=alpha_bar,
        )

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))

        score = torch.zeros_like(x_t)
        active_x = x_t.detach()[:, :, :active_dims]
        active_x0 = x0_hat[:, :, :active_dims]

        alpha_step = torch.clamp(alpha / torch.clamp(alpha_prev, min=self.config.flow_eps), min=self.config.flow_eps)
        if self.config.grad_calc == "backprop":
            score = self._backprop_clean_score(
                result,
                x_t,
                active_dims,
                policy_inputs,
                context,
                alpha_bar=alpha_bar,
            )
            score_numerator = beta * score[:, :, :active_dims]
            active_prev = (
                active_x + float(score_scale) * score_numerator
            ) / torch.sqrt(alpha_step)
        elif self.config.optimize_space == "accel":
            horizon = x_t.shape[1]
            opt_horizon = self._interpolation_knot_count(horizon)
            active_x_opt = self._control_point_resample(active_x[0], opt_horizon)
            active_code = self._trajectory_to_accel_code(active_x_opt)
            clean_code = result.mean.to(device=x_t.device, dtype=x_t.dtype)
            score_numerator_code = sqrt_alpha * clean_code - active_code
            score_code = score_numerator_code / beta
            score[:, :, :active_dims] = self._interpolate_control_points(
                self._accel_code_to_trajectory(score_code.unsqueeze(0))[0],
                horizon,
            ).unsqueeze(0)
            prev_code = (
                active_code + float(score_scale) * score_numerator_code
            ) / torch.sqrt(alpha_step)
            active_prev = self._interpolate_control_points(
                self._accel_code_to_trajectory(prev_code.unsqueeze(0))[0],
                horizon,
            ).unsqueeze(0)
        else:
            score_numerator = sqrt_alpha * active_x0 - active_x
            score[:, :, :active_dims] = score_numerator / beta
            active_prev = (active_x + float(score_scale) * score_numerator) / torch.sqrt(alpha_step)

        next_x = x_t.detach().clone()
        next_x[:, :, :active_dims] = active_prev

        diagnostics = self._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
        diagnostics.update(
            {
                "update_mode": "mbd_score",
                "ddim_iteration": int(iteration),
                "ddim_num_iterations": int(num_iterations),
                "alpha_bar": float(alpha_bar),
                "alpha_bar_prev": float(alpha_bar_prev),
                "clean_sample_std": float(clean_std),
                "score_scale": float(score_scale),
                "mbd_step_delta_norm": float(torch.linalg.vector_norm((next_x - x_t).detach()).cpu()),
            }
        )
        return next_x, diagnostics

    def step_mbd_score_action_prox(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
        score_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        x0_hat, result, active_dims, proposal_std = self._optimize_action_prox_chunk(
            x_t,
            policy_inputs,
            context,
            alpha_bar=alpha_bar,
            final=(int(iteration) == int(num_iterations) - 1),
        )

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))

        active_x = x_t.detach()[:, :, :active_dims]
        active_x0 = x0_hat[:, :, :active_dims]
        score_numerator = sqrt_alpha * active_x0 - active_x
        score = torch.zeros_like(x_t)
        score[:, :, :active_dims] = score_numerator / beta

        # Additive score-space steering seam: the hook sees this level's candidate
        # population and returns an (already scaled) score-field addend, blended
        # into the numerator so the DDIM update and the logged score agree.
        addend_fn = context.get("score_addend")
        steer_addend = None
        if addend_fn is not None:
            steer_addend = addend_fn(
                samples=self._interpolate_control_points(result.samples, x_t.shape[1]),
                weights=result.weights,
                x_t=x_t,
                base_score=score,
                iteration=iteration,
                num_iterations=num_iterations,
                alpha_bar=float(alpha_bar),
            )
        if steer_addend is not None:
            addend = torch.as_tensor(
                steer_addend, device=x_t.device, dtype=x_t.dtype)
            if addend.ndim == 2:
                addend = addend.unsqueeze(0)
            # H20: a non-finite addend silently poisons the whole plan, and 0 * NaN is NaN, so
            # even gamma=0 was not a safe identity. Drop the level's steering and say so rather
            # than propagate NaN into the executed action.
            if not bool(torch.isfinite(addend).all()):
                print("[planner] non-finite steering addend at this level; steering dropped",
                      flush=True)
                addend = torch.zeros_like(addend)
            score_numerator = score_numerator + beta * addend[:, :, :active_dims]
            score[:, :, :active_dims] = score_numerator / beta

        alpha_step = torch.clamp(
            alpha / torch.clamp(alpha_prev, min=self.config.flow_eps),
            min=self.config.flow_eps,
        )
        active_prev = (
            active_x + float(score_scale) * score_numerator
        ) / torch.sqrt(alpha_step)

        next_x = x_t.detach().clone()
        next_x[:, :, :active_dims] = active_prev
        next_x = self._ancestral_noise(
            next_x, active_dims, iteration=iteration,
            num_iterations=num_iterations, alpha_bar_prev=alpha_bar_prev)

        diagnostics = self._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
        diagnostics.update(
            {
                "update_mode": "mbd_score_action_prox",
                "ddim_iteration": int(iteration),
                "ddim_num_iterations": int(num_iterations),
                "alpha_bar": float(alpha_bar),
                "alpha_bar_prev": float(alpha_bar_prev),
                "active_dims": int(active_dims),
                "proposal_center": "noisy_action_div_sqrt_alpha",
                "proposal_noise_scale": float(proposal_std),
                "clean_sample_std": float(proposal_std),
                "score_scale": float(score_scale),
                "mbd_step_delta_norm": float(torch.linalg.vector_norm((next_x - x_t).detach()).cpu()),
                "ancestral_eta": float(self.config.ancestral_eta),
                "logit_norm": self.config.logit_norm,
            }
        )
        if steer_addend is not None:
            diagnostics["score_addend_norm"] = float(
                torch.linalg.vector_norm(
                    torch.as_tensor(steer_addend, dtype=x_t.dtype)).cpu())
        inject = context.get("inject")
        if inject is not None:
            diagnostics["inject_rho"] = float(inject.get("rho", 0.0))
            if getattr(self, "_last_inject_share", None) is not None:
                diagnostics["inject_weight_share"] = self._last_inject_share
            self._add_inject_trace_diagnostics(diagnostics)
        return next_x, diagnostics

    def step_mbd_score_action_warm(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
        score_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        next_x, diagnostics = self.step_mbd_score_action_prox(
            x_t,
            policy_inputs,
            context,
            iteration=iteration,
            num_iterations=num_iterations,
            score_scale=score_scale,
        )
        diagnostics = dict(diagnostics)
        diagnostics["update_mode"] = "mbd_score_action_warm"
        return next_x, diagnostics

    def estimate_mbd_score(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        score, _, diagnostics = self.estimate_mbd_score_terms(
            x_t,
            policy_inputs,
            context,
            iteration=iteration,
            num_iterations=num_iterations,
        )
        return score, diagnostics

    def estimate_mbd_score_terms(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        self._rank_alpha = float(alpha_bar)      # read by _ranking_scalar for the prior schedule
        x0_hat, result, active_dims, clean_std = self._optimize_ddim_clean_chunk(
            x_t,
            policy_inputs,
            context,
            alpha_bar=alpha_bar,
        )

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))

        score = torch.zeros_like(x_t)
        numerator = torch.zeros_like(x_t)
        active_x = x_t.detach()[:, :, :active_dims]
        active_x0 = x0_hat[:, :, :active_dims]

        if self.config.grad_calc == "backprop":
            score = self._backprop_clean_score(
                result,
                x_t,
                active_dims,
                policy_inputs,
                context,
                alpha_bar=alpha_bar,
            )
            numerator[:, :, :active_dims] = beta * score[:, :, :active_dims]
        elif self.config.optimize_space == "accel":
            horizon = x_t.shape[1]
            opt_horizon = self._interpolation_knot_count(horizon)
            active_x_opt = self._control_point_resample(active_x[0], opt_horizon)
            active_code = self._trajectory_to_accel_code(active_x_opt)
            clean_code = result.mean.to(device=x_t.device, dtype=x_t.dtype)
            score_code = (sqrt_alpha * clean_code - active_code) / beta
            score[:, :, :active_dims] = self._interpolate_control_points(
                self._accel_code_to_trajectory(score_code.unsqueeze(0))[0],
                horizon,
            ).unsqueeze(0)
            numerator[:, :, :active_dims] = beta * score[:, :, :active_dims]
        else:
            active_numerator = sqrt_alpha * active_x0 - active_x
            numerator[:, :, :active_dims] = active_numerator
            score[:, :, :active_dims] = active_numerator / beta

        diagnostics = self._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
        diagnostics.update(
            {
                "update_mode": "estimate_mbd_score",
                "ddim_iteration": int(iteration),
                "ddim_num_iterations": int(num_iterations),
                "alpha_bar": float(alpha_bar),
                "alpha_bar_prev": float(alpha_bar_prev),
                "active_dims": int(active_dims),
                "clean_sample_std": float(clean_std),
            }
        )
        return score, numerator, diagnostics

    def estimate_mbd_score_action_prox(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        score, _, diagnostics = self.estimate_mbd_score_action_prox_terms(
            x_t,
            policy_inputs,
            context,
            iteration=iteration,
            num_iterations=num_iterations,
        )
        return score, diagnostics

    def estimate_mbd_score_action_prox_terms(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        x0_hat, result, active_dims, proposal_std = self._optimize_action_prox_chunk(
            x_t,
            policy_inputs,
            context,
            alpha_bar=alpha_bar,
            final=(int(iteration) == int(num_iterations) - 1),
        )

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))

        score = torch.zeros_like(x_t)
        numerator = torch.zeros_like(x_t)
        active_x = x_t.detach()[:, :, :active_dims]
        active_x0 = x0_hat[:, :, :active_dims]
        active_numerator = sqrt_alpha * active_x0 - active_x
        numerator[:, :, :active_dims] = active_numerator
        score[:, :, :active_dims] = active_numerator / beta

        diagnostics = self._diagnostics(result=result, target=x0_hat, x_t=x_t, score=score)
        diagnostics.update(
            {
                "update_mode": "estimate_mbd_score_action_prox",
                "ddim_iteration": int(iteration),
                "ddim_num_iterations": int(num_iterations),
                "alpha_bar": float(alpha_bar),
                "alpha_bar_prev": float(alpha_bar_prev),
                "active_dims": int(active_dims),
                "proposal_center": "noisy_action_div_sqrt_alpha",
                "proposal_noise_scale": float(proposal_std),
                "clean_sample_std": float(proposal_std),
            }
        )
        tilt = context.get("task_tilt")
        if tilt is not None:
            diagnostics["task_tilt_weight"] = float(tilt["weight"])
            diagnostics["task_tilt_authority"] = float(tilt.get("authority", 1.0))
            if getattr(self, "_last_tilt_lambda", None) is not None:
                diagnostics["task_tilt_lambda_eff"] = self._last_tilt_lambda
            if getattr(self, "_last_tilt_base_cost", None) is not None:
                diagnostics["task_tilt_base_cost_mean"] = self._last_tilt_base_cost
        inject = context.get("inject")
        if inject is not None:
            diagnostics["inject_rho"] = float(inject.get("rho", 0.0))
            if getattr(self, "_last_inject_share", None) is not None:
                diagnostics["inject_weight_share"] = self._last_inject_share
            self._add_inject_trace_diagnostics(diagnostics)
        return score, numerator, diagnostics

    def estimate_mbd_score_action_warm(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        score, diagnostics = self.estimate_mbd_score_action_prox(
            x_t,
            policy_inputs,
            context,
            iteration=iteration,
            num_iterations=num_iterations,
        )
        diagnostics = dict(diagnostics)
        diagnostics["update_mode"] = "estimate_mbd_score_action_warm"
        return score, diagnostics

    def estimate_mbd_score_action_warm_terms(
        self,
        x_t: torch.Tensor,
        policy_inputs: dict[str, Any],
        context: dict[str, Any],
        *,
        iteration: int,
        num_iterations: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        score, numerator, diagnostics = self.estimate_mbd_score_action_prox_terms(
            x_t,
            policy_inputs,
            context,
            iteration=iteration,
            num_iterations=num_iterations,
        )
        diagnostics = dict(diagnostics)
        diagnostics["update_mode"] = "estimate_mbd_score_action_warm"
        return score, numerator, diagnostics

    def _ancestral_noise(
        self,
        x_prev: torch.Tensor,
        active_dims: int,
        *,
        iteration: int,
        num_iterations: int,
        alpha_bar_prev: float,
    ) -> torch.Tensor:
        """Marginal re-noising between denoise levels (opt-in via ancestral_eta).

        If x0_hat samples the level's smoothed posterior, adding
        eta*sqrt(1-abar_prev)*z distributes x_{k-1} as the forward marginal at
        the destination level (MBD's Monte-Carlo ancestral scheme; eta=0 is the
        current deterministic chain). Never applied at the final level, and only
        to the active dims. One draw per level per chain.
        """
        eta = float(self.config.ancestral_eta)
        if eta <= 0.0 or iteration + 1 >= num_iterations:
            return x_prev
        sigma = eta * math.sqrt(max(1.0 - float(alpha_bar_prev), 0.0))
        if sigma <= 0.0:
            return x_prev
        noised = x_prev.clone()
        active = x_prev[:, :, :active_dims]
        noised[:, :, :active_dims] = active + sigma * torch.randn_like(active)
        return noised

    def step_from_mbd_residual(
        self,
        x_t: torch.Tensor,
        base_numerator: torch.Tensor,
        residual_score: torch.Tensor,
        *,
        iteration: int,
        num_iterations: int,
        base_scale: float = 1.0,
        residual_scale: float = 1.0,
        active_dims: int | None = None,
    ) -> torch.Tensor:
        """Apply score residual while preserving the direct MBD base update."""
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        if base_numerator.shape != x_t.shape or residual_score.shape != x_t.shape:
            raise ValueError(
                "base_numerator and residual_score must match x_t: "
                f"base={tuple(base_numerator.shape)}, residual={tuple(residual_score.shape)}, "
                f"x_t={tuple(x_t.shape)}"
            )
        if active_dims is None:
            active_dims = min(self.config.action_dims, x_t.shape[-1])
        active_dims = min(int(active_dims), x_t.shape[-1])

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)
        alpha_step = torch.clamp(
            alpha / torch.clamp(alpha_prev, min=self.config.flow_eps),
            min=self.config.flow_eps,
        )

        if isinstance(residual_scale, torch.Tensor):
            # Per-channel gamma: broadcast over the active action dims.
            res_scale = residual_scale.to(device=x_t.device, dtype=x_t.dtype)[..., :active_dims]
        else:
            res_scale = float(residual_scale)
        active_x = x_t.detach()[:, :, :active_dims]
        active_prev = (
            active_x
            + float(base_scale) * base_numerator[:, :, :active_dims]
            + res_scale * beta * residual_score[:, :, :active_dims]
        ) / torch.sqrt(alpha_step)
        next_x = x_t.detach().clone()
        next_x[:, :, :active_dims] = active_prev
        return self._ancestral_noise(
            next_x, active_dims, iteration=iteration,
            num_iterations=num_iterations, alpha_bar_prev=alpha_bar_prev)

    def step_from_score(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        *,
        iteration: int,
        num_iterations: int,
        update_mode: str,
        score_scale: float = 1.0,
        active_dims: int | None = None,
    ) -> torch.Tensor:
        alpha_bar, alpha_bar_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_iterations,
            num_train_timesteps=self.config.ddim_num_train_timesteps,
        )
        if score.shape != x_t.shape:
            raise ValueError(
                f"score shape must match x_t shape, got {tuple(score.shape)} vs {tuple(x_t.shape)}"
            )
        if active_dims is None:
            active_dims = min(self.config.action_dims, x_t.shape[-1])
        active_dims = min(int(active_dims), x_t.shape[-1])

        alpha = torch.as_tensor(alpha_bar, device=x_t.device, dtype=x_t.dtype)
        alpha_prev = torch.as_tensor(alpha_bar_prev, device=x_t.device, dtype=x_t.dtype)
        beta = torch.clamp(1.0 - alpha, min=self.config.flow_eps)

        active_x = x_t.detach()[:, :, :active_dims]
        active_score = score[:, :, :active_dims]
        next_x = x_t.detach().clone()

        if update_mode == "mbd_score":
            alpha_step = torch.clamp(
                alpha / torch.clamp(alpha_prev, min=self.config.flow_eps),
                min=self.config.flow_eps,
            )
            active_prev = (
                active_x + float(score_scale) * beta * active_score
            ) / torch.sqrt(alpha_step)
        elif update_mode == "ddim":
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=self.config.flow_eps))
            sqrt_beta = torch.sqrt(beta)
            x0_hat = (active_x + beta * active_score) / sqrt_alpha
            eps_hat = -sqrt_beta * active_score
            active_prev = (
                torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * x0_hat
                + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * eps_hat
            )
            active_prev = active_x + float(score_scale) * (active_prev - active_x)
        else:
            raise ValueError(
                "External score updates support update_mode='ddim' or 'mbd_score', "
                f"got {update_mode!r}."
            )

        next_x[:, :, :active_dims] = active_prev
        return self._ancestral_noise(
            next_x, active_dims, iteration=iteration,
            num_iterations=num_iterations, alpha_bar_prev=alpha_bar_prev)
