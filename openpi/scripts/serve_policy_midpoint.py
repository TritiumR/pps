"""
Serve a policy using the Midpoint method (2nd order Runge-Kutta) for denoising.

The Midpoint method provides better accuracy than Euler at the cost of 2x denoise calls per step.

Usage:
    python scripts/serve_policy_midpoint.py \
        --base_model_name pi05_droid \
        --base_checkpoint_dir checkpoints/pi05_droid \
        --num_steps 10 \
        --port 8000

Note: num_steps is the number of integration steps. Each step requires 2 denoise calls,
so total denoise calls = 2 * num_steps.
"""

import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config

from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy_midpoint script."""

    # Model configuration
    base_model_name: str
    base_checkpoint_dir: str

    # Number of integration steps (each step = 2 denoise calls for midpoint method)
    num_steps: int = 10

    # Environment to serve the policy for.
    env: EnvMode = EnvMode.DROID

    # If provided, will be used in case the "prompt" key is not present in the data.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000

    # Record the policy's behavior for debugging.
    record: bool = False


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    base_policy = _policy_config.create_trained_policy(
        _config.get_config(args.base_model_name),
        args.base_checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    return base_policy


def main(args: Args) -> None:
    base_policy = create_policy(args)
    policy_metadata = base_policy.metadata

    # Record the policy's behavior.
    if args.record:
        base_policy = _policy.PolicyRecorder(base_policy, "midpoint_policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating Midpoint server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("Using %d integration steps (= %d denoise calls per inference)", args.num_steps, args.num_steps * 2)

    server = websocket_policy_server.WebsocketMidpointServer(
        base_policy=base_policy,
        args=args,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))






