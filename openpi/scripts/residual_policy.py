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
    """Arguments for the serve_policy script."""

    base_model_name: str
    residual_model_name: str
    base_checkpoint_dir: str
    residual_checkpoint_dir: str

    num_vla_steps: int = 10
    num_residual_steps: int = 10

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False


def create_policy(args: Args) -> tuple[_policy.Policy, _policy.Policy]:
    """Create a policy from the given arguments."""
    residual_policy = _policy_config.create_trained_policy(
        _config.get_config(args.residual_model_name),
        args.residual_checkpoint_dir,
    )
    base_policy = _policy_config.create_trained_policy(
        _config.get_config(args.base_model_name),
        args.base_checkpoint_dir,
        default_prompt=args.default_prompt,
    )

    return base_policy, residual_policy


def main(args: Args) -> None:
    base_policy, residual_policy = create_policy(args)
    policy_metadata = base_policy.metadata

    # Record the policy's behavior.
    if args.record:
        base_policy = _policy.PolicyRecorder(base_policy, "base_policy_records")
        residual_policy = _policy.PolicyRecorder(
            residual_policy, "residual_policy_records"
        )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketResidualServer(
        vla_policy=base_policy,
        residual_policy=residual_policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        num_vla_steps=args.num_vla_steps,
        num_residual_steps=args.num_residual_steps,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
