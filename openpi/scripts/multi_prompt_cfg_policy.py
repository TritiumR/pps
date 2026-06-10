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
    """Arguments for the multi-prompt CFG server."""

    model_name: str
    checkpoint_dir: str

    # Positive prompt (conditional)
    prompt_positive: str
    # Negative prompt (unconditional or alternative condition)
    # Use empty string "" for unconditional guidance
    prompt_negative: str = ""

    # CFG parameters
    guidance_scale: float = 1.0
    guidance_start_step: float = 0.0  # Start applying CFG at this denoising timestep
    num_steps: int = 10

    # Dynamic guidance scale options
    use_decreasing_guidance: bool = False
    use_increasing_guidance: bool = False

    # Environment mode
    env: EnvMode = EnvMode.DROID

    # Port to serve the policy on
    port: int = 8000
    # Record policy behavior for debugging
    record: bool = False


def create_policy(args: Args) -> _policy.Policy:
    """Create a single policy from the given arguments."""
    # Note: We don't set default_prompt here since we'll inject prompts dynamically
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.model_name),
        args.checkpoint_dir,
    )
    return policy


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior
    if args.record:
        policy = _policy.PolicyRecorder(policy, "multi_prompt_cfg_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketMultiPromptCFGServer(
        policy=policy,
        args=args,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))