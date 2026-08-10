"""Custom robosuite environments for the MuJoCo evaluation stack.

Importing this package registers every environment it defines with robosuite's
``REGISTERED_ENVS`` (subclasses self-register through ``EnvMeta``), which is what
``robomimic``'s ``create_env_from_metadata`` resolves ``env_args["env_name"]`` against.
"""

from .sort_can import SortCanTwoBin  # noqa: F401
from .sort_can_tray import SortCanTray  # noqa: F401
