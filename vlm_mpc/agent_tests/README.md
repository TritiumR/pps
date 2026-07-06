# vlm_mpc/agent_tests

Staging area for **new throwaway `_`-prefixed experiment scripts** specific to `vlm_mpc/`, before they're
refactored into the shared library + a proper `tasks/<name>.py` driver.

Workflow:
1. Prototype here as `_<thing>.py` (standalone is fine — copy the runtime/overlay/control bootstrap).
2. Once it's confirmed worth keeping, lift the shared parts into the library and turn the entry point
   into `vlm_mpc/tasks/<name>.py` (`add_args` + `run`), registered in `main.py`.
3. Move (don't delete) the original probe to `vlm_mpc/archive/` if it's worth keeping for reference.

This is the package-scoped complement to the top-level `agent_tests/` (which holds cross-cutting probes).
Nothing here is a stable entry point; don't import from it.
