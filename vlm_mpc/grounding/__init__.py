"""Grounding contract + source registry for base controllers.

Front-ends produce a ``Grounding`` (see ``api``); the base driver consumes it. ``get_source`` resolves a
source by name, importing each source lazily so heavy/front-end dependencies stay local to the source.
"""
from vlm_mpc.grounding.api import Grounding, GroundingSource, SceneObject, Stage

__all__ = ["Grounding", "GroundingSource", "SceneObject", "Stage", "get_source"]


def get_source(name: str, **kwargs) -> GroundingSource:
    """Return a grounding source by name (``gt`` / ``rekep_fake`` / ``rekep_real``; VoxPoser/MOKA to follow)."""
    if name == "gt":
        from vlm_mpc.grounding.gt import GTGrounding
        return GTGrounding(grasp_obj=kwargs.get("grasp_obj", "pear"), place_obj=kwargs.get("place_obj", "scale"))
    if name in ("rekep_fake", "rekep_real"):
        from vlm_mpc.grounding.rekep import RekepGrounding
        return RekepGrounding(vlm="real" if name == "rekep_real" else "fake",
                              task_key=kwargs.get("task_key", "weight"),
                              place_obj=kwargs.get("place_obj", "scale"))
    raise ValueError(f"unknown grounding source: {name!r}")
