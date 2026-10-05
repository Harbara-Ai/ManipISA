"""Adapters are loaded explicitly; importing manipisa does not start Isaac Sim."""

__all__ = ["DualUr5WujiAdapter", "DualUr5WujiConfig", "SideBinding", "ToolFrame", "RigidTransform", "WujiContact", "RoboDojoAdapter"]


def __getattr__(name):
    if name == "RoboDojoAdapter":
        from .robodojo import RoboDojoAdapter
        return RoboDojoAdapter
    if name in __all__:
        from . import dual_ur5_wuji
        return getattr(dual_ur5_wuji, name)
    raise AttributeError(name)
