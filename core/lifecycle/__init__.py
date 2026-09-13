"""应用生命周期基座。

目前只包含关闭协调器（ShutdownCoordinator）。放在 core 层而不是
plugins/annotation 下，是因为"生命周期纪律"对所有插件都成立，不只是标注界面。
"""
from .shutdown_coordinator import (  # noqa: F401
    Drainable,
    ShutdownCoordinator,
    register_drainable,
    shutdown_all,
    unregister_drainable,
)

__all__ = [
    "Drainable",
    "ShutdownCoordinator",
    "register_drainable",
    "unregister_drainable",
    "shutdown_all",
]
