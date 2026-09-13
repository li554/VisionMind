"""
VisionMind 插件化架构启动入口
"""
import sys
import os

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    # 解析命令行参数
    test_config = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--test' and i < len(sys.argv) - 1:
            test_config = sys.argv[i + 1]
            sys.argv = sys.argv[:i]  # 移除 --test 参数，避免 Qt 解析报错
            break

    # 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    # DSH Web 侧栏(QWebEngineView)要求共享 OpenGL 上下文,须在 QApplication
    # 创建前设置
    if not QApplication.testAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)
    app.setApplicationName("VisionMind")
    app.setOrganizationName("VisionMind")

    # 全局禁止 SpinBox/DoubleSpinBox 鼠标滚轮改值
    from PySide6.QtCore import QObject, QEvent
    from qfluentwidgets import SpinBox, DoubleSpinBox
    from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox

    _spin_classes = (SpinBox, DoubleSpinBox, QSpinBox, QDoubleSpinBox)

    class _WheelBlocker(QObject):
        def eventFilter(self, obj, event):
            if isinstance(obj, _spin_classes) and event.type() == QEvent.Type.Wheel:
                return True
            if not isinstance(obj, QObject):
                return False
            return super().eventFilter(obj, event)

    app.installEventFilter(_WheelBlocker(app))

    # 设置应用图标(多尺寸,Windows 任务栏/Alt-Tab 需要 16/32 等小尺寸)
    from core.common.icons import build_app_icon
    app_icon = build_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    # 安装全局异常处理器（包含 Qt 消息处理器，过滤 QFont::setPointSize 等警告）
    from core.common.exception_handler import install_global_exception_handler
    install_global_exception_handler()

    # 1. 初始化核心组件
    from core.event_bus import EventBus, StateRouter
    from core.service.project_service import ProjectService
    from core.plugin_manager import PluginContext, PluginManager
    from core.main_window import MainWindow
    from core.common.action_registry import ActionRegistry

    event_bus = EventBus()
    state_router = StateRouter(event_bus)  # 全局状态路由器：唯一订阅 app:state_changed
    project_service = ProjectService()
    main_window = MainWindow()

    # 初始化 ActionRegistry 单例
    registry = ActionRegistry.instance()

    # 2. 构建 PluginContext
    context = PluginContext(main_window, event_bus)
    context.set_state_router(state_router)
    main_window.set_event_bus(event_bus)
    context.project_service = project_service
    project_service.set_event_bus(event_bus)  # 项目域服务发布 dashboard:projects_changed 事件

    # 3. 初始化 PluginManager
    plugins_dir = os.path.join(PROJECT_ROOT, "plugins")
    config_path = os.path.join(PROJECT_ROOT, "core", "plugins.json")

    pm = PluginManager(plugins_dir, config_path, context)
    pm.load_config()       # 读 core/plugins.json
    pm.discover_plugins()  # 扫描 plugins/ 目录
    pm.load_all()          # 按依赖顺序加载

    # 4. 创建并注册系统配置界面
    from core.view.config_interface import ConfigInterface
    from qfluentwidgets import FluentIcon as FIF
    config_interface = ConfigInterface()
    config_interface.set_plugin_manager(pm)
    config_interface.set_project_service(project_service)
    main_window.add_plugin_interface(config_interface, FIF.SETTING, "系统配置", order=100, position="bottom")

    # 5. 注册内置 Action 和输入模拟器
    from core.common.builtin_actions import BuiltinActions
    from core.common.input_simulator import InputSimulator
    registry.register_instance(BuiltinActions())
    registry.register_instance(InputSimulator(main_window), prefix="input")
    # 注册 MainWindow 上的 @action 方法（如 switch_to）
    registry.register_instance(main_window, prefix="main")
    # 注册项目域服务（项目管理 action：project.*），并注册旧全名别名做过渡兼容
    registry.register_instance(project_service, prefix="project")
    for old, new in {
        "dashboard.project.delete_project_data": "project.delete_project_data",
        "dashboard.project.create_project_data": "project.create_project_data",
        "dashboard.project.load_project_data": "project.load_project_data",
        "dashboard.project.update_project_info": "project.update_project_info",
        "dashboard.project.get_image_count": "project.get_image_count",
    }.items():
        registry.register_alias(old, new)
    print(f"[ActionRegistry] 已注册 {len(registry)} 个 action")

    # 6. 显示主窗口
    main_window.show()

    # 无边框/透明窗口在 setWindowIcon 时原生句柄可能尚未创建，Windows
    # 任务栏图标会回落成默认程序图标；show 后补设一次确保图标生效
    if not app_icon.isNull():
        main_window.setWindowIcon(app_icon)
        app.setWindowIcon(app_icon)

    # 7. 自动化测试模式
    if test_config:
        from PySide6.QtCore import QTimer
        from core.common.test_engine import TestEngine

        config_file = os.path.join(PROJECT_ROOT, "config", test_config) if not os.path.isabs(test_config) else test_config
        if not os.path.exists(config_file):
            print(f"[TestRunner] 配置文件不存在: {config_file}")
        else:
            engine = TestEngine(main_window)
            if engine.load_scenarios(config_file):
                # 延迟 2 秒启动测试，等待界面完全加载
                QTimer.singleShot(2000, engine.start_tests)
                engine.all_tests_completed.connect(lambda s, t: print(f"\n[TestRunner] 测试完成: {s}/{t}"))
                print(f"[TestRunner] 已加载测试配置: {config_file}")
                print(f"[TestRunner] 共 {len(engine.scenarios)} 个场景，2 秒后开始执行...")

    # 8. 运行应用
    # 关闭兜底：正常路径是 MainWindow.closeEvent 排空后台工作；但若窗口不是通过
    # 标题栏关闭（或 Qt 自行退出），closeEvent 可能不触发。ShutdownCoordinator
    # 的 shutdown() 是幂等的，因此这里再挂一次作为最后一道防线，确保退出期不会
    # 出现 "QThread: Destroyed while thread '' is still running"。
    from core.lifecycle import ShutdownCoordinator
    app.aboutToQuit.connect(lambda: ShutdownCoordinator.instance().shutdown(5000))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
