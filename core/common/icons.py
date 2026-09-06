"""
语义化自定义图标（AppIcon）

QFluentWidgets 内置的 FluentIcon 没有贴合"打点/矩形/多边形/旋转框"等标注语义的图形，
此前工具栏图标多为随意选择（剪刀表示矩形、指纹表示编辑等）。

本模块提供一套与 FluentIconBase 兼容的自定义矢量图标：
- 统一 24x24 视框、2px 圆角描边风格；
- 明暗主题自动切换（{name}_light.svg 为亮色主题用的深色图形，
  {name}_dark.svg 为暗色主题用的白色图形）；
- 可直接用于 ToolButton / Action / PushButton 等任何接受 FluentIconBase 的位置。

用法::

    from core.common.icons import AppIcon

    btn = ToolButton(AppIcon.RECT, self)      # 矩形标注
    act = Action(AppIcon.MODEL, "模型管理")    # 模型管理
"""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from qfluentwidgets import FluentIconBase, Theme, qconfig

# core/common/icons.py -> 项目根目录
_SVG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "icons", "svg"))
_ICON_DIR = os.path.normpath(os.path.join(_SVG_DIR, ".."))


def build_app_icon() -> QIcon:
    """构建多尺寸应用图标。

    Windows 任务栏/Alt-Tab 会按 16/32/48 等小尺寸请求窗口图标；直接使用
    app.png(794x794) 或仅含单个 256px PNG 条目的 app.ico 时，小尺寸请求
    偶尔会回落成默认的程序图标。这里从 app.png 逐尺寸缩放出多档 pixmap
    合成 QIcon，确保任何尺寸请求都能命中，且不受图标缓存/句柄时序影响。
    """
    icon = QIcon()
    png_path = os.path.join(_ICON_DIR, "app.png")
    base = QPixmap(png_path)
    if base.isNull():
        ico_path = os.path.join(_ICON_DIR, "app.ico")
        if os.path.exists(ico_path):
            return QIcon(ico_path)
        return QIcon()
    for size in (16, 20, 24, 32, 48, 64, 128, 256):
        pm = base.scaled(
            size, size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if not pm.isNull():
            icon.addPixmap(pm)
    return icon


class AppIcon(FluentIconBase):
    """自定义 FluentIconBase 图标（按主题选择描边颜色的 SVG 文件）"""

    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name

    def path(self, theme: Theme = Theme.AUTO) -> str:
        theme = qconfig.theme if theme == Theme.AUTO else theme
        suffix = "dark" if theme == Theme.DARK else "light"
        return os.path.join(_SVG_DIR, f"{self._name}_{suffix}.svg")

    def icon(self, theme: Theme = Theme.AUTO, colorIsReverse: bool = False) -> QIcon:
        return QIcon(self.path(theme))

    def themeIcon(self) -> QIcon:
        return self.icon(qconfig.theme)


# ---- 标注工具栏 ----
POINT = AppIcon("point")        # AI 点工具（SAM 交互打点）：十字准星 + 中心圆点
AI_RECT = AppIcon("ai_rect")    # AI 矩形框工具：矩形 + 星芒（AI 魔棒）
RECT = AppIcon("rect")          # 矩形标注：圆角矩形
POLYGON = AppIcon("polygon")    # 多边形标注：五边形轮廓
OBB = AppIcon("obb")            # 旋转检测：斜置矩形
CURSOR = AppIcon("cursor")      # 编辑模式：鼠标指针
ROI = AppIcon("roi")            # ROI 区域选择：虚线选框

# ---- 通用 ----
FIT_VIEW = AppIcon("fit_view")  # 重置/适配视图：四角括线 + 中心方块
MODEL = AppIcon("model")        # 模型管理：芯片

# 类级别名：两种用法均可（AppIcon.RECT / 直接导入 RECT）
AppIcon.POINT = POINT
AppIcon.AI_RECT = AI_RECT
AppIcon.RECT = RECT
AppIcon.POLYGON = POLYGON
AppIcon.OBB = OBB
AppIcon.CURSOR = CURSOR
AppIcon.ROI = ROI
AppIcon.FIT_VIEW = FIT_VIEW
AppIcon.MODEL = MODEL

# ---- 第二批（Agent 侧栏 / 导航 / 通用）----
ATTACH = AppIcon("attach")        # 附加文件：回形针
IMAGE = AppIcon("image")          # 图片：相框 + 山 + 太阳
SEARCH = AppIcon("search")        # 搜索：放大镜
HISTORY = AppIcon("history")      # 历史/会话：时钟 + 逆时针箭头
AGENT = AppIcon("agent")          # AI 助手：星芒
DEBUG = AppIcon("debug")          # 调试：终端方框
CLEAR = AppIcon("clear")          # 清空：橡皮擦
THEME = AppIcon("theme")          # 主题切换：日月
USER = AppIcon("user")            # 用户：人形
DASHBOARD = AppIcon("dashboard")  # 仪表盘：四宫格
TAG = AppIcon("tag")              # 标注：标签

AppIcon.ATTACH = ATTACH
AppIcon.IMAGE = IMAGE
AppIcon.SEARCH = SEARCH
AppIcon.HISTORY = HISTORY
AppIcon.AGENT = AGENT
AppIcon.DEBUG = DEBUG
AppIcon.CLEAR = CLEAR
AppIcon.THEME = THEME
AppIcon.USER = USER
AppIcon.DASHBOARD = DASHBOARD
AppIcon.TAG = TAG
