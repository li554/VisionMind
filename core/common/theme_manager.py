"""
主题管理器 - 管理应用程序的白天/夜晚主题切换
"""
from PySide6.QtCore import QObject, Signal
from qfluentwidgets import setTheme, Theme

from .settings import settings


class ThemeManager(QObject):
    """
    主题管理器，负责管理应用程序的白天/夜晚主题切换
    """
    theme_changed = Signal(str)  # 主题改变时发出的信号，参数为主题名称 'light' 或 'dark'

    def __init__(self):
        super().__init__()
        self._current_theme = 'dark'  # 默认主题
        self._load_theme()

    def _load_theme(self):
        """从设置中加载主题配置"""
        saved_theme = settings.get('theme', 'dark')
        self._current_theme = saved_theme if saved_theme in ['light', 'dark'] else 'dark'
        self._apply_theme(self._current_theme, save=False)

    def _apply_theme(self, theme: str, save: bool = True):
        """
        应用主题
        
        Args:
            theme: 主题名称，'light' 或 'dark'
            save: 是否保存到设置文件
        """
        if theme == 'light':
            setTheme(Theme.LIGHT)
        else:
            setTheme(Theme.DARK)
        
        self._current_theme = theme
        
        if save:
            settings.set('theme', theme)
        
        # 发出主题改变信号
        self.theme_changed.emit(theme)

    def set_theme(self, theme: str):
        """
        设置主题
        
        Args:
            theme: 主题名称，'light' 或 'dark'
        """
        if theme not in ['light', 'dark']:
            theme = 'dark'
        
        if theme != self._current_theme:
            self._apply_theme(theme)

    def toggle_theme(self):
        """切换主题（白天/夜晚）"""
        new_theme = 'light' if self._current_theme == 'dark' else 'dark'
        self.set_theme(new_theme)

    def get_current_theme(self) -> str:
        """
        获取当前主题
        
        Returns:
            当前主题名称，'light' 或 'dark'
        """
        return self._current_theme

    def is_dark_theme(self) -> bool:
        """
        判断当前是否为暗色主题
        
        Returns:
            True 如果是暗色主题，否则 False
        """
        return self._current_theme == 'dark'

    def is_light_theme(self) -> bool:
        """
        判断当前是否为亮色主题
        
        Returns:
            True 如果是亮色主题，否则 False
        """
        return self._current_theme == 'light'


# 全局主题管理器实例
theme_manager = ThemeManager()
