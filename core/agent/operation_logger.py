"""
全程序操作记录器 — 记录 VisionMind 的所有用户操作

覆盖所有模块：项目管理、导航、标注、自动标注、类别管理、筛选、视图、模型、导出、系统。
每条记录包含：action + params + 完整上下文 + 操作前后状态快照 + 耗时。

与 ActionRecorder 并存：
- ActionRecorder：用于测试回放，只在录制模式下工作
- OperationLogger：始终后台记录，用于 Agent 模式分析
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

from PySide6.QtCore import QObject, Signal


class OperationLogger(QObject):
    """全程序操作记录器（全局单例）"""

    operation_recorded = Signal(dict)  # 新操作记录

    _instance: Optional['OperationLogger'] = None

    def __init__(self):
        super().__init__()
        self._history: deque = deque(maxlen=500)  # 内存中保留最近 500 条
        self._project_dir: Optional[str] = None
        self._session_start = time.time()
        self._unsaved_count = 0  # 未落盘的记录数（限流保存）
        self._save_interval = 20  # 每 N 条记录落盘一次

    @classmethod
    def instance(cls) -> 'OperationLogger':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_project_dir(self, project_dir: str):
        """设置当前项目目录（切换项目时调用）"""
        self._project_dir = project_dir
        self._unsaved_count = 0
        # 加载已有历史
        self._load_history()

    def flush(self):
        """立即将未落盘的记录写入磁盘"""
        if self._unsaved_count > 0:
            self._unsaved_count = 0
            self._save_history()

    def record(self, action_name: str, params: dict = None,
               context: dict = None, before_state: dict = None,
               after_state: dict = None):
        """
        记录一次操作

        Args:
            action_name: action 名称
            params: action 参数
            context: 当前上下文（项目、图片、工具、类别等）
            before_state: 操作前状态快照
            after_state: 操作后状态快照
        """
        now = datetime.now()
        record = {
            "id": f"op_{now.strftime('%Y%m%d_%H%M%S')}_{len(self._history):03d}",
            "timestamp": now.isoformat(),
            "action": action_name,
            "params": self._sanitize(params or {}),
            "context": self._sanitize(context or {}),
            "before_state": self._sanitize(before_state or {}),
            "after_state": self._sanitize(after_state or {}),
        }

        self._history.append(record)
        self.operation_recorded.emit(record)
        self._unsaved_count += 1
        if self._unsaved_count >= self._save_interval:
            self._unsaved_count = 0
            self._save_history()

    # 敏感字段名集合：记录到操作历史/会话日志时需掩码，防止 API Key 等凭据泄露
    _SENSITIVE_KEYS = {"api_key", "ai_api_key", "secret", "token", "password", "apikey"}

    @staticmethod
    def _mask_value(value: str) -> str:
        """对敏感字符串值做掩码，保留前缀和末尾少量字符以便辨识。"""
        if not isinstance(value, str) or len(value) <= 8:
            return "***"
        return f"{value[:6]}***{value[-4:]}"

    @staticmethod
    def _sanitize(value):
        """递归将不可 JSON 序列化的对象转换为可序列化形式（如 Qt widget）。

        同时对敏感字段（api_key、token、secret 等）做掩码处理，
        防止凭据明文写入操作历史文件或会话日志。
        """
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                key_lower = str(k).lower()
                if key_lower in OperationLogger._SENSITIVE_KEYS and isinstance(v, str) and v:
                    result[k] = OperationLogger._mask_value(v)
                else:
                    result[k] = OperationLogger._sanitize(v)
            return result
        if isinstance(value, (list, tuple, set)):
            return [OperationLogger._sanitize(v) for v in value]
        try:
            return str(value)
        except Exception:
            return f"<{type(value).__name__}>"

    def get_history(self, count: int = 50, time_range: float = None,
                    action_filter: str = None, category_filter: str = None) -> List[dict]:
        """
        查询操作历史

        Args:
            count: 返回条数
            time_range: 时间范围（秒），如 3600 = 最近 1 小时
            action_filter: 按 action 名称过滤
            category_filter: 按操作分类过滤
        """
        records = list(self._history)

        if time_range:
            cutoff = time.time() - time_range
            records = [
                r for r in records
                if datetime.fromisoformat(r["timestamp"]).timestamp() >= cutoff
            ]

        if action_filter:
            records = [r for r in records if action_filter in r["action"]]

        if category_filter:
            records = [
                r for r in records
                if r.get("context", {}).get("category") == category_filter
            ]

        return records[-count:]

    def get_stats(self) -> dict:
        """获取操作统计"""
        records = list(self._history)
        if not records:
            return {"total": 0}

        # 按 action 分类统计
        action_counts = {}
        for r in records:
            action = r["action"]
            action_counts[action] = action_counts.get(action, 0) + 1

        # 按分类统计
        category_counts = {}
        for r in records:
            cat = r.get("context", {}).get("category", "未知")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # 时间统计
        timestamps = [
            datetime.fromisoformat(r["timestamp"]).timestamp()
            for r in records
        ]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0

        return {
            "total": len(records),
            "action_counts": action_counts,
            "category_counts": category_counts,
            "duration_seconds": round(duration, 1),
            "first_record": records[0]["timestamp"],
            "last_record": records[-1]["timestamp"],
        }

    def analyze_patterns(self) -> dict:
        """
        分析操作模式，返回重复操作、连续行为等

        Returns:
            patterns: 检测到的模式列表
        """
        records = list(self._history)
        if len(records) < 3:
            return {"patterns": [], "message": "记录不足，至少需要 3 条操作记录"}

        patterns = []

        # 检测连续相同操作
        streak = 1
        streak_action = records[0]["action"]
        streak_params = []
        for i in range(1, len(records)):
            if records[i]["action"] == streak_action:
                streak += 1
                streak_params.append(records[i].get("params", {}))
            else:
                if streak >= 3:
                    patterns.append({
                        "type": "repeated_action",
                        "action": streak_action,
                        "count": streak,
                        "params_sample": streak_params[:3] if streak_params else [],
                        "message": f"连续 {streak} 次执行 {streak_action}",
                    })
                streak = 1
                streak_action = records[i]["action"]
                streak_params = []

        # 检查最后一段连续
        if streak >= 3:
            patterns.append({
                "type": "repeated_action",
                "action": streak_action,
                "count": streak,
                "params_sample": streak_params[:3] if streak_params else [],
                "message": f"连续 {streak} 次执行 {streak_action}",
            })

        # 检测类别修改趋势
        category_changes = [
            r for r in records
            if r["action"] == "change_annotation_category"
        ]
        if len(category_changes) >= 3:
            target_categories = [
                r.get("params", {}).get("category", "")
                for r in category_changes
            ]
            target_categories = [c for c in target_categories if c]
            if target_categories:
                from collections import Counter
                counts = Counter(target_categories)
                most_common = counts.most_common(1)[0]
                if most_common[1] >= 3:
                    patterns.append({
                        "type": "category_trend",
                        "target_category": most_common[0],
                        "count": most_common[1],
                        "message": f"最近 {len(category_changes)} 次类别修改中，{most_common[1]} 次改为 '{most_common[0]}'",
                    })

        return {"patterns": patterns}

    def _save_history(self):
        """持久化历史到磁盘"""
        if not self._project_dir:
            return
        try:
            history_path = os.path.join(self._project_dir, ".operation_history.json")
            records = list(self._history)
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[OperationLogger] 保存历史失败: {e}")

    def _load_history(self):
        """从磁盘加载历史"""
        if not self._project_dir:
            return
        try:
            history_path = os.path.join(self._project_dir, ".operation_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                self._history.clear()
                for r in records[-500:]:  # 只加载最近 500 条
                    self._history.append(r)
                print(f"[OperationLogger] 加载了 {len(self._history)} 条历史记录")
        except Exception as e:
            print(f"[OperationLogger] 加载历史失败: {e}")


def capture_state_snapshot() -> dict:
    """在主线程采集轻量应用状态快照（供操作记录器记录操作前后状态）

    返回：interface / image_index / total_images / annotation_count / categories 等
    """
    try:
        def _do():
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if not app:
                return {}
            main_window = None
            for widget in app.topLevelWidgets():
                if hasattr(widget, '_plugin_interfaces'):
                    main_window = widget
                    break
            if not main_window:
                return {}
            snap = {"interface": ""}
            current = main_window.stackedWidget.currentWidget()
            if current:
                snap["interface"] = current.objectName()
            for w in main_window._plugin_interfaces:
                if hasattr(w, 'current_image_path'):
                    snap["image_index"] = w.file_list.currentRow() if hasattr(w, 'file_list') else -1
                    snap["total_images"] = w.file_list.count() if hasattr(w, 'file_list') else 0
                    snap["categories"] = list(getattr(w, 'categories', []) or [])
                    draw_area = getattr(w, 'draw_area', None)
                    if draw_area:
                        snap["annotation_count"] = len(draw_area.annotations or [])
                    break
            return snap

        from core.common.main_thread_dispatcher import run_on_main
        return run_on_main(_do)
    except Exception as e:
        return {"snapshot_error": f"{type(e).__name__}: {e}"}

