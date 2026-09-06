"""
操作录制器 — 拦截 ActionRegistry.call() 记录用户操作

录制用户的一切操作，生成标准 test_*.json 格式的测试流程。
"""

import json
import time
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal


class ActionRecorder(QObject):
    """操作录制器"""

    recording_changed = Signal(bool)  # 录制状态变化
    step_recorded = Signal(str, dict)  # action_name, step

    _instance: Optional['ActionRecorder'] = None

    # 不需要录制的 action 前缀/名称
    RECORD_SKIP_PREFIXES = (
        "wait", "print", "print_result", "assert", "input.",
        "loop", "if", "close_app", "start_timer", "stop_timer",
        "assert_timer", "assert_ui_responsive",
    )
    RECORD_SKIP_NAMES = {
        "set_variable", "update_inference_btn_state",
        "update_log", "update_progress", "update_metrics",
        "update_curve_visualization", "connect_signals",
        "get_settings", "set_settings", "get_selected",
        "generation_started", "generation_finished", "generation_error",
        "update_result_grid_layout",
        "toggle_recording",  # 避免录制器自身被录制
        "refresh_projects",  # 刷新操作为自动触发，不应录制
        # 以下 action 有对话框，需在对话框完成后手动 record_raw_step，
        # 此处跳过 wrapper 的自动录制（自动录制不含对话框结果参数）
        "set_as_example",              # 需手动录制含 example_name
        "change_annotation_category",  # 需手动录制含 category
        "add_category",                # 需手动录制含 category_name
        "rename_category",             # 需手动录制含 new_name
        "clear_annotations",           # 需手动录制含 skip_confirm
        "delete_category",             # 需手动录制含 category_name + skip_confirm
        "delete_image_file",           # 需手动录制含 path + skip_confirm
    }

    def __init__(self):
        super().__init__()
        self._recording = False
        self._steps: List[dict] = []
        self._scenario_name = ""
        self._description = ""
        self._last_step_time: Optional[float] = None

    @classmethod
    def instance(cls) -> 'ActionRecorder':
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, scenario_name: str = "录制流程", description: str = ""):
        """开始录制"""
        self._recording = True
        self._steps = []
        self._scenario_name = scenario_name
        self._description = description
        self._last_step_time = time.time()
        # 激活自动交互录制器
        self._activate_auto_recorder()
        self.recording_changed.emit(True)
        print(f"[ActionRecorder] 开始录制: {scenario_name}")

    def stop(self) -> dict:
        """停止录制，返回测试场景 dict"""
        self._recording = False
        # 停用自动交互录制器
        self._deactivate_auto_recorder()
        self.recording_changed.emit(False)
        scenario = self._build_scenario()
        print(f"[ActionRecorder] 停止录制，共 {len(self._steps)} 步")
        return scenario

    def _activate_auto_recorder(self):
        """激活自动交互录制器"""
        try:
            from core.common.auto_interaction_recorder import AutoInteractionRecorder
            self._auto_recorder = AutoInteractionRecorder()
            self._auto_recorder.activate()
        except Exception as e:
            print(f"[ActionRecorder] 激活自动交互录制器失败: {e}")
            self._auto_recorder = None

    def _deactivate_auto_recorder(self):
        """停用自动交互录制器"""
        if hasattr(self, '_auto_recorder') and self._auto_recorder:
            self._auto_recorder.deactivate()
            self._auto_recorder = None

    def record(self, action_name: str, params: dict):
        """记录一个 action 调用（由 @action wrapper 自动调用）

        自动将参数归入 params 字段，并从 ActionMeta 获取 description。
        """
        if not self._recording:
            return

        # 过滤不需要录制的 action
        for prefix in self.RECORD_SKIP_PREFIXES:
            if action_name.startswith(prefix):
                return
        if action_name in self.RECORD_SKIP_NAMES:
            return

        now = time.time()
        elapsed_ms = int((now - self._last_step_time) * 1000) if self._last_step_time else 0
        self._last_step_time = now

        # 构建参数字典
        record_params = {}
        for k, v in params.items():
            if v is not None and k not in ("self", "kwargs"):
                try:
                    json.dumps(v)
                    record_params[k] = v
                except (TypeError, ValueError):
                    record_params[k] = str(v)

        step = {
            "action": action_name,
            "description": f"录制: {action_name}",
            "params": record_params if record_params else {},
            "timestamp": round(now, 3),
            "wait_after": max(elapsed_ms, 100),
        }

        self._steps.append(step)
        self.step_recorded.emit(action_name, step)

    def record_step(self, action_name: str, params: dict = None,
                    description: str = "", expected_result: str = ""):
        """记录一个步骤（推荐使用此方法手动录制）

        Args:
            action_name: action 名称
            params: 参数字典
            description: 操作描述（用户可读）
            expected_result: 预期结果描述
        """
        if not self._recording:
            return

        now = time.time()
        elapsed_ms = int((now - self._last_step_time) * 1000) if self._last_step_time else 0
        self._last_step_time = now

        step = {
            "action": action_name,
            "description": description or f"录制: {action_name}",
            "params": params or {},
            "timestamp": round(now, 3),
            "wait_after": max(elapsed_ms, 100),
        }
        if expected_result:
            step["expected_result"] = expected_result

        self._steps.append(step)
        self.step_recorded.emit(action_name, step)

    def record_raw_step(self, step: dict):
        """手动写入一个原始步骤（用于拖拽等非 action 操作）"""
        if not self._recording:
            return
        now = time.time()
        elapsed_ms = int((now - self._last_step_time) * 1000) if self._last_step_time else 0
        self._last_step_time = now
        if "wait_after" not in step:
            step["wait_after"] = max(elapsed_ms, 100)
        if "timestamp" not in step:
            step["timestamp"] = round(now, 3)
        self._steps.append(step)
        action_name = step.get("action", "")
        self.step_recorded.emit(action_name, step)

    def _build_scenario(self) -> dict:
        """构建标准测试场景 dict"""
        return {
            "name": self._scenario_name,
            "description": self._description,
            "enabled": True,
            "steps": self._steps,
            "on_complete": {"action": "print_result", "message": f"{self._scenario_name} 完成"}
        }

    def save_to_file(self, file_path: str, scenario: dict = None):
        """保存为 test_*.json"""
        if scenario is None:
            scenario = self._build_scenario()
        config = {
            "test_scenarios": [scenario],
            "settings": {
                "default_wait_after_action": 200,
                "fail_on_error": False
            }
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        print(f"[ActionRecorder] 已保存到: {file_path}")

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def step_count(self) -> int:
        return len(self._steps)
