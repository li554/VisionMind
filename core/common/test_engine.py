"""
自动化测试执行引擎
支持通过配置文件定义测试流程，自动执行界面操作测试
"""

import json
import os
from typing import Dict, List, Any, Optional
from PySide6.QtCore import QObject, Signal, QTimer, Qt, QCoreApplication
from PySide6.QtGui import QColor


class TestEngine(QObject):
    """测试执行引擎"""
    
    test_started = Signal(str)  # scenario_name
    test_step_started = Signal(str, str)  # action, description
    test_step_completed = Signal(str, str, bool)  # action, description, success
    test_completed = Signal(str, bool, str)  # scenario_name, success, message
    all_tests_completed = Signal(int, int)  # success_count, total_count
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._init_state()
        self._ensure_actions_registered()

    def _get_interface(self, name):
        """通过 objectName 获取插件界面（兼容新插件架构）"""
        return self.main_window.find_interface(name)

    @property
    def annotation_interface(self):
        return self._get_interface('AnnotationInterface')

    @property
    def dashboard_interface(self):
        return self._get_interface('DashboardInterface')

    @property
    def version_interface(self):
        return self._get_interface('VersionInterface')

    @property
    def inference_interface(self):
        return self._get_interface('InferenceInterface')

    def _ensure_actions_registered(self):
        """确保 ActionRegistry 中的 action 已注册，并设置旧名称别名"""
        from core.common.action_registry import ActionRegistry
        registry = ActionRegistry.instance()

        # 注册 BuiltinActions（可能已被 main.py 无 engine 注册，此处覆盖）
        from core.common.builtin_actions import BuiltinActions
        # 查找已有 BuiltinActions 实例并设置 engine
        found = False
        for _name, (inst, _bound, _meta) in list(registry._actions.items()):
            if isinstance(inst, BuiltinActions):
                inst._engine = self
                found = True
                break
        if not found:
            builtin = BuiltinActions(engine=self)
            registry.register_instance(builtin)

        # 注册 InputSimulator（如果尚未注册）
        if not registry.has_action('input.mouse'):
            from core.common.input_simulator import InputSimulator
            simulator = InputSimulator(self.main_window)
            registry.register_instance(simulator, prefix="input")

        # 注册旧名称到新注册表的别名（兼容现有测试配置）
        self._register_aliases(registry)

    def _register_aliases(self, registry):
        """注册旧名称别名，确保兼容性

        @action 注解的 action 在 plugin_manager.register_interface() 时自动注册，
        带 prefix（如 annotation.select_image）。这里注册无前缀别名。
        """
        # 动态生成别名：prefix.xxx → xxx
        for name in list(registry._actions.keys()):
            if '.' in name:
                short_name = name.split('.', 1)[1]
                if short_name not in registry._actions and short_name not in registry._aliases:
                    registry.register_alias(short_name, name)

        # 兼容别名：测试配置常使用 "action": "assert"，对应注册名 assert.generic
        if registry.has_action('assert.generic'):
            registry.register_alias('assert', 'assert.generic')

    def _init_state(self):
        self.scenarios = []
        self.current_scenario = None
        self.current_step_index = 0
        self.loop_index = 0
        self.in_loop = False
        self.loop_steps = []
        self.loop_count = 0
        self._wait_duration = None  # 由 wait action 设置的等待时长

        self.success_count = 0
        self.error_count = 0
        self.results = []

        # 步骤级统计（用于检测未知动作、执行失败等，不受场景级 fail_on_error 影响）
        self.step_total = 0
        self.step_failed = 0
        self.step_results = []  # 每步执行结果明细

        self._projects = []
        self._variables = {}  # 存储用户定义的变量

        # 计时器存储
        self._timers = {}

        # 回放辅助组件
        self._dialog_interceptor = None  # ReplayDialogInterceptor
        self._step_panel = None  # ReplayStepPanel
        self._is_paused = False
        self._scenario_step_failed_start = 0  # 当前场景开始时的步骤失败数
    
    def load_scenarios(self, config_path: str) -> bool:
        """加载测试场景配置
        
        支持加载单个文件或目录中的所有json文件
        """
        try:
            if os.path.isdir(config_path):
                # 加载目录中的所有json文件
                all_scenarios = []
                all_settings = {}
                
                for filename in sorted(os.listdir(config_path)):
                    if filename.endswith('.json') and filename.startswith('test_'):
                        file_path = os.path.join(config_path, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8-sig') as f:
                                config = json.load(f)
                            
                            scenarios = config.get('test_scenarios', [])
                            settings = config.get('settings', {})
                            
                            # 为每个场景添加来源标记
                            for scenario in scenarios:
                                scenario['_source_file'] = filename
                            
                            all_scenarios.extend(scenarios)
                            all_settings.update(settings)
                            
                            print(f"[TestEngine] 从 {filename} 加载了 {len(scenarios)} 个场景")
                        except Exception as e:
                            print(f"[TestEngine] 加载 {filename} 失败: {e}")
                
                self.scenarios = [s for s in all_scenarios if s.get('enabled', True)]
                self.settings = all_settings
                
                print(f"[TestEngine] 总共加载了 {len(self.scenarios)} 个测试场景")
                return True
            else:
                # 加载单个文件
                with open(config_path, 'r', encoding='utf-8-sig') as f:
                    config = json.load(f)
                
                self.scenarios = config.get('test_scenarios', [])
                self.settings = config.get('settings', {})
                
                self.scenarios = [s for s in self.scenarios if s.get('enabled', True)]
                
                print(f"[TestEngine] 从 {config_path} 加载了 {len(self.scenarios)} 个测试场景")
                return True
        except Exception as e:
            print(f"[TestEngine] 加载配置失败: {e}")
            return False

    def load_scenario_dict(self, scenario: dict, settings: dict = None):
        """从 dict 加载录制结果作为测试场景

        Args:
            scenario: ActionRecorder.stop() 返回的场景 dict
            settings: 可选的测试设置
        """
        if settings:
            self.settings = settings
        else:
            self.settings = {"default_wait_after_action": 200, "fail_on_error": False}

        if scenario.get('enabled', True):
            self.scenarios = [scenario]
        else:
            self.scenarios = []

        print(f"[TestEngine] 从 dict 加载了 {len(self.scenarios)} 个测试场景")
        return True

    def get_projects(self) -> List[str]:
        """获取项目列表"""
        if not self._projects:
            from ..service.project_service import ProjectService
            service = ProjectService()
            all_projects = service.get_all_projects()
            self._projects = [p['name'] for p in all_projects]
        return self._projects
    
    def start_tests(self):
        """开始执行所有测试"""
        if not self.scenarios:
            print("[TestEngine] 没有可执行的测试场景")
            return

        self.success_count = 0
        self.error_count = 0
        self.results = []
        self.step_total = 0
        self.step_failed = 0
        self.step_results = []
        self._variables = {}
        self._is_paused = False

        # 激活对话框自动确认拦截器
        self._activate_dialog_interceptor()

        # 显示回放步骤面板
        self._show_step_panel()

        print("\n" + "=" * 60)
        print("[TestEngine] 开始执行自动化测试")
        print("=" * 60)
        print(f"测试场景数量: {len(self.scenarios)}")
        print("=" * 60)

        self._run_next_scenario()
    
    def _run_next_scenario(self):
        """执行下一个测试场景"""
        if not self.scenarios:
            self._on_all_tests_completed()
            return

        self.current_scenario = self.scenarios.pop(0)
        scenario_name = self.current_scenario.get('name', 'Unknown')

        print(f"\n{'=' * 60}")
        print(f"[TestEngine] 开始场景: {scenario_name}")
        print(f"描述: {self.current_scenario.get('description', '')}")
        print("=" * 60)

        self.test_started.emit(scenario_name)
        self.current_step_index = 0
        self.loop_index = 0
        self.in_loop = False
        # 记录场景开始时的步骤失败数，用于判断本场景是否有步骤失败
        self._scenario_step_failed_start = self.step_failed

        # 更新步骤面板
        steps = self.current_scenario.get('steps', [])
        if self._step_panel:
            self._step_panel.set_steps(steps, scenario_name)

        if steps:
            self._execute_steps(steps)
        else:
            self._start_loop()
    
    def _execute_steps(self, steps: List[Dict]):
        """执行步骤列表"""
        if self.current_step_index >= len(steps):
            if self.in_loop:
                self.loop_index += 1
                if self.loop_index < self.loop_count:
                    self.current_step_index = 0
                    QTimer.singleShot(100, lambda: self._execute_steps(steps))
                else:
                    self._on_scenario_completed(True, "循环执行完成")
            else:
                self._start_loop()
            return
        
        step = steps[self.current_step_index]
        step = self._resolve_variables(step)
        
        action = step.get('action', '')
        description = step.get('description', '')
        
        self.test_step_started.emit(action, description)
        print(f"  [Step] {action}: {description}")

        # 更新步骤面板：当前步骤
        if self._step_panel:
            self._step_panel.set_current_step(self.current_step_index)

        # 设置对话框拦截器的当前 action（用于自动填入值）
        if self._dialog_interceptor and self._dialog_interceptor.is_active:
            # 新格式优先从 params 字段取参，兼容旧格式
            interceptor_params = step.get('params', {})
            if not interceptor_params:
                interceptor_params = {k: v for k, v in step.items() if k not in ('action', 'description', 'wait_after', 'params', 'expected_result')}
            self._dialog_interceptor.set_current_action(action, interceptor_params)

            # 预读后续连续的对话框操作步骤，传给 interceptor
            # 当 action 内部调用 dialog.exec_() 阻塞事件循环时，
            # interceptor 能在对话框 Show 时自动执行这些操作
            DIALOG_OP_ACTIONS = self._dialog_interceptor.DIALOG_OP_ACTIONS
            pending_ops = []
            next_idx = self.current_step_index + 1
            while next_idx < len(steps):
                next_step = steps[next_idx]
                next_action = next_step.get('action', '')
                if next_action in DIALOG_OP_ACTIONS:
                    op_params = next_step.get('params', {})
                    if not op_params:
                        op_params = {k: v for k, v in next_step.items()
                                     if k not in ('action', 'description', 'wait_after', 'params', 'expected_result')}
                    pending_ops.append({
                        'action': next_action,
                        'params': op_params,
                        'description': next_step.get('description', ''),
                    })
                    next_idx += 1
                else:
                    break
            self._dialog_interceptor.set_pending_ops(pending_ops)

        success = False
        error_msg = ""

        from core.common.action_registry import ActionRegistry
        registry = ActionRegistry.instance()
        if registry.has_action(action):
            try:
                # 新格式优先从 params 字段取参，兼容旧格式
                params = step.get('params', {})
                if not params:
                    params = {k: v for k, v in step.items() if k not in ('action', 'description', 'wait_after', 'params', 'expected_result')}
                result = registry.call(action, params=params)
                success = result if isinstance(result, bool) else True
                if not success:
                    error_msg = f"动作返回失败: {action}"
            except Exception as e:
                print(f"    [Error] ActionRegistry 调用异常: {e}")
                import traceback
                traceback.print_exc()
                success = False
                error_msg = f"动作执行异常: {action} - {e}"
        else:
            print(f"    [Error] 未知动作: {action}")
            success = False
            error_msg = f"未知动作: {action}"

        self.test_step_completed.emit(action, description, success)

        # 更新步骤面板：标记完成
        if self._step_panel:
            self._step_panel.set_step_completed(self.current_step_index, success)

        # 清除对话框拦截器的当前 action
        if self._dialog_interceptor and self._dialog_interceptor.is_active:
            self._dialog_interceptor.clear_current_action()

        # 步骤级失败检测：未知动作、执行异常、返回 False 都计入失败
        self.step_total += 1
        if not success:
            self.step_failed += 1
            scenario_name = self.current_scenario.get('name', 'Unknown') if self.current_scenario else 'Unknown'
            self.step_results.append({
                'scenario': scenario_name,
                'step_index': self.current_step_index,
                'action': action,
                'description': description,
                'success': False,
                'message': error_msg
            })
            print(f"    [Step FAIL] {error_msg}")
            # fail_on_error 为 True 时立即终止当前场景
            if self.settings.get('fail_on_error', False):
                self._on_scenario_completed(False, error_msg)
                return
        
        wait_time = step.get('wait_after', self.settings.get('default_wait_after_action', 200))
        # 如果 wait action 设置了等待时长，使用该时长
        if self._wait_duration is not None:
            wait_time = self._wait_duration
            self._wait_duration = None
        self.current_step_index += 1
        QTimer.singleShot(wait_time, lambda: self._execute_steps(steps))
    
    def _resolve_variables(self, step: Dict) -> Dict:
        """解析步骤中的变量"""
        step_str = json.dumps(step)
        projects = self.get_projects()
        project_count = len(projects)
        
        # 替换简单变量
        step_str = step_str.replace('${loop_index}', str(self.loop_index))
        step_str = step_str.replace('${project_count}', str(project_count))
        
        # 替换用户自定义变量 ${var_name}
        import re
        for var_name, var_value in self._variables.items():
            step_str = step_str.replace(f'${{{var_name}}}', str(var_value))
        
        # 处理表达式 ${loop_index % project_count}
        expr_pattern = r'\$\{([^}]+)\}'
        
        def eval_expr(match):
            expr = match.group(1)
            # 替换变量名
            expr = expr.replace('loop_index', str(self.loop_index))
            expr = expr.replace('project_count', str(project_count))
            # 替换用户变量
            for var_name, var_value in self._variables.items():
                expr = expr.replace(var_name, str(var_value))
            try:
                # 安全地计算简单算术表达式
                result = eval(expr, {"__builtins__": {}}, {})
                return str(result)
            except:
                return match.group(0)
        
        step_str = re.sub(expr_pattern, eval_expr, step_str)
        
        result = json.loads(step_str)
        
        # 确保 project_index 是整数
        if 'project_index' in result:
            try:
                result['project_index'] = int(result['project_index'])
            except (ValueError, TypeError):
                result['project_index'] = 0
        
        # 确保 image_index 是整数
        if 'image_index' in result:
            try:
                result['image_index'] = int(result['image_index'])
            except (ValueError, TypeError):
                result['image_index'] = 0
        
        return result
    
    def _start_loop(self):
        """开始循环执行"""
        loop_config = self.current_scenario.get('loop')
        if not loop_config:
            self._on_scenario_completed(True, "场景执行完成")
            return
        
        self.loop_count = loop_config.get('count', 1)
        self.loop_steps = loop_config.get('steps', [])
        self.loop_index = 0
        self.in_loop = True
        self.current_step_index = 0
        
        print(f"\n  [Loop] 开始循环执行，次数: {self.loop_count}")
        
        if self.loop_steps:
            QTimer.singleShot(100, lambda: self._execute_steps(self.loop_steps))
        else:
            self._on_scenario_completed(True, "循环步骤为空")
    
    def _on_scenario_completed(self, success: bool, message: str):
        """场景执行完成"""
        scenario_name = self.current_scenario.get('name', 'Unknown') if self.current_scenario else 'Unknown'

        # 如果本场景执行期间有任何步骤失败，整体标记为失败
        scenario_step_failed = self.step_failed - getattr(self, '_scenario_step_failed_start', 0)
        if success and scenario_step_failed > 0:
            success = False
            message = f"{message}（含 {scenario_step_failed} 个失败步骤）"

        if success:
            self.success_count += 1
            print(f"\n[TestEngine] 场景 '{scenario_name}' 完成: {message}")
        else:
            self.error_count += 1
            print(f"\n[TestEngine] 场景 '{scenario_name}' 失败: {message}")

        self.results.append({
            'name': scenario_name,
            'success': success,
            'message': message
        })

        self.test_completed.emit(scenario_name, success, message)
        
        on_complete = self.current_scenario.get('on_complete') if self.current_scenario else None
        if on_complete:
            from core.common.action_registry import ActionRegistry
            registry = ActionRegistry.instance()
            action = on_complete.get('action', '')
            if registry.has_action(action):
                try:
                    registry.call(action, params=on_complete)
                except Exception as e:
                    print(f"  [Error] on_complete 执行失败: {e}")
        
        QTimer.singleShot(500, self._run_next_scenario)
    
    def _on_all_tests_completed(self):
        """所有测试完成"""
        # 停用对话框拦截器
        self._deactivate_dialog_interceptor()

        # 隐藏步骤面板
        if self._step_panel:
            self._step_panel.hide_panel()

        total = self.success_count + self.error_count
        print("\n" + "=" * 60)
        print("[TestEngine] 所有测试完成!")
        print("=" * 60)
        print(f"总场景数: {total}")
        print(f"成功: {self.success_count}")
        print(f"失败: {self.error_count}")
        if total > 0:
            print(f"场景成功率: {self.success_count / total * 100:.1f}%")
        print("-" * 60)
        print(f"步骤总数: {self.step_total}")
        print(f"步骤失败: {self.step_failed}")
        if self.step_total > 0:
            print(f"步骤成功率: {(self.step_total - self.step_failed) / self.step_total * 100:.1f}%")
        # 打印失败步骤明细
        if self.step_results:
            print("-" * 60)
            print("失败步骤明细:")
            for r in self.step_results:
                print(f"  [{r['scenario']}] 步骤#{r['step_index']} {r['action']}: {r['message']}")
        print("=" * 60)

        self.all_tests_completed.emit(self.success_count, total)

        # 自动关闭程序（有任一失败都返回非0退出码）
        if self.settings.get('auto_close_on_complete', False):
            print("\n[TestEngine] 自动关闭程序...")
            import sys
            from PySide6.QtWidgets import QApplication
            QApplication.quit()
            sys.exit(0 if (self.error_count == 0 and self.step_failed == 0) else 1)
    
    # ==================== TestEngine 级别的内置 action（仅保留调度所需） ====================

    # ==================== 回放辅助方法 ====================

    def _activate_dialog_interceptor(self):
        """激活回放对话框自动确认拦截器"""
        try:
            from core.common.replay_dialog_interceptor import ReplayDialogInterceptor
            auto_accept_delay = self.settings.get('auto_accept_delay', 800)
            self._dialog_interceptor = ReplayDialogInterceptor(
                auto_accept_delay=auto_accept_delay
            )
            self._dialog_interceptor.activate()
            print("[TestEngine] 对话框自动确认拦截器已激活")
        except Exception as e:
            print(f"[TestEngine] 激活对话框拦截器失败: {e}")

    def _deactivate_dialog_interceptor(self):
        """停用回放对话框自动确认拦截器"""
        if self._dialog_interceptor:
            self._dialog_interceptor.deactivate()
            self._dialog_interceptor = None
            print("[TestEngine] 对话框自动确认拦截器已停用")

    def _show_step_panel(self):
        """显示回放步骤进度面板"""
        try:
            from core.common.replay_step_panel import ReplayStepPanel
            self._step_panel = ReplayStepPanel(parent=self.main_window)
            self._step_panel.stop_requested.connect(self._on_stop_requested)
            self._step_panel.pause_requested.connect(self._on_pause_requested)
            # 在主窗口右上角显示
            if self.main_window:
                x = self.main_window.x() + self.main_window.width() - 320
                y = self.main_window.y() + 60
            else:
                x, y = 100, 100
            self._step_panel.show_panel(x, y)
            print("[TestEngine] 回放步骤面板已显示")
        except Exception as e:
            print(f"[TestEngine] 显示步骤面板失败: {e}")

    def _on_stop_requested(self):
        """用户点击停止按钮"""
        print("[TestEngine] 用户请求停止回放")
        self.scenarios.clear()
        # 标记当前步骤为失败
        if self._step_panel and self.current_step_index >= 0:
            self._step_panel.set_step_completed(self.current_step_index, False)
        self._on_all_tests_completed()

    def _on_pause_requested(self, is_paused: bool):
        """用户点击暂停/继续按钮"""
        self._is_paused = is_paused
        state = "暂停" if is_paused else "继续"
        print(f"[TestEngine] 用户请求{state}回放")

