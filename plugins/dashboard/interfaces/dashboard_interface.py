import os
import sys
import traceback

from PySide6.QtCore import Qt, Signal, QSize, QThread, QObject, Slot, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QFileDialog, QDialog, QFrame
from qfluentwidgets import (PrimaryPushButton, PushButton, FluentIcon,
                            BodyLabel, CaptionLabel, SubtitleLabel, TransparentPushButton,
                            LineEdit, ComboBox, InfoBar, MessageBox, TransparentToolButton)
from core.common.custom_card import SolidCardWidget as CardWidget

from core.common.widgets.base import ProgressDialog
from core.common.settings import settings
from core.common.action_registry import action
from ..services.dashboard_service import DashboardService


class DatasetImportWorker(QObject):
    progress = Signal(int)
    finished = Signal(dict, str)  # result dict and project_name
    error = Signal(str)

    def __init__(self, service, source_dir, project_name, output_root):
        super().__init__()
        self.service = service
        self.source_dir = source_dir
        self.project_name = project_name
        self.output_root = output_root

    def run(self):
        try:
            res = self.service.import_and_normalize(
                self.source_dir,
                project_name=self.project_name,
                output_root=self.output_root,
                force=True,
                progress_callback=self.progress.emit
            )
            self.finished.emit(res, self.project_name)
        except Exception as e:
            if settings.DEBUG:
                raise
            self.error.emit(str(e))

class ProjectCard(CardWidget):
    project_clicked = Signal(str)
    import_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, project_info, parent=None):
        super().__init__(parent)
        self.project_name = project_info['name']
        from core.service.project_service import normalize_task_type, TASK_TYPE_DISPLAY
        self.task_type = normalize_task_type(project_info.get('task_type', 'det'))
        self.setFixedHeight(220)
        self.setCursor(Qt.PointingHandCursor)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)

        # Header with Title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        self.title_label = SubtitleLabel(self.project_name)
        self.title_label.setProperty("class", "ProjectTitle")
        title_layout.addWidget(self.title_label)

        task_text = TASK_TYPE_DISPLAY.get(self.task_type, "目标检测")
        self.type_label = CaptionLabel(task_text)
        self.type_label.setProperty("class", "ProjectSubtitle")
        title_layout.addWidget(self.type_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        # Delete button - Using ToolButton for better icon display
        self.del_btn = TransparentToolButton(FluentIcon.DELETE, self)
        self.del_btn.setFixedSize(32, 32)
        self.del_btn.setIconSize(QSize(16, 16))
        self.del_btn.setToolTip("删除项目")
        self.del_btn.setObjectName("ProjectDeleteButton")
        self.del_btn.clicked.connect(self._on_delete_clicked)
        header_layout.addWidget(self.del_btn)

        self.layout.addLayout(header_layout)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        line.setObjectName("ProjectDivider")
        self.layout.addWidget(line)

        # Content Info
        info_layout = QGridLayout()
        info_layout.setVerticalSpacing(8)

        def add_info(row, label_text, value_text):
            lbl = CaptionLabel(label_text)
            lbl.setProperty("class", "ProjectInfoLabel")
            val = BodyLabel(str(value_text))
            val.setProperty("class", "ProjectInfoValue")
            info_layout.addWidget(lbl, row, 0)
            info_layout.addWidget(val, row, 1)

        add_info(0, "图片数量", project_info.get('image_count', 0))
        add_info(1, "创建时间", project_info.get('created_at', '').split(' ')[0])

        self.layout.addLayout(info_layout)
        self.layout.addStretch()

        # Footer Actions
        footer = QHBoxLayout()
        footer.setSpacing(10)

        self.import_btn = TransparentPushButton(FluentIcon.ADD, "导入图片", self)
        self.import_btn.setObjectName("ProjectImportButton")
        self.import_btn.clicked.connect(self._on_import_clicked)
        footer.addWidget(self.import_btn)

        footer.addStretch()

        self.enter_btn = PrimaryPushButton("进入项目", self)
        self.enter_btn.setFixedWidth(100)
        self.enter_btn.clicked.connect(self._on_enter_clicked)
        footer.addWidget(self.enter_btn)

        self.layout.addLayout(footer)

    def _on_delete_clicked(self):
        """删除按钮点击处理 - 带异常捕获"""
        try:
            self.delete_requested.emit(self.project_name)
        except Exception as e:
            print(f"[ProjectCard] Error in delete click: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def _on_import_clicked(self):
        """导入按钮点击处理 - 带异常捕获"""
        try:
            self.import_requested.emit(self.project_name)
        except Exception as e:
            print(f"[ProjectCard] Error in import click: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def _on_enter_clicked(self):
        """进入按钮点击处理 - 带异常捕获"""
        try:
            self.project_clicked.emit(self.project_name)
        except Exception as e:
            print(f"[ProjectCard] Error in enter click: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def mouseReleaseEvent(self, event):
        """鼠标释放事件 - 带异常捕获"""
        try:
            super().mouseReleaseEvent(event)
            self.project_clicked.emit(self.project_name)
        except Exception as e:
            print(f"[ProjectCard] Error in mouse release: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

class DashboardInterface(QFrame):
    project_selected = Signal(str)

    def __init__(self, project_service, parent=None):
        super().__init__(parent)
        self.dashboard_service = DashboardService(project_service)
        self.setObjectName("DashboardInterface")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(30, 30, 30, 30)

        # Header
        self.header = QHBoxLayout()
        self.title_lbl = SubtitleLabel("项目大厅")
        self.title_lbl.setProperty("class", "InterfaceTitle")
        self.header.addWidget(self.title_lbl)
        self.header.addStretch(1)
        self.btn_create = PrimaryPushButton(FluentIcon.ADD, "创建新项目", self)
        self.btn_create.clicked.connect(self.show_create_dialog)
        self.header.addWidget(self.btn_create)
        self.layout.addLayout(self.header)

        # Scroll area for project cards
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("DashboardScroll")

        self._init_project_grid()

    def get_menubar_config(self):
        """返回项目大厅的菜单配置"""
        return [
            {
                'title': '文件(&F)',
                'actions': [
                    ('创建项目', self.show_create_dialog),
                    None,  # 分隔符
                    ('导入图片', self._menu_import_images),
                ]
            },
            {
                'title': '录制(&R)',
                'actions': [
                    ('开始录制', self._menu_start_recording),
                    ('停止录制', self._menu_stop_recording),
                ]
            },
            {
                'title': '视图(&V)',
                'actions': [
                    ('刷新项目列表', self.refresh_projects),
                ]
            }
        ]

    def _menu_import_images(self):
        """菜单：导入图片（需要先选择项目）"""
        projects = self.dashboard_service.get_all_projects()
        if not projects:
            InfoBar.warning("提示", "请先创建项目", parent=self)
            return

        # 弹出项目选择对话框
        from PySide6.QtWidgets import QInputDialog
        names = [p['name'] for p in projects]
        name, ok = QInputDialog.getItem(self, "导入图片", "选择项目", names, 0, False)
        if ok and name:
            self.import_images(name)

    def _menu_start_recording(self):
        """菜单：开始录制"""
        from core.common.action_recorder import ActionRecorder
        recorder = ActionRecorder.instance()
        if recorder.is_recording:
            InfoBar.warning("录制", "已经在录制中", parent=self)
            return
        recorder.start(scenario_name="录制流程")
        recorder.record_raw_step({
            "action": "switch_to",
            "params": {"target": "dashboard"},
            "comment": "切换到项目大厅"
        })
        InfoBar.info("录制", "开始录制", parent=self)

    def _menu_stop_recording(self):
        """菜单：停止录制"""
        # 检查是否已有 ProjectService 实例，没有则创建
        if not hasattr(self, 'project_service') or self.project_service is None:
            from core.service.project_service import ProjectService
            self.project_service = ProjectService()
        result = self.project_service.stop_recording()
        if not result.get("success"):
            InfoBar.warning("录制", result.get("error", "当前未在录制"), parent=self)
            return
        scenario = result.get("scenario", {})
        step_count = result.get("step_count", 0)
        if step_count == 0:
            InfoBar.info("录制", "录制已停止，无操作记录", parent=self)
            return
        # 保存录制结果（UI 操作）
        import os
        from datetime import datetime
        from PySide6.QtWidgets import QFileDialog
        from core.common.action_recorder import ActionRecorder
        default_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        scenarios_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_scenarios")
        os.makedirs(scenarios_dir, exist_ok=True)
        default_path = os.path.join(scenarios_dir, default_name)
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存录制流程", default_path, "JSON文件 (*.json)"
        )
        if file_path:
            recorder = ActionRecorder.instance()
            recorder.save_to_file(file_path, scenario)
            InfoBar.success("录制已保存", f"共 {step_count} 步，已保存到 {os.path.basename(file_path)}", parent=self)
        else:
            InfoBar.info("录制", f"录制已停止，共 {step_count} 步（未保存）", parent=self)

    def _init_project_grid(self):
        """初始化项目网格"""
        self.container = QFrame()
        self.container.setObjectName("projectContainer")
        self.container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(20)
        self.grid.setContentsMargins(0, 10, 0, 0)

        self.scroll.setWidget(self.container)
        self.layout.addWidget(self.scroll)

        # resize 防抖:拖拽侧栏等连续 resize 时每个像素都全量重建卡片,
        # 重建与嵌套 resize 互相打断,最终网格可能停留在空态(卡片"消失")
        self._resize_debounce = QTimer(self)
        self._resize_debounce.setSingleShot(True)
        self._resize_debounce.setInterval(150)
        self._resize_debounce.timeout.connect(self._on_resize_settled)
        self._last_cols = None

        self.refresh_projects()

    def _compute_cols(self) -> int:
        width = self.width()
        if width > 1600:
            return 4
        if width > 1200:
            return 3
        if width > 800:
            return 2
        return 1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_debounce.start()

    def _on_resize_settled(self):
        # 列数没变化就不重建,避免无意义的全量刷新
        cols = self._compute_cols()
        if cols != self._last_cols:
            self.refresh_projects()

    @action("project.refresh_projects", description="刷新当前显示的项目列表。\n- 无参数\n- 从数据库/配置重新加载所有项目\n- 刷新后更新项目卡片显示", category="项目", scope="ui")
    def refresh_projects(self):
        # Clear grid
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        projects = self.dashboard_service.get_all_projects()
        if not projects:
            self._last_cols = None
            return "错误: 没有找到项目"

        # Dynamic columns based on width
        cols = self._compute_cols()
        self._last_cols = cols

        for i, info in enumerate(projects):
            card = ProjectCard(info)
            card.project_clicked.connect(self._on_card_clicked)
            card.import_requested.connect(self._on_card_import)
            card.delete_requested.connect(self._on_card_delete)
            self.grid.addWidget(card, i // cols, i % cols)

        # Add empty widgets to keep cards aligned to the top-left if there are few projects
        # This is a bit of a hack for QGridLayout, but works
        if projects:
            row_count = (len(projects) + cols - 1) // cols
            self.grid.setRowStretch(row_count, 1)
            for c in range(cols):
                self.grid.setColumnStretch(c, 1)

    @action("nav.click_project_card", description="点击指定项目的卡片以进入该项目。\n- project_name: 项目名称\n- project_index: 项目索引（从0开始），与 project_name 二选一\n- 进入项目后会切换到标注界面", category="导航", params={"project_name": "str", "project_index": "int"})
    def _on_card_clicked(self, project_name: str = "", project_index: int = -1):
        """处理卡片点击事件 - 带详细异常捕获"""
        try:
            if not project_name and project_index < 0:
                return "错误: 请提供 project_name 或有效的 project_index"

            # action调用路径：通过索引查找项目名
            if not project_name and project_index >= 0:
                all_projects = self.dashboard_service.get_all_projects()
                projects = [p['name'] for p in all_projects]
                if project_index < len(projects):
                    project_name = projects[project_index]
                else:
                    return "错误: project_index %d 超出范围（共 %d 个项目）" % (project_index, len(projects))

            # 检查是否正在导入中（import_thread 不为 None 表示导入尚未完成）
            if hasattr(self, 'import_thread') and self.import_thread is not None:
                InfoBar.warning("导入进行中", "请等待图片导入完成后再进入项目", parent=self)
                return

            print(f"[Dashboard] Project card clicked: {project_name}")
            self.project_selected.emit(project_name)
        except Exception as e:
            print(f"[Dashboard] Error handling card click for '{project_name}': {e}", file=sys.stderr)
            traceback.print_exc()
            return "错误: 点击项目卡片失败: %s" % e

    def _on_card_import(self, project_name):
        """处理卡片导入请求 - 带详细异常捕获"""
        try:
            print(f"[Dashboard] Import requested for project: {project_name}")
            self.import_images(project_name)
        except Exception as e:
            print(f"[Dashboard] Error handling import for '{project_name}': {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def _on_card_delete(self, project_name):
        """处理卡片删除请求 - 带详细异常捕获"""
        try:
            print(f"[Dashboard] Delete requested for project: {project_name}")
            self.delete_project(project_name)
        except Exception as e:
            print(f"[Dashboard] Error handling delete for '{project_name}': {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    @action("project.delete_project", description="删除指定的项目及其所有数据。\n- project_name: 要删除的项目名称\n- skip_confirm: 是否跳过确认对话框（默认 false）\n- 谨慎使用，删除不可恢复", category="项目", params={"project_name": "str", "skip_confirm": "bool"}, scope="ui")
    def delete_project(self, project_name: str = "", skip_confirm: bool = False):
        if not project_name:
            return "错误: 项目名称不能为空"
        if not skip_confirm:
            msg = MessageBox(
                "删除项目",
                f"确定要删除项目 '{project_name}' 吗？此操作不可撤销，所有图片和标注都将被删除。",
                self
            )
            msg.yesButton.setText("确定删除")
            msg.cancelButton.setText("取消")
            if not msg.exec():
                return

        if self.dashboard_service.delete_project(project_name):
            InfoBar.success("删除成功", f"项目 {project_name} 已删除", duration=2000, parent=self)
            parent = self.parent()
            if parent and hasattr(parent, "on_project_deleted"):
                parent.on_project_deleted(project_name)
            self.refresh_projects()
        else:
            InfoBar.error("删除失败", "无法删除项目文件夹", parent=self)

    @action("project.import_images", background=True, description="从指定目录导入图片到项目中。\n- project_name: 目标项目名称\n- source_dir: 图片源目录路径\n- 导入的图片会自动复制到项目目录", category="项目", params={"project_name": "str", "source_dir": "str"}, scope="ui")
    def import_images(self, project_name: str = "", source_dir: str = ""):
        """导入图片到项目"""
        if not project_name:
            return "错误: 项目名称不能为空"

        # source_dir 有值时跳过文件选择对话框
        if source_dir:
            dir_path = source_dir
        else:
            dir_path = QFileDialog.getExistingDirectory(self, "选择导入目录", "")
            if not dir_path:
                return

        # 获取项目路径
        project_info = self.dashboard_service.load_project(project_name)
        if not project_info:
            InfoBar.error("错误", "无法加载项目信息", parent=self)
            return

        project_path = project_info['path']

        # 提示用户
        msg = MessageBox(
            "导入数据",
            f"是否将 '{os.path.basename(dir_path)}' 导入并解析到项目 '{project_name}'？\n\n"
            "系统将自动识别数据集结构（YOLO/LabelMe/普通目录）并完成规范化导入。",
            self
        )
        msg.yesButton.setText("开始导入")
        msg.cancelButton.setText("取消")

        if msg.exec():
            # 创建进度对话框
            self.progress_dialog = ProgressDialog("正在导入数据...", "请稍候，正在解析并拷贝图片...", self)
            self.progress_dialog.show()

            # 创建工作线程
            self.import_thread = QThread()
            self.import_worker = DatasetImportWorker(
                self.dashboard_service,
                dir_path,
                project_name,
                os.path.dirname(project_path)
            )
            self.import_worker.moveToThread(self.import_thread)

            # 连接信号 - 使用 @Slot 装饰器的方法而不是 lambda
            self.import_thread.started.connect(self.import_worker.run)
            self.import_worker.progress.connect(self.progress_dialog.setValue)
            self.import_worker.finished.connect(self.on_import_finished)
            self.import_worker.error.connect(self.on_import_error)

            # 注意：不在这里连接 quit()，而是在 on_import_finished/on_import_error 中处理
            # 这样可以确保线程真正结束后再清理状态

            self.import_thread.start()

    @Slot(dict, str)
    def on_import_finished(self, res, project_name):
        # 先确保线程真正退出，再更新 UI
        if hasattr(self, 'import_thread') and self.import_thread:
            self.import_thread.quit()
            self.import_thread.wait(5000)
            self.import_thread = None
            self.import_worker = None

        self.progress_dialog.close()

        if res['status'] in ['success', 'exists']:
            if res['status'] == 'success':
                InfoBar.success("导入成功", res['message'], duration=3000, parent=self)
            else:
                InfoBar.info("提示", res['message'], duration=3000, parent=self)

            # 刷新项目列表以更新图片数量
            self.dashboard_service.update_project_info(project_name, {"image_count": self.dashboard_service.get_image_count(project_name)})
            self.refresh_projects()
        else:
            InfoBar.error("导入失败", res['message'], parent=self)

    @Slot(str)
    def on_import_error(self, error_msg):
        # 先确保线程真正退出，再更新 UI
        if hasattr(self, 'import_thread') and self.import_thread:
            self.import_thread.quit()
            self.import_thread.wait(5000)
            self.import_thread = None
            self.import_worker = None

        self.progress_dialog.close()
        InfoBar.error("导入出错", error_msg, parent=self)

    @action("project.create_project", description="创建一个新的标注项目。\n- project_name: 项目名称\n- task_type: 任务类型，可选 seg(分割)/det(检测)/obb(旋转框)\n- 创建后自动切换到新项目", category="项目", params={"project_name": "str", "task_type": "str"}, scope="ui")
    def show_create_dialog(self, project_name: str = "", task_type: str = ""):
        """创建新项目 — 有参数时直接创建，无参数时弹对话框"""
        if project_name or task_type:
            if not project_name:
                return "错误: 项目名称不能为空"
            if not task_type:
                return "错误: 任务类型不能为空"
            if task_type not in ("seg", "det", "obb"):
                return "错误: 无效的任务类型 '%s'，可选值: seg(分割), det(检测), obb(旋转框)" % task_type
            res = self.dashboard_service.create_project(project_name, task_type)
            if res['status'] == 'success':
                InfoBar.success("创建成功", f"项目 {project_name} 已就绪", duration=2000, parent=self)
                self.refresh_projects()
            else:
                InfoBar.error("创建失败", res['message'], parent=self)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("新建项目")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Title
        title = SubtitleLabel("创建新视觉项目")
        layout.addWidget(title)

        # Name
        name_layout = QVBoxLayout()
        name_layout.setSpacing(8)
        name_layout.addWidget(BodyLabel("项目名称"))
        name_input = LineEdit()
        name_input.setPlaceholderText("例如: Defect_Detection_2024")
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)

        # Task Type
        task_layout = QVBoxLayout()
        task_layout.setSpacing(8)
        task_layout.addWidget(BodyLabel("任务类型"))
        task_combo = ComboBox()
        task_combo.addItems(["目标检测", "目标分割", "旋转目标检测"])
        task_layout.addWidget(task_combo)
        layout.addLayout(task_layout)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()

        cancel_btn = PushButton("取消")
        ok_btn = PrimaryPushButton("立即创建")

        cancel_btn.clicked.connect(dialog.reject)
        ok_btn.clicked.connect(dialog.accept)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        if dialog.exec():
            name = name_input.text().strip()
            if name:
                task_type = ("det", "seg", "obb")[task_combo.currentIndex()] \
                    if 0 <= task_combo.currentIndex() < 3 else "seg"
                res = self.dashboard_service.create_project(name, task_type)
                if res['status'] == 'success':
                    InfoBar.success("创建成功", f"项目 {name} 已就绪", duration=2000, parent=self)
                    self.refresh_projects()
                else:
                    InfoBar.error("创建失败", res['message'], parent=self)
