"""
Markdown 文档编辑器窗口 — 阅读/编辑/保存 reports/ 目录下的文档

- 编辑区：QPlainTextEdit（Markdown 源文本，可自由修改）
- 预览区：QTextBrowser（Markdown 渲染后的 HTML 预览，随编辑实时刷新）
- 工具栏：新建 / 打开 / 保存 / 退出
- 单例复用：get_or_create() 保证多次打开共用同一窗口
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QPlainTextEdit, QTextBrowser, QFileDialog, QLabel, QWidget,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, StrongBodyLabel, CaptionLabel, InfoBar,
)


def _md_to_html(text: str) -> str:
    """将 Markdown 文本转换为带样式的 HTML"""
    if not text:
        return ""
    try:
        import markdown as _md
        html = _md.markdown(
            text,
            extensions=["fenced_code", "nl2br", "tables"],
        )
        html = html.replace(
            "<pre>",
            '<pre style="background:rgba(0,0,0,0.25);padding:8px;border-radius:6px;'
            'overflow-x:auto;font-size:12px;font-family:Consolas,monospace;">'
        )
        html = html.replace(
            "<code>",
            '<code style="background:rgba(0,0,0,0.15);padding:1px 4px;border-radius:3px;font-size:12px;">'
        )
        html = html.replace("</pre><code>", "</pre>")
        html = html.replace(
            "<table>",
            '<table style="border-collapse:collapse;width:100%;font-size:12px;">'
        )
        html = html.replace(
            "<td>",
            '<td style="border:1px solid rgba(255,255,255,0.15);padding:4px 8px;">'
        )
        html = html.replace(
            "<th>",
            '<th style="border:1px solid rgba(255,255,255,0.15);padding:4px 8px;font-weight:600;">'
        )
        html = html.replace(
            "<blockquote>",
            '<blockquote style="border-left:3px solid #2563eb;padding:4px 12px;margin:8px 0;color:#9ca3af;">'
        )
        return html
    except Exception:
        return text.replace("\n", "<br>")


class DocumentEditorDialog(QDialog):
    """Markdown 文档编辑器对话框"""

    _instance = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DocumentEditorDialog")
        self.setWindowTitle("Markdown 文档编辑器")
        self.resize(960, 640)
        self.setModal(False)
        self._current_path = ""
        self._setup_ui()
        self._load_style()

    @classmethod
    def get_or_create(cls):
        """获取共享实例（若已存在则复用并置顶）"""
        if cls._instance is None:
            cls._instance = cls()
        cls._instance.raise_()
        cls._instance.show()
        cls._instance.activateWindow()
        return cls._instance

    def _load_style(self):
        self.setStyleSheet("""
            DocumentEditorDialog {
                background-color: #1a1a1a;
                border: 1px solid #333333;
            }
            #EditorToolbar {
                background-color: #222222;
                border-bottom: 1px solid #333333;
            }
            #DocTitle {
                color: #ffffff;
                font-weight: bold;
            }
            #DocPathLabel {
                color: #9ca3af;
                font-size: 11px;
            }
            #EditorEdit {
                background-color: #141414;
                color: #d4d4d4;
                border: none;
                font-family: "Consolas", "Microsoft YaHei", monospace;
                font-size: 13px;
                padding: 8px;
            }
            #EditorPreview {
                background-color: #181818;
                color: #e5e7eb;
                border: none;
                padding: 8px;
                font-size: 13px;
            }
        """)

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 工具栏
        toolbar = QWidget()
        toolbar.setObjectName("EditorToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        toolbar_layout.setSpacing(8)

        title = StrongBodyLabel("Markdown 文档编辑器")
        title.setObjectName("DocTitle")
        toolbar_layout.addWidget(title)

        self._path_label = CaptionLabel("未保存新文档")
        self._path_label.setObjectName("DocPathLabel")
        self._path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        toolbar_layout.addWidget(self._path_label, 1)

        self._btn_new = PushButton("新建")
        self._btn_new.clicked.connect(self._on_new)
        toolbar_layout.addWidget(self._btn_new)

        self._btn_open = PushButton("打开")
        self._btn_open.clicked.connect(self._on_open_dialog)
        toolbar_layout.addWidget(self._btn_open)

        self._btn_save = PrimaryPushButton("保存")
        self._btn_save.clicked.connect(self._on_save)
        toolbar_layout.addWidget(self._btn_save)

        self._btn_close = PushButton("关闭")
        self._btn_close.clicked.connect(self.close)
        toolbar_layout.addWidget(self._btn_close)

        root.addWidget(toolbar)

        # 编辑 / 预览 分栏
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._edit = QPlainTextEdit()
        self._edit.setObjectName("EditorEdit")
        self._edit.setFont(QFont("Consolas", 11))
        splitter.addWidget(self._edit)

        self._preview = QTextBrowser()
        self._preview.setObjectName("EditorPreview")
        self._preview.setOpenExternalLinks(True)
        splitter.addWidget(self._preview)
        splitter.setSizes([460, 480])

        root.addWidget(splitter, 1)

        self._edit.textChanged.connect(self._refresh_preview)

    def open_file(self, file_path: str = ""):
        """打开指定文件；为空时尝试打开最近文档，无文档则新建"""
        if not file_path:
            from core.agent.tools.document_tools import get_recent_doc_path
            file_path = get_recent_doc_path()
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self._current_path = os.path.normpath(file_path)
                self._edit.blockSignals(True)
                self._edit.setPlainText(content)
                self._edit.blockSignals(False)
                self._path_label.setText(self._current_path)
                self._refresh_preview()
                self.setWindowTitle(f"Markdown 文档编辑器 - {os.path.basename(self._current_path)}")
                return
            except Exception as e:
                InfoBar.error("打开失败", f"{type(e).__name__}: {e}", parent=self)
        self._on_new()

    def _on_new(self):
        self._current_path = ""
        self._edit.blockSignals(True)
        self._edit.setPlainText("# 新建文档\n\n在此输入 Markdown 内容...\n")
        self._edit.blockSignals(False)
        self._path_label.setText("未保存新文档")
        self.setWindowTitle("Markdown 文档编辑器 - 未命名")
        self._refresh_preview()

    def _on_open_dialog(self):
        from core.agent.tools.document_tools import get_reports_dir
        default_dir = get_reports_dir()
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 Markdown 文档", default_dir, "Markdown (*.md);;所有文件 (*)"
        )
        if path:
            self.open_file(path)

    def _on_save(self):
        if not self._current_path:
            from core.agent.tools.document_tools import get_reports_dir
            default_name = os.path.join(get_reports_dir(), "analysis_report.md")
            path, _ = QFileDialog.getSaveFileName(
                self, "保存 Markdown 文档", default_name, "Markdown (*.md)"
            )
            if not path:
                return
            if not path.endswith(".md"):
                path += ".md"
            self._current_path = path
        try:
            os.makedirs(os.path.dirname(self._current_path), exist_ok=True)
            with open(self._current_path, "w", encoding="utf-8") as f:
                f.write(self._edit.toPlainText())
            self._path_label.setText(self._current_path)
            self.setWindowTitle(f"Markdown 文档编辑器 - {os.path.basename(self._current_path)}")
            InfoBar.success("保存成功", self._current_path, parent=self)
        except Exception as e:
            InfoBar.error("保存失败", f"{type(e).__name__}: {e}", parent=self)

    def _refresh_preview(self):
        html = _md_to_html(self._edit.toPlainText())
        self._preview.setHtml(html)
