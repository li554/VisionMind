"""
文档工具 — 生成/读取/列出/打开 Markdown 文档

文档统一保存在当前项目目录下的 reports/ 子目录中；
未打开项目时使用用户目录下的 VisionLeeQT_Reports 兜底。

与界面文档编辑器联动：
- generate_doc / read_doc / list_docs：纯文件操作，可在任意线程执行
- open_doc：通过主线程调度打开文档编辑器窗口（非模态）
"""

import json
import os

from core.corecoder.tools import Tool


def get_reports_dir() -> str:
    """获取文档保存目录（当前项目 reports/ 或用户目录兜底）"""
    try:
        from core.common.project_settings import project_settings
        proj_path = project_settings.path
        if proj_path:
            d = os.path.join(proj_path, "reports")
            os.makedirs(d, exist_ok=True)
            return d
    except Exception:
        pass
    fallback = os.path.join(os.path.expanduser("~"), "VisionLeeQT_Reports")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _safe_name(filename: str) -> str:
    """规范化文件名，防止路径穿越"""
    name = os.path.basename(filename or "").strip()
    if not name:
        raise ValueError("文件名不能为空")
    if not name.endswith(".md"):
        name += ".md"
    return name


class GenerateDocTool(Tool):
    read_only = False
    """生成 Markdown 文档（覆盖或追加）"""

    name = "generate_doc"
    description = (
        "生成 Markdown 分析报告文档（平台文档生成专用工具，**必须优先于 write_file**），"
        "保存到当前项目 reports/ 目录并返回完整路径，目录不存在时自动创建。"
        "mode=overwrite 覆盖写入；mode=append 追加到已有文档末尾。"
        "适合输出数据分析结论、标注统计、图像分析报告等结构化文档。"
        "生成报告必须使用本工具，严禁用 write_file 或脚本写 md 文件替代。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "文件名（如 analysis_report.md，可不带扩展名）",
            },
            "content": {
                "type": "string",
                "description": "Markdown 格式的文档内容（支持标题/表格/代码块等）",
            },
            "mode": {
                "type": "string",
                "enum": ["overwrite", "append"],
                "description": "写入模式，默认 overwrite",
            },
        },
        "required": ["filename", "content"],
    }

    def execute(self, filename: str, content: str, mode: str = "overwrite") -> str:
        try:
            safe_name = _safe_name(filename)
            reports_dir = get_reports_dir()
            file_path = os.path.join(reports_dir, safe_name)
            if mode == "append" and os.path.exists(file_path):
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write("\n" + content)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            return json.dumps({
                "status": "ok",
                "file_path": file_path,
                "mode": mode,
                "chars": len(content),
                "message": f"文档已写入 {file_path}",
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


class ReadDocTool(Tool):
    read_only = True
    """读取 Markdown 文档内容"""

    name = "read_doc"
    description = "读取项目 reports/ 目录下指定 Markdown 文档的完整内容，用于验证已生成报告的准确性。"
    parameters = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "文件名（如 analysis_report.md）",
            },
        },
        "required": ["filename"],
    }

    def execute(self, filename: str) -> str:
        try:
            safe_name = _safe_name(filename)
            reports_dir = get_reports_dir()
            file_path = os.path.join(reports_dir, safe_name)
            if not os.path.exists(file_path):
                return json.dumps({"status": "error", "message": f"文档不存在: {file_path}"}, ensure_ascii=False)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return json.dumps({
                "status": "ok",
                "file_path": file_path,
                "chars": len(content),
                "content": content,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


class ListDocsTool(Tool):
    read_only = True
    """列出已生成的所有 Markdown 文档"""

    name = "list_docs"
    description = "列出当前项目 reports/ 目录下所有已生成的 Markdown 文档及大小，用于确认已有报告清单。"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self) -> str:
        try:
            reports_dir = get_reports_dir()
            docs = []
            if os.path.isdir(reports_dir):
                for fname in sorted(os.listdir(reports_dir)):
                    if fname.endswith(".md"):
                        fpath = os.path.join(reports_dir, fname)
                        docs.append({
                            "filename": fname,
                            "size": os.path.getsize(fpath),
                        })
            return json.dumps({
                "status": "ok",
                "reports_dir": reports_dir,
                "documents": docs,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


class OpenDocTool(Tool):
    read_only = True
    """打开文档编辑器窗口"""

    name = "open_doc"
    description = (
        "在界面上打开 Markdown 文档编辑器窗口（可阅读和修改 reports/ 目录下的文档）。"
        "filename 省略时打开最近一次生成的文档；打开后用户可在窗口中编辑并保存。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "要打开的文件名（如 analysis_report.md），省略则打开最近文档",
            },
        },
        "required": [],
    }

    def execute(self, filename: str = "") -> str:
        try:
            from core.common.main_thread_dispatcher import run_on_main

            def _do_open():
                from core.common.widgets.document_editor_dialog import DocumentEditorDialog
                dlg = DocumentEditorDialog.get_or_create()
                target = ""
                if filename:
                    safe_name = _safe_name(filename)
                    target = os.path.join(get_reports_dir(), safe_name)
                dlg.open_file(target)
                return True

            run_on_main(_do_open)
            return json.dumps({"status": "ok", "message": "文档编辑器已打开"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


def get_recent_doc_path() -> str:
    """获取最近修改的 md 文档路径（供编辑器默认打开）"""
    try:
        reports_dir = get_reports_dir()
        if not os.path.isdir(reports_dir):
            return ""
        md_files = [
            os.path.join(reports_dir, f)
            for f in os.listdir(reports_dir)
            if f.endswith(".md")
        ]
        if not md_files:
            return ""
        return max(md_files, key=os.path.getmtime)
    except Exception:
        return ""
