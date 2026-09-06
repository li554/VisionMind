"""
python 脚本执行工具 — 受限沙箱环境

允许 Agent 编写 python 脚本对图片进行深度分析
（cv2 梯度/灰度/边缘提取/膨胀腐蚀/直方图等），返回可序列化结果。

安全约束：
- 仅允许导入白名单模块：cv2、numpy、json、math、statistics、collections、re
- 不提供 open / os / sys / subprocess / 网络 等能力
- 提供 save_result_image() 将可视化结果保存到项目 reports/ 目录
- 注入当前图片上下文：image_path / image_width / image_height /
  annotations / categories / image_index
- 约定：脚本中给变量 result 赋值（可 JSON 序列化），执行后返回该值
- 超时保护（默认 30 秒）

示例脚本（灰度统计 + Canny 边缘提取）：
    import cv2
    import numpy as np
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    result = {
        "width": gray.shape[1],
        "height": gray.shape[0],
        "mean_gray": round(float(gray.mean()), 2),
        "std_gray": round(float(gray.std()), 2),
        "edge_ratio": round(float((edges > 0).mean()), 4),
    }
"""

import io
import json
import queue
import sys
import threading

from core.corecoder.tools import Tool

_TIMEOUT_SECONDS = 30

_ALLOWED_MODULES = {
    "cv2", "numpy", "np", "json", "math", "statistics",
    "collections", "re", "collections.abc",
}


def _restricted_import(name, *args, **kwargs):
    """受限导入：仅允许白名单模块"""
    base = name.split(".")[0]
    if base not in _ALLOWED_MODULES:
        raise ImportError(f"模块 '{name}' 不在沙箱白名单中，仅允许: cv2/numpy/json/math/statistics/collections/re")
    return __import__(name, *args, **kwargs)


def _collect_context() -> dict:
    """在主线程收集当前图片/标注上下文"""
    try:
        from PySide6.QtWidgets import QApplication

        def _do():
            ctx = {}
            app = QApplication.instance()
            if not app:
                return ctx
            main_window = None
            for widget in app.topLevelWidgets():
                if hasattr(widget, '_plugin_interfaces'):
                    main_window = widget
                    break
            if not main_window:
                return ctx
            current = main_window.stackedWidget.currentWidget()
            if not current:
                return ctx
            for w in main_window._plugin_interfaces:
                if hasattr(w, 'current_image_path'):
                    ctx["image_path"] = w.current_image_path or ""
                    ctx["image_index"] = w.file_list.currentRow() if hasattr(w, 'file_list') else -1
                    ctx["categories"] = list(getattr(w, 'categories', []) or [])
                    draw_area = getattr(w, 'draw_area', None)
                    if draw_area:
                        ctx["annotations"] = [
                            {"label": a.get("label", ""), "bbox": a.get("bbox", [])}
                            for a in (draw_area.annotations or []) if isinstance(a, dict)
                        ]
                        if draw_area.original_image_size and not draw_area.original_image_size.isNull():
                            ctx["image_width"] = draw_area.original_image_size.width()
                            ctx["image_height"] = draw_area.original_image_size.height()
                    break
            return ctx

        from core.common.main_thread_dispatcher import run_on_main
        return run_on_main(_do)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _build_sandbox_globals() -> dict:
    """构建受限执行的全局命名空间"""
    allowed_builtins = {
        "abs": abs, "len": len, "min": min, "max": max, "sum": sum,
        "sorted": sorted, "range": range, "round": round,
        "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "enumerate": enumerate, "zip": zip, "isinstance": isinstance,
        "type": type, "pow": pow, "reversed": reversed,
        "Exception": Exception, "ValueError": ValueError,
        "TypeError": TypeError, "KeyError": KeyError,
        "True": True, "False": False, "None": None,
        "__import__": _restricted_import,
        "print": print,
    }
    return {"__builtins__": allowed_builtins}


def _run_script(code: str, context: dict, timeout: int = _TIMEOUT_SECONDS):
    """在子线程中执行脚本，返回 (status, result_or_error, stdout_text)"""
    sandbox_globals = _build_sandbox_globals()
    sandbox_globals.update({k: v for k, v in context.items() if k != "error"})

    out_buf = io.StringIO()

    def _save_result_image(img, name: str) -> str:
        """保存可视化图片到 reports/ 目录"""
        import os
        try:
            from core.agent.tools.document_tools import get_reports_dir
            d = get_reports_dir()
            safe = os.path.basename(str(name))
            if not safe.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                safe += ".png"
            path = os.path.join(d, safe)
            import cv2
            cv2.imwrite(path, img)
            return path
        except Exception as e:
            return f"保存失败: {type(e).__name__}: {e}"

    sandbox_globals["save_result_image"] = _save_result_image

    box = queue.Queue()

    def _target():
        old_stdout = sys.stdout
        sys.stdout = out_buf
        try:
            exec(code, sandbox_globals)
            box.put(("ok", sandbox_globals.get("result")))
        except Exception as e:
            box.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            sys.stdout = old_stdout

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    stdout_text = out_buf.getvalue()
    if t.is_alive():
        return "error", f"脚本执行超时（>{timeout}s）", stdout_text
    if box.empty():
        return "error", "脚本未返回有效结果", stdout_text
    status, payload = box.get_nowait()
    return status, payload, stdout_text


def _json_dumps(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


class RunScriptTool(Tool):
    read_only = False
    """执行 python 脚本进行图像深度分析"""

    name = "run_script"
    description = (
        "在受限 python 沙箱中执行脚本，可导入 cv2/numpy 对当前图片进行深度分析"
        "（梯度 Sobel/Scharr、灰度直方图、Canny 边缘提取、膨胀腐蚀、阈值分割等）。"
        "注入的上下文变量：image_path(当前图片路径)、image_width、image_height、"
        "annotations(当前标注列表)、categories(类别列表)、image_index。"
        "脚本中把最终结果赋给变量 result（dict/list/str/数字/布尔，需可 JSON 序列化），"
        "print 输出会一并返回。"
        "保存可视化结果图必须调用 save_result_image(img, 'name.png')，"
        "它会自动保存到项目 reports/ 目录（该目录已预创建，无需 mkdir）。"
        "沙箱无 open/os/sys/subprocess 能力，脚本内禁止使用 os.makedirs 等文件系统操作。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 python 代码（自动注入 image_path 等上下文，结果赋值给 result 变量）",
            },
            "timeout": {
                "type": "integer",
                "description": "超时秒数，默认 30",
            },
        },
        "required": ["code"],
    }

    def execute(self, code: str, timeout: int = _TIMEOUT_SECONDS) -> str:
        if not code or not code.strip():
            return json.dumps({"status": "error", "message": "code 不能为空"}, ensure_ascii=False)
        try:
            # 预创建 reports 目录，保证脚本内 save_result_image / cv2.imwrite 可直接使用
            try:
                from core.agent.tools.document_tools import get_reports_dir
                get_reports_dir()
            except Exception:
                pass
            context = _collect_context()
            if "error" in context:
                return json.dumps({"status": "error", "message": f"获取上下文失败: {context['error']}"}, ensure_ascii=False)
            status, payload, stdout_text = _run_script(code, context, timeout=timeout)
            result = {
                "status": status,
                "stdout": stdout_text.strip()[:2000],
                "context_keys": sorted(k for k in context if k != "error"),
            }
            if status == "ok":
                result["result"] = payload
                return json.dumps(result, ensure_ascii=False, indent=2, default=str)
            result["error"] = payload
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
