"""view_image 工具 — agent 按需读图(机制 B)

工具只返回文本标记 [IMG_VIEW: path, mode, ...]。
真正的图片编码与 user 消息注入由 llm.chat 包装层完成(visionmind_agent.py)。
"""

from core.corecoder.tools import Tool


class ViewImageTool(Tool):
    read_only = True
    """请求在对话中查看一张图片(支持 4 种读图模式)"""

    name = "view_image"
    description = (
        "请求在对话中查看一张图片。mode 可选:\n"
        "- original: 原图\n"
        "- local: 指定标注的区域裁剪(需 annotation_index)\n"
        "- overlay: 全图叠加标注渲染\n"
        "- overlay_local: 指定标注区域叠加渲染(需 annotation_index)\n"
        "- roi: 项目配置的 ROI 区域裁剪，传入目标图片路径即可裁剪其 ROI 区域\n"
        "path 必填，为项目图片路径（调用前先通过 manual.get_state 获取当前图片路径传入）。该工具只发起查看请求,图片会在下一轮对话中展示。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "图片路径(必填)"},
            "mode": {
                "type": "string",
                "enum": ["original", "local", "overlay", "overlay_local", "roi"],
                "description": "读图模式",
            },
            "annotation_index": {
                "type": "integer",
                "description": "local/overlay_local 模式下的标注索引",
            },
            "padding": {
                "type": "number",
                "description": "裁剪 padding(像素),默认标注框长边 10%",
            },
        },
        "required": ["mode", "path"],
    }

    def execute(self, path: str = "", mode: str = "original",
                annotation_index: int | None = None,
                padding: float | None = None) -> str:
        if mode not in ("original", "local", "overlay", "overlay_local", "roi"):
            return f"错误: 不支持的 mode: {mode}"
        if mode in ("local", "overlay_local") and annotation_index is None:
            return "错误: local/overlay_local 模式需要 annotation_index 参数"
        if not path or not isinstance(path, str):
            return "错误: 必须提供 path 参数（可先调用 manual.get_state 获取当前图片路径）"
        parts = [f"[IMG_VIEW: {path}", f"mode={mode}"]
        if annotation_index is not None:
            parts.append(f"annotation={annotation_index}")
        if padding is not None:
            parts.append(f"padding={padding}")
        return ", ".join(parts) + "]"
