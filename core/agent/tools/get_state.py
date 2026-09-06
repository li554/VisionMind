"""
get_state 工具 — 获取 VisionMind 当前状态快照

返回当前界面、图片、类别、标注数量等状态信息，
用于 AI 验证操作是否成功。
"""

import json
from core.corecoder.tools import Tool


class GetStateTool(Tool):
    read_only = True
    """获取 VisionMind 当前状态快照"""

    name = "get_state"
    description = (
        "获取当前应用状态快照，包括当前界面、当前图片、当前类别、标注数量等。"
        "用于验证操作是否成功（如切换界面后确认 current_interface 是否正确）。"
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self) -> str:
        """在主线程获取状态快照"""
        from core.common.main_thread_dispatcher import run_on_main

        def _do_get_state():
            state = {}

            try:
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if not app:
                    return json.dumps({"error": "无法获取应用实例"}, ensure_ascii=False)

                # 查找主窗口
                main_window = None
                for widget in app.topLevelWidgets():
                    if hasattr(widget, '_plugin_interfaces'):
                        main_window = widget
                        break

                if not main_window:
                    return json.dumps({"error": "无法找到主窗口"}, ensure_ascii=False)

                # 当前界面
                current = main_window.stackedWidget.currentWidget()
                if current:
                    state["current_interface"] = current.objectName()
                    if current in main_window._plugin_interfaces:
                        _, text, _, _ = main_window._plugin_interfaces[current]
                        state["interface_display_name"] = text

                # 标注相关状态（如果在标注界面）
                if state.get("current_interface") == "AnnotationInterface":
                    for w in main_window._plugin_interfaces:
                        if hasattr(w, 'current_project_name'):
                            state["current_project"] = w.current_project_name
                            state["current_image_index"] = w.file_list.currentRow()
                            state["total_images"] = w.file_list.count()
                            state["current_category"] = getattr(w, 'current_category', '')
                            if hasattr(w, 'current_image_path'):
                                state["current_image_path"] = w.current_image_path
                            if hasattr(w, 'categories'):
                                state["categories"] = list(w.categories)
                            if hasattr(w, 'draw_area'):
                                draw_area = w.draw_area
                                state["annotation_count"] = len(draw_area.annotations)
                                annotations = []
                                for ann in draw_area.annotations:
                                    if isinstance(ann, dict):
                                        annotations.append({
                                            "label": ann.get("label", ""),
                                            "bbox": ann.get("bbox", []),
                                            "shape_type": "polygon" if ann.get("polygons") else "bbox",
                                            "has_polygons": bool(ann.get("polygons")),
                                        })
                                state["annotations"] = annotations
                                hidden_count = len(draw_area.hidden_indices) if hasattr(draw_area, 'hidden_indices') else 0
                                state["hidden_count"] = hidden_count
                                state["visible_count"] = len(draw_area.annotations) - hidden_count
                                state["drawing_mode"] = str(draw_area.mode)
                                mode_names = {0: "edit", 1: "rect", 2: "polygon", 3: "sam", 4: "obb", 5: "roi", 6: "ai_rect"}
                                state["drawing_mode_name"] = mode_names.get(draw_area.mode, "unknown")
                                state["task_mode"] = getattr(draw_area, 'task_mode', '')
                                if draw_area.original_image_size and not draw_area.original_image_size.isNull():
                                    state["image_width"] = draw_area.original_image_size.width()
                                    state["image_height"] = draw_area.original_image_size.height()
                                if draw_area.sam_points:
                                    state["sam_point_count"] = len(draw_area.sam_points)
                            # 未加载项目时需要引导 Agent 先加载,而不是盲目跳界面
                            if not w.current_project_name:
                                state["requires_project"] = True
                                state["projects"] = _collect_projects(main_window)
                            break

                # Dashboard 状态
                elif state.get("current_interface") == "DashboardInterface":
                    state["projects"] = _collect_projects(main_window)

            except Exception as e:
                state["error"] = f"{type(e).__name__}: {e}"

            return json.dumps(state, ensure_ascii=False, indent=2)

        def _collect_projects(mw):
            """从项目大厅收集现有项目名列表,供 Agent 选用。"""
            try:
                for w in mw._plugin_interfaces:
                    if hasattr(w, 'dashboard_service'):
                        svc = w.dashboard_service
                        all_projects = svc.get_all_projects()
                        return [p['name'] for p in all_projects]
            except Exception:
                pass
            return []

        return run_on_main(_do_get_state)
