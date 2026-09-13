"""AutoAnnotationService — 自动标注领域服务

职责: 推理（文本/模型/示例/交互/细化）、模型管理、示例库、示例选择、
     提示词、规则、批量 AI 转换、推理后收尾算法（去重、新类别发现）
AI action: 本服务方法上的 @action(..., scope="agent") 为自动标注的核心逻辑
依赖: ProjectContext（共享状态）+ ModelManager（QThread 信号，Phase 2 注入）
"""

import copy
import json
import os
import shutil
import time
import traceback
import cv2
import numpy as np
from core.backend.core import ModelFactory, example_predict, interactive_predict
from core.backend.defaults import resolve_example_params, resolve_interactive_params
from core.backend.executor import run_inference
from core.backend.unified_inference import prepare_support_sets, unified_inference
from core.event_bus import StateType
from core.backend.formats import save_annotations
from core.backend.path_resolver import get_custom_models, save_custom_models
from core.backend.utils import process_mask_results, convert_bbox_to_polygon, calculate_iou
from core.common.action_registry import action
from core.common.image_utils import imread_unicode, imwrite_unicode
from core.common.settings import settings
from core.common.project_settings import project_settings, RULES_SCHEMA_VERSION
from core.service.common_service import ExampleSelectionManager, PromptManager
from core.service.project_service import ProjectService
from . import _convert_numpy_types
from .foreground_check import mask_to_polygon, mask_to_bbox
from .model_manager import ModelManager
from .project_context import ProjectContext


class AutoAnnotationService:
    """自动标注领域服务：推理、模型、示例库、提示词、规则、批量（仅模型相关）"""

    def __init__(self, context=None):
        self.context = context or ProjectContext()
        self.manual = None  # ManualAnnotationService 交叉引用（Phase 2 注入）
        self.project_service = None  # 门面注入（save_rules 使用）

        # ====== 状态变量: 模型选择 ======
        self.model_type = ""
        self.interactive_model_type = ""

        # ====== 状态变量: 批量标注 ======
        self.one_click_running = False
        self.one_click_paused = False
        self.batch_cancel_requested = False
        self.batch_image_list = []
        self.batch_current_idx = 0
        self.batch_mode = None
        self.batch_prompt_data = None
        self.batch_rules = None

        # ====== 状态变量: 示例库 & 模型加载 ======
        self.example_library = []
        self.support_sets_dir = None
        self.plain_model_type = ""
        self.example_selector = ExampleSelectionManager()
        self.model_manager = ModelManager()  # 后台线程池加载模型（不再持有 QThread 引用）

    # ====== 区域: 推理核心 ======
    # 原 inference_service: predict_by_text/model/example_current，已合并为单个 predict_current
    #   （texts / model_type / selected_examples 三选一输入，统一走 unified_inference）

    def predict_current(self, image_path, texts=None, model_type="", selected_examples=None,
                        rules=None, task_mode="seg", current_category=None):
        """统一预测入口：texts / model_type / selected_examples 三选一传入，其余保持 None。
        - texts: text 模式（字符串或提示词列表，兼容逗号分隔）
        - model_type: model 模式（普通模型名称，权重由 ModelFactory 解析）；text/example 模式时为分割模型
        - selected_examples: example 模式（参考示例列表）
        返回原始推理结果（未去重），由上层统一去重与发现新类别。
        """
        rules = dict(rules or {})

        if texts is not None:
            rules['text'] = texts if isinstance(texts, str) else ', '.join(str(t) for t in texts)
            params = resolve_example_params(model_type=model_type)
            text_list = [t.strip() for t in texts.split(',') if t.strip()] if isinstance(texts, str) else (
                texts if isinstance(texts, list) else [str(texts)]
            )

            def inference_fn(img):
                results = unified_inference(query_images=[img], texts=text_list, **params)
                # 传入提示词列表，让每个实例按其命中的提示词索引映射到对应类别，
                # 而不是把所有提示词用逗号拼成一个类别
                return process_mask_results(results[0], texts=text_list, rules=rules, image=img)

            res = run_inference(image_path, inference_fn, True, True, params.get("img_size"))
            if res['status'] == 'success':
                for ann in res['annotations']:
                    if not ann.get('label'):
                        ann['label'] = rules['text']
                self._normalize_annotations(res['annotations'], task_mode, current_category or "object")
            return res

        if selected_examples is not None:
            rules.pop('text', None)
            params = resolve_example_params(model_type=model_type)
            support_paths = []
            # 示例若为 ROI 示例（记录 is_roi/roi_bbox），查询图裁剪到示例自身的 ROI，
            # 保证查询与示例裁剪图尺度一致（不依赖当前项目 ROI 是否仍设置）
            example_roi = None
            for ex in (selected_examples if isinstance(selected_examples, list) else [selected_examples]):
                if isinstance(ex, dict):
                    support_paths.append(ex.get('folder_path') or ex.get('image_path', ''))
                    if example_roi is None and ex.get('is_roi') and ex.get('roi_bbox'):
                        example_roi = ex.get('roi_bbox')
                elif isinstance(ex, str):
                    support_paths.append(ex)
            support_paths = [p for p in support_paths if p]
            support_images, support_infos, support_rules = prepare_support_sets(support_paths)
            if support_images is None:
                return {"status": "error", "message": "No valid reference examples provided"}

            def inference_fn(img):
                valid_keys = {'model_type', 'refine', 'smooth', 'smooth_epsilon', 'img_size', 'mode', 'verbose'}
                filtered = {k: v for k, v in params.items() if k in valid_keys}
                # mode 兜底与规则配置对话框一致（对话框无 rules 时默认显示 Reuse）：
                # 规则里配了什么就用什么，未配置时兜底 reuse（trtsam3 支持，sam3 内部自动转 grid）
                results = unified_inference(
                    support_images=support_images, support_infos=support_infos,
                    query_images=[img], mode=(rules.get("mode") or "reuse") if rules else "reuse", **filtered
                )
                return process_mask_results(results[0], rules=rules, support_info=support_rules, image=img)

            res = run_inference(image_path, inference_fn, True, True, params.get("img_size"), roi=example_roi)
            if res['status'] == 'success' and res.get('annotations'):
                example_label = selected_examples[0].get('category') or \
                                selected_examples[0].get('annotation', {}).get('label', current_category or 'default')
                for ann in res['annotations']:
                    ann['label'] = example_label
                self._normalize_annotations(res['annotations'], task_mode, example_label)
            return res

        if model_type:
            img_size = settings.get("example_img_size", 640)

            def inference_fn(img):
                results = unified_inference(
                    query_images=[img], model_type=model_type, conf=0.5, iou=0.7, refine=False,
                    smooth=settings.get("mask_smooth_enabled", True),
                    smooth_epsilon=settings.get("mask_smooth_epsilon", 0.001),
                    img_size=img_size,
                )
                return process_mask_results(results[0])

            res = run_inference(image_path, inference_fn, True, True, img_size)
            if res['status'] == 'success' and res.get('annotations'):
                self._normalize_annotations(res['annotations'], task_mode, current_category or "object")
            return res

        return {"status": "error", "message": "未指定任何预测输入（texts/model_type/selected_examples 三选一）"}

    def predict_mask_with_points(self, image_path, points_data, model_type=None, refine=None, use_roi=True, img_size=None):
        try:
            params = resolve_interactive_params(model_type, refine, img_size=img_size)
            roi = project_settings.get("roi") if use_roi else None
            if roi:
                rx, ry, rw, rh = roi
                points = [[[p[0] - rx, p[1] - ry] for p in points_data]]
            else:
                points = [[[p[0], p[1]] for p in points_data]]
            labels = [[p[2] for p in points_data]]
            def inference_fn(img):
                results = interactive_predict(img, points=points, labels=labels, **params)
                return process_mask_results(results, single_result=True)
            result = run_inference(image_path, inference_fn, use_roi=use_roi, use_slice=False, img_size=params.get("img_size"))
            if result.get('status') == 'success' and result.get('annotations'):
                result = result["annotations"][0]
                result["status"] = "success"
            return _convert_numpy_types(result)
        except Exception as e:
            if settings.DEBUG:
                raise
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def predict_mask_with_bbox(self, image_path, bbox, model_type=None, refine=None, use_roi=True, img_size=None):
        """基于矩形框进行交互式分割推理，返回标注 dict（成功时 status=success）"""
        try:
            params = resolve_interactive_params(model_type, refine, img_size=img_size)
            roi = project_settings.get("roi") if use_roi else None
            adjusted_bbox = bbox
            if roi:
                rx, ry, rw, rh = roi
                adjusted_bbox = [bbox[0] - rx, bbox[1] - ry, bbox[2], bbox[3]]
            def inference_fn(img):
                results = interactive_predict(img, bboxes=[adjusted_bbox], **params)
                return process_mask_results(results, bbox=adjusted_bbox, single_result=True)
            result = run_inference(image_path, inference_fn, use_roi=use_roi, use_slice=False, img_size=params.get("img_size"))
            if result.get('status') == 'success' and result.get('annotations'):
                result = result["annotations"][0]
                result["status"] = "success"
            return _convert_numpy_types(result)
        except Exception as e:
            if settings.DEBUG:
                raise
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def predict_similar_in_image(self, image_path, reference_annotations, rules=None, model_type=None, refine=None, use_roi=True, use_slice=True, img_size=None):
        try:
            params = resolve_example_params(model_type, refine, img_size=img_size)

            if not isinstance(reference_annotations, list):
                reference_annotations = [reference_annotations]

            bboxes = []
            for ann in reference_annotations:
                if 'bbox' in ann:
                    bboxes.append(ann['bbox'])

            if not bboxes:
                return {"status": "error", "message": "No valid bboxes in reference annotations"}

            def inference_fn(img):
                extra_kwargs = {}
                actual_model_type = params.get("model_type", model_type or settings.get("example_model", "trtsam3"))

                if actual_model_type.startswith("fs_"):
                    support_images = []
                    support_boxes = []
                    support_masks = []

                    for ann, bbox in zip(reference_annotations, bboxes):
                        if ann.get('is_sliced') and 'slice' in ann:
                            slice_info = ann['slice']
                            slice_img = imread_unicode(slice_info['image_path'])
                            if slice_img is not None:
                                support_images.append(slice_img)
                                support_boxes.append([
                                    bbox[0], bbox[1],
                                    bbox[0] + bbox[2], bbox[1] + bbox[3]
                                ])
                                mask = np.zeros(slice_img.shape[:2], dtype=np.uint8)
                                if 'polygons' in ann and ann['polygons']:
                                    slice_bbox = slice_info['slice_bbox']
                                    for poly in ann['polygons']:
                                        adjusted_poly = [[p[0] - slice_bbox[0], p[1] - slice_bbox[1]] for p in poly]
                                        pts = np.array(adjusted_poly, dtype=np.int32).reshape((-1, 1, 2))
                                        cv2.fillPoly(mask, [pts], 1)
                                else:
                                    x, y, w, h = [int(v) for v in bbox]
                                    x = max(0, x)
                                    y = max(0, y)
                                    w = min(w, slice_img.shape[1] - x)
                                    h = min(h, slice_img.shape[0] - y)
                                    if w > 0 and h > 0:
                                        mask[y:y+h, x:x+w] = 1
                                support_masks.append(mask)
                        else:
                            if 'image_path' in ann:
                                ref_img = imread_unicode(ann['image_path'])
                            elif 'folder_path' in ann:
                                folder_path = ann['folder_path']
                                ref_img = None
                                for f in os.listdir(folder_path):
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                        ref_img = imread_unicode(os.path.join(folder_path, f))
                                        break
                            else:
                                ref_img = img

                            support_images.append(ref_img if ref_img is not None else img)
                            support_boxes.append([
                                bbox[0], bbox[1],
                                bbox[0] + bbox[2], bbox[1] + bbox[3]
                            ])

                            mask = np.zeros((ref_img if ref_img is not None else img).shape[:2], dtype=np.uint8)
                            if 'polygons' in ann and ann['polygons']:
                                for poly in ann['polygons']:
                                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                                    cv2.fillPoly(mask, [pts], 1)
                            else:
                                x, y, w, h = [int(v) for v in bbox]
                                x = max(0, x)
                                y = max(0, y)
                                w = min(w, mask.shape[1] - x)
                                h = min(h, mask.shape[0] - y)
                                if w > 0 and h > 0:
                                    mask[y:y+h, x:x+w] = 1
                            support_masks.append(mask)

                    extra_kwargs['support_images'] = support_images
                    extra_kwargs['support_boxes'] = support_boxes
                    extra_kwargs['support_masks'] = support_masks

                inference_params = dict(params)
                inference_params['model_type'] = actual_model_type

                _roi = project_settings.get("roi") if use_roi else None
                _adjusted_bboxes = bboxes
                if _roi:
                    _rx, _ry, _rw, _rh = _roi
                    _adjusted_bboxes = [[b[0] - _rx, b[1] - _ry, b[2], b[3]] for b in bboxes]

                results = example_predict(
                    img,
                    bboxes=_adjusted_bboxes,
                    **inference_params,
                    **extra_kwargs
                )

                force_label = reference_annotations[0].get('label', 'defect')
                return process_mask_results(results, label=force_label, rules=rules, image=img)

            result = run_inference(image_path, inference_fn, use_roi=use_roi, use_slice=use_slice, img_size=img_size or params.get("img_size"))
            return _convert_numpy_types(result)

        except Exception as e:
            if settings.DEBUG:
                raise
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def refine_annotation(self, image_path, annotations, index,
                          refine_method="vitmatte", use_onnx=False):
        if index < 0 or index >= len(annotations):
            return {"status": "error", "message": f"索引 {index} 无效"}
        ann = annotations[index]
        polygons = ann.get('polygons')
        if not polygons:
            return {"status": "error", "message": "标注没有多边形数据"}
        return self.refine_annotation_polygons(image_path, polygons, refine_method=refine_method)

    # ====== 区域: AI 复合 action ======
    # 原 inference_service: annotate_by_text/model/example_current，已合并为单个 annotate_by_current(mode=...)

    @action("annotate.by_current", background=True, description="按模式对当前图片进行自动标注并写入数据。\n- mode: 标注模式（必填），text=基于文本标注 / example=基于示例标注 / model=基于普通模型标注\n- image_path: 目标图片路径（可选，缺省时自动使用当前打开的图片）\n- text: mode=text 时的描述文本，留空则自动取提示词库中选中的提示词\n- prompt_list: 提示词库列表（[{text, checked}]），可选，缺省使用当前项目提示词库\n- model_type: mode=model 时为普通模型名称（先用 resource.set_model(role='plain', ...) 选择）；text/example 时为分割模型\n- library: mode=example 时的示例库列表，可选，缺省使用当前示例库\n- reference_annotation: 参考标注（选填，库为空时使用）\n- rules: 当前规则配置（可选），用于生成更新后的规则\n- 本方法负责变更数据：将预测结果与磁盘已有标注去重合并后写入当前图片的标注数据（保存），返回新增 annotations/new_categories/rules(更新后)/prompt/examples\n- 通常通过 batch.one_click_by_current 调用，该方法会自动将结果应用到画布并刷新界面\n- 用户未明确标注方式时，可根据提示词库/示例库状态自行判断采用 text 或 example 模式；不确定时可询问用户", category="自动标注", params={"image_path": "str", "mode": {"type": "str", "required": True}, "text": "str", "prompt_list": "list", "model_type": "str", "library": "list", "reference_annotation": "dict", "rules": "dict"}, scope="agent")
    def annotate_by_current(self, image_path=None, mode="", text="", prompt_list=None, model_type="",
                            library=None, reference_annotation=None,
                            existing_annotations=None, categories=None, rules=None,
                            task_mode=None, current_category=None, _persist=True):
        task_mode = task_mode or self.context.task_mode or "seg"
        """
        按 mode 执行完整的自动标注流程：解析输入 → 更新规则 → 预测 → 去重 → 发现新类别，
        并将结果合并写入当前图片的标注数据（数据变更）。image_path/prompt_list/library 缺省时
        自动回退到当前项目状态；返回所有需要的数据，界面刷新由调用方（如 batch.one_click_by_current
        发布 annotation:annotations_changed 信号）负责。
        """
        mode = (mode or "").strip()
        if not image_path:
            if not self.manual or not self.manual.current_image_path:
                return {"status": "error", "message": "未指定 image_path 且当前没有打开图片，请先选择一张图片"}
            image_path = self.manual.current_image_path
        if prompt_list is None:
            prompt_list = self.context.prompt_library
        if library is None:
            library = self.example_library
        new_rules = dict(rules or {})
        texts = None
        selected_examples = None
        predict_model_type = None
        extra = {}

        if mode == "text":
            prompt_res = self.resolve_text_prompt(text, prompt_list)
            if prompt_res['status'] != 'success':
                return prompt_res
            texts = prompt_res['prompt']
            new_rules['text'] = texts
            extra['prompt'] = texts
            predict_model_type = model_type or self.model_type
        elif mode == "model":
            plain_type = model_type or self.plain_model_type
            if not plain_type or plain_type not in ModelFactory.get_plain_models():
                return {"status": "error", "message": "未选择可用的普通模型，请先在模型菜单中配置或使用 add_custom_model 注册"}
            predict_model_type = plain_type
        elif mode == "example":
            example_res = self.resolve_example_for_current(
                library or [], reference_annotation, image_path, current_category
            )
            if example_res['status'] != 'success':
                return example_res
            selected_examples = example_res['examples']
            new_rules.pop('text', None)
            extra['examples'] = selected_examples
            extra['has_memory_example'] = any(not ex.get('folder_path') for ex in selected_examples)
            predict_model_type = model_type or self.model_type
        else:
            return {"status": "error", "message": "未知的标注模式: %s（可选 text/model/example）" % mode}

        res = self.predict_current(
            image_path,
            texts=texts if mode == "text" else None,
            model_type=predict_model_type,
            selected_examples=selected_examples if mode == "example" else None,
            rules=new_rules,
            task_mode=task_mode,
            current_category=current_category,
        )
        if res['status'] != 'success':
            return res

        if existing_annotations is None:
            existing_annotations = self._load_existing_annotations(image_path, task_mode)
        detected_count = len(res.get('annotations') or [])
        filtered = self._filter_duplicate_annotations(res['annotations'], existing_annotations or [])
        new_categories = self._find_new_categories(filtered, categories or {})

        if _persist and filtered:
            merged = list(existing_annotations or []) + filtered
            if self.manual:
                save_res = self.manual.save_current(merged, image_path, task_mode)
            else:
                save_res = {"status": "error", "message": "manual 服务不可用，无法写入标注数据"}
            if save_res['status'] != 'success':
                return {
                    "status": "error",
                    "message": f"标注已生成但写入数据失败: {save_res.get('message', '未知错误')}",
                    "annotations": filtered,
                    "new_categories": new_categories,
                }
            # 关键: save_current 只写盘不同步内存。画布刷新事件
            # (annotations_changed) 的签名去重基于内存 current_annotations,
            # 不同步会签名不变 → 事件处理器跳过刷新 → 检测到目标但画布不渲染
            #
            # 但**必须带 for_path 校验**（审计 D2）：批量路径下 image_path 常常不是屏幕
            # 上那张图，无条件写回会把"最后一张批量图"的标注塞进"当前图"的状态里，
            # 随后任何一次保存都会把它写进当前图的标注文件 —— 静默数据损坏。
            # 这里改用只读真值的受控入口：路径不匹配时直接拒绝写回（画布保持不动）。
            if self.manual is not None:
                self.manual.set_current_annotations(list(merged), for_path=image_path)

        result = {
            "status": "success",
            "annotations": filtered,
            "detected": detected_count,
            "duplicates": detected_count - len(filtered),
            "new_categories": new_categories,
            "rules": new_rules,
            "saved": bool(_persist and filtered),
        }
        result.update(extra)
        return _convert_numpy_types(result)

    def _load_existing_annotations(self, image_path, task_mode):
        """加载当前图片已保存到磁盘的标注（作为合并/去重的基础数据）"""
        if not self.manual:
            return []
        res, _ = self.manual.load_annotations_for_image(
            image_path, task_mode, strict=False,
            output_dir=self.context.output_dir,
            is_non_project_mode=self.context._is_non_project_mode,
            non_project_format=self.context._non_project_format,
        )
        if res['status'] == 'success':
            return res['annotations']
        return []

    # ====== 区域: 推理辅助 ======
    # 原 inference_service: resolve_example_for_current / resolve_text_prompt /
    #   predict_by_*_current_with_result / refine_annotation_polygons

    def resolve_example_for_current(self, library, reference_annotation, current_image_path, current_category):
        """
        解析当前图片的示例列表。
        - 示例库非空时使用示例库
        - 示例库为空但有 reference_annotation 时构建内存示例
        Returns: {"status": "success", "examples": list} or {"status": "error", "message": str}
        """
        if library:
            return {"status": "success", "examples": list(library)}
        if reference_annotation:
            example_label = reference_annotation.get('label', current_category or 'default')
            example = {
                'image_path': current_image_path,
                'annotation': reference_annotation,
                'category': example_label
            }
            return {"status": "success", "examples": [example]}
        return {"status": "error", "message": "示例库为空且没有参考示例"}

    def resolve_text_prompt(self, text, prompt_list=None):
        """
        解析文本标注的提示词。
        - text 不为空时直接使用
        - text 为空时从 prompt_list 中获取选中的提示词
        Returns: {"status": "success", "prompt": str} or {"status": "error", "message": str}
        """
        if text and text.strip():
            return {"status": "success", "prompt": text.strip()}
        if prompt_list:
            selected = [p.get('text', '') for p in prompt_list if isinstance(p, dict) and p.get('checked', False)]
            if selected:
                return {"status": "success", "prompt": ", ".join(selected)}
        return {"status": "error", "message": "提示词库中没有选中的提示词，请先添加并勾选提示词"}

    def refine_annotation_polygons(self, image_path, polygons, refine_method="vitmatte", bbox=None):
        if refine_method == "hybrid":
            return self._refine_hybrid_polygons(image_path, polygons, bbox)
        if not polygons:
            return {"status": "error", "message": "没有多边形数据"}
        img = imread_unicode(image_path)
        if img is None:
            return {"status": "error", "message": "图片加载失败"}
        try:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in polygons:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            refine_model_type = "vitmatte" if refine_method == "vitmatte" else "cascadepsp"
            refiner = ModelFactory.create(refine_model_type)
            refined = refiner.predict(img, masks=[mask],
                                      L=settings.get("refine_image_size", 896),
                                      trimap_thickness=settings.get("refine_trimap_thickness", 20),
                                      fast=settings.get("refine_fast", False),
                                      use_roi=settings.get("refine_use_roi", True),
                                      margin=settings.get("refine_roi_margin", 0.15))
            refined_mask = (refined[0].masks.data[0].numpy() * 255).astype(np.uint8)
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                new_polygons = [largest.squeeze(1).tolist()]
                xs = [p[0] for p in new_polygons[0]]
                ys = [p[1] for p in new_polygons[0]]
                new_bbox = [float(min(xs)), float(min(ys)),
                            float(max(xs) - min(xs)), float(max(ys) - min(ys))]
                return {"status": "success", "polygons": new_polygons, "bbox": new_bbox}
            return {"status": "error", "message": "细化后无有效轮廓"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _refine_hybrid_polygons(self, image_path, polygons, bbox=None):
        """hybrid 细化：经后端 HybridWrapper（小目标CV/中大目标SAM2）返回新 polygons/bbox。"""
        img = imread_unicode(image_path)
        if img is None:
            return {"status": "error", "message": "图片加载失败"}
        box = self._resolve_refine_bbox(img, polygons, bbox)
        if box is None:
            return {"status": "error", "message": "标注无有效区域"}
        try:
            hybrid = ModelFactory.create("hybrid")
            results = hybrid.predict(img, bboxes=[box])
        except Exception as e:
            return {"status": "error", "message": "hybrid 推理失败: %s" % e}
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return {"status": "error", "message": "hybrid 分割未检出目标，保持原标注"}
        mask = results[0].masks.data[0].numpy()
        method = getattr(results[0], "methods", [""])[0] if hasattr(results[0], "methods") else ""
        polygon = mask_to_polygon(mask)
        new_bbox = mask_to_bbox(mask)
        if polygon is None or new_bbox is None:
            return {"status": "error", "message": "hybrid 分割无有效轮廓"}
        return {"status": "success", "polygons": [polygon], "bbox": new_bbox, "method": method}

    def _resolve_refine_bbox(self, img, polygons, bbox=None):
        """解析 hybrid 细化的目标框：优先用标注 bbox，否则由 polygons 掩码外接框得到。"""
        if bbox and len(bbox) >= 4:
            try:
                bw, bh = float(bbox[2]), float(bbox[3])
            except (TypeError, ValueError):
                bw, bh = 0.0, 0.0
            if bw > 0 and bh > 0:
                return [int(round(float(bbox[0]))), int(round(float(bbox[1]))),
                        int(round(bw)), int(round(bh))]
        if polygons:
            H, W = img.shape[:2]
            mask = np.zeros((H, W), dtype=np.uint8)
            for poly in polygons:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            return mask_to_bbox(mask)
        return None

    # ====== 区域: 内部归一化 ======
    # 原 inference_service: _normalize_annotations / _normalize_annotation / _convert_annotation_mode

    def _normalize_annotations(self, annotations, task_mode, default_label="object"):
        """归一化并转换推理结果标注格式"""
        for ann in annotations:
            normalized = self._normalize_annotation(ann, default_label)
            if task_mode in ('obb', 'det'):
                normalized = self._convert_annotation_mode(normalized, task_mode)
            ann.clear()
            ann.update(normalized)

    def _normalize_annotation(self, ann, default_label="object"):
        """归一化单个标注数据结构，确保包含 bbox 和 label 字段"""
        result = dict(ann)
        pts = result.get('points')
        if pts and 'bbox' not in result:
            if result.get('type') == 'rect' and len(pts) >= 2:
                p1, p2 = pts[0], pts[1]
                x1, y1 = p1
                x2, y2 = p2
                result['bbox'] = [int(min(x1, x2)), int(min(y1, y2)),
                                  int(abs(x2 - x1)), int(abs(y2 - y1))]
            else:
                flat_pts = []
                for p in pts:
                    if hasattr(p, '__len__') and len(p) >= 2:
                        flat_pts.append([float(p[0]), float(p[1])])
                if flat_pts:
                    result['polygons'] = [flat_pts]
                    xs = [p[0] for p in flat_pts]
                    ys = [p[1] for p in flat_pts]
                    result['bbox'] = [float(min(xs)), float(min(ys)),
                                      float(max(xs) - min(xs)), float(max(ys) - min(ys))]
        if 'label' not in result:
            result['label'] = default_label or "object"
        return result

    def _convert_annotation_mode(self, ann, target_mode):
        """转换标注格式为指定模式（obb / det），返回新字典"""
        result = dict(ann)
        if target_mode == "obb":
            if result.get('polygons'):
                all_pts = []
                for p in result['polygons']:
                    all_pts.extend(p)
                if all_pts:
                    pts = np.array(all_pts, dtype=np.float32)
                    if len(pts) > 4:
                        rect = cv2.minAreaRect(pts)
                        box = cv2.boxPoints(rect).tolist()
                        result['polygons'] = [box]
            elif result.get('bbox'):
                x, y, w, h = result['bbox']
                result['polygons'] = [[(float(x), float(y)), (float(x + w), float(y)),
                                       (float(x + w), float(y + h)), (float(x), float(y + h))]]
            if result.get('polygons'):
                all_pts = []
                for p in result['polygons']:
                    all_pts.extend(p)
                if all_pts:
                    xs = [pt[0] for pt in all_pts]
                    ys = [pt[1] for pt in all_pts]
                    result['bbox'] = [float(min(xs)), float(min(ys)),
                                      float(max(xs) - min(xs)), float(max(ys) - min(ys))]
            result['shape_type'] = 'rotation'
        elif target_mode == "det":
            if result.get('bbox'):
                result['shape_type'] = 'rectangle'
            elif result.get('polygons'):
                pts = result['polygons'][0]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                result['bbox'] = [float(min(xs)), float(min(ys)),
                                  float(max(xs) - min(xs)), float(max(ys) - min(ys))]
                result['shape_type'] = 'rectangle'
        return result

    # ====== 区域: 示例库管理 ======
    # 原 resource_service: _load_support_sets_dir / set_non_project_mode / load_example_library /
    #   get_example_library / list_examples_data / add_example / remove_example / clear_example_library /
    #   list_examples_text / _create_target_centered_slice / _convert_annotation_for_slice(原 _adjust_annotation_for_slice)

    def _load_support_sets_dir(self):
        """从当前加载的项目获取 support_sets 目录"""
        project_service = ProjectService()
        self.support_sets_dir = project_service.get_project_support_sets_dir()
        if not self.support_sets_dir:
            # 新项目尚无 support_sets 目录:基于当前项目名预置路径,
            # add_example 会自动创建(load_example_library 对不存在目录安全)
            project_name = settings.get("current_project_name")
            if project_name:
                self.support_sets_dir = os.path.join(
                    project_service.base_dir, project_name, "support_sets"
                )
            else:
                default_dir = os.path.join(project_service.base_dir, "support_sets")
                if os.path.exists(default_dir):
                    self.support_sets_dir = default_dir
        self.load_example_library()

    def set_non_project_mode(self, save_dir):
        """设置非项目模式下的示例库目录"""
        if save_dir:
            self.support_sets_dir = os.path.join(save_dir, "support_sets")
            if os.path.exists(self.support_sets_dir):
                self.load_example_library()
            else:
                self.example_library = []
        else:
            self._load_support_sets_dir()

    def load_example_library(self):
        """从磁盘扫描 support_sets 目录加载示例库"""
        try:
            self.example_library = []
            if not self.support_sets_dir or not os.path.exists(self.support_sets_dir):
                return
            for folder_name in os.listdir(self.support_sets_dir):
                folder_path = os.path.join(self.support_sets_dir, folder_name)
                if not os.path.isdir(folder_path):
                    continue
                info_path = os.path.join(folder_path, "info.json")
                if os.path.exists(info_path):
                    try:
                        if os.path.getsize(info_path) > 50 * 1024 * 1024:
                            continue
                        with open(info_path, 'r', encoding='utf-8-sig') as f:
                            info = json.load(f)
                        img_path = info.get('image_path')
                        if not img_path or not os.path.exists(img_path):
                            for fname in os.listdir(folder_path):
                                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    img_path = os.path.join(folder_path, fname)
                                    break
                        if img_path:
                            info['image_path'] = img_path
                            info['folder_path'] = folder_path
                            self.example_library.append(info)
                    except Exception as e:
                        if settings.DEBUG:
                            raise
        except Exception as e:
            if settings.DEBUG:
                raise
            self.example_library = []
        # 示例库刷新后恢复用户勾选的示例（持久化到项目设置）
        self.load_selected_examples()

    def get_example_library(self):
        """获取示例库数据列表"""
        return list(self.example_library) if self.example_library else []

    def list_examples_data(self):
        """返回结构化示例库数据（用于 action 返回值）"""
        return [
            {
                "index": i,
                "name": item.get('name', f'示例{i}'),
                "category": item.get('category', '未知'),
                "image_path": item.get('image_path', ''),
                "folder_path": item.get('folder_path', ''),
            }
            for i, item in enumerate(self.example_library or [])
        ]

    def add_example(self, example_item, roi_bbox=None):
        """添加示例到库中，可选 ROI / 切片模式"""
        try:
            if not self.support_sets_dir:
                self._load_support_sets_dir()
            if not self.support_sets_dir:
                return {"status": "error",
                        "message": "未加载项目，无法确定示例库保存目录"}
            timestamp = int(time.time() * 1000)
            folder_name = f"set_{timestamp}"
            folder_path = os.path.join(self.support_sets_dir, folder_name)
            os.makedirs(self.support_sets_dir, exist_ok=True)
            os.makedirs(folder_path, exist_ok=True)

            orig_img_path = example_item['image_path']
            ext = os.path.splitext(orig_img_path)[1]
            orig_base_name = os.path.splitext(os.path.basename(orig_img_path))[0]

            ann = example_item['annotation'].copy()
            img = imread_unicode(orig_img_path)
            h, w = img.shape[:2]

            slice_enabled = settings.get("slice_inference_enabled", False)
            slice_processed = False

            if slice_enabled:
                slice_rows = settings.get("slice_rows", 2)
                slice_cols = settings.get("slice_cols", 2)
                slice_overlap = settings.get("slice_overlap_pixels", 50)
                target_bbox = ann.get('bbox', [0, 0, w, h])
                slice_info = self._create_target_centered_slice(img, target_bbox, slice_rows, slice_cols, slice_overlap)
                slice_img = slice_info['image']
                slice_bbox = slice_info['bbox']
                adjusted_ann = self._convert_annotation_for_slice(ann, slice_bbox)
                if adjusted_ann is not None:
                    slice_processed = True
                    debug_img = slice_img.copy()
                    if len(debug_img.shape) == 2:
                        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
                    if 'bbox' in adjusted_ann:
                        bx, by, bw, bh = adjusted_ann['bbox']
                        cv2.rectangle(debug_img, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 255, 0), 2)
                    if 'polygons' in adjusted_ann:
                        for poly in adjusted_ann['polygons']:
                            if len(poly) >= 3:
                                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                                cv2.polylines(debug_img, [pts], True, (0, 0, 255), 2)
                    imwrite_unicode(os.path.join(folder_path, f"{orig_base_name}_slice_debug.jpg"), debug_img)
                    slice_img_name = f"{orig_base_name}_slice{ext}"
                    slice_img_path = os.path.join(folder_path, slice_img_name)
                    imwrite_unicode(slice_img_path, slice_img)
                    slice_label_name = f"{orig_base_name}_slice.txt"
                    slice_label_path = os.path.join(folder_path, slice_label_name)
                    label_name = example_item.get('category', adjusted_ann.get('label', 'target'))
                    polygons = adjusted_ann.get('polygons', [])
                    instances = [{
                        'bbox': adjusted_ann['bbox'],
                        'label': label_name,
                        'class_id': example_item.get('class_id', 0),
                        'polygons': polygons,
                    }]
                    save_data = {
                        "annotations": instances,
                        "image_info": {"width": slice_bbox[2], "height": slice_bbox[3]},
                        "categories": [{"id": 0, "name": label_name}],
                    }
                    save_annotations(save_data, slice_label_path, format_type="yolo",
                                    mode="seg" if polygons else "det")
                    shutil.copy2(orig_img_path, os.path.join(folder_path, f"{orig_base_name}{ext}"))
                    serializable_ann = ann.copy()
                    serializable_ann.pop('mask', None)
                    serializable_ann['polygons'] = ann.get('polygons', [])
                    serializable_ann = _convert_numpy_types(serializable_ann)
                    info = {
                        'image_path': os.path.join(folder_path, f"{orig_base_name}{ext}"),
                        'annotation': serializable_ann,
                        'name': example_item.get('name', folder_name),
                        'category': example_item.get('category', ann.get('label', 'target')),
                        'created_at': timestamp, 'is_sliced': True,
                        'slice_config': {'rows': slice_rows, 'cols': slice_cols, 'overlap': slice_overlap},
                        'slice': {
                            'image_path': slice_img_path,
                            'bbox': slice_bbox,
                            'slice_bbox': adjusted_ann.get('bbox', slice_bbox),
                            'slice_polygons': _convert_numpy_types(adjusted_ann.get('polygons', [])),
                            'target_centered': True,
                        },
                    }

            if not slice_processed and roi_bbox:
                dest_img_name = f"{orig_base_name}{ext}"
                dest_img_path = os.path.join(folder_path, dest_img_name)
                rx, ry, rw, rh = roi_bbox
                rx, ry = max(0, min(rx, img.shape[1] - 1)), max(0, min(ry, img.shape[0] - 1))
                rw, rh = max(1, min(rw, img.shape[1] - rx)), max(1, min(rh, img.shape[0] - ry))
                crop = img[ry:ry + rh, rx:rx + rw]
                imwrite_unicode(dest_img_path, crop)
                w, h = rw, rh
                if 'bbox' in ann:
                    bx, by, bw, bh = ann['bbox']
                    nx1, ny1 = max(rx, bx), max(ry, by)
                    nx2, ny2 = min(rx + rw, bx + bw), min(ry + rh, by + bh)
                    ann['bbox'] = [nx1 - rx, ny1 - ry, nx2 - nx1, ny2 - ny1] if nx2 > nx1 and ny2 > ny1 else [0, 0, rw, rh]
                if 'polygons' in ann:
                    ann['polygons'] = [[[max(0, min(w, p[0] - rx)), max(0, min(h, p[1] - ry))] for p in poly] for poly in ann['polygons']]
                label_name = example_item.get('category', ann.get('label', 'target'))
                polygons = ann.get('polygons', [])
                instances = [{'bbox': ann['bbox'], 'label': label_name, 'class_id': example_item.get('class_id', 0), 'polygons': polygons}]
                save_data = {"annotations": instances, "image_info": {"width": w, "height": h}, "categories": [{"id": 0, "name": label_name}]}
                save_annotations(save_data, os.path.join(folder_path, f"{orig_base_name}.txt"), format_type="yolo", mode="seg" if polygons else "det")
                serializable_ann = ann.copy()
                serializable_ann.pop('mask', None)
                serializable_ann['polygons'] = polygons
                serializable_ann = _convert_numpy_types(serializable_ann)
                info = {'image_path': dest_img_path, 'annotation': serializable_ann,
                        'name': example_item.get('name', folder_name),
                        'category': label_name, 'created_at': timestamp, 'is_roi': True, 'roi_bbox': roi_bbox}

            if not slice_processed and not roi_bbox:
                dest_img_path = os.path.join(folder_path, f"{orig_base_name}{ext}")
                shutil.copy2(orig_img_path, dest_img_path)
                label_name = example_item.get('category', ann.get('label', 'target'))
                polygons = ann.get('polygons', [])
                instances = [{'bbox': ann['bbox'], 'label': label_name, 'class_id': example_item.get('class_id', 0), 'polygons': polygons}]
                save_data = {"annotations": instances, "image_info": {"width": w, "height": h}, "categories": [{"id": 0, "name": label_name}]}
                save_annotations(save_data, os.path.join(folder_path, f"{orig_base_name}.txt"), format_type="yolo", mode="seg" if polygons else "det")
                serializable_ann = ann.copy()
                serializable_ann.pop('mask', None)
                serializable_ann['polygons'] = polygons
                serializable_ann = _convert_numpy_types(serializable_ann)
                info = {'image_path': dest_img_path, 'annotation': serializable_ann,
                        'name': example_item.get('name', folder_name),
                        'category': label_name, 'created_at': timestamp}

            with open(os.path.join(folder_path, "info.json"), 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=4)
            info['folder_path'] = folder_path
            self.example_library.append(info)
            return {"status": "success", "folder_path": folder_path}
        except Exception as e:
            if settings.DEBUG:
                raise
            return {"status": "error", "message": str(e)}

    @action("resource.remove_example", description="从示例库中删除指定索引的示例。\n- index：示例索引（从0开始），可通过 list_examples 获取\n- 删除后不可恢复", category="自动标注", params={"index": "int"}, scope="agent")
    def remove_example(self, index):
        """从库中删除示例及其文件夹"""
        if 0 <= index < len(self.example_library):
            item = self.example_library.pop(index)
            name = item.get('name')
            if name:
                self.example_selector.remove_from_selection(name)
                self._persist_selected_examples()
            folder_path = item.get('folder_path')
            if folder_path and os.path.exists(folder_path):
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    if settings.DEBUG:
                        raise
                    print(f"[AnnotationService] Failed to delete example folder: {e}")

    @action("resource.clear_example_library", description="清空当前项目的所有示例。\n- 删除所有 support_sets 目录下的示例文件夹\n- 操作不可恢复\n- 清空后用 one_click_by_example 之前需重新添加示例", category="自动标注", scope="agent")
    def clear_example_library(self):
        """清空所有示例"""
        try:
            for item in self.example_library:
                folder_path = item.get('folder_path')
                if folder_path and os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
            self.example_library = []
            self.example_selector.clear_selected_examples()
            self._persist_selected_examples()
            return {"status": "success"}
        except Exception as e:
            if settings.DEBUG:
                raise
            return {"status": "error", "message": str(e)}

    def list_examples_text(self):
        """列出示例库文本"""
        library = self.example_library
        if not library:
            return "示例库为空"
        lines = [f"示例库共 {len(library)} 个示例:"]
        for i, item in enumerate(library):
            lines.append(f"  [{i}] {item.get('name', f'示例{i}')} (类别: {item.get('category', '未知')})")
        return "\n".join(lines)

    def _create_target_centered_slice(self, image, target_bbox, rows, cols, overlap):
        """创建以目标为中心的单个切片，返回 {'image', 'bbox', 'target_centered'}"""
        h, w = image.shape[:2]
        tx, ty, tw, th = target_bbox
        tx = int(max(0, min(tx, w - 1)))
        ty = int(max(0, min(ty, h - 1)))
        tw = int(max(1, min(tw, w - tx)))
        th = int(max(1, min(th, h - ty)))
        target_cx = tx + tw / 2
        target_cy = ty + th / 2
        slice_w = int((w + (cols - 1) * overlap) / cols) if cols > 0 else w
        slice_h = int((h + (rows - 1) * overlap) / rows) if rows > 0 else h
        x_start = int(target_cx - slice_w / 2)
        y_start = int(target_cy - slice_h / 2)
        if x_start + slice_w > w:
            x_start = w - slice_w
        if x_start < 0:
            x_start = 0
        if y_start + slice_h > h:
            y_start = h - slice_h
        if y_start < 0:
            y_start = 0
        if tx < x_start:
            x_start = int(max(0, tx))
        if tx + tw > x_start + slice_w:
            slice_w = int(min(w - x_start, tx + tw - x_start))
        if ty < y_start:
            y_start = int(max(0, ty))
        if ty + th > y_start + slice_h:
            slice_h = int(min(h - y_start, ty + th - y_start))
        slice_w = int(min(slice_w, w - x_start))
        slice_h = int(min(slice_h, h - y_start))
        return {
            'image': image[y_start:y_start + slice_h, x_start:x_start + slice_w],
            'bbox': [x_start, y_start, slice_w, slice_h],
            'target_centered': True,
        }

    def _convert_annotation_for_slice(self, annotation, slice_bbox):
        """将标注坐标转换为相对于切片的坐标，切片外标注返回 None（原 _adjust_annotation_for_slice）"""
        adjusted = dict(annotation)
        sx, sy, sw, sh = slice_bbox
        if 'bbox' in adjusted:
            bx, by, bw, bh = adjusted['bbox']
            nx1 = max(sx, bx)
            ny1 = max(sy, by)
            nx2 = min(sx + sw, bx + bw)
            ny2 = min(sy + sh, by + bh)
            if nx2 > nx1 and ny2 > ny1:
                adjusted['bbox'] = [nx1 - sx, ny1 - sy, nx2 - nx1, ny2 - ny1]
            else:
                return None
        if 'polygons' in adjusted:
            new_polys = []
            for poly in adjusted['polygons']:
                new_poly = [[p[0] - sx, p[1] - sy] for p in poly]
                if any(0 <= p[0] < sw and 0 <= p[1] < sh for p in new_poly):
                    clipped = [[max(0, min(sw - 1, p[0])), max(0, min(sh - 1, p[1]))] for p in new_poly]
                    new_polys.append(clipped)
            if new_polys:
                adjusted['polygons'] = new_polys
            else:
                return None
        return adjusted

    # ====== 区域: 示例选择 ======
    # 原 resource_service: get_selected_example_items

    def get_selected_example_items(self):
        """根据选中的名称返回对应的完整示例项列表"""
        return ExampleSelectionManager.get_examples_by_names(
            self.example_selector.get_selected_examples(),
            self.example_library or []
        )

    # ====== 区域: 规则管理 ======
    # 原 resource_service: get_rules_text / _set_rule(原 set_rule，命令版 set_rule(key, value) 见「批量标注命令」区域)

    VALID_RULE_KEYS = {"text", "max_instances", "conf_threshold", "area_range",
                       "width_range", "height_range", "aspect_ratio_range",
                       "gray_range", "mode"}

    def get_rules_text(self, rules):
        if not rules:
            return "当前无规则配置"
        return json.dumps(rules, ensure_ascii=False, indent=2)

    def _set_rule(self, rules, key, value):
        if key not in self.VALID_RULE_KEYS:
            return rules, f"错误: 不支持的规则字段 '{key}'，可选: {', '.join(sorted(self.VALID_RULE_KEYS))}"
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed = value
        new_rules = dict(rules or {})
        new_rules[key] = parsed
        return new_rules, f"规则 '{key}' 已设置为: {value}"

    # ====== 区域: 自定义模型管理 ======
    # 原 resource_service: list_custom_models_text / list_builtin_models_text / add_custom_model / remove_custom_model

    def list_custom_models_text(self, role=""):
        roles = [role] if role else ("interactive", "auto", "plain", "refine")
        lines = []
        for r in roles:
            models = get_custom_models(r)
            if not models:
                continue
            role_label = {"interactive": "交互模型", "auto": "自动标注模型", "plain": "普通模型", "refine": "细化模型"}.get(r, r)
            lines.append(f"[{role_label}]")
            for i, m in enumerate(models):
                lines.append(f"  [{i}] {m.get('name')} (类别: {m.get('category')}, 权重: {m.get('weight_path')})")
        return "\n".join(lines) if lines else "当前没有已注册的自定义模型。可通过 add_custom_model 添加。"

    def list_builtin_models_text(self, role=""):
        role_map = {
            "interactive": (ModelFactory.get_interactive_models, "交互模型"),
            "auto": (ModelFactory.get_auto_models, "自动标注模型"),
            "plain": (ModelFactory.get_plain_models, "普通模型"),
            "refine": (ModelFactory.get_refine_models, "细化模型"),
        }
        targets = {role: role_map[role]} if role in role_map else role_map
        lines = []
        for r, (getter, label) in targets.items():
            models = getter()
            builtin = [m for m in models if m not in ModelFactory._custom_weight_paths]
            if builtin:
                lines.append(f"[{label}]")
                for i, m in enumerate(builtin):
                    lines.append(f"  [{i}] {m}")
        return "\n".join(lines) if lines else "无可用内置模型。"

    @action("resource.add_custom_model", description="添加自定义模型，无需打开对话框。\n- name：模型名称（不可与已有模型重名）\n- category：模型类别，如 sam、sam2、sam_onnx、yoloe、yolo_trained 等，可通过 list_builtin_models 查看可选类别\n- weight_path：权重文件绝对路径（文件必须存在）\n- role：模型角色，必填，需严格按用户语义映射：auto=示例模型（对应界面菜单『示例模型』，用于基于示例的一键自动标注 one_click_by_example）；interactive=交互模型（交互式分割）；plain=普通模型（已训练YOLO，mode=model）；refine=细化模型。注意：用户说『增加一个示例模型』对应 role=auto，切勿选成 interactive\n- 添加后自动刷新模型菜单，可通过 set_model(role=interactive/auto/plain) 切换使用", category="自动标注", params={"name": "str", "category": "str", "weight_path": "str", "role": "str"}, scope="agent")
    def add_custom_model(self, name, category, weight_path, role="auto"):
        import os
        if not name or not name.strip():
            return "错误: 模型名称不能为空"
        name = name.strip()
        if not weight_path or not os.path.exists(weight_path):
            return "错误: 权重文件路径无效或文件不存在"
        if role not in ("interactive", "auto", "plain", "refine"):
            return "错误: 角色必须为 interactive、auto、plain 或 refine"
        all_models = (ModelFactory.get_interactive_models() +
                      ModelFactory.get_auto_models() +
                      ModelFactory.get_plain_models() +
                      ModelFactory.get_refine_models())
        if name in all_models:
            return f"错误: 模型名称 '{name}' 已存在"
        ok = ModelFactory.register_custom_model(name, category, weight_path, role=role)
        if not ok:
            return f"错误: 未知的模型类别 '{category}'"
        models = get_custom_models(role)
        models.append({"name": name, "category": category, "weight_path": weight_path})
        save_custom_models(role, models)
        return f"已成功添加自定义模型: {name} (类别: {category}, 角色: {role})"

    @action("resource.remove_custom_model", description="删除指定的自定义模型。\n- name：要删除的模型名称\n- role：模型角色，可选 interactive/auto/plain/refine，默认 auto\n- 删除后自动刷新模型菜单\n- 可通过 list_custom_models 查看所有自定义模型及其名称", category="自动标注", params={"name": "str", "role": "str"}, scope="agent")
    def remove_custom_model(self, name, role="auto"):
        if role not in ("interactive", "auto", "plain", "refine"):
            return "错误: 角色必须为 interactive、auto、plain 或 refine"
        ModelFactory.unregister_custom_model(name, role=role)
        models = get_custom_models(role)
        new_models = [m for m in models if m.get("name") != name]
        if len(new_models) == len(models):
            return f"错误: 未找到自定义模型 '{name}' (角色: {role})"
        save_custom_models(role, new_models)
        return f"已成功删除自定义模型: {name}"

    # ====== 区域: 批量 AI 转换 ======
    # 原 annotation_io: batch_ai_convert

    # `find_and_load_annotations` / `save_image_annotations` 只存在于
    # ManualAnnotationService 上（本类没有这两个方法）。原先以 `self.` 调用 ->
    # 第一次迭代必 AttributeError，"AI 全量转换"100% 不可用（审计 D9）。
    # 这里显式委托给 manual 服务，并在缺失时给出可诊断的失败而不是崩栈。
    def _load_annotations_for_convert(self, img_path, mode, fmt, output_dir):
        manual = getattr(self, "manual", None)
        if manual is None or not hasattr(manual, "find_and_load_annotations"):
            return {"status": "error", "annotations": [],
                    "message": "manual 服务不可用，无法读取待转换标注"}
        return manual.find_and_load_annotations(
            img_path, mode, fmt, output_dir=output_dir, is_non_project_mode=True)

    def _save_converted_annotations(self, img_path, annotations, target_dir, fmt, categories):
        manual = getattr(self, "manual", None)
        if manual is None or not hasattr(manual, "save_image_annotations"):
            return {"status": "error",
                    "message": "manual 服务不可用，无法写入转换后的标注"}
        return manual.save_image_annotations(
            img_path, annotations, output_dir=target_dir, fmt=fmt,
            categories=categories, project_mode=False)

    def batch_ai_convert(self, image_paths, source_mode='det', target_mode='seg',
                          output_dir=None, categories=None, progress_callback=None,
                          cancel_event=None):
        result = {"converted_count": 0, "empty_results_count": 0,
                  "images_with_empty_results": 0, "total": len(image_paths),
                  "processed_count": 0, "cancelled": False, "errors": []}
        # 取消标记挂到 service 上，界面的"取消"按钮经 `cancel_batch_convert()` 置位
        # （审计 D9 余项：原先取消按钮从未接线，按下去毫无作用）。
        self._batch_cancel_event = cancel_event
        try:
            for i, img_path in enumerate(image_paths):
                # 协作式取消：在**图片边界**检查，绝不打断正在写的那张图，
                # 保证已落盘的数据始终是完整的（不会留半张图的标注）
                if cancel_event is not None and cancel_event.is_set():
                    result["cancelled"] = True
                    print("[batch_ai_convert] 已按用户请求取消，处理到第 %d/%d 张"
                          % (i, len(image_paths)))
                    break
                result["processed_count"] = i
                if progress_callback:
                    progress_callback(i, len(image_paths))
                mode = source_mode
                fmt = 'yolo' if mode == 'det' else ('yoloobb' if mode == 'obb' else 'yoloseg')
                ann_dir = output_dir or settings.get("output_dir", "")
                res = self._load_annotations_for_convert(img_path, mode, fmt, output_dir=ann_dir)
                if res['status'] != 'success' or not res['annotations']:
                    continue
                anns = res['annotations']
                to_convert_bboxes = []
                for ann in anns:
                    if ann.get('bbox'):
                        to_convert_bboxes.append(ann['bbox'])
                if not to_convert_bboxes:
                    continue
                image = imread_unicode(img_path)
                if image is None:
                    continue
                image_had_empty = False
                final_annotations = []
                for ann in anns:
                    bbox = ann.get('bbox')
                    if not bbox:
                        continue
                    refine = settings.get("high_precision", True)
                    conv = convert_bbox_to_polygon(image=image, bbox=bbox, model_type=None, refine=refine)
                    if conv.get('status') == 'success' and conv.get('polygons'):
                        polygons = conv['polygons']
                        if target_mode == 'det':
                            all_pts = []
                            for p in polygons:
                                all_pts.extend(p)
                            if all_pts:
                                xs = [p[0] for p in all_pts]
                                ys = [p[1] for p in all_pts]
                                bbox = [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))]
                            else:
                                bbox = [0, 0, 0, 0]
                        else:
                            if target_mode == 'obb':
                                all_pts = []
                                for p in polygons:
                                    all_pts.extend(p)
                                if all_pts:
                                    pts = np.array(all_pts, dtype=np.float32)
                                    if len(pts) > 4:
                                        rect = cv2.minAreaRect(pts)
                                        polygons = [cv2.boxPoints(rect).tolist()]
                        final_ann = {'bbox': bbox, 'label': ann.get('label', 'unknown'),
                                     'polygons': polygons,
                                     'shape_type': 'rectangle' if target_mode == 'det' else 'polygon'}
                        final_annotations.append(final_ann)
                        result["converted_count"] += 1
                    else:
                        image_had_empty = True
                        result["empty_results_count"] += 1
                        bx, by, bw, bh = bbox
                        poly = [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]
                        final_annotations.append({'bbox': bbox, 'label': ann.get('label', 'unknown'),
                                                 'polygons': [poly],
                                                 'shape_type': 'rectangle' if target_mode == 'det' else 'polygon'})
                if image_had_empty:
                    result["images_with_empty_results"] += 1
                if target_mode == 'obb':
                    for ann in final_annotations:
                        ann = self._convert_annotation_mode(ann, 'obb')
                target_fmt = {'seg': 'yoloseg', 'obb': 'yoloobb', 'det': 'yolo'}.get(target_mode, 'yolo')
                target_dir = os.path.join(ann_dir, target_mode)
                os.makedirs(target_dir, exist_ok=True)
                self._save_converted_annotations(img_path, final_annotations,
                                                target_dir, target_fmt, categories)
            result["processed_count"] = i + 1
            if progress_callback:
                progress_callback(i + 1, len(image_paths))
        finally:
            self._batch_cancel_event = None
        if not result["cancelled"] and progress_callback:
            progress_callback(len(image_paths), len(image_paths))
        return result

    # ====== 区域: 推理后收尾算法 ======
    # 原 annotation_io: _calculate_annotations_iou / _filter_duplicate_annotations /
    #   _filter_duplicate_annotations_hungarian / _filter_and_add_annotations /
    #   _find_new_categories(原 _discover_new_categories) / _discover_new_categories(兼容别名) /
    #   _normalize_yolo_inference_ann / _convert_single_ann_to_obb / _convert_single_ann_to_det

    def _calculate_annotations_iou(self, ann1, ann2):
        """计算两个标注之间的 IoU：优先 polygon mask 精确计算，退化到 bbox IoU"""
        polygons1 = ann1.get('polygons')
        polygons2 = ann2.get('polygons')
        if polygons1 and polygons2:
            try:
                arr1 = np.asarray(polygons1[0])
                arr2 = np.asarray(polygons2[0])
            except (TypeError, ValueError, IndexError):
                pass
            else:
                if arr1.size > 0 and arr2.size > 0:
                    bbox1 = ann1.get('bbox')
                    bbox2 = ann2.get('bbox')
                    if bbox1 and bbox2:
                        x1, y1, w1, h1 = bbox1
                        x2, y2, w2, h2 = bbox2
                        ux = float(min(x1, x2))
                        uy = float(min(y1, y2))
                        ux2 = float(max(x1 + w1, x2 + w2))
                        uy2 = float(max(y1 + h1, y2 + h2))
                        uw, uh = int(np.ceil(ux2 - ux)), int(np.ceil(uy2 - uy))
                        if uw > 0 and uh > 0:
                            mask1 = np.zeros((uh, uw), dtype=np.uint8)
                            mask2 = np.zeros((uh, uw), dtype=np.uint8)
                            for poly in polygons1:
                                pts = np.asarray(poly, dtype=np.float64)
                                if pts.ndim == 3 and pts.shape[-1] >= 2:
                                    pts = pts.reshape(-1, pts.shape[-1])
                                if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 3:
                                    continue
                                pts = pts[:, :2].copy()
                                pts[:, 0] -= ux
                                pts[:, 1] -= uy
                                cv2.fillPoly(mask1, [np.round(pts).astype(np.int32)], 1)
                            for poly in polygons2:
                                pts = np.asarray(poly, dtype=np.float64)
                                if pts.ndim == 3 and pts.shape[-1] >= 2:
                                    pts = pts.reshape(-1, pts.shape[-1])
                                if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 3:
                                    continue
                                pts = pts[:, :2].copy()
                                pts[:, 0] -= ux
                                pts[:, 1] -= uy
                                cv2.fillPoly(mask2, [np.round(pts).astype(np.int32)], 1)
                            inter = np.logical_and(mask1, mask2).sum()
                            union = np.logical_or(mask1, mask2).sum()
                            if union > 0:
                                return float(inter) / float(union)
        bbox1, bbox2 = ann1.get('bbox'), ann2.get('bbox')
        if bbox1 and bbox2:
            return calculate_iou(bbox1, bbox2)
        return 0.0

    def _filter_duplicate_annotations(self, new_annotations, existing_annotations, iou_threshold=0.4):
        """使用匈牙利匹配算法过滤与已有标注重复的新标注，返回不重复的标注列表"""
        if not existing_annotations:
            return list(new_annotations)
        if not new_annotations:
            return []
        n_new, n_exist = len(new_annotations), len(existing_annotations)
        iou_matrix = np.zeros((n_new, n_exist))
        for i, new_ann in enumerate(new_annotations):
            for j, exist_ann in enumerate(existing_annotations):
                iou_matrix[i, j] = self._calculate_annotations_iou(new_ann, exist_ann)
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(1 - iou_matrix)
        matched_indices = set()
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] > iou_threshold:
                matched_indices.add(i)
        return [ann for i, ann in enumerate(new_annotations) if i not in matched_indices]

    def _filter_duplicate_annotations_hungarian(self, new_annotations, existing_annotations, iou_threshold=0.4):
        """使用匈牙利算法过滤与已有标注重复的新标注"""
        return self._filter_duplicate_annotations(new_annotations, existing_annotations, iou_threshold)

    def _filter_and_add_annotations(self, new_annotations, existing_annotations, iou_threshold=0.4):
        """过滤新标注后追加到已有标注列表"""
        filtered = self._filter_duplicate_annotations(new_annotations, existing_annotations, iou_threshold)
        return list(existing_annotations) + list(filtered)

    def _discover_new_categories(self, annotations, existing_categories):
        """从标注中发现新类别并返回 (label, color) 列表（兼容别名，Phase 4 删除）"""
        return self._find_new_categories(annotations, existing_categories)

    def _get_category_color(self, label):
        """从固定颜色表获取类别颜色（md5 哈希保证跨进程一致性，返回 list 格式用于序列化）"""
        import hashlib
        from . import CATEGORY_COLOR_PALETTE
        idx = int(hashlib.md5(label.encode()).hexdigest(), 16) % len(CATEGORY_COLOR_PALETTE)
        return list(CATEGORY_COLOR_PALETTE[idx])

    def _find_new_categories(self, annotations, existing_categories):
        """从标注中提取不在现有类别中的新类别，返回 [(label, color_list), ...]"""
        new_cats = {}
        for ann in annotations:
            label = ann.get('label')
            if label and label not in existing_categories and label not in new_cats:
                new_cats[label] = self._get_category_color(label)
        return [(label, color) for label, color in new_cats.items()]

    def _normalize_yolo_inference_ann(self, ann, current_category="defect"):
        """归一化 YOLO 训练模型推理结果，统一补齐 label 和 bbox"""
        self._normalize_annotation(ann, current_category)

    def _convert_single_ann_to_obb(self, ann):
        """将单个标注就地转换为 OBB 格式"""
        data = self._convert_annotation_mode(ann, 'obb')
        ann.clear()
        ann.update(data)

    def _convert_single_ann_to_det(self, ann):
        """将单个标注就地转换为检测 (det) 格式"""
        data = self._convert_annotation_mode(ann, 'det')
        ann.clear()
        ann.update(data)

    # ====== 区域: 模型工具 ======
    # 原 annotation_io: ensure_yolo_model

    def ensure_yolo_model(self, model_path):
        """兼容旧接口：校验模型文件是否存在（已废弃，保留避免外部引用报错）。"""
        result = {"exists": False, "model_path": model_path, "error": None}
        if model_path and os.path.exists(model_path):
            result["exists"] = True
        elif model_path:
            result["error"] = f"模型文件不存在: {model_path}"
        else:
            result["error"] = "未提供模型路径"
        return result

    # ====== 区域: 示例条目构建 ======
    # 原 annotation_io: build_example_item

    def build_example_item(self, image_path, annotation, name,
                           default_category=None, roi_bbox=None):
        final_label = annotation.get('label') or default_category or "default"
        return {'image_path': image_path, 'annotation': annotation,
                'name': name, 'category': final_label}

    # ====== 区域: 批量标注命令（原 BatchVM）======
    # set_model_type / set_interactive_model_type / change_model / select_trained_model /
    #   ensure_yolo_model_loaded / list_custom_models / list_builtin_models /
    #   one_click_by_current / process_batch_all / cancel_batch / convert_all_with_ai /
    #   _discover_categories / discover_categories / merge_annotations
    # 信号已弃（model_changed / batch_running_changed / batch_progress_changed 未在界面连接）

    def update_model_type(self, model_type):
        """设置自动标注模型类型。"""
        self.model_type = model_type

    def update_interactive_model_type(self, model_type):
        """设置交互式模型类型。"""
        self.interactive_model_type = model_type

    def change_model(self, model_name, is_interactive=False, role=None):
        """切换当前模型并触发异步加载。返回 True 表示模型已切换。
        role: None(自动按 is_interactive) / "interactive" / "example" / "plain" / "refine"
        """
        if role is None:
            role = "interactive" if is_interactive else "example"
        attr_map = {
            "interactive": "interactive_model_type",
            "example": "model_type",
            "plain": "plain_model_type",
            "refine": "model_type",
        }
        attr = attr_map.get(role, "model_type")
        if getattr(self, attr) == model_name:
            return False
        setattr(self, attr, model_name)
        if self.model_manager is not None:
            self.model_manager.load_model_async(model_name, is_interactive=(role == "interactive"))
        return True

    @action("resource.set_model", description="设置指定角色的模型。\n- role：模型角色（必填），枚举含义如下：\n    - interactive=交互式分割模型（SAM 等），用于交互式分割操作\n    - auto=自动标注模型，用于 mode=example（基于示例）的一键自动标注\n    - plain=普通模型（已训练的 YOLO 等），用于 mode=model（基于普通模型）的一键自动标注\n    - refine=细化模型，用于标注细化操作\n- model_name：模型名称，必须存在于对应角色的可用模型列表中（plain 需先用 add_custom_model 注册，可用 resource.list_custom_models(role='plain') / resource.list_builtin_models(role='plain') 查看）\n- 提示：若要执行一键标注 mode=model，必须先 set_model(role='plain', ...) 选择普通模型；若 role='plain' 列表为空则当前没有普通模型，不应使用 mode=model\n- 切换前自动释放旧模型的资源\n- 切换后异步加载新模型\n- 可通过 get_state / list_custom_models 查看当前模型\n- 影响相应角色的一键标注操作", category="自动标注", params={"role": {"type": "str", "required": True, "description": "模型角色：interactive/auto/plain/refine，含义见描述"}, "model_name": "str"}, scope="agent")
    def set_model(self, role: str = "auto", model_name: str = ""):
        """设置指定角色的模型。成功返回 True，失败返回错误字符串。"""
        configs = {
            "interactive": {
                "models": ModelFactory.get_interactive_models,
                "role": "interactive",
                "setting": "default_interactive_model",
                "label": "交互模型",
            },
            "auto": {
                "models": ModelFactory.get_auto_models,
                "role": "example",
                "setting": "example_model",
                "label": "自动标注模型",
            },
            "plain": {
                "models": ModelFactory.get_plain_models,
                "role": "plain",
                "setting": "plain_model",
                "label": "普通模型",
            },
        }
        cfg = configs.get(role)
        if cfg is None:
            return "错误: 未知的模型角色，可选 interactive/auto/plain"
        if not model_name:
            return "错误: 未指定模型名称"
        if model_name not in cfg["models"]():
            hint = "，请先使用 add_custom_model 添加" if role == "plain" else ""
            return f"错误: {cfg['label']} '{model_name}' 未注册{hint}"
        changed = self.change_model(model_name, is_interactive=(role == "interactive"), role=cfg["role"])
        if changed:
            settings.set(cfg["setting"], model_name)
        return True

    def update_plain_model_type(self, model_type):
        """设置普通模型类型（不触发加载）。"""
        self.plain_model_type = model_type

    @action("resource.select_trained_model", description="选择已训练模型权重文件，自动注册为普通模型并设为当前模型。\n- model_path: 模型文件路径（如 .pt 文件）\n- 注册的模型会出现在模型菜单的普通模型子菜单中\n- 可通过 set_model(role=plain) 切换，或 one_click_by_model 批量标注", category="自动标注", params={"model_path": "str"}, scope="agent")
    def select_trained_model(self, model_path):
        """选择已训练模型权重文件，自动注册为普通模型并设为当前模型。成功返回 True，失败返回错误字符串。"""
        if not model_path or not os.path.exists(model_path):
            return "错误: 模型文件不存在"
        name = os.path.splitext(os.path.basename(model_path))[0]
        existing = ModelFactory.get_custom_model_info(name)
        if not existing:
            ok = ModelFactory.register_custom_model(name, "yolo_trained", model_path, role="plain")
            if not ok:
                return "错误: 注册普通模型失败"
            models = get_custom_models("plain")
            models.append({"name": name, "category": "yolo_trained", "weight_path": model_path})
            save_custom_models("plain", models)
        self.plain_model_type = name
        settings.set("plain_model", name)
        settings.set("last_trained_model_path", model_path)
        return True

    def ensure_yolo_model_loaded(self):
        """校验普通模型可用（当前已选或从最近使用路径迁移）。失败返回 False。"""
        if self.plain_model_type and self.plain_model_type in ModelFactory.get_plain_models():
            return True
        last_model = settings.get("last_trained_model_path", "")
        if last_model and os.path.exists(last_model):
            try:
                result = self.select_trained_model(last_model)
                return result is True
            except Exception:
                return False
        return False

    def is_plain_model_available(self):
        """校验当前已选择的普通模型可用（已选择且已注册）。"""
        return self.ensure_yolo_model_loaded()

    @action("resource.list_custom_models", read_only=True, description="列出所有自定义模型及其详细信息。\n- role：可选过滤参数，枚举含义：interactive=交互式分割模型、auto=自动标注模型(example模式用)、plain=普通模型(model模式用)、refine=细化模型；留空显示全部\n- 返回值：每个自定义模型的名称、类别、权重路径和角色\n- 自定义模型是通过 add_custom_model 添加的模型", category="自动标注", params={"role": "str"}, scope="agent")
    def list_custom_models(self, role=""):
        return self.list_custom_models_text(role)

    @action("resource.list_builtin_models", read_only=True, description="列出所有内置模型，按角色分组。\n- role：可选过滤参数，枚举含义：interactive=交互式分割模型、auto=自动标注模型(example模式用)、plain=普通模型(model模式用)、refine=细化模型；留空显示全部\n- 返回值：每个内置模型的名称和类别\n- 可通过 set_enabled_models 控制哪些内置模型显示在菜单中", category="自动标注", params={"role": "str"}, scope="agent")
    def list_builtin_models(self, role=""):
        return self.list_builtin_models_text(role)

    def convert_all_with_ai(self, image_paths, source_mode='det', target_mode='seg',
                            progress_callback=None, cancel_event=None):
        """AI 全量转换标注。返回 service 结果 dict。

        `cancel_event`：可选的 threading.Event，由界面上的"取消"按钮置位（审计 D9 余项）。
        原先这个对话框的取消按钮**没有接到任何逻辑**，按下去毫无作用；现在把
        event 一路传到逐图循环，在每个图片边界检查并提前收尾。
        """
        return self.batch_ai_convert(
            image_paths, source_mode, target_mode,
            output_dir=self.context.output_dir,
            categories=self.context.categories,
            progress_callback=progress_callback,
            cancel_event=cancel_event
        )

    def cancel_batch_convert(self):
        """请求取消正在进行的 AI 全量转换（下一个图片边界生效）。"""
        ev = getattr(self, "_batch_cancel_event", None)
        if ev is not None:
            ev.set()
            return True
        return False

    @action("batch.one_click_by_current", background=True, description="按模式对当前图片进行自动标注并写入数据。\n- mode: 标注模式（必填），text=基于文本 / example=基于示例 / model=基于普通模型\n- text: mode=text 时的描述文本，留空则使用提示词库中选中的提示词\n- mode=model 时需先通过 resource.set_model(role='plain', ...) 选择普通模型（详见 set_model 的 role 说明）\n- 标注结果自动保存到磁盘，标注结果已通过 annotation:annotations_changed 事件自动刷新界面\n- 用户仅说自动标注时：优先基于文本，提示词库无选中提示词则基于示例；示例库也为空且无普通模型（role='plain'）时，**停止并询问用户**标注方式，不得自行扫描磁盘注册模型\n- 返回「模型正在后台加载，请稍候...」时最多重试 1 次，仍加载中则告知用户稍后重试，不得连续重试", category="自动标注", params={"mode": {"type": "str", "required": True}, "text": "str"}, scope="agent")
    def one_click_by_current(self, mode="", text="", reference_annotation=None, task_mode=None):
        """按 mode 标注当前图片（text/model/example），结果写入磁盘。返回含 annotations/new_categories 的结果字典。"""
        # 任务模式跟随当前项目/画布(手动路径传 draw_area.task_mode,agent 路径
        # 用 context),否则 det 项目会被硬编码 seg 渲染成掩码
        task_mode = task_mode or self.context.task_mode or "seg"
        mode = (mode or "").strip()
        if mode not in ("text", "model", "example"):
            return {"status": "error", "message": "未知的标注模式: %s（可选 text/model/example）" % mode}
        if self.model_manager is not None and self.model_manager.is_model_loading():
            return {"status": "error", "message": "模型正在后台加载，请稍候..."}
        if not self.manual or not self.manual.current_image_path:
            return {"status": "error", "message": "请先选择一张图片"}
        if mode == "model" and not self.is_plain_model_available():
            return {"status": "error", "message": "未选择普通模型，请先在模型菜单中配置"}
        # 示例模式优先使用用户勾选的示例（与 process_batch_all 一致），
        # 未选择时回退整个示例库，避免忽略选择导致多个示例同时命中产生重复结果
        library = self.example_library
        if mode == "example":
            pre_selected = self.get_selected_example_items()
            if pre_selected:
                library = pre_selected
        res = self.annotate_by_current(
            self.manual.current_image_path,
            mode=mode, text=text,
            prompt_list=self.context.prompt_library,
            model_type=self.plain_model_type if mode == "model" else self.model_type,
            library=library,
            reference_annotation=reference_annotation,
            existing_annotations=None,
            categories=self.context.categories,
            rules=self.context.current_project_rules,
            task_mode=task_mode,
            current_category=self.context.current_category,
            _persist=True,
        )
        if res['status'] == 'success' and mode == "text":
            self.context.current_project_rules = res['rules']
            self.save_rules(self.context.current_project_rules)
            # 仅当外部显式传入 text 时才持久化到提示词库，避免把库内已勾选提示词
            # 的逗号拼接串拆分后重复写回，导致后续自动标注误用额外提示词。
            if text and text.strip():
                self.add_prompt(text)
        # 自动标注写入数据后发布信号，驱动界面自动刷新（不再依赖调用方手动调 refresh_view）
        if res.get('status') == 'success' and self.manual is not None:
            self.manual._emit("annotation.annotations_changed",
                              {"image_path": self.manual.current_image_path, "source": "one_click_by_current"})
        return res

    @action("batch.one_click_by_all", background=True, description="按模式对全部图片进行批量自动标注。\n- mode: 标注模式（必填），text=基于文本 / example=基于示例 / model=基于普通模型\n- text: mode=text 时的描述文本（可选），留空则使用提示词库中选中的提示词\n- mode=model 时需先在模型菜单选择普通模型\n- 标注结果已通过 annotation:annotations_changed 事件自动刷新界面", category="自动标注", params={"mode": {"type": "str", "required": True}, "text": "str"}, scope="agent")
    def one_click_by_all(self, mode: str = "", text: str = "", selected_examples: list = None):
        """批量处理全部图片。返回 {status, total, processed, errors}。"""
        image_files = list(self.manual.image_files) if self.manual is not None else []
        return self.process_batch_all(
            mode=mode, text=text, selected_examples=selected_examples,
            image_files=image_files, task_mode=self.context.task_mode,
        )

    def process_batch_all(self, mode, text="", selected_examples=None, image_files=None,
                          task_mode="seg", model_type=None, progress_callback=None):
        """批量处理所有图片，内部统一走 annotate_by_current。
        mode: text / model / example
        返回 {status, total, processed, errors}。
        """
        if not image_files:
            return {"status": "error", "total": 0, "processed": 0, "errors": ["请先打开包含图片的目录"]}
        if self.model_manager is not None and self.model_manager.is_model_loading():
            return {"status": "error", "total": len(image_files), "processed": 0, "errors": ["模型正在后台加载，请稍候..."]}

        # 预校验 + 输入解析（与 one_click_by_current 对齐）
        rules = dict(self.context.current_project_rules or {})
        predict_model_type = model_type

        if mode == "text":
            if not text:
                return {"status": "error", "total": len(image_files), "processed": 0, "errors": ["缺少文本提示词"]}
            rules['text'] = text
            self.context.current_project_rules = rules
            self.save_rules(rules)
            predict_model_type = model_type or self.model_type
        elif mode == "model":
            mt = model_type or self.plain_model_type
            if not mt or mt not in ModelFactory.get_plain_models():
                return {"status": "error", "total": len(image_files), "processed": 0, "errors": ["未选择可用的普通模型，请先在模型菜单中配置"]}
            predict_model_type = mt
        elif mode == "example":
            if selected_examples is None:
                pre_selected = self.get_selected_example_items()
                selected_examples = pre_selected if pre_selected else list(self.example_library)
            if not selected_examples:
                return {"status": "error", "total": len(image_files), "processed": 0, "errors": ["示例库为空且没有参考示例"]}
            example_label = selected_examples[0].get('category') or \
                            selected_examples[0].get('annotation', {}).get('label', self.context.current_category or 'default')
            # 浅拷贝后再补充 category，避免污染示例库中的共享 dict
            selected_examples = [dict(ex) for ex in selected_examples]
            for ex in selected_examples:
                if 'category' not in ex:
                    ex['category'] = example_label
            rules.pop('text', None)
            self.context.current_project_rules = rules
            self.save_rules(rules)
            predict_model_type = model_type or self.model_type
        else:
            return {"status": "error", "total": len(image_files), "processed": 0, "errors": [f"未知的标注模式: {mode}"]}

        # 逐图走 annotate_by_current（复用单图路径的解析→预测→去重→保存）
        self.one_click_running = True
        self.batch_cancel_requested = False
        total = len(image_files)
        processed = 0
        errors = []
        # 通知界面弹出进度对话框：主线程发起（菜单）时事件同步分发，
        # agent 后台线程发起时 EventBus 会把事件 marshal 回主线程，两条路径都能显示
        self._publish_batch_progress({"phase": "start", "mode": mode, "total": total})
        try:
            for img_path in image_files:
                if self.batch_cancel_requested:
                    break
                res = self.annotate_by_current(
                    image_path=img_path,
                    mode=mode,
                    text=text if mode == "text" else "",
                    prompt_list=self.context.prompt_library,
                    model_type=predict_model_type,
                    library=self.example_library if mode == "example" else None,
                    reference_annotation=None,
                    existing_annotations=None,
                    categories=self.context.categories,
                    rules=rules,
                    task_mode=task_mode,
                    current_category=self.context.current_category,
                    _persist=True,
                )
                if res['status'] == 'success':
                    for label, color in res.get('new_categories', []):
                        self.context.categories[label] = color
                else:
                    errors.append(f"{os.path.basename(img_path)}: {res.get('message', '未知错误')}")
                processed += 1
                self._publish_batch_progress({
                    "phase": "progress", "mode": mode,
                    "processed": processed, "total": total, "image": img_path,
                })
                if progress_callback is not None:
                    progress_callback(processed, total, img_path, res)
        finally:
            canceled = self.batch_cancel_requested
            self.one_click_running = False
            self.one_click_paused = False
            self.batch_cancel_requested = False
            self._publish_batch_progress({
                "phase": "finished", "mode": mode, "canceled": canceled,
                "processed": processed, "total": total, "errors": len(errors),
            })
        return {"status": "success", "total": total, "processed": processed,
                "canceled": canceled, "errors": errors}

    def _publish_batch_progress(self, payload):
        """发布批量自动标注进度事件（界面订阅后显示/更新进度对话框）。

        进度事件是纯 UI 反馈，发布失败（未注册事件总线等）不应影响标注本身。
        """
        if self.manual is None:
            return
        try:
            self.manual._emit(StateType.BATCH_PROGRESS, payload)
        except Exception as e:
            print(f"[AutoAnnotationService] 发布批量标注进度失败: {e}")

    def cancel_batch(self):
        self.batch_cancel_requested = True
        self.one_click_running = False
        self.one_click_paused = False

    def _discover_categories(self, annotations):
        """从新标注中发现未登记类别并加入类别字典。返回新类别标签列表。"""
        discovered = self._find_new_categories(annotations, self.context.categories)
        new_labels = []
        for label, color in discovered:
            self.context.categories[label] = color
            new_labels.append(label)
        return new_labels

    def discover_categories(self, annotations):
        """公共命令：从新标注中发现并登记新类别。返回新类别标签列表。"""
        return self._discover_categories(annotations)

    def merge_annotations(self, new_annotations, existing_annotations):
        """将新标注与现有标注去重合并。返回新增标注列表。"""
        return self._filter_and_add_annotations(
            new_annotations, existing_annotations
        )

    # ====== 区域: 示例库 & 提示词 & 规则命令（原 BatchVM）======
    # list_examples / load_support_sets / set_examples / get_selected_examples /
    #   clear_examples_selection / set_as_example / set_as_example_by_index /
    #   list_prompts / add_prompt / delete_prompt / clear_prompts /
    #   load_prompt_library / save_prompt_library / get_rules / set_rule /
    #   save_rules / _save_non_project_config
    # get_example_library / remove_example / clear_example_library /
    #   add_custom_model / remove_custom_model 已位于上方相应区域

    @action("resource.list_examples", read_only=True, description="列出当前示例库中的所有示例。\n- 返回值：结构化示例列表（每个示例包含 index、name、category、image_path、folder_path）\n- 示例可通过 set_as_example 添加\n- 可用 set_examples 选择要使用的示例\n- 结合 one_click_by_example 使用", category="自动标注", scope="agent")
    def list_examples(self):
        return self.list_examples_data()

    def load_support_sets(self):
        """加载支持集示例库目录。返回示例数量。"""
        self._load_support_sets_dir()
        return len(self.example_library)

    @action("resource.set_examples", description="选择要用于 one_click_by_example 的示例。\n- example_names：示例名称列表，可通过 list_examples 获取可用示例名称\n- 选择后 one_click_by_example 将使用这些示例进行自动标注\n- 返回值包含选中数量和无效名称信息", category="自动标注", params={"example_names": "list"}, scope="agent")
    def set_examples(self, example_names=None):
        if not example_names:
            self.example_selector.clear_selected_examples()
            self._persist_selected_examples()
            return {"status": "success", "message": "已清空示例选择", "selected": []}
        library = self.example_library or []
        available_names = ExampleSelectionManager.get_available_names(library)
        result = self.example_selector.set_selected_examples(example_names, available_names)
        # 仅在校验通过时落盘：若名称无效（如示例库为空），保持磁盘上已有的选中值，
        # 避免把空列表覆盖回配置文件
        if result.get("status") == "success":
            self._persist_selected_examples()
        return result

    @action("resource.get_selected_examples", read_only=True, description="获取当前选中的示例名称列表。\n- 返回值：当前已选示例的名称列表\n- 可通过 set_examples 修改选择", category="自动标注", scope="agent")
    def get_selected_examples(self):
        return self.example_selector.get_selected_examples()

    @action("resource.clear_examples_selection", description="清空当前示例选择。\n- 取消所有已选示例\n- 之后 one_click_by_example 将自动选择或弹出选择对话框", category="自动标注", scope="agent")
    def clear_examples_selection(self):
        self.example_selector.clear_selected_examples()
        self._persist_selected_examples()

    def _persist_selected_examples(self):
        """将示例选中状态写入项目设置（非项目模式仅内存，不落盘）。"""
        if getattr(self.context, "_is_non_project_mode", False):
            return
        try:
            project_settings.set("selected_examples", list(self.example_selector.get_selected_examples()))
        except Exception as e:
            print(f"[AutoAnnotationService] 示例选中持久化失败: {e}")

    def load_selected_examples(self):
        """项目加载/示例库刷新后，从项目设置恢复示例选中状态（非项目模式跳过）。"""
        if getattr(self.context, "_is_non_project_mode", False):
            return
        try:
            names = project_settings.get("selected_examples") or []
            available = ExampleSelectionManager.get_available_names(self.example_library or [])
            valid = [n for n in names if n in available]
            if valid:
                self.example_selector.set_selected_examples(valid, available)
            else:
                self.example_selector.clear_selected_examples()
        except Exception as e:
            print(f"[AutoAnnotationService] 示例选中恢复失败: {e}")

    def _add_example_from_annotation(self, annotation, example_name="", image_path=None, roi_bbox=None):
        """将标注加入示例库并设为参考示例。返回示例名称，失败返回 None。"""
        # 深拷贝后再修改，避免污染画布/缓存中的共享标注数据
        annotation = copy.deepcopy(annotation) if isinstance(annotation, dict) else annotation
        if self.manual is not None:
            self.manual.reference_annotation = annotation
        name = example_name or f"示例_{len(self.example_library) + 1}"
        final_label = annotation.get('label') or self.context.current_category
        annotation['label'] = final_label
        example_item = self.build_example_item(
            image_path, annotation, name, default_category=self.context.current_category
        )
        result = self.add_example(example_item, roi_bbox=roi_bbox)
        if isinstance(result, dict) and result.get("status") == "error":
            # 保存失败必须让上层知道,否则 UI 假成功且示例库保持为空
            self._last_example_error = result.get("message", "未知错误")
            print(f"[AutoAnnotationService] 示例保存失败: {self._last_example_error}")
            return None
        return name

    def set_as_example_by_index(self, index, annotations, image_path, example_name="", persistent_roi=None):
        """按索引将标注加入示例库（含 ROI 判定）。返回示例名称，索引无效返回 None。"""
        if index < 0 or index >= len(annotations):
            return None
        ann = annotations[index]
        roi_bbox = None
        if persistent_roi:
            ann_bbox = ann.get('bbox')
            if ann_bbox and self.manual is not None and self.manual.check_annotation_in_roi(ann_bbox, persistent_roi):
                roi_bbox = persistent_roi
        return self._add_example_from_annotation(ann, example_name, image_path, roi_bbox)

    @action("resource.set_as_example", description="将当前图片第 index 个已保存标注设为支持集示例（用于 few-shot 自动标注）。\n- index：标注索引（从0开始），对应当前图片已保存的标注列表\n- example_name：示例名称（必需），不可留空\n- 示例存储在项目目录下的 support_sets/ 中\n- 设置后可用 one_click_by_example() 进行相似目标标注", category="标注", params={"index": "int", "example_name": "str"}, scope="agent")
    def set_as_example(self, index: int = -1, example_name: str = ""):
        """将当前图片第 index 个已保存标注设为示例。成功返回 True，失败返回错误字符串。"""
        if not example_name:
            return "错误: 需要 example_name 参数"
        if self.manual is None or not self.manual.current_image_path:
            return "错误: 当前没有打开图片"
        annotations = self.manual.get_cached_annotations(
            self.manual.current_image_path,
            self.context.output_dir,
            self.context._is_non_project_mode,
            self.context._non_project_format,
        )
        if not annotations:
            return "错误: 当前图片没有已保存的标注，请先保存标注后再设为示例"
        if index < 0 or index >= len(annotations):
            return "错误: 标注索引无效"
        name = self.set_as_example_by_index(index, annotations, self.manual.current_image_path, example_name)
        if name is None:
            err = getattr(self, "_last_example_error", "")
            self._last_example_error = ""
            return f"错误: {err or '标注索引无效或示例保存失败'}"
        return True

    @action("resource.list_prompts", read_only=True, description="列出所有提示词及其状态。\n- 返回值：每个提示词的索引、文本和选中状态\n- 提示词用于 one_click_by_text 自动标注\n- 可通过 add_prompt / delete_prompt 管理", category="自动标注", scope="agent")
    def list_prompts(self):
        return PromptManager.list_prompts_text(self.context.prompt_library)

    @action("resource.add_prompt", description="添加一个或多个提示词到提示词库。\n- text：提示词文本，支持逗号分割同时添加多个\n- 添加后自动保存到项目设置\n- 提示词用于 one_click_by_text 自动标注", category="自动标注", params={"text": "str"}, scope="agent")
    def add_prompt(self, text):
        if not text:
            return "错误: 提示词不能为空"
        old_len = len(self.context.prompt_library)
        self.context.prompt_library = PromptManager.add_prompt(self.context.prompt_library, text)
        self.save_prompt_library()
        count = len(self.context.prompt_library) - old_len
        return f"已添加 {count} 个提示词，当前共 {len(self.context.prompt_library)} 个"

    @action("resource.delete_prompt", description="从提示词库中删除指定索引的提示词。\n- index：提示词索引（从0开始），可通过 list_prompts 获取", category="自动标注", params={"index": "int"}, scope="agent")
    def remove_prompt(self, index):
        # 兼容 agent 以字符串传入索引（如 "0"），统一转为 int 后再做比较
        try:
            index = int(index)
        except (TypeError, ValueError):
            return f"错误: 索引 {index} 无效"
        self.context.prompt_library, msg = PromptManager.delete_prompt(
            self.context.prompt_library, index
        )
        if self.context.prompt_library is not None:
            self.save_prompt_library()
        return msg

    @action("resource.clear_prompts", description="清空所有提示词。\n- 删除提示词库中所有条目\n- 操作不可恢复\n- 清空后用 one_click_by_text 前需重新添加提示词", category="自动标注", scope="agent")
    def clear_prompts(self):
        self.context.prompt_library, msg = PromptManager.clear_prompts(
            self.context.prompt_library
        )
        self.save_prompt_library()
        return msg

    def load_prompt_library(self):
        self.context.prompt_library = PromptManager.load_prompt_library(project_settings)

    def save_prompt_library(self):
        PromptManager.save_prompt_library(self.context.prompt_library, project_settings)

    @action("resource.get_rules", read_only=True, description="获取当前项目的自动标注规则配置（含启用状态）。\n"
                "- 返回值：每个规则字段包含 value（当前值）和 enabled（是否启用）\n"
                "- enabled=true 表示该规则已启用并会在自动标注时应用过滤；enabled=false 表示未启用（不参与过滤）\n"
                "- 规则启用状态与界面复选框一致：置信度默认启用，其他规则默认关闭\n"
                "- text（文本描述）、mode（模式）不算过滤规则，始终存在\n"
                "- 全部过滤规则均为绝对值规则，任何标注模式下都生效：\n"
                "  max_instances=最大保留数量；area_range=[min,max] 面积范围（像素²）；width_range/height_range=[min,max] 宽/高范围（像素）；aspect_ratio_range=[min,max] 宽高比 W/H；gray_range=[min,max] 区域平均灰度(0~255)；conf_threshold=置信度下限(0~1)\n"
                "- 禁用某规则：调用 set_rule(key=\"规则名\", value=\"disabled\")；启用：传入实际值如 set_rule(key=\"area_range\", value=\"[100, 50000]\")",
            category="自动标注", scope="agent")
    def get_rules(self):
        rules = self.context.current_project_rules or {}
        # 所有可能的规则字段（mode 不算过滤规则，text 始终存在）
        all_keys = ["text", "max_instances", "conf_threshold", "area_range",
                    "width_range", "height_range", "aspect_ratio_range",
                    "gray_range", "mode"]
        result = {}
        for key in all_keys:
            if key in rules:
                result[key] = {"value": rules[key], "enabled": True}
            else:
                result[key] = {"value": None, "enabled": False}
        return json.dumps(result, ensure_ascii=False, indent=2)

    @action("resource.set_rule", description="设置或禁用自动标注规则的某个字段。\n"
                "- key：规则字段名，可选值：text（文本描述）、max_instances（最大保留数量）、conf_threshold（置信度阈值 0-1）、area_range（面积范围 [min,max]，像素²）、width_range（宽度范围 [min,max]，像素）、height_range（高度范围 [min,max]，像素）、aspect_ratio_range（宽高比 W/H 范围 [min,max]）、gray_range（灰度范围 [min,max]，0~255）\n"
                "- value：字段值，数字范围类用 JSON 数组如 [0, 10000]；传入 \"disabled\" 表示禁用该规则（从规则中移除，不再参与过滤）\n"
                "- 启用某规则：传入实际值，如 set_rule(key=\"area_range\", value=\"[100, 50000]\")\n"
                "- 禁用某规则：set_rule(key=\"area_range\", value=\"disabled\")\n"
                "- 所有过滤规则都是绝对值规则，任何标注模式下都生效\n"
                "- 设置后立即持久化，后续 one_click_by_text 等标注操作会使用新规则",
            category="自动标注", params={"key": "str", "value": "str"}, scope="agent")
    def set_rule(self, key, value):
        # value="disabled" 表示禁用规则：从 rules 字典中删除该字段
        if value == "disabled":
            if key not in self.VALID_RULE_KEYS:
                return f"错误: 不支持的规则字段 '{key}'，可选: {', '.join(sorted(self.VALID_RULE_KEYS))}"
            new_rules = dict(self.context.current_project_rules or {})
            if key in new_rules:
                del new_rules[key]
                self.context.current_project_rules = new_rules
                self.save_rules(new_rules)
                return f"规则 '{key}' 已禁用"
            return f"规则 '{key}' 本来就未启用"
        new_rules, msg = self._set_rule(self.context.current_project_rules, key, value)
        self.context.current_project_rules = new_rules
        self.save_rules(new_rules)
        return msg

    def save_rules(self, rules=None):
        if rules is not None:
            self.context.current_project_rules = rules
        if self.context._is_non_project_mode:
            self._save_non_project_config()
            return
        ps = self.project_service
        if not self.context.current_project_name or ps is None:
            return
        ps.update_project_rules(self.context.current_project_name, self.context.current_project_rules)
        # 通知规则配置对话框(若打开)刷新:agent set_rule / 推理后规则更新都会走这里
        if self.manual is not None:
            self.manual._emit(StateType.RULES_CHANGED, {
                "rules": self.context.current_project_rules or {},
            })

    def _save_non_project_config(self):
        if not self.context._is_non_project_mode or not self.context.output_dir:
            return
        config = {
            "task_mode": self.context.task_mode,
            "categories": {name: color.name() if hasattr(color, "name") else color
                           for name, color in self.context.categories.items()},
            "current_category": self.context.current_category or "",
            "export_format": self.context._non_project_format or "labelme",
            "rules": self.context.current_project_rules,
            "rules_schema": RULES_SCHEMA_VERSION,
            "hard_samples": [{"path": p, "description": d} for p, d in self.context.hard_samples.items()],
        }
        if self.manual is not None:
            self.manual._save_non_project_config_file(self.context.output_dir, config)
