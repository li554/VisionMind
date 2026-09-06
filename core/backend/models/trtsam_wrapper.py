import os
import cv2
import torch
import gc
import numpy as np
import trtsam.sam3 as trtsam3
from tokenizers import Tokenizer
from .base import ModelInterface, MockBoxes, MockMasks, MockResults
from .registry import register_model
from ..utils import _cleanup_runs

@register_model(
    "trtsam3", img_size=1008,
    weight_path_key="vision_encoder,text_encoder,geometry_encoder,decoder,tokenizer_path",
    category="trtsam3", role="example",
)
class TRTSAMWrapper(ModelInterface):
    """TensorRT SAM 模型包装器"""
    _infer_instance = None
    _tokenizer_instance = None
    _engine_paths = None  # 缓存解析后的引擎路径

    def __init__(self):
        self.infer = None
        self.tokenizer = None

    @staticmethod
    def _parse_weight_path(weight_path: str):
        """从 weight_path 逗号 split 解析路径
        格式: vision_encoder,text_encoder,geometry_encoder,decoder,tokenizer_path
        """
        parts = weight_path.split(",") if weight_path else []
        vision_encoder = parts[0] if len(parts) > 0 else ""
        text_encoder = parts[1] if len(parts) > 1 else ""
        geometry_encoder = parts[2] if len(parts) > 2 else ""
        decoder = parts[3] if len(parts) > 3 else ""
        tokenizer_path = parts[4] if len(parts) > 4 else ""
        return vision_encoder, text_encoder, geometry_encoder, decoder, tokenizer_path

    @classmethod
    def get_infer_instance(cls, engine_paths=None):
        if engine_paths is None:
            engine_paths = cls._engine_paths
        if engine_paths is None:
            print("[TRTSAMWrapper] 错误: 未提供引擎路径")
            return None

        vision_encoder, text_encoder, geometry_encoder, decoder, _ = engine_paths

        if cls._infer_instance is None:
            engines = [vision_encoder, text_encoder, geometry_encoder, decoder]
            missing = [os.path.basename(e) for e in engines if not os.path.exists(e)]
            if missing:
                print(f"[TRTSAMWrapper] 错误: 缺少 TensorRT 引擎文件: {', '.join(missing)}")
                return None

            print(f"[TRTSAMWrapper] 正在加载 TensorRT 引擎...")
            try:
                cls._infer_instance = trtsam3.Sam3Infer.create_instance(vision_encoder, text_encoder, geometry_encoder, decoder, 0)
                if cls._infer_instance is None:
                    print(f"[TRTSAMWrapper] 错误: trtsam3.Sam3Infer.create_instance 返回了 None")
                else:
                    print(f"[TRTSAMWrapper] 引擎加载完成。")
            except Exception as e:
                print(f"[TRTSAMWrapper] 引擎加载失败: {e}")
                cls._infer_instance = None
        return cls._infer_instance

    @classmethod
    def get_tokenizer_instance(cls, tokenizer_path=""):
        if cls._tokenizer_instance is None:
            if tokenizer_path and os.path.exists(tokenizer_path):
                try:
                    print(f"[TRTSAMWrapper] 正在从 {tokenizer_path} 加载分词器...")
                    cls._tokenizer_instance = Tokenizer.from_file(tokenizer_path)
                    cls._tokenizer_instance.enable_padding(length=32, pad_id=49407)
                    cls._tokenizer_instance.enable_truncation(max_length=32)
                    print(f"[TRTSAMWrapper] 分词器加载完成。")
                except Exception as e:
                    print(f"[TRTSAMWrapper] 加载 {tokenizer_path} 失败: {e}")
            else:
                print(f"[TRTSAMWrapper] 未找到分词器文件: {tokenizer_path}")
        return cls._tokenizer_instance

    def load(self, weight_path: str = "") -> bool:
        # 解析 weight_path 获取各模块路径
        engine_paths = self._parse_weight_path(weight_path)
        TRTSAMWrapper._engine_paths = engine_paths
        vision_encoder, text_encoder, geometry_encoder, decoder, tokenizer_path = engine_paths
        self.infer = self.get_infer_instance(engine_paths)
        self.tokenizer = self.get_tokenizer_instance(tokenizer_path)
        return self.infer is not None and self.tokenizer is not None

    def reset_cache(self):
        """重置模型缓存"""
        pass

    def _normalize_text_list(self, texts):
        if not texts:
            return []
        if isinstance(texts, str):
            return [t.strip() for t in texts.split(',') if t.strip()]
        return [str(t).strip() for t in texts if t and str(t).strip()]

    def _register_text(self, text, tokenizer, infer, registered_texts):
        if not text or text in registered_texts:
            return
        encoded = tokenizer.encode(text)
        try:
            infer.setup_text_inputs(text, encoded.ids, encoded.attention_mask)
            registered_texts.add(text)
        except Exception as e:
            print(f"[TRTSAMWrapper] setup_text_inputs 失败 ({text}): {e}")

    def _resolve_class_id(self, detection, text_list):
        """与官方 demo 一致：优先使用引擎返回的 class_name / class_id"""
        class_name = getattr(detection, 'class_name', None)
        if class_name and text_list and class_name in text_list:
            return text_list.index(class_name)

        class_id = getattr(detection, 'class_id', None)
        if class_id is not None and int(class_id) >= 0:
            return int(class_id)
        return 0

    def _apply_nms(self, detections, conf, nms_threshold):
        """按 class_name 分组 NMS，避免不同提示词互相抑制"""
        if len(detections) <= 1:
            return detections

        grouped = {}
        for db in detections:
            key = getattr(db, 'class_name', None) or str(self._resolve_class_id(db, []))
            grouped.setdefault(key, []).append(db)

        filtered = []
        for group in grouped.values():
            if len(group) <= 1:
                filtered.extend(group)
                continue

            nms_boxes = []
            nms_scores = []
            for db in group:
                l, t, r, b = db.box.left, db.box.top, db.box.right, db.box.bottom
                nms_boxes.append([float(l), float(t), float(r - l), float(b - t)])
                nms_scores.append(float(db.score))

            indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf, nms_threshold)
            if len(indices) > 0:
                if isinstance(indices, (list, np.ndarray)):
                    indices = np.array(indices).flatten()
                filtered.extend(group[idx] for idx in indices)
        return filtered

    def setup_geometry_input(self, image: np.ndarray, label: str, prompt_data: list):
        """
        为 TensorRT SAM 3 设置几何提示词（支持图像特征缓存）
        prompt_data: [("pos", [x1, y1, x2, y2]), ...]
        """
        if self.infer is None:
            return False
        
        try:
            self.infer.setup_geometry_input(image, label, prompt_data)
            return True
        except Exception as e:
            print(f"[TRTSAMWrapper] setup_geometry_input 失败: {e}")
            return False

    def predict(self, image, bboxes=None, points=None, labels=None, texts=None, conf=0.5, **kwargs):
        geom_label = kwargs.get('geom_label')
        infer = self.infer
        tokenizer = self.tokenizer
        
        if infer is None or tokenizer is None:
            print("[TRTSAMWrapper] 错误: 模型或分词器未加载")
            return []

        prompts = []
        registered_texts = set()
        text_list = self._normalize_text_list(texts)
        text_to_class_id = {idx: text for idx, text in enumerate(text_list)}

        for text in text_list:
            self._register_text(text, tokenizer, infer, registered_texts)
            if not bboxes and not points:
                prompts.append(trtsam3.Sam3PromptUnit(text))

        if bboxes:
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                xyxy = [float(x), float(y), float(x + w), float(y + h)]
                label = text_list[i] if i < len(text_list) else (text_list[0] if text_list else "")
                if label:
                    self._register_text(label, tokenizer, infer, registered_texts)
                prompts.append(trtsam3.Sam3PromptUnit(label, [("pos", xyxy)]))
        elif points:
            pts_array = np.array(points)
            lbls_array = np.array(labels) if labels is not None else None
            if pts_array.ndim == 2:
                pts_array = pts_array[np.newaxis, :]
                if lbls_array is not None:
                    lbls_array = lbls_array[np.newaxis, :]
            
            for i in range(pts_array.shape[0]):
                geometry_prompts = []
                b_pts = pts_array[i]
                b_lbls = lbls_array[i] if lbls_array is not None else [1] * len(b_pts)
                for pt, lbl in zip(b_pts, b_lbls):
                    type_str = "pos" if int(lbl) == 1 else "neg"
                    geometry_prompts.append((type_str, [float(pt[0]), float(pt[1]), float(pt[0]), float(pt[1])]))
                if geometry_prompts:
                    label = text_list[i] if i < len(text_list) else (text_list[0] if text_list else "")
                    if label:
                        self._register_text(label, tokenizer, infer, registered_texts)
                    prompts.append(trtsam3.Sam3PromptUnit(label, geometry_prompts))

        if not prompts and geom_label is None:
            return []

        merged_res_list = [[] for _ in range(1)]

        if prompts:
            try:
                input_data = trtsam3.Sam3Input(image, prompts, conf)
                results_list = infer.forwards([input_data], return_mask=True)
                for i, res in enumerate(results_list):
                    if i < len(merged_res_list):
                        merged_res_list[i].extend(res)
            except Exception as e:
                print(f"[TRTSAMWrapper] 基础推理失败: {e}")
        
        if geom_label:
            geom_labels = geom_label if isinstance(geom_label, list) else [geom_label]
            input_data_pure = trtsam3.Sam3Input(image, [], conf)
            
            for g_idx, g_label in enumerate(geom_labels):
                g_label_str = g_label if isinstance(g_label, str) else ""
                if not g_label_str:
                    continue
                
                try:
                    results_list = infer.forwards([input_data_pure], geom_label=g_label_str, return_mask=True)
                    for i, res in enumerate(results_list):
                        if i < len(merged_res_list):
                            for r in res:
                                r.class_id = g_idx
                            merged_res_list[i].extend(res)
                except Exception as e:
                    error_msg = str(e)
                    if "Geometry cache not found" in error_msg:
                        print(f"[TRTSAMWrapper] 警告: 找不到标签为 '{g_label_str}' 的几何缓存，跳过。")
                    else:
                        print(f"[TRTSAMWrapper] 几何推理失败 (label={g_label_str}): {e}")
                    continue
        
        nms_threshold = kwargs.get('nms_threshold', 0.5)
        for i in range(len(merged_res_list)):
            merged_res_list[i] = self._apply_nms(merged_res_list[i], conf, nms_threshold)
        
        final_results = []
        for res in merged_res_list:
            if not res:
                final_results.append(MockResults(None, None, image))
                continue

            boxes_xyxy, boxes_conf, boxes_cls, masks_data = [], [], [], []
            names_dict = dict(text_to_class_id)

            for db in res:
                class_id = self._resolve_class_id(db, text_list)
                class_name = getattr(db, 'class_name', None)
                if class_name:
                    names_dict[class_id] = class_name
                elif class_id in text_to_class_id:
                    names_dict[class_id] = text_to_class_id[class_id]

                boxes_xyxy.append([db.box.left, db.box.top, db.box.right, db.box.bottom])
                boxes_conf.append(db.score)
                boxes_cls.append(class_id)

                if db.segmentation and db.segmentation.mask is not None:
                    m = db.segmentation.mask
                    if m.ndim == 3:
                        m = m[0] if m.shape[0] == 1 else m[:, :, 0]
                    m = m.astype(np.float32) / (255.0 if m.dtype == np.uint8 and m.max() > 1 else 1.0)
                    h, w = image.shape[:2]
                    full_mask = np.zeros((h, w), dtype=np.float32)
                    l, t, r, b = map(int, [db.box.left, db.box.top, db.box.right, db.box.bottom])
                    l, t, r, b = max(0, l), max(0, t), min(w, r), min(h, b)
                    bw, bh = r - l, b - t
                    if bw > 0 and bh > 0:
                        roi_mask = cv2.resize(m, (bw, bh), interpolation=cv2.INTER_LINEAR)
                        full_mask[t:b, l:r] = (roi_mask > 0.5).astype(np.float32)
                    masks_data.append(full_mask)
                else:
                    masks_data.append(np.zeros((image.shape[0], image.shape[1]), dtype=np.float32))

            final_results.append(MockResults(
                MockBoxes(boxes_xyxy, boxes_conf, boxes_cls),
                MockMasks(np.stack(masks_data) if masks_data else None),
                image,
                names=names_dict if names_dict else None
            ))

        _cleanup_runs()
        return final_results

    def release(self):
        if TRTSAMWrapper._infer_instance is not None:
            print("[TRTSAMWrapper] 彻底销毁 TensorRT 引擎实例")
            TRTSAMWrapper._infer_instance = None
            self.infer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
