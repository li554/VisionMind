import os
import json
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

from core.common.image_utils import imread_unicode
from .core import example_predict, get_example_predictor
from .utils import (
    grid_concatenate_mode, copy_paste_mode, imwrite_unicode
)


def unified_inference(support_images=None, support_infos=None, query_images=None,
                      texts=None, bboxes=None, points=None, labels=None,
                      conf=0.5, model_type="sam3", rules=None, mode="reuse",
                      verbose=True, refine=True, smooth=False,
                      smooth_epsilon=0.001, img_size=640, geom_label=None, **kwargs):
    """统一推理编排：文本 / 示例 / 普通模型三类输入走同一入口。

    - 示例模式: 传 support_images + support_infos（支持集），内部按 mode 分派 fs_/reuse/grid/copy_paste
    - 文本模式: 不传支持集，仅传 texts
    - 普通模型模式: 不传支持集，仅传 model_type（普通模型名，权重由 ModelFactory 解析）
    返回列表的列表: all_results[i] 为第 i 张查询图像的推理结果列表。
    """
    print(f"DEBUG: unified_inference called. model_type={model_type}, mode={mode}")
    query_images = query_images or []

    # 无支持集直通分支：文本 / 普通模型模式，逐图转发 example_predict
    if not support_images:
        if verbose:
            print(f"开始统一推理（无支持集）: 模型={model_type}, 图像数={len(query_images)}")
        all_results = []
        for q_image in query_images:
            if q_image is None:
                print("[unified_inference] 错误: 图像为空")
                continue
            results = example_predict(
                q_image, bboxes=bboxes, points=points, labels=labels, texts=texts,
                conf=conf, model_type=model_type, geom_label=geom_label,
                refine=refine, smooth=smooth, smooth_epsilon=smooth_epsilon,
                img_size=img_size, **kwargs
            )
            all_results.append(results)
        return all_results

    if verbose:
        print(f"开始自动标注, 模式: {mode}")

    # For sam3, force grid mode if reuse is requested as sam3 doesn't support reuse yet
    if model_type == "sam3" and mode == "reuse":
        if verbose:
            print("[auto_annotate] SAM3 目前不支持 reuse 模式，将自动切换为 grid 模式处理")
        mode = "grid"
        
    # Pre-process for FS models (Move out of reuse mode block)
    support_masks = []
    if model_type.startswith("fs_"):
        # Ensure we have polygons list of correct length

        for img, polygons_list in zip(support_images, support_infos["polygons"]):
            if not polygons_list: continue
        
            for poly in polygons_list:
                # Create mask
                mask = np.zeros(img.shape[:2], dtype=np.uint8)

                # Draw polygons
                for p in poly:
                    # Ensure points are integer and correct shape
                    pts = np.array(p, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 1)
                
                support_masks.append(mask)
        
        if verbose:
             print(f"[auto_annotate] FS模式: 准备了 {len(support_masks)} 个支持样本")

    # 3. 如果是 reuse 模式，预提取支持集的几何特征
    all_geom_features = None

    if mode == "reuse":
        if model_type == "trtsam3":
            model = get_example_predictor().get_model("trtsam3")
            
            # 复杂的文本标签分配策略：增强跨图示例的独立特征表示
            original_text = rules.get('text', "") if rules else ""
            sub_prompts = [p.strip() for p in original_text.split(',')] if original_text else []
            
            # 统计有效示例数量
            valid_support_count = sum(len(bboxes) for bboxes in support_infos["boxes"] if bboxes)
            
            # 准备标签列表
            final_labels = []
            if original_text:
                if len(sub_prompts) >= valid_support_count:
                    # 如果分割后的文本够用，直接取前 N 个
                    final_labels = sub_prompts[:valid_support_count]
                else:
                    # 如果不够用，则全部使用原始文本填充
                    final_labels = [original_text] * valid_support_count
                
                # 确保每个标签都是唯一的（通过在末尾添加空格）
                seen_labels = {}
                for i in range(len(final_labels)):
                    base_label = final_labels[i]
                    if base_label in seen_labels:
                        seen_labels[base_label] += 1
                        # 增加空格使其不一致，但不影响 Tokenizer 语义
                        final_labels[i] = base_label + (" " * seen_labels[base_label])
                    else:
                        seen_labels[base_label] = 0
            
            # 5. 确保全局标签唯一性（通过空格）
            global_seen_labels = {}
            
            print(f"Reuse 模式: 正在为 {len(support_images)} 张支持图像缓存特征 (TensorRT)...")
            registered_geom_labels = []
            current_valid_idx = 0
            # 同时遍历支持集图像及其对应的 BBoxes, Class IDs 和 Labels
            for idx, (img, bboxes) in enumerate(zip(
                support_images, 
                support_infos["boxes"],
            )):
                if not bboxes: continue
                
                # 根据策略分配基础标签 (geom_label)
                if final_labels and current_valid_idx < len(final_labels):
                    base_label = final_labels[current_valid_idx].strip()
                else:
                    base_label = f"geom_{idx}"
                
                # --- 直接使用原始图像和标注 ---
                input_img = img
                if bboxes:
                    for bbox in bboxes:
                        x, y, w, h = bbox
                        # 核心修复：针对 TensorRT 静态 Batch=1 引擎
                        # 将每个 BBox 作为独立的几何提示注册，避免 Batch Size 超过 1
                        single_box = [("pos", [float(x), float(y), float(x + w), float(y + h)])]
                        
                        # 采用空格数量来代表不同目标示例，不改变原始语义
                        if base_label not in global_seen_labels:
                            global_seen_labels[base_label] = 0
                            unique_label = base_label
                        else:
                            global_seen_labels[base_label] += 1
                            unique_label = base_label + (" " * global_seen_labels[base_label])
                        
                        model.setup_geometry_input(input_img, unique_label, single_box)
                        if unique_label not in registered_geom_labels:
                            registered_geom_labels.append(unique_label)
                
                current_valid_idx += 1

            all_geom_features = registered_geom_labels
            print(f"特征缓存完成，共注册 {len(all_geom_features)} 个几何提示。")
    
    all_results = []
    for i, query_image in enumerate(query_images):
        if verbose:
            print(f"处理查询图像 {i+1}/{len(query_images)}")
        
        if query_image is None: 
            print(f"[auto_annotate] 错误: 图像 为空")
            continue
        
        img_h, img_w = query_image.shape[:2]
        if verbose:
            print(f"[auto_annotate] 查询图像尺寸: {img_w}x{img_h}, 模式: {mode}, 模型: {model_type}")

        # 3. 准备推理结果
        results = []
        if verbose:
            print(f"[DEBUG] auto_annotate: Checking model_type={model_type}")
        if model_type.startswith("fs_"):
             if verbose:
                 print(f"[DEBUG] auto_annotate: Calling example_predict for FS mode")
                 import sys
                 sys.stderr.write(f"[DEBUG] example_predict is {example_predict}\n")
             results = example_predict(query_image, model_type=model_type, refine=refine, smooth=smooth, smooth_epsilon=smooth_epsilon,
                                              support_images=support_images,
                                              support_masks=support_masks,    
                                              conf=rules.get("conf_threshold", 0.5),
                                              img_size=img_size)
             if verbose:
                 print(f"[DEBUG] auto_annotate: example_predict returned {results}")
        elif mode == "reuse" and all_geom_features is not None:
            texts = rules.get('text', None) if rules else None
            # 对于 trtsam3，geom_label 是字符串 label
            geom_label = all_geom_features
            if verbose:
                print(f"[auto_annotate] 正在进行 reuse 模式推理，提示词: {texts}, 几何标签数: {len(geom_label)}")
            results = example_predict(query_image, texts=texts, model_type=model_type, 
                                             refine=refine, smooth=smooth, smooth_epsilon=smooth_epsilon, img_size=img_size, geom_label=geom_label)
        else:
            if mode == "grid":
                concat_img, adjusted_bboxes, img_regions = grid_concatenate_mode(
                    query_image, support_images, support_infos["boxes"]
                )
            else: # copy_paste
                concat_img, adjusted_bboxes, img_regions = copy_paste_mode(
                    query_image, support_images, support_infos["boxes"]
                )

            if verbose:
                debug_dir = os.path.join(os.getcwd(), "debug_output")
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"debug_concat_{mode}_{i}.jpg")
                # 在拼接图上画出提示框，确认坐标是否正确
                debug_vis = concat_img.copy()
                for bbox in adjusted_bboxes:
                    x, y, w, h = bbox
                    cv2.rectangle(debug_vis, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                imwrite_unicode(debug_path, debug_vis)
                print(f"[auto_annotate] 已保存拼接调试图: {debug_path}")

            texts = rules.get('text', None) if rules else None
            if verbose:
                print(f"[auto_annotate] 正在进行拼接模式({mode})推理，拼接图尺寸: {concat_img.shape[1]}x{concat_img.shape[0]}, 提示框数: {len(adjusted_bboxes)}")

            results = example_predict(concat_img, bboxes=adjusted_bboxes, texts=texts, model_type=model_type, 
                                             refine=refine, smooth=smooth, smooth_epsilon=smooth_epsilon, img_size=img_size, 
                                             region=[0, 0, query_image.shape[1], query_image.shape[0]])
        all_results.append(results)

        if verbose:
            if results and len(results) > 0:
                num_polygons = len(results[0].polygons) if hasattr(results[0], 'polygons') and results[0].polygons is not None else 0
                print(f"[auto_annotate] 模型原始推理完成，检测到目标数：{num_polygons}")
            else:
                print(f"[auto_annotate] 模型原始推理完成，未返回任何结果")

    return all_results


def prepare_support_sets(support_paths, read_image_fn=None):
    """
    准备示例推理的支持集数据
    
    从支持集文件夹中读取 info.json 和图像，提取标注信息，
    计算统计规则（平均面积、宽高比、中心点），供 unified_inference 使用。
    
    Args:
        support_paths: 支持集路径列表（每个路径指向一个包含 info.json 的文件夹）
        read_image_fn: 图像读取函数，默认使用 imread_unicode
    
    Returns:
        tuple: (support_images, support_infos, support_rules)
            - support_images: 支持集图像列表
            - support_infos: dict，包含 boxes 和 polygons
            - support_rules: dict，包含 avg_area, avg_width, avg_height, avg_aspect_ratio, image_size
        （示例统计仅为界面参考值；过滤规则已全部改为绝对阈值，不再依赖这些统计）
        如果失败返回 (None, None, None)
    """
    if read_image_fn is None:
        read_image_fn = imread_unicode

    support_infos = {"boxes": [], "polygons": []}
    support_images = []
    all_bboxes = []
    all_polygons = []
    all_areas = []
    all_widths = []
    all_heights = []
    all_aspect_ratios = []
    image_sizes = []

    for support_path in support_paths:
        info_path = os.path.join(support_path, "info.json")
        if not os.path.exists(info_path):
            continue

        with open(info_path, 'r', encoding='utf-8-sig') as f:
            info = json.load(f)

        # 确定要读取的图像路径
        if info.get('is_sliced') and 'slice' in info:
            image_path = info['slice']['image_path']
        else:
            image_path = info['image_path']

        # 读取支持集图像
        support_img = read_image_fn(image_path)
        if support_img is None:
            continue
        support_images.append(support_img)
        image_sizes.append((support_img.shape[1], support_img.shape[0]))

        # 提取标注信息
        # 如果是切片模式，优先使用 slice 中的标注信息
        if info.get('is_sliced') and 'slice' in info:
            slice_info = info['slice']
            # 使用调整后的标注（相对于切片坐标）
            bbox = slice_info.get('slice_bbox')
            polygons = slice_info.get('slice_polygons', [])
        else:
            annotation = info.get('annotation', {})
            bbox = annotation.get('bbox')
            polygons = annotation.get('polygons', [])

        if bbox:
            all_bboxes.append([bbox])
            # 计算面积和宽高比
            x, y, w, h = bbox
            area = w * h
            all_areas.append(area)
            all_widths.append(w)
            all_heights.append(h)
            aspect_ratio = w / h if h > 0 else 1.0
            all_aspect_ratios.append(aspect_ratio)
        else:
            all_bboxes.append([])

        all_polygons.append(polygons if polygons else [])

    if not support_images:
        return (None, None, None)

    support_infos["boxes"] = all_bboxes
    support_infos["polygons"] = all_polygons

    # 计算支持集参考统计（仅供智能调参界面展示参考值，不再参与规则过滤）
    support_rules = {}
    if all_areas:
        support_rules["avg_area"] = sum(all_areas) / len(all_areas)
        support_rules["avg_width"] = sum(all_widths) / len(all_widths)
        support_rules["avg_height"] = sum(all_heights) / len(all_heights)
        support_rules["avg_aspect_ratio"] = sum(all_aspect_ratios) / len(all_aspect_ratios)
        support_rules["image_size"] = image_sizes[0] if image_sizes else None

    return (support_images, support_infos, support_rules)
