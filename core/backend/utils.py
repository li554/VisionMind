import json
import os
import shutil
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import requests
from tqdm import tqdm


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """读取图像，支持中文路径"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, flags)
        return image
    except Exception as e:
        print(f"[imread_unicode] Failed to read image {path}: {e}")
        return None

def imwrite_unicode(path: str, image: np.ndarray, params: list = None) -> bool:
    """保存图像，支持中文路径"""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            encode_param = cv2.IMWRITE_JPEG_QUALITY
            quality = 95
            if params and len(params) >= 2:
                quality = params[1]
            success, buffer = cv2.imencode(ext, image, [encode_param, quality])
        elif ext == '.png':
            encode_param = cv2.IMWRITE_PNG_COMPRESSION
            compression = 3
            if params and len(params) >= 2:
                compression = params[1]
            success, buffer = cv2.imencode(ext, image, [encode_param, compression])
        else:
            success, buffer = cv2.imencode(ext, image)
        
        if success:
            buffer.tofile(path)
            return True
        return False
    except Exception as e:
        print(f"[imwrite_unicode] Failed to write image {path}: {e}")
        return False

def _cleanup_runs():
    """清理临时推理文件"""
    from .path_resolver import _PROJECT_ROOT
    runs_dir = os.path.join(_PROJECT_ROOT, "runs")
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir, ignore_errors=True)

def download_file(url, save_path):
    """下载文件并显示进度条"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        filename = os.path.basename(save_path)
        print(f"[ExamplePredictor] 正在从 {url} 下载 {filename}...")
        
        with open(save_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))
        print(f"[ExamplePredictor] {filename} 下载完成。")
        return True
    except Exception as e:
        print(f"[ExamplePredictor] 下载失败: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def generate_distinct_colors(n):
    """生成n个视觉上区分明显的颜色"""
    if n <= 0:
        return []
    
    colors = []
    predefined_colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [255, 0, 255], [0, 255, 255], [255, 165, 0], [128, 0, 128],
    ]
    
    if n <= len(predefined_colors):
        return predefined_colors[:n]
    
    for i in range(n):
        hue = int(i * (180 / n))
        saturation = 200 + (i % 3) * 20 if (i % 3) != 0 else 255
        value = 200 + (i % 2) * 50 if (i % 2) != 0 else 255
        saturation = min(saturation, 255)
        value = min(value, 255)
        hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        colors.append(rgb.tolist())
    
    return colors

def create_class_colored_mask(instances: List[Dict[str, Any]], image_shape: Tuple, num_classes: int = 80) -> np.ndarray:
    """按类别着色的多类别语义分割掩码，使用 polygons"""
    if not instances:
        return np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    
    colored_mask = np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    class_colors = generate_distinct_colors(num_classes)
    
    for inst in instances:
        class_id = inst.get('class_id', 0)
        color = class_colors[class_id % num_classes]
        
        if 'polygons' in inst and inst['polygons']:
            for poly in inst['polygons']:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(colored_mask, [pts], color)
        
    return colored_mask

def create_visualization(image: np.ndarray, instances: List[Dict[str, Any]], alpha: float = 0.5) -> np.ndarray:
    """在原图上叠加掩码进行可视化渲染"""
    if not instances:
        return image.copy()
    
    overlay = image.copy()
    colored_mask = create_class_colored_mask(instances, image.shape)
    
    mask_indices = np.any(colored_mask > 0, axis=-1)
    
    overlay[mask_indices] = cv2.addWeighted(image[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)
    
    for inst in instances:
        color = generate_distinct_colors(80)[inst.get('class_id', 0) % 80]
        
        if 'polygons' in inst and inst['polygons']:
            poly = inst['polygons'][0]
            if len(poly) >= 4:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], True, color, 2)
            else:
                x, y, w, h = inst['bbox']
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        else:
            x, y, w, h = inst['bbox']
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        
    return overlay

def merge_multi_contours(contours: List[np.ndarray]) -> np.ndarray:
    """
    将多个分离的轮廓合并为一个单一的多边形点集。
    使用桥接技术：寻找最近点对并插入路径，不增加额外面积。
    优化版本：使用简化轮廓和空间索引加速最近邻搜索。
    """
    if not contours:
        return np.array([], dtype=np.float32)
    if len(contours) == 1:
        return contours[0].reshape(-1, 2).astype(np.float32)

    simplified_contours = []
    for c in contours:
        pts = c.reshape(-1, 2)
        if len(pts) > 100:
            perimeter = cv2.arcLength(c, True)
            epsilon = 0.002 * perimeter
            approx = cv2.approxPolyDP(c, epsilon, True)
            simplified_contours.append(approx.reshape(-1, 2))
        else:
            simplified_contours.append(pts)

    main_poly = simplified_contours[0].tolist()
    others = [c.tolist() for c in simplified_contours[1:]]

    while others:
        min_dist = float('inf')
        best_pair = None

        main_pts = np.array(main_poly)
        
        for o_idx, other in enumerate(others):
            other_pts = np.array(other)
            
            for i in range(0, len(main_poly), max(1, len(main_poly) // 20)):
                p1 = main_poly[i]
                diffs = other_pts - p1
                dists = np.sum(diffs ** 2, axis=1)
                j = np.argmin(dists)
                dist = dists[j]
                
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (i, j, o_idx)

        if best_pair:
            m_idx, o_idx, other_list_idx = best_pair
            other = others.pop(other_list_idx)
            
            rotated_other = other[o_idx:] + other[:o_idx+1]
            new_poly = main_poly[:m_idx+1] + rotated_other + main_poly[m_idx:]
            main_poly = new_poly

    return np.array(main_poly, dtype=np.float32)

def create_colored_combined_mask(masks: List[np.ndarray], image_shape: Tuple) -> np.ndarray:
    """创建彩色合并掩码，为每个实例分配不同颜色"""
    if not masks:
        return np.zeros((*image_shape[:2], 3), dtype=np.uint8)

    colored_combined_mask = np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    colors = generate_distinct_colors(len(masks))

    for j, (mask, color) in enumerate(zip(masks, colors)):
        colored_instance = np.zeros_like(colored_combined_mask)
        colored_instance[mask > 0] = color
        colored_combined_mask = np.where(mask[..., None] > 0, colored_instance, colored_combined_mask)

    return colored_combined_mask

def apply_rules(instances: List[Dict[str, Any]], rules: Optional[Dict[str, Any]] = None,
                support_info: Optional[Dict[str, Any]] = None,
                image: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """根据规则过滤实例

    Args:
        instances: 实例列表
        rules: 过滤规则字典
        support_info: 参考信息（有示例时使用相对模式）
        image: 原始图像（灰度过滤需要）
    """
    if rules is None:
        rules = {}

    for inst in instances:
        has_polygons = 'polygons' in inst and inst['polygons'] and len(inst['polygons']) > 0

        if has_polygons:
            all_polys = []
            for p in inst['polygons']:
                pts = np.array(p, dtype=np.int32).reshape((-1, 1, 2))
                all_polys.append(pts)

            poly_areas = [cv2.contourArea(p) for p in all_polys]
            if poly_areas:
                inst['bbox'] = cv2.boundingRect(np.vstack(all_polys))
                inst['bbox'] = [int(inst['bbox'][0]), int(inst['bbox'][1]), int(inst['bbox'][2]), int(inst['bbox'][3])]
                inst['_calc_area'] = sum(poly_areas)
        else:
            inst['bbox'] = [0, 0, 0, 0]
            inst['_calc_area'] = 0

    filtered = [i for i in instances if i.get('_calc_area', 0) > 0 and i.get('bbox', [0,0,0,0])[2] >= 2]

    conf_thres = rules.get('conf_threshold', rules.get('conf_thres'))
    if conf_thres is not None:
        filtered = [i for i in filtered if i.get('conf', 1.0) >= conf_thres]

    if 'area_range' in rules and support_info and 'avg_area' in support_info:
        l, h = rules['area_range']
        target = support_info['avg_area']
        filtered = [i for i in filtered if target*(1-l) <= i['_calc_area'] <= target*(1+h)]

    if 'width_range' in rules and support_info and 'avg_width' in support_info:
        l, h = rules['width_range']
        target = support_info['avg_width']
        filtered = [i for i in filtered if target*(1-l) <= i['bbox'][2] <= target*(1+h)]

    if 'height_range' in rules and support_info and 'avg_height' in support_info:
        l, h = rules['height_range']
        target = support_info['avg_height']
        filtered = [i for i in filtered if target*(1-l) <= i['bbox'][3] <= target*(1+h)]

    if 'aspect_ratio_range' in rules and support_info and 'avg_aspect_ratio' in support_info:
        l, h = rules['aspect_ratio_range']
        target = support_info['avg_aspect_ratio']
        filtered = [i for i in filtered if target*(1-l) <= (i['bbox'][2]/i['bbox'][3] if i['bbox'][3]!=0 else 0) <= target*(1+h)]

    if 'center_range' in rules and support_info and 'avg_center_x' in support_info:
        mx, my = rules['center_range']
        tx, ty = support_info['avg_center_x'], support_info['avg_center_y']
        filtered = [i for i in filtered if abs((i['bbox'][0]+i['bbox'][2]/2)-tx) <= mx and abs((i['bbox'][1]+i['bbox'][3]/2)-ty) <= my]

    if 'gray_range' in rules:
        l, h = rules['gray_range']
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            img_h, img_w = gray_image.shape[:2]
            new_filtered = []
            for i in filtered:
                x, y, w, ht = i['bbox']
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(img_w, int(x + w)), min(img_h, int(y + ht))
                if x2 <= x1 or y2 <= y1:
                    continue
                region = gray_image[y1:y2, x1:x2]
                # 若有多边形，仅计算多边形内的灰度均值
                has_polys = 'polygons' in i and i['polygons'] and len(i['polygons']) > 0
                if has_polys:
                    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                    for poly in i['polygons']:
                        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                        pts_shifted = pts - np.array([[[x1, y1]]])
                        cv2.fillPoly(mask, [pts_shifted], 255)
                    if np.any(mask > 0):
                        avg_gray = float(np.mean(region[mask > 0]))
                    else:
                        avg_gray = float(np.mean(region))
                else:
                    avg_gray = float(np.mean(region))
                if l <= avg_gray <= h:
                    new_filtered.append(i)
            filtered = new_filtered

    if 'max_instances' in rules:
        filtered.sort(key=lambda x: x.get('conf', 0), reverse=True)
        filtered = filtered[:rules['max_instances']]

    for i in filtered:
        if '_calc_area' in i: del i['_calc_area']

    return filtered

def create_grid_from_images(images: List[np.ndarray], 
                            bboxes_list: Optional[List[List[List[int]]]] = None,
                            target_width: Optional[int] = None,
                            first_at_top_left: bool = False) -> Tuple[np.ndarray, List[List[int]], List[Tuple[int, int, int, int]]]:
    """更紧凑的图像拼接函数（最小面积方式）"""
    if not images:
        return np.zeros((0, 0, 3), dtype=np.uint8), [], []
        
    indexed_images = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        indexed_images.append({
            'index': i,
            'img': img,
            'h': h,
            'w': w,
            'bboxes': bboxes_list[i] if bboxes_list and i < len(bboxes_list) else []
        })
    
    if first_at_top_left:
        anchor = indexed_images[0]
        others = indexed_images[1:]
        others.sort(key=lambda x: x['h'], reverse=True)
        to_pack = others
    else:
        indexed_images.sort(key=lambda x: x['h'], reverse=True)
        anchor = None
        to_pack = indexed_images

    max_single_w = max(item['w'] for item in indexed_images)
    
    if target_width is None:
        total_area = sum(item['h'] * item['w'] for item in indexed_images)
        ideal_side = int(np.sqrt(total_area))
        target_width = max(ideal_side, max_single_w)
        
        if first_at_top_left:
            target_width = max(target_width, indexed_images[0]['w'])

    best_rows = []
    best_ratio_diff = float('inf')
    
    search_step = max(10, max_single_w // 4)
    search_range = [target_width + i * search_step for i in range(5)]
    
    for trial_w in search_range:
        current_rows = []
        if first_at_top_left:
            current_rows.append({'h': anchor['h'], 'w': anchor['w'], 'items': [anchor]})
        
        for item in to_pack:
            placed = False
            for row in current_rows:
                if row['w'] + item['w'] <= trial_w:
                    row['items'].append(item)
                    row['w'] += item['w']
                    if item['h'] > row['h']: row['h'] = item['h']
                    placed = True
                    break
            if not placed:
                current_rows.append({'h': item['h'], 'w': item['w'], 'items': [item]})
        
        cur_h = sum(r['h'] for r in current_rows)
        cur_w = max(r['w'] for r in current_rows)
        ratio = cur_w / cur_h if cur_h > 0 else 1
        diff = abs(ratio - 1.0)
        
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_rows = current_rows
            if diff < 0.05: break
            
    rows = best_rows
    
    final_w = max(row['w'] for row in rows)
    final_h = sum(row['h'] for row in rows)
    
    grid_img = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    adjusted_bboxes = []
    regions = [None] * len(images)
    
    current_y = 0
    for row in rows:
        current_x = 0
        for item in row['items']:
            h, w = item['h'], item['w']
            grid_img[current_y:current_y + h, current_x:current_x + w] = item['img']
            
            regions[item['index']] = (current_x, current_y, w, h)
            
            for bbox in item['bboxes']:
                bx, by, bw, bh = bbox
                adjusted_bboxes.append([bx + current_x, by + current_y, bw, bh])
            
            current_x += w
        current_y += row['h']
                
    return grid_img, adjusted_bboxes, regions

def grid_concatenate_mode(query_image: np.ndarray,
                          support_images: List[np.ndarray],
                          support_bboxes: List[List[List[int]]]) -> Tuple[np.ndarray, List[List[int]], List[Tuple[int, int, int, int]]]:
    """网格拼接模式：将查询图像和支持图像拼接成一张图片"""
    all_images = [query_image] + support_images
    support_bboxes.insert(0, [])
    return create_grid_from_images(all_images, support_bboxes, first_at_top_left=True)

def copy_paste_mode(query_image: np.ndarray,
                    support_images: List[np.ndarray],
                    support_bboxes: List[List[List[int]]],
                    pad_ratio: float = 0.30) -> Tuple[np.ndarray, List[List[int]], List[Tuple[int, int, int, int]]]:
    """复制粘贴模式"""
    obj_regions = []
    obj_bboxes_list = []
    
    for i, bboxes in enumerate(support_bboxes):
        img = support_images[i]
        for bbox in bboxes:
            x, y, w, h = bbox
            
            pad_w = int(w * pad_ratio)
            pad_h = int(h * pad_ratio)
            
            x1, y1 = int(max(0, x - pad_w)), int(max(0, y - pad_h))
            x2, y2 = int(min(img.shape[1], x + w + pad_w)), int(min(img.shape[0], y + h + pad_h))

            obj_regions.append(img[y1:y2, x1:x2])
            obj_bboxes_list.append([[x - x1, y - y1, w, h]])

    if not obj_regions:
        h, w = query_image.shape[:2]
        return query_image, [0, 0, 0, 0], [(0, 0, w, h)]

    total_obj_area = sum(img.shape[0] * img.shape[1] for img in obj_regions)
    query_h, query_w = query_image.shape[:2]
    target_grid_w = max(int(total_obj_area / query_h), max(img.shape[1] for img in obj_regions))
    
    support_grid, grid_bboxes, _ = create_grid_from_images(obj_regions, obj_bboxes_list, target_width=target_grid_w)

    grid_h, grid_w = support_grid.shape[:2]
    
    canvas_h = max(query_h, grid_h)
    canvas_w = query_w + grid_w
    concatenated_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    concatenated_image[:query_h, :query_w] = query_image
    concatenated_image[:grid_h, query_w:query_w + grid_w] = support_grid
    
    adjusted_bboxes = []
    
    for bbox in grid_bboxes:
        gx, gy, gw, gh = bbox
        adjusted_bboxes.append([
            int(query_w + gx),
            int(gy),
            int(gw),
            int(gh)
        ])

    image_regions = [
        (0, 0, query_w, query_h),
        (query_w, 0, grid_w, grid_h)
    ]

    return concatenated_image, adjusted_bboxes, image_regions

def find_label_file(image_path: str, label_format: str = "auto") -> Optional[str]:
    """
    寻找图像对应的标签文件

    搜索策略（按优先级）：
    1. 同级目录下的 labels/Annotations/masks/coco_annotations 子目录
    2. 父目录下的 labels/Annotations/masks/coco_annotations 目录
    3. 图像所在目录本身
    4. 对于 train/val/test 分割结构，在同级目录下搜索
    5. 统一的 COCO 标注文件 _annotations.coco.json
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    abs_image_path = os.path.abspath(image_path)
    img_dir = os.path.dirname(abs_image_path)
    parent_dir = os.path.dirname(img_dir)
    grandparent_dir = os.path.dirname(parent_dir)

    search_dirs = []

    # 1. 图像所在目录的标注子目录
    search_dirs.extend([
        os.path.join(img_dir, "labels"),
        os.path.join(img_dir, "Annotations"),
        os.path.join(img_dir, "masks"),
        os.path.join(img_dir, "coco_annotations"),
    ])

    # 2. 父目录下的标注目录（适用于 images/labels 平行结构）
    search_dirs.extend([
        os.path.join(parent_dir, "labels"),
        os.path.join(parent_dir, "Annotations"),
        os.path.join(parent_dir, "masks"),
        os.path.join(parent_dir, "coco_annotations"),
    ])

    # 3. 图像所在目录本身
    search_dirs.append(img_dir)

    # 4. 处理 train/val/test 分割结构
    # 如果图像路径包含这些分割名称，尝试在同级目录中搜索
    img_dir_name = os.path.basename(img_dir).lower()
    split_names = ["train", "val", "test", "training", "validation", "testing"]

    if img_dir_name in split_names:
        # 在同级其他分割目录中搜索标注
        for split in split_names:
            sibling_img_dir = os.path.join(parent_dir, split)
            if sibling_img_dir != img_dir:
                search_dirs.extend([
                    os.path.join(sibling_img_dir, "labels"),
                    os.path.join(sibling_img_dir, "Annotations"),
                    os.path.join(sibling_img_dir, "masks"),
                    os.path.join(sibling_img_dir, "coco_annotations"),
                ])

    # 去重并过滤不存在的目录
    valid_dirs = []
    seen = set()
    for d in search_dirs:
        if not d:
            continue
        abs_d = os.path.abspath(d)
        if os.path.exists(abs_d) and abs_d not in seen:
            valid_dirs.append(abs_d)
            seen.add(abs_d)

    # 搜索标注文件（按优先级）
    annotation_exts = ['.txt', '.json', '.xml']
    if label_format != "auto":
        # 如果指定了格式，优先搜索该格式
        priority_ext = f".{label_format}" if not label_format.startswith('.') else label_format
        if priority_ext in annotation_exts:
            annotation_exts.remove(priority_ext)
            annotation_exts.insert(0, priority_ext)

    for s_dir in valid_dirs:
        for ext in annotation_exts:
            p = os.path.join(s_dir, base + ext)
            if os.path.exists(p):
                return p

    # 搜索统一的 COCO 标注文件 _annotations.coco.json
    for s_dir in valid_dirs:
        coco_file = os.path.join(s_dir, "_annotations.coco.json")
        if os.path.exists(coco_file):
            return coco_file

    # 搜索掩码图像文件
    mask_exts = ['.png', '.jpg']
    for s_dir in valid_dirs:
        for ext in mask_exts:
            p = os.path.join(s_dir, base + ext)
            if os.path.exists(p):
                # 确保找到的文件不是输入图像本身
                abs_p = os.path.abspath(p)
                if abs_p != abs_image_path:
                    return p

    return None


def polygons_to_mask(polygons: List[List[List[int]]], bbox: List[int], padding: int = 10) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    将多边形转换为掩码（仅在bbox周边区域）

    Args:
        polygons: list of list of points
        bbox: [x, y, w, h]
        padding: 边缘填充像素

    Returns:
        mask: numpy array
        offset: (x_offset, y_offset) 掩码相对于原图的偏移
    """
    x, y, w, h = bbox
    # 添加padding
    x_min = max(0, int(x) - padding)
    y_min = max(0, int(y) - padding)
    x_max = int(x + w) + padding
    y_max = int(y + h) + padding

    mask_w = x_max - x_min
    mask_h = y_max - y_min

    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

    for poly in polygons:
        if not poly:
            continue
        # 将多边形坐标转换为掩码坐标
        pts = np.array([[p[0] - x_min, p[1] - y_min] for p in poly], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)

    return mask, (x_min, y_min)


def mask_to_polygons(mask: np.ndarray, x_offset: int, y_offset: int, epsilon_factor: float = 0.001) -> List[List[List[int]]]:
    """
    将掩码转换为多边形

    Args:
        mask: numpy array
        x_offset, y_offset: 掩码相对于原图的偏移
        epsilon_factor: 多边形简化因子

    Returns:
        polygons: list of list of points
    """
    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue

        # 简化多边形
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 转换回原始坐标
        points = [[int(p[0][0] + x_offset), int(p[0][1] + y_offset)] for p in approx]
        polygons.append(points)

    return polygons


def clip_polygon_to_bbox(polygon: List[List[float]], bbox: List[int]) -> Optional[List[List[float]]]:
    """
    将多边形裁剪到bbox范围内
    使用Sutherland-Hodgman算法

    Args:
        polygon: 多边形点列表 [[x, y], ...]
        bbox: [x, y, w, h]

    Returns:
        裁剪后的多边形，如果点数少于3则返回None
    """
    if polygon is None or len(polygon) < 3:
        return polygon

    x, y, w, h = bbox
    xmin, ymin = x, y
    xmax, ymax = x + w, y + h

    # 定义裁剪边界的四条边 (左, 右, 上, 下)
    edges = [
        (xmin, 'left'),
        (xmax, 'right'),
        (ymin, 'top'),
        (ymax, 'bottom')
    ]

    def inside(p, val, edge_type):
        """检查点是否在裁剪边内部"""
        if edge_type == 'left':
            return p[0] >= val
        elif edge_type == 'right':
            return p[0] <= val
        elif edge_type == 'top':
            return p[1] >= val
        elif edge_type == 'bottom':
            return p[1] <= val
        return True

    def intersect(p1, p2, val, edge_type):
        """计算线段与裁剪边的交点"""
        x1, y1 = p1
        x2, y2 = p2

        if edge_type in ('left', 'right'):
            x = val
            if x2 != x1:
                y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            else:
                y = y1
            return [x, y]
        else:  # top or bottom
            y = val
            if y2 != y1:
                x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
            else:
                x = x1
            return [x, y]

    # 当前多边形顶点列表
    output_list = polygon[:]

    # 对每条边进行裁剪
    for val, edge_type in edges:
        input_list = output_list
        output_list = []

        if input_list is None or len(input_list) == 0:
            break

        s = input_list[-1]  # 上一个点

        for p in input_list:
            if inside(p, val, edge_type):
                if not inside(s, val, edge_type):
                    # 从外到内，添加交点
                    output_list.append(intersect(s, p, val, edge_type))
                output_list.append(p)
            elif inside(s, val, edge_type):
                # 从内到外，添加交点
                output_list.append(intersect(s, p, val, edge_type))
            s = p

    return output_list if len(output_list) >= 3 else None


def process_mask_results(results, label: Optional[str] = None, bbox: Optional[List[int]] = None,
                        rules: Optional[Dict] = None, single_result: bool = False,
                        support_info=None, image: Optional[np.ndarray] = None,
                        texts: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Unified function to process segmentation results.
    Process polygons directly from results (mask-based processing is deprecated).

    Args:
        results: 分割模型返回的结果
        label: 指定的标签
        bbox: 指定的边界框
        rules: 过滤规则
        single_result: 是否只返回单个结果
        support_info: 参考信息（有示例时使用相对模式）
        image: 原始图像（灰度过滤需要）
        texts: 文本提示词列表。当传入时，每个实例按其命中的提示词索引(cls)映射到对应类别，
            实现多提示词时"哪个提示词命中的结果就归哪个类别"。

    Returns:
        处理后的标注结果字典
    """
    if results and hasattr(results[0], 'polygons') and results[0].polygons is not None and len(results[0].polygons) > 0:
        annotations = []

        boxes = results[0].boxes.xywh.cpu().numpy() if hasattr(results[0], 'boxes') and results[
            0].boxes is not None else None
        cls_data = results[0].boxes.cls.cpu().numpy() if hasattr(results[0], 'boxes') and results[
            0].boxes is not None and results[0].boxes.cls is not None else None
        conf_data = results[0].boxes.conf.cpu().numpy() if hasattr(results[0], 'boxes') and results[
            0].boxes is not None and results[0].boxes.conf is not None else None
        names = results[0].names if hasattr(results[0], 'names') else {}

        all_polygons = results[0].polygons

        if all_polygons is None:
            return {"status": "error", "message": "No polygons found in results"}

        num_masks = 1 if single_result else len(all_polygons)

        for i in range(num_masks):
            polygons_list = all_polygons[i]

            if not polygons_list or len(polygons_list) == 0:
                continue

            polygon = polygons_list[0]

            if len(polygon) == 0:
                continue

            if i == 0 and bbox is not None:
                curr_bbox = bbox
            elif boxes is not None and i < len(boxes):
                cx, cy, w, h = boxes[i]
                curr_bbox = [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]
            else:
                x, y, bw, bh = cv2.boundingRect(polygon.astype(np.float32))
                curr_bbox = [int(x), int(y), int(bw), int(bh)]

            # 如果是单结果模式且有指定bbox，裁剪多边形到bbox范围内
            if single_result and bbox is not None:
                polygon = clip_polygon_to_bbox(polygon, bbox)
                if polygon is None or len(polygon) < 3:
                    continue
                # 更新bbox为裁剪后的多边形bbox
                x, y, bw, bh = cv2.boundingRect(np.array(polygon, dtype=np.float32))
                curr_bbox = [int(x), int(y), int(bw), int(bh)]

            ann = {
                "polygons": [polygon],
                "bbox": curr_bbox,
                "conf": float(conf_data[i]) if conf_data is not None and i < len(conf_data) else 1.0
            }

            final_label = label
            # 传入 texts 时优先按实例命中的提示词索引(cls)映射到对应提示词类别
            if not final_label and texts is not None:
                if cls_data is not None and i < len(cls_data):
                    class_id = int(cls_data[i])
                    if 0 <= class_id < len(texts):
                        final_label = texts[class_id]
                # clss 无效但只有一个提示词时，直接使用该提示词
                if not final_label and len(texts) == 1:
                    final_label = texts[0]
            # 未显式指定 label / texts 时才回退到模型 names
            if not final_label and texts is None and cls_data is not None and i < len(cls_data):
                class_id = int(cls_data[i])
                if hasattr(names, 'get'):
                    final_label = names.get(class_id, f"class_{class_id}")
                elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                    final_label = names[class_id]
                else:
                    final_label = f"class_{class_id}"

            if final_label:
                ann["label"] = final_label
            annotations.append(ann)

        if rules:
            annotations = apply_rules(annotations, rules, support_info, image)

        return {"status": "success", "annotations": annotations}

    return {"status": "error", "message": "No objects found"}


def convert_bbox_to_polygon(image: np.ndarray, bbox: List[int], model_type: str,
                           refine: bool = True, smooth: bool = False,
                           smooth_epsilon: float = 0.001) -> Dict[str, Any]:
    """
    Convert a single bounding box to polygon using interactive segmentation model.

    Args:
        image: 输入图像
        bbox: 边界框 [x, y, w, h]
        model_type: 模型类型
        refine: 是否细化
        smooth: 是否平滑
        smooth_epsilon: 平滑因子

    Returns:
        处理后的标注结果字典
    """
    try:
        from .core import interactive_predict

        results = interactive_predict(image, bboxes=[bbox], model_type=model_type, refine=refine,
                         smooth=smooth, smooth_epsilon=smooth_epsilon)

        # segment() already processes masks and extracts polygons in BaseSegmentor.predict()
        # Use process_mask_results to handle the results consistently
        return process_mask_results(
            results,
            bbox=bbox,
            single_result=True
        )
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


def calculate_iou(bbox1, bbox2) -> float:
    """
    计算两个边界框的IOU (Intersection over Union)
    
    Args:
        bbox1: [x, y, w, h] 格式边界框
        bbox2: [x, y, w, h] 格式边界框
        
    Returns:
        IOU值，范围[0, 1]
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # 计算交集区域
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # 计算并集区域
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area
