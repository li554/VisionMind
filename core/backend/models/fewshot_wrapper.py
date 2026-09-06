import numpy as np
import torch
import gc
import cv2
import os
from typing import List, Any, Optional

from .base import ModelInterface, MockBoxes, MockMasks, MockResults
from .registry import register_model
from ..utils import _cleanup_runs

# 可视化调试开关
DEBUG_VIS = False
DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "debug_fewshot")
if DEBUG_VIS and not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)


def mask_to_uniform_points(mask, num_points=4):
    """从掩码中提取均匀分布的点"""
    if mask is None or np.sum(mask) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    # 获取所有掩码点的坐标 (y, x)
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.float32)
        
    # 如果点数少于要求的数量，直接返回所有点
    if len(coords) <= num_points:
        return coords[:, ::-1].astype(np.float32)
        
    # 使用简单的采样策略：取质心和周边的点
    # 或者使用更均匀的采样，这里采用简单的随机选择以保证性能，
    # 实际应用中可以考虑 K-Means 或者更稳定的采样
    indices = np.linspace(0, len(coords) - 1, num_points, dtype=int)
    sampled_coords = coords[indices]
    
    # 转换为 (x, y) 格式
    return sampled_coords[:, ::-1].astype(np.float32)

def adapt_point_array(pts, current_shape, origin_shape):
    """将点坐标从 current_shape 映射到 origin_shape，考虑等比缩放并填充到正方形的情况"""
    if pts is None or len(pts) == 0:
        return pts
    
    h_curr, w_curr = current_shape[:2]
    h_orig, w_orig = origin_shape[:2]
    
    # 基于最大边计算缩放因子，因为图像是被等比缩放并填充到 (window_size, window_size) 的
    # 这里假设 origin_shape 是特征图，它代表了完整的 (window_size, window_size) 画布
    scale_x = w_orig / max(h_curr, w_curr)
    scale_y = h_orig / max(h_curr, w_curr)
    
    new_pts = pts.copy().astype(np.float32)
    new_pts[:, 0] *= scale_x
    new_pts[:, 1] *= scale_y
    
    return new_pts

def adapt_box_array(boxes, current_shape, origin_shape):
    """将边界框从 current_shape 映射到 origin_shape"""
    if boxes is None or len(boxes) == 0:
        return boxes
    
    h_curr, w_curr = current_shape[:2]
    h_orig, w_orig = origin_shape[:2]
    
    scale_x = w_orig / w_curr
    scale_y = h_orig / h_curr
    
    boxes = np.array(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        # 单个框 [x, y, w, h]
        boxes[0] *= scale_x
        boxes[1] *= scale_y
        boxes[2] *= scale_x
        boxes[3] *= scale_y
    else:
        # 多个框 [N, 4]
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        
    return boxes

def get_similarity(emb, features):
    """计算嵌入向量与特征图的相似度"""
    # emb: [C] or [N, C]
    # features: [C, H, W] or [B, C, H, W]
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
        
    # 处理特征图维度，确保是 [C, H, W]
    if features.ndim == 4:
        features = features[0]
        
    if emb.ndim == 2:
        emb = emb.mean(dim=0) # [C]
        
    # 归一化以计算余弦相似度
    # 注意 features 现在的维度是 [C, H, W]，归一化应该在 C 维度 (dim=0)
    emb = torch.nn.functional.normalize(emb, dim=0)
    features_norm = torch.nn.functional.normalize(features, dim=0)
    
    # 计算点积相似度
    # features_norm: [C, H, W], emb: [C]
    sim = torch.einsum('c,chw->hw', emb, features_norm)
    return sim

def graphcut_binarize(img: np.ndarray, iter_count: int = 5) -> np.ndarray:
    """
    GraphCut binarization using OpenCV GrabCut.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image in [0,1], shape (H,W)
    iter_count : int
        Number of GrabCut iterations

    Returns
    -------
    mask : np.ndarray
        Binary mask (0/1), shape (H,W)
    """

    assert img.ndim == 2
    assert img.min() >= 0 and img.max() <= 1

    H, W = img.shape

    # 转成8bit
    img8 = (img * 255).astype(np.uint8)

    # 变成3通道
    img3 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    # 初始mask
    mask = np.zeros((H, W), np.uint8)

    # 用Otsu仅做初始化
    _, thresh = cv2.threshold(
        img8, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask[thresh == 255] = cv2.GC_PR_FGD
    mask[thresh == 0]   = cv2.GC_PR_BGD

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    # 运行GrabCut
    cv2.grabCut(
        img3,
        mask,
        None,
        bgModel,
        fgModel,
        iter_count,
        cv2.GC_INIT_WITH_MASK
    )

    # 生成最终二值mask
    result_mask = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        1, 0
    ).astype(np.uint8)

    return result_mask

@register_model("fs_sam3", img_size=1008, weight_path_key="sam3_path", category="fssam", role="example")
@register_model("fs_mobilesam_onnx", img_size=1024, weight_path_key="mobile_sam_onnx_encoder_path,mobile_sam_onnx_decoder_path", category="fssam", role="example")
@register_model("fs_sam_onnx_b", img_size=1024, weight_path_key="sam_onnx_b_encoder_path,sam_onnx_b_decoder_path", category="fssam", role="example")
@register_model("fs_sam_onnx_l", img_size=1024, weight_path_key="sam_onnx_l_encoder_path,sam_onnx_l_decoder_path", category="fssam", role="example")
@register_model("fs_hqsam_onnx_b", img_size=1024, weight_path_key="hqsam_onnx_b_encoder_path,hqsam_onnx_b_decoder_path", category="fssam", role="example")
@register_model("fs_hqsam_onnx_l", img_size=1024, weight_path_key="hqsam_onnx_l_encoder_path,hqsam_onnx_l_decoder_path", category="fssam", role="example")
@register_model("fs_sam2_onnx_b", img_size=1024, weight_path_key="sam2_onnx_b_encoder_path,sam2_onnx_b_decoder_path", category="fssam", role="example")
@register_model("fs_sam2_onnx_l", img_size=1024, weight_path_key="sam2_onnx_l_encoder_path,sam2_onnx_l_decoder_path", category="fssam", role="example")
class FSWrapper(ModelInterface):
    """Few Shot SAM 模型包装器"""
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.window_size = 640

    def get_features(self, image: np.ndarray) -> torch.Tensor:
        """提取图像特征"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.get_features(image)

    def reset_cache(self):
        """重置模型缓存"""
        if self.model:
            self.model.reset_cache()

    def load(self, weight_path: str) -> bool:
        try:
            print(f"[FSWrapper] Loading {self.model_type} from {weight_path}...")
            # Import locally to avoid circular import
            from ..core import ModelFactory
            # fs_X 包装的是基础模型 X,若直接 create(self.model_type) 会再次实例化
            # FSWrapper 并调用 load() 导致无限递归
            base_type = self.model_type[3:] if self.model_type.startswith("fs_") else self.model_type
            self.model = ModelFactory.create(base_type, weight_path=weight_path)
            return self.model is not None
        except Exception as e:
            print(f"[FSWrapper] 加载 {self.model_type} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, image, bboxes=None, points=None, labels=None, texts=None, conf=0.5, img_size=640, **kwargs):
        support_images = kwargs.get('support_images', None)
        support_boxes = kwargs.get('support_boxes', None)
        support_masks = kwargs.get('support_masks', None) # Optional: direct masks
        
        # 如果没有支持集，直接返回空或使用底层模型普通预测
        if support_images is None or len(support_images) == 0:
            if self.model:
                return self.model.predict(image, bboxes, points, labels, texts, conf, img_size=img_size, **kwargs)
            return [MockResults(None, None, image)]
            
        return self._predict_few_shot(image, support_images, support_boxes, support_masks, threshold=conf)

    def get_point_embeddings(self, features, points):
        """提取指定点的特征嵌入"""
        # 确保 features 是 torch tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        
        if len(points) == 0:
            # 如果没有点，返回空嵌入，注意 C 维度
            c = features.shape[1] if features.ndim == 4 else features.shape[0]
            return torch.zeros((0, c), device=features.device)
            
        if features.ndim == 4:
            features = features[0] # [C, H, W]
            
        c, h, w = features.shape
        
        # points 是特征图坐标 [N, 2]
        # 使用双线性插值采样
        if isinstance(points, np.ndarray):
            points_torch = torch.from_numpy(points).to(features.device)
        else:
            points_torch = points.to(features.device)
        
        # 归一化到 [-1, 1] 用于 grid_sample
        # grid_sample 期望 grid 为 [N, H_out, W_out, 2]
        grid = points_torch.view(1, 1, -1, 2).float()
        grid[..., 0] = (grid[..., 0] / (w - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (h - 1)) * 2 - 1
        
        # features: [1, C, H, W], grid: [1, 1, N, 2]
        sampled = torch.nn.functional.grid_sample(
            features.unsqueeze(0), 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
        # sampled: [1, C, 1, N] -> [C, N] -> [N, C]
        return sampled.view(c, -1).t()

    def get_support_embeddings(self, support_features, support_masks):
        """从支持掩码中提取正负样本嵌入"""
        pos_embeddings = []
        neg_embeddings = []
        for feature, mask in zip(support_features, support_masks):
            # 获取掩码的边界框
            coords = np.argwhere(mask > 0)
            if len(coords) == 0:
                continue
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # 1. 正样本点从掩码内部提取四个点 (确保点在掩码内)
            # 使用距离变换寻找掩码“最深”的位置，保证中心点在掩码内
            dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
            cx, cy = max_loc
            
            # 在最深点周围采样，并校验是否仍在掩码内
            pos_points_list = [[float(cx), float(cy)]]
            offsets = [[-3, -3], [3, -3], [-3, 3], [3, 3]]
            for dx, dy in offsets:
                nx, ny = int(cx + dx), int(cy + dy)
                if 0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0] and mask[ny, nx] > 0:
                    pos_points_list.append([float(nx), float(ny)])
            
            # 如果点数不足4个，使用通用的均匀采样补齐
            if len(pos_points_list) < 4:
                extra_needed = 4 - len(pos_points_list)
                extra_pts = mask_to_uniform_points(mask, num_points=extra_needed)
                if len(extra_pts) > 0:
                    pos_points_list.extend(extra_pts.tolist())
            
            pos_points = np.array(pos_points_list[:4], dtype=np.float32)
            
            # 映射到特征图坐标 (mask -> feature)
            pos_points_feat = adapt_point_array(pos_points, mask.shape[:2], feature.shape[-2:])
            pos_embeddings.append(self.get_point_embeddings(feature, pos_points_feat))
            
            # 2. 负样本提取掩码边界四个角以及四个角方向上按step像素间隔的N个点
            # 最终点数为4+4N
            N = 2 # 默认外扩点数
            step = 20 # 默认间隔
            corners = [
                [x1, y1], [x2, y1], [x1, y2], [x2, y2]
            ]
            directions = [
                [-1, -1], [1, -1], [-1, 1], [1, 1]
            ]
            
            neg_points_list = []
            for corner, direction in zip(corners, directions):
                neg_points_list.append(corner) # 添加角点 (4)
                for i in range(1, N + 1):
                    # 按方向外扩 (4*N)
                    ext_point = [
                        corner[0] + direction[0] * i * step,
                        corner[1] + direction[1] * i * step
                    ]
                    neg_points_list.append(ext_point)
            
            neg_points = np.array(neg_points_list, dtype=np.float32)
            # 映射到特征图坐标 (mask -> feature)
            neg_points_feat = adapt_point_array(neg_points, mask.shape[:2], feature.shape[-2:])
            neg_embeddings.append(self.get_point_embeddings(feature, neg_points_feat))
            
        return pos_embeddings, neg_embeddings


    def resize_and_pad_image(self, image):
        """将图像缩放并填充到 window_size x window_size"""
        h, w = image.shape[:2]
        scale = self.window_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建填充后的图像
        padded = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized
        return padded

    def merge_results(self, results):
        """合并多个预测结果"""
        if not results:
            return None
            
        valid_results = [res for res in results if res is not None and res.boxes is not None and len(res.boxes) > 0]
        if not valid_results:
            return MockResults(None, None, results[0].orig_img if results else None)

        merged_boxes_xyxy = []
        merged_boxes_conf = []
        merged_boxes_cls = []
        merged_masks_data = []
        merged_names = {}
        
        for res in valid_results:
            # 确保在 cat 之前维度一致
            xyxy = res.boxes.xyxy
            conf = res.boxes.conf
            cls = res.boxes.cls
            
            if xyxy.ndim == 1: xyxy = xyxy.unsqueeze(0)
            if conf.ndim == 0: conf = conf.unsqueeze(0)
            if cls.ndim == 0: cls = cls.unsqueeze(0)
            
            merged_boxes_xyxy.append(xyxy)
            merged_boxes_conf.append(conf)
            merged_boxes_cls.append(cls)
            
            if res.masks is not None and res.masks.data is not None:
                mask_data = res.masks.data
                if mask_data.ndim == 2: # [H, W] -> [1, H, W]
                    mask_data = mask_data.unsqueeze(0)
                merged_masks_data.append(mask_data)
                
            if res.names:
                merged_names.update(res.names)
            
        # 合并后的结果
        boxes = MockBoxes(
            torch.cat(merged_boxes_xyxy, dim=0),
            torch.cat(merged_boxes_conf, dim=0),
            torch.cat(merged_boxes_cls, dim=0)
        )
            
        masks = None
        if merged_masks_data:
            masks = MockMasks(torch.cat(merged_masks_data, dim=0))
            
        return MockResults(
            boxes=boxes,
            masks=masks,
            orig_img=results[0].orig_img,
            names=merged_names
        )



    def _predict_few_shot(self, query_image, support_images, support_boxes, support_masks=None, threshold=0.5):
        """
        Few-shot prediction logic using similarity maps and SAM refinement.
        """
        if support_images is None or len(support_images) == 0:
            return [MockResults(None, None, query_image)]

        query_image_shape = query_image.shape[:2]
        # 1. 缩放并填充图像，提取特征
        query_image_resized = self.resize_and_pad_image(query_image)
        # 提取查询图像特征 (对应 window_size x window_size)
        query_features = self.get_features(query_image_resized)
        
        # 2. 提取支持集特征和掩码
        support_features = []
        actual_support_masks = []
        for i, img in enumerate(support_images):
            img_resized = self.resize_and_pad_image(img)
            feat = self.get_features(img_resized)
            support_features.append(feat)
            
            # 如果没有提供 masks，则从 boxes 生成
            mask = None
            if support_masks is not None and i < len(support_masks) and support_masks[i] is not None:
                mask = support_masks[i]
            elif support_boxes is not None and i < len(support_boxes) and support_boxes[i] is not None:
                # 从 bbox 生成简单的矩形掩码
                box = support_boxes[i]
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                x, y, w, h = map(int, box)
                mask[y:y+h, x:x+w] = 1
            
            if mask is not None:
                actual_support_masks.append(mask)

        if not actual_support_masks:
            return [MockResults(None, None, query_image)]

        # 3. 获取支持集嵌入
        pos_embeddings, neg_embeddings = self.get_support_embeddings(support_features, actual_support_masks)
        
        # 4. 计算相似度图
        # 正样本相似度
        pos_sim_maps = []
        for i, embeddings in enumerate(pos_embeddings):
            sim_map = get_similarity(embeddings, query_features) # [H_feat, W_feat]
            pos_sim_maps.append(sim_map)
            if DEBUG_VIS:
                sim_vis = (sim_map.detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"pos_sim_{i}.png"), cv2.applyColorMap(sim_vis, cv2.COLORMAP_JET))
        
        # 负样本相似度
        neg_sim_maps = []
        for i, embeddings in enumerate(neg_embeddings):
            sim_map = get_similarity(embeddings, query_features)
            neg_sim_maps.append(sim_map)
            if DEBUG_VIS:
                sim_vis = (sim_map.detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"neg_sim_{i}.png"), cv2.applyColorMap(sim_vis, cv2.COLORMAP_JET))
            
        # 合并相似度图
        if pos_sim_maps:
            pos_sim_map = torch.stack(pos_sim_maps, dim=0).mean(dim=0)
        else:
            pos_sim_map = torch.zeros(query_features.shape[-2:], device=query_features.device)
            
        if neg_sim_maps:
            neg_sim_map = torch.stack(neg_sim_maps, dim=0).mean(dim=0)
        else:
            neg_sim_map = torch.zeros(query_features.shape[-2:], device=query_features.device)
            
        # 抑制负样本区域
        # 只有当负样本相似度显著较高（例如 > 0.5）时才进行抑制，
        # 避免背景中的轻微相似度噪声导致正样本区域的得分被削弱
        neg_suppression = torch.where(neg_sim_map > (1-threshold), neg_sim_map, torch.zeros_like(neg_sim_map))
        sim_map = pos_sim_map - neg_suppression
        sim_map = torch.clamp(sim_map, min=0) # [H_feat, W_feat]

        if DEBUG_VIS:
            # 保存聚合后的相似度图
            sim_map_vis = (sim_map.detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(DEBUG_DIR, "sim_map_final.png"), cv2.applyColorMap(sim_map_vis, cv2.COLORMAP_JET))
        
        # 5. 生成候选掩码
        sim_map_np = sim_map.detach().cpu().numpy()
        
        # 特征图对应的是 (window_size, window_size) 的正方形画布
        # 但查询图像内容只占了其中的一部分，需要裁剪出有效区域再 resize 回原图
        fh, fw = sim_map_np.shape
        h_q, w_q = query_image_shape
        ratio_h = h_q / max(h_q, w_q)
        ratio_w = w_q / max(h_q, w_q)
        fh_valid, fw_valid = int(fh * ratio_h), int(fw * ratio_w)
        sim_map_valid = sim_map_np[:fh_valid, :fw_valid]
        
        # Resize 到原始图像大小
        sim_map_full = cv2.resize(sim_map_valid, query_image_shape[::-1], interpolation=cv2.INTER_LINEAR)
        
        # 5. 基于 K-means 的多簇自适应二值化
        # 聚类成 3 个簇（背景、弱相关、强相关），取最亮的簇
        binary_map = graphcut_binarize(sim_map_full)

        if DEBUG_VIS:
            cv2.imwrite(os.path.join(DEBUG_DIR, "binary_map.png"), binary_map * 255)
        
        # 轮廓查找
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        all_results = []
        # 针对每个候选区域，使用 SAM 进行精细化预测
        debug_prompt_img = query_image.copy() if DEBUG_VIS else None

        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 100: # 过滤太小的区域
                continue
                
            # 提取该区域的点和框作为 Prompt
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = [x, y, w, h]
            
            # 采样几个点
            mask_roi = np.zeros(query_image_shape, dtype=np.uint8)
            cv2.drawContours(mask_roi, [cnt], -1, 1, thickness=cv2.FILLED)
            points = mask_to_uniform_points(mask_roi, num_points=5)
            
            if DEBUG_VIS and debug_prompt_img is not None:
                # 可视化框
                cv2.rectangle(debug_prompt_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 可视化点
                for pt in points:
                    cv2.circle(debug_prompt_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            # 针对不同模型类型适配 Prompt 格式
            # SAM 模型通常对点数和标签数有严格匹配要求
            # 同时也要求 bboxes 和 points 的 batch 维度一致
            pts = points[:5] if len(points) > 5 else points
            lbls = np.ones(len(pts), dtype=np.int32)
            
            print(f"[FSWrapper] Calling predict with bbox={bbox}, pts shape={pts.shape}, lbls shape={lbls.shape}")
            
            # 使用底层模型进行预测
            try:
                # 封装成 list 以匹配 bboxes=[bbox] 的 batch=1 结构
                # 这样 bboxes 是 [1, 4], points 是 [1, N, 2], labels 是 [1, N]
                res = self.model.predict(query_image, bboxes=[bbox], points=[pts], labels=[lbls])
                print(f"[FSWrapper] predict returned: {res}")
                if res:
                    all_results.append(res[0])
            except Exception as e:
                print(f"[FSWrapper] SAM refinement predict failed: {e}")
                import traceback
                traceback.print_exc()

        if DEBUG_VIS and debug_prompt_img is not None:
            cv2.imwrite(os.path.join(DEBUG_DIR, "prompts_vis.png"), debug_prompt_img)
                
        # 6. 合并所有结果
        if not all_results:
            return [MockResults(None, None, query_image)]
            
        return [self.merge_results(all_results)]
        
    def release(self):
        self.model.release()
        self.model = None

