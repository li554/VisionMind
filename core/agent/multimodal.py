"""多模态图片编解码与引用解析(纯数据,机制 A/B 共用)。

- ImageCodec: 图片压缩、缩放、ROI 裁剪、标注叠加渲染、base64 编码
- ReferenceResolver: @语法解析与资源解析(Task 4/5 实现)
"""

from __future__ import annotations

import base64
import os


class ImageCodec:
    """图片编码工具(纯函数,无状态)"""

    @staticmethod
    def _load_rgb(path: str):
        import cv2
        import numpy as np
        img = cv2.imread(path)
        if img is None:
            try:
                data = np.fromfile(path, dtype=np.uint8)
                if data.size:
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except OSError:
                pass
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        return img

    @staticmethod
    def _to_data_url(img, quality: int = 80) -> str:
        import cv2
        ok, buf = cv2.imencode(
            ".jpg", img,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _resize_long_side(img, max_side: int):
        import cv2
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return img
        scale = max_side / longest
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _fit_side(img, max_side: int = 0, min_side: int = 0, max_upscale: float = 0):
        """双向适配尺寸：最长边超过 max_side 时缩小（INTER_AREA），
        最短边小于 min_side 时放大（LANCZOS4）。0 表示该方向不限制。

        max_upscale 限制最大放大倍数：极小目标不再被盲目放大到 min_side
        （如 5px→384px 达 55× 只会越插值越糊），而是最多放大到 min_side
        与 max_upscale 二者的较小效果（以实际比例为准）。
        """
        import cv2
        h, w = img.shape[:2]
        if max_side and max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        if min_side and min(h, w) < min_side:
            scale = min_side / min(h, w)
            if max_upscale and scale > max_upscale:
                scale = max_upscale
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_LANCZOS4)
        return img

    @staticmethod
    def _clamp_bbox(img, bbox) -> tuple:
        h, w = img.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        return x, y, bw, bh

    @staticmethod
    def _apply_padding(img, bbox, padding_ratio: float = 0.1, padding_px: int | None = None) -> tuple:
        h, w = img.shape[:2]
        x, y, bw, bh = ImageCodec._clamp_bbox(img, bbox)
        pad = padding_px if padding_px is not None else int(max(bw, bh) * padding_ratio)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        return x0, y0, x1, y1

    @classmethod
    def encode_original(cls, path: str, max_side: int = 1536, quality: int = 80) -> str:
        """① 原图: 缩放最长边 + JPEG 压缩 → data_url"""
        img = cls._load_rgb(path)
        img = cls._resize_long_side(img, max_side)
        return cls._to_data_url(img, quality)

    @classmethod
    def encode_crop(cls, path: str, bbox, padding_ratio: float = 0.1,
                    padding_px: int | None = None, max_side: int = 1536,
                    min_side: int = 0, max_upscale: float = 0) -> str:
        """② 局部: 标注框裁剪(+padding) → data_url

        min_side>0 时小裁剪块会显式放大（LANCZOS4）；max_upscale 限制最大放大倍数。
        """
        img = cls._load_rgb(path)
        x0, y0, x1, y1 = cls._apply_padding(img, bbox, padding_ratio, padding_px)
        crop = img[y0:y1, x0:x1]
        crop = cls._fit_side(crop, max_side, min_side, max_upscale)
        return cls._to_data_url(crop)

    @classmethod
    def encode_overlay(cls, path: str, annotations: list, max_side: int = 1536,
                       show_index: bool = False) -> str:
        """③ 渲染图: 全图叠加标注(矩形/多边形 + 类别标签 + 可选索引号) → data_url

        show_index=True 时在框左上角绘制从 1 开始的索引号，供批量审查精确引用。
        """
        img = cls._load_rgb(path)
        img = cls._draw_annotations(img, annotations, show_index=show_index)
        img = cls._resize_long_side(img, max_side)
        return cls._to_data_url(img)

    @classmethod
    def encode_overlay_crop(cls, path: str, annotations: list, bbox,
                            padding_ratio: float = 0.1, padding_px: int | None = None,
                            max_side: int = 1536, min_side: int = 0,
                            max_upscale: float = 0) -> str:
        """④ 渲染局部: 叠加标注后按标注框裁剪(+padding) → data_url

        min_side>0 时小裁剪块会显式放大（LANCZOS4）；max_upscale 限制最大放大倍数。
        """
        img = cls._load_rgb(path)
        img = cls._draw_annotations(img, annotations)
        x0, y0, x1, y1 = cls._apply_padding(img, bbox, padding_ratio, padding_px)
        crop = img[y0:y1, x0:x1]
        crop = cls._fit_side(crop, max_side, min_side, max_upscale)
        return cls._to_data_url(crop)

    @classmethod
    def montage_crops(cls, entries: list, cell_side: int = 96, max_cols: int = 8,
                      max_cells: int = 48, padding_ratio: float = 0.3,
                      padding_px: int | None = None) -> str:
        """⑤ 极小目标拼贴: 多个 (path, bbox) 裁剪块放大后拼成网格 → data_url

        每个裁剪块缩放到 cell_side（LANCZOS4），格内左上角绘制从 1 开始的
        编号（ASCII），编号到实例的映射由调用方在 prompt 文本中给出。
        单块过小时（如 5×5px）放大后仍模糊——拼贴的意义在于让 VLM 一次看到
        该类别全部极小实例的聚合规律（颜色/亮度/位置一致性）。
        """
        import cv2
        import numpy as np
        entries = list(entries)[:max_cells]
        if not entries:
            raise ValueError("montage_crops 需要至少一个 (path, bbox)")
        cols = min(max_cols, max(1, int(np.ceil(np.sqrt(len(entries))))))
        cols = min(cols, len(entries))
        rows = int(np.ceil(len(entries) / cols))
        canvas = np.full((rows * cell_side, cols * cell_side, 3), 32, dtype=np.uint8)
        for i, (path, bbox) in enumerate(entries):
            img = cls._load_rgb(path)
            x0, y0, x1, y1 = cls._apply_padding(img, bbox, padding_ratio, padding_px)
            cell = img[y0:y1, x0:x1]
            cell = cls._fit_side(cell, max_side=cell_side, min_side=cell_side)
            r, c = divmod(i, cols)
            y, x = r * cell_side, c * cell_side
            canvas[y:y + cell_side, x:x + cell_side] = cell[:cell_side, :cell_side]
            cv2.rectangle(canvas, (x, y), (x + cell_side - 1, y + cell_side - 1), (70, 70, 70), 1)
            cv2.putText(canvas, str(i + 1), (x + 4, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return cls._to_data_url(canvas)

    @staticmethod
    def _draw_annotations(img, annotations: list, show_index: bool = False):
        """叠加渲染标注。遵循标注标识规则: [主]常规色、仅[副]蓝色。

        标注 dict 字段: label(str), bbox([x,y,w,h]), polygons(可选), is_secondary(bool 可选)。
        show_index=True 时在框左上角绘制从 1 开始的索引号，标签右移让位。
        """
        import cv2
        import numpy as np
        text_ops = []
        for i, ann in enumerate(annotations):
            label = ann.get("label")
            is_secondary = bool(ann.get("is_secondary"))
            color = (255, 0, 0) if is_secondary else (0, 200, 0)  # BGR: 蓝=副, 绿=主
            prefix = "[副] " if is_secondary else "[主] "
            polygons = ann.get("polygons")
            bbox = ann.get("bbox")
            if polygons:
                for poly in polygons:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], True, color, 2)
            elif bbox:
                x, y, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            if bbox:
                x, y = int(bbox[0]), int(bbox[1])
                if show_index:
                    text_ops.append((f"{i + 1}", (x, max(10, y - 5)), color))
                if label:
                    text_ops.append((prefix + label, (x + (26 if show_index else 0), max(10, y - 5)), color))
        if text_ops:
            img = ImageCodec._draw_texts(img, text_ops)
        return img

    @staticmethod
    def _draw_texts(img, text_ops: list):
        """用 PIL 绘制文本(支持中文标签),字体不可用时回退 cv2.putText。"""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            import cv2
            for text, org, color in text_ops:
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return img
        font = None
        for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"):
            try:
                font = ImageFont.truetype(f"C:/Windows/Fonts/{name}", 20)
                break
            except OSError:
                continue
        if font is None:
            import cv2
            for text, org, color in text_ops:
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return img
        import numpy as np
        rgb = img[:, :, ::-1].copy()
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        for text, org, color in text_ops:
            draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
        return np.asarray(pil)[:, :, ::-1].copy()


class ReferenceError(Exception):
    """引用语法或资源解析错误"""


class ReferenceResolver:
    """解析 @引用语法 → 资源规格(纯数据)。

    - no_service=True 时仅做语法解析(单测用),不查询 service 层。
      该标志由 Task 5 的 resolve/resolve_message 消费,本阶段仅语法解析。
    - 数据查询统一走 ActionRegistry(service 层 action),不直接读磁盘。
    """

    def __init__(self, no_service: bool = False):
        self.no_service = no_service

    def _parse_grammar(self, token: str) -> dict:
        """解析单个 @token,返回资源规格 dict。非法语法抛 ReferenceError。

        语法兼容两种形态:
        - 带分隔符: `@原图:img_003.jpg` / `@局部:图3标注2` / `@类别:缺陷` / `@搜索:裂纹`
        - 无分隔符: `@图3`(索引) / `@难样本`(裸关键字,无目标)
        """
        import re
        m = re.match(r"^@(原图|图|局部|渲染图|渲染局部|类别|难样本|搜索|报告|文件|ROI)(?::\s*|\s+)?(.*)$", token)
        if not m:
            raise ReferenceError(f"无法识别的引用语法: {token}")
        kind, target = m.group(1), m.group(2).strip()
        target = target.rstrip("，,。;；")

        if kind == "文件":
            if not target:
                raise ReferenceError("文件引用缺少路径,如 @文件:D:\\project\\images\\001.jpg")
            return {"kind": "file", "target": target}
        if kind == "ROI":
            return {"kind": "roi"}
        if kind in ("原图", "图"):
            if not target:
                raise ReferenceError(f"{kind} 引用缺少目标,如 @图3 或 @原图:img_003.jpg")
            return {"kind": "original", "target": target}
        if kind == "渲染图":
            if not target:
                raise ReferenceError("渲染图引用缺少目标,如 @渲染图:img_003.jpg")
            return {"kind": "overlay", "target": target}
        if kind in ("局部", "渲染局部"):
            # 格式: 图名标注N 或 图片索引标注N(如 图3标注2 / img_003.jpg标注2)
            m2 = re.match(r"^(图\d+)(?:标注)(\d+)$", target)
            m3 = re.match(r"^(.+?\.(?:jpg|jpeg|png|bmp|webp))(?:标注)(\d+)$", target, re.IGNORECASE)
            if m2:
                # 剥离"图"前缀,产出纯索引(与 @图3 的 target="3" 一致)
                return {"kind": "local" if kind == "局部" else "overlay_local",
                        "image": m2.group(1)[1:], "annotation": int(m2.group(2))}
            if m3:
                return {"kind": "local" if kind == "局部" else "overlay_local",
                        "image": m3.group(1), "annotation": int(m3.group(2))}
            raise ReferenceError(
                f"标注引用必须带图名,如 @局部:图3标注2 或 @局部:img_003.jpg标注2, 收到: {target}")
        if kind == "类别":
            if not target:
                raise ReferenceError("类别引用缺少类别名,如 @类别:缺陷")
            return {"kind": "category", "target": target}
        if kind == "难样本":
            return {"kind": "hard_samples"}
        if kind == "搜索":
            if not target:
                raise ReferenceError("搜索引用缺少关键词,如 @搜索:裂纹")
            return {"kind": "search", "target": target}
        if kind == "报告":
            if not target:
                raise ReferenceError("报告引用缺少名称,如 @报告:分析")
            return {"kind": "report", "target": target}
        raise ReferenceError(f"未知引用类型: {kind}")

    # ---- 资源解析(经 ActionRegistry 调 service 层 action) ----

    def resolve(self, spec: dict) -> dict:
        """把资源规格解析为实际数据。

        返回:
          {"type": "image", "path": str, "data_url": str|None, "caption": str}   个体引用
          {"type": "paths", "paths": [str], "caption": str}                      集合引用(路径列表)
          {"type": "text", "text": str}                                          纯文本引用
        """
        if self.no_service:
            raise ReferenceError("no_service 模式不支持 resolve(仅语法解析)")
        kind = spec["kind"]
        if kind == "file":
            path = self._resolve_file_path(spec["target"])
            return {"type": "image", "path": path,
                    "data_url": ImageCodec.encode_original(path), "caption": f"[图片: {path}]"}
        if kind == "roi":
            state = self._call_action("annotation.manual.get_state")
            path = (state or {}).get("current_image_path")
            if not path:
                raise ReferenceError("当前无选中图片,无法引用 ROI")
            from core.common.project_settings import project_settings
            roi = project_settings.get("roi")
            if not roi or len(roi) < 4:
                raise ReferenceError("当前项目未配置 ROI 区域")
            data_url = ImageCodec.encode_crop(path, roi, padding_ratio=0.0)
            return {"type": "image", "path": path, "data_url": data_url,
                    "caption": f"[ROI区域 {roi}: {path}]"}
        if kind == "original":
            path = self._resolve_image_path(spec["target"])
            return {"type": "image", "path": path,
                    "data_url": ImageCodec.encode_original(path), "caption": f"[图片: {path}]"}
        if kind == "overlay":
            path = self._resolve_image_path(spec["target"])
            anns = self._load_annotations(path)
            return {"type": "image", "path": path,
                    "data_url": ImageCodec.encode_overlay(path, anns),
                    "caption": f"[标注渲染图: {path}]"}
        if kind in ("local", "overlay_local"):
            img_spec = spec["image"]
            ann_idx = spec["annotation"]
            path = self._resolve_image_path(img_spec)
            anns = self._load_annotations(path)
            if ann_idx < 0 or ann_idx >= len(anns):
                raise ReferenceError(f"标注索引 {ann_idx} 超出范围(共 {len(anns)} 个标注)")
            bbox = anns[ann_idx].get("bbox")
            if not bbox:
                raise ReferenceError(f"标注 {ann_idx} 无 bbox,无法裁剪")
            if kind == "local":
                data_url = ImageCodec.encode_crop(path, bbox)
            else:
                data_url = ImageCodec.encode_overlay_crop(path, anns, bbox)
            return {"type": "image", "path": path, "data_url": data_url,
                    "caption": f"[标注{ann_idx}区域: {path}]"}
        if kind == "category":
            paths = self._query_category_paths(spec["target"])
            return {"type": "paths", "paths": paths,
                    "caption": f"类别「{spec['target']}」的图片({len(paths)} 张):\n" + "\n".join(paths)}
        if kind == "hard_samples":
            paths = self._query_hard_samples()
            return {"type": "paths", "paths": paths,
                    "caption": "难样本图片:\n" + "\n".join(paths)}
        if kind == "search":
            paths = self._query_search_paths(spec["target"])
            return {"type": "paths", "paths": paths,
                    "caption": f"搜索「{spec['target']}」结果({len(paths)} 张):\n" + "\n".join(paths)}
        if kind == "report":
            return {"type": "text", "text": f"[报告: {spec['target']}]"}
        raise ReferenceError(f"未知引用规格: {spec}")

    def resolve_message(self, text: str):
        """解析整条消息文本: 提取 @引用,返回 content(str 或多模态 list)。

        - 无引用 → 原样返回 str
        - 有个体引用(图片) → 返回 [{"type":"text",...}, {"type":"image_url",...}, ...]
        - 仅集合/文本引用 → 文本中插入路径列表(返回 str)
        - 解析失败 → 在文本中插入错误提示(返回 str)
        """
        import re
        tokens = re.findall(
            r"@(?:原图|图|局部|渲染图|渲染局部|类别|难样本|搜索|报告|文件|ROI)[^\s@，,。;；]*", text)
        if not tokens:
            return text

        results = []
        for token in tokens:
            try:
                spec = self._parse_grammar(token)
                results.append((token, self.resolve(spec)))
            except Exception as e:
                results.append((token, {"type": "text", "text": f"[引用错误: {e}]"}))

        # 按 token 长度降序排序,避免前缀重叠导致误替换(如 @图3 vs @图30)
        results.sort(key=lambda x: len(x[0]), reverse=True)

        has_image = any(r["type"] == "image" for _, r in results)
        if not has_image:
            new_text = text
            for token, r in results:
                caption = r.get("caption", r.get("text", ""))
                new_text = new_text.replace(token, "\n" + caption)
            return new_text

        # 存在图片引用 → 构造多模态 content list
        remaining = text
        for token, _ in results:
            remaining = remaining.replace(token, "")
        content = []
        if remaining.strip():
            content.append({"type": "text", "text": remaining.strip()})
        for _, r in results:
            if r["type"] == "image":
                content.append({"type": "text", "text": r["caption"]})
                content.append({"type": "image_url", "image_url": {"url": r["data_url"]}})
            elif r["type"] == "paths":
                content.append({"type": "text", "text": r["caption"]})
            elif r["type"] == "text":
                content.append({"type": "text", "text": r["text"]})
        return content

    # ---- 数据源: 统一经 ActionRegistry 调 service 层 action ----
    # 已核实: annotation.manual.get_state 不返回 image_files(只有 image_count),
    # 全量枚举图片用 annotation.manual.search_images(keyword="") → {"files": [[path, hidden], ...]}
    # get_state().hard_samples 是 dict {path: description},不是列表。

    @staticmethod
    def _action_registry():
        from core.common.action_registry import ActionRegistry
        return ActionRegistry.instance()

    def _call_action(self, action: str, **kwargs):
        registry = self._action_registry()
        return registry.call(action, **kwargs)

    def _all_image_files(self) -> list:
        """枚举当前项目全部图片路径(经 search_images 空关键词全量返回)。"""
        result = self._call_action("annotation.manual.search_images", keyword="")
        files = (result.get("files") or []) if isinstance(result, dict) else []
        out = []
        for entry in files:
            p = entry[0] if isinstance(entry, (list, tuple)) else entry
            if p:
                out.append(p)
        return out

    def _resolve_image_path(self, target: str) -> str:
        """target 为索引 → 取当前项目图片列表;为文件名 → 匹配当前项目/全项目。

        兼容 `_parse_grammar` 产生的三种 target 形态:
        - "3"(@图3 纯索引)
        - "图3"(@局部:图3标注N 的 image 字段,需剥掉前缀)
        - "img_003.jpg"(@原图:/@局部:img_003.jpg标注N 的文件名)
        """
        files = self._all_image_files()
        idx_str = target
        if idx_str.startswith("图") and idx_str[1:].isdigit():
            idx_str = idx_str[1:]
        if idx_str.isdigit():
            idx = int(idx_str)
            if idx < 0 or idx >= len(files):
                raise ReferenceError(f"图片索引 {idx} 超出范围(共 {len(files)} 张)")
            return files[idx]
        for f in files:
            if os.path.basename(f) == target:
                return f
        # 按文件名关键词兜底(子串匹配)
        for f in files:
            if target.lower() in os.path.basename(f).lower():
                return f
        raise ReferenceError(f"未找到图片: {target}")

    def _resolve_file_path(self, target: str) -> str:
        """文件引用: 完整路径直接使用,否则按文件名匹配项目图片。"""
        if os.path.isfile(target):
            return target
        return self._resolve_image_path(target)

    def _load_annotations(self, path: str) -> list:
        """读单张图片标注 → 简洁标注列表(经 service 层 action,复用缓存)。"""
        result = self._call_action("annotation.manual.get_annotations_for_image", image_path=path)
        anns = (result or {}).get("annotations") or []
        return anns

    def _query_category_paths(self, category: str) -> list:
        return [f for f in self._all_image_files()
                if any(a["label"] == category for a in self._load_annotations(f))]

    def _query_hard_samples(self) -> list:
        state = self._call_action("annotation.manual.get_state")
        hard = (state or {}).get("hard_samples") or {}
        return [p for p in hard.keys() if p]

    def _query_search_paths(self, keyword: str) -> list:
        result = self._call_action("annotation.manual.search_images", keyword=keyword)
        files = (result.get("files") or []) if isinstance(result, dict) else []
        out = []
        for entry in files:
            p = entry[0] if isinstance(entry, (list, tuple)) else entry
            out.append(p)
        return out


def _load_annotations_static(path: str) -> list:
    """纯函数版本读标注(通过 ReferenceResolver 走 service action,复用缓存)。"""
    return ReferenceResolver()._load_annotations(path)


def replace_image_urls_with_captions(messages: list) -> list:
    """把多模态图片消息收敛为 view_image tool 消息中的结构化描述，移除 base64。

    图片由 ``assistant(tool_calls: view_image) → tool([已加载图片: path])`` 发起，
    其后 agent 注入一条含 image_url 的 user 消息再回复解读。这里把解读回填到
    该 view_image 的 tool 消息，并删除含 base64 的注入 user 消息。

    - 保留 assistant(tool_calls) 与 tool 消息
    - 删除注入的图片 user 消息(消除 base64)
    - 解读取"图片消息之后最后一个纯文本 assistant"(跳过执行计划/tool_calls)
    - 匹配 tool 消息(内容以 [已加载图片: 开头)并从中提取 path 回填结构化描述
    - 无对应 tool 时把图片 user 消息收敛为该结构化文本兜底
    """
    result: list = []
    i = 0
    n = len(messages)

    def _last_text_assistant_after(start: int) -> str:
        """自 start 起找最后一条纯文本 assistant(非 tool_calls)的文本。"""
        cap = ""
        for idx in range(start, n):
            m = messages[idx]
            if not isinstance(m, dict):
                continue
            if m.get("role") != "assistant":
                continue
            if m.get("tool_calls"):
                continue  # 执行计划/工具调用轮,非图片解读
            txt = m.get("content")
            if isinstance(txt, str) and txt.strip():
                cap = txt.strip()
        return cap

    while i < n:
        m = messages[i]
        content = m.get("content")
        is_img_user = (
            isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(content, list)
            and any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)
        )
        if not is_img_user:
            result.append(m)
            i += 1
            continue

        caption = _last_text_assistant_after(i + 1)

        # 找对应的 view_image tool 消息(内容以 [已加载图片: 开头)回填描述
        path = ""
        target_tool = None
        for im in reversed(result):
            if im.get("role") != "tool":
                continue
            imc = im.get("content")
            if not isinstance(imc, str):
                continue
            if imc.startswith("[已加载图片:"):
                target_tool = im
                path = imc[len("[已加载图片:"):].rstrip("]").strip().split(",")[0].strip()
                break

        if caption:
            desc_text = caption
            if len(desc_text) > 300:
                desc_text = desc_text[:300] + "..."
            structured = f"[图片: path={path}, 描述={desc_text}]" if path else f"[图片: {desc_text}]"
        else:
            structured = f"[图片: path={path}, 无文字解读]" if path else "[图片: 无文字解读]"

        if target_tool is not None:
            target_tool["content"] = structured
        else:
            # 无对应 tool：该图片消息无独立归属。若已提取到解读/路径则收敛为
            # 结构化 user 文本保留语义；否则(如 @图N 指令引用)直接丢弃——
            # 解读已回填到 view_image tool,避免残留 [图片: 无文字解读] 冗余行。
            if path or caption:
                result.append({"role": "user", "content": structured})

        i += 1

    return result


def build_image_user_message(marker: str) -> dict | None:
    """把 view_image 工具标记解析为 user 图片消息。

    marker 格式: [IMG_VIEW: path, mode=..., annotation=..., padding=...]
    返回 {"role": "user", "content": [text, image]} 或 None(解析/编码失败)。
    """
    import re
    m = re.match(
        r"\[IMG_VIEW:\s*(.+?),\s*mode=(\w+)(?:,\s*annotation=(\d+))?(?:,\s*padding=([\d.]+))?\]",
        marker)
    if not m:
        return None
    path, mode = m.group(1).strip(), m.group(2)
    annotation_index = int(m.group(3)) if m.group(3) else None
    padding = float(m.group(4)) if m.group(4) else None
    try:
        if mode == "original":
            url = ImageCodec.encode_original(path)
        elif mode == "overlay":
            anns = _load_annotations_static(path)
            url = ImageCodec.encode_overlay(path, anns)
        elif mode == "local":
            anns = _load_annotations_static(path)
            if annotation_index is None or not 0 <= annotation_index < len(anns):
                return None
            bbox = anns[annotation_index].get("bbox")
            if bbox is None:
                return None
            url = ImageCodec.encode_crop(path, bbox, padding_px=padding)
        elif mode == "overlay_local":
            anns = _load_annotations_static(path)
            if annotation_index is None or not 0 <= annotation_index < len(anns):
                return None
            bbox = anns[annotation_index].get("bbox")
            if bbox is None:
                return None
            url = ImageCodec.encode_overlay_crop(path, anns, bbox, padding_px=padding)
        elif mode == "roi":
            from core.common.project_settings import project_settings
            roi = project_settings.get("roi")
            if not roi or len(roi) < 4:
                return None
            cur = path
            if cur == "ROI" or not cur or not os.path.isfile(cur):
                state = ReferenceResolver()._call_action("annotation.manual.get_state")
                cur = (state or {}).get("current_image_path")
            if not cur:
                return None
            url = ImageCodec.encode_crop(cur, roi, padding_ratio=0.0)
            path = cur
        else:
            return None
    except Exception:
        return None
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"[系统: 已按 view_image 请求加载图片 {path} 到对话,请基于图片继续分析]"},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }
