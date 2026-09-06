import os
from typing import Optional
import cv2
import numpy as np


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
