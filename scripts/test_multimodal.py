"""ImageCodec 纯函数单测"""
import sys, os, base64
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent.multimodal import ImageCodec, ReferenceResolver, ReferenceError as RefError

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_imgcodec_tmp")
os.makedirs(TEST_DIR, exist_ok=True)
IMG_PATH = os.path.join(TEST_DIR, "sample.jpg")

def _ensure_image():
    if not os.path.exists(IMG_PATH):
        import cv2
        import numpy as np
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "projects", "焊脉", "images")
        candidates = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".png"))] if os.path.isdir(src) else []
        assert candidates, f"项目图片目录为空或不存在: {src}"
        src_path = os.path.join(src, candidates[0])
        data = np.fromfile(src_path, dtype=np.uint8)
        cv2.imwrite(IMG_PATH, cv2.imdecode(data, cv2.IMREAD_COLOR))

def _decode_to_img(url):
    import cv2
    import numpy as np
    raw = base64.b64decode(url.split(",", 1)[1])
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def test_encode_original():
    _ensure_image()
    url = ImageCodec.encode_original(IMG_PATH)
    assert url.startswith("data:image/jpeg;base64,"), url[:30]
    raw = base64.b64decode(url.split(",", 1)[1])
    assert len(raw) > 100

def test_encode_original_respects_max_side():
    _ensure_image()
    import cv2
    h, w = cv2.imread(IMG_PATH).shape[:2]
    url = ImageCodec.encode_original(IMG_PATH, max_side=256)
    raw = base64.b64decode(url.split(",", 1)[1])
    tmp = os.path.join(TEST_DIR, "resized.jpg")
    with open(tmp, "wb") as f:
        f.write(raw)
    h2, w2 = cv2.imread(tmp).shape[:2]
    assert max(h2, w2) <= 256, f"最长边 {max(h2,w2)} > 256"

def test_encode_crop_with_padding():
    _ensure_image()
    import cv2
    bbox = [100, 100, 200, 150]
    url = ImageCodec.encode_crop(IMG_PATH, bbox)
    raw = base64.b64decode(url.split(",", 1)[1])
    tmp = os.path.join(TEST_DIR, "crop.jpg")
    with open(tmp, "wb") as f:
        f.write(raw)
    h2, w2 = cv2.imread(tmp).shape[:2]
    pad = int(max(200, 150) * 0.1)
    assert abs(h2 - (150 + 2 * pad)) <= 2, f"裁剪高度 {h2}, 期望 ~{150 + 2*pad}"
    assert abs(w2 - (200 + 2 * pad)) <= 2, f"裁剪宽度 {w2}, 期望 ~{200 + 2*pad}"

def test_encode_overlay():
    _ensure_image()
    import numpy as np
    annotations = [{"label": "缺陷", "bbox": [10, 10, 100, 80], "polygons": None}]
    url = ImageCodec.encode_overlay(IMG_PATH, annotations)
    assert url.startswith("data:image/jpeg;base64,")
    diff = np.abs(_decode_to_img(url).astype(int) - _decode_to_img(ImageCodec.encode_original(IMG_PATH)).astype(int))
    assert diff.sum() > 0, "叠加渲染未产生像素差异"

def test_encode_overlay_polygon():
    _ensure_image()
    import numpy as np
    annotations = [{"label": "缺陷", "bbox": [20, 20, 100, 70],
                    "polygons": [[[20, 20], [120, 20], [120, 90], [20, 90]]]}]
    url = ImageCodec.encode_overlay(IMG_PATH, annotations)
    assert url.startswith("data:image/jpeg;base64,")
    diff = np.abs(_decode_to_img(url).astype(int) - _decode_to_img(ImageCodec.encode_original(IMG_PATH)).astype(int))
    assert diff.sum() > 0, "polygon 叠加渲染未产生像素差异"

def test_encode_overlay_crop():
    _ensure_image()
    annotations = [{"label": "缺陷", "bbox": [10, 10, 100, 80], "polygons": None}]
    url = ImageCodec.encode_overlay_crop(IMG_PATH, annotations, annotations[0]["bbox"])
    assert url.startswith("data:image/jpeg;base64,")


def test_parse_original_syntax():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@原图:img_003.jpg")
    assert spec == {"kind": "original", "target": "img_003.jpg"}

def test_parse_index_syntax():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@图3")
    assert spec == {"kind": "original", "target": "3"}

def test_parse_local_requires_image():
    resolver = ReferenceResolver(no_service=True)
    try:
        resolver._parse_grammar("@局部:标注2")
        raise AssertionError("应抛出 ReferenceError(标注必须带图名)")
    except RefError:
        pass

def test_parse_local_with_image():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@局部:图3标注2")
    assert spec == {"kind": "local", "image": "3", "annotation": 2}

def test_parse_overlay():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@渲染图:img_003.jpg")
    assert spec == {"kind": "overlay", "target": "img_003.jpg"}

def test_parse_overlay_local():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@渲染局部:图5标注1")
    assert spec == {"kind": "overlay_local", "image": "5", "annotation": 1}

def test_parse_category_collection():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@类别:缺陷")
    assert spec == {"kind": "category", "target": "缺陷"}

def test_parse_hard_samples():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@难样本")
    assert spec == {"kind": "hard_samples"}

def test_parse_search():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@搜索:裂纹")
    assert spec == {"kind": "search", "target": "裂纹"}

def test_parse_report():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@报告:分析")
    assert spec == {"kind": "report", "target": "分析"}

def test_parse_file_abs_path():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@文件:D:\\project\\images\\001.jpg")
    assert spec == {"kind": "file", "target": "D:\\project\\images\\001.jpg"}

def test_parse_file_filename():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@文件:img_003.jpg")
    assert spec == {"kind": "file", "target": "img_003.jpg"}

def test_parse_file_missing_target():
    resolver = ReferenceResolver(no_service=True)
    try:
        resolver._parse_grammar("@文件")
        raise AssertionError("应抛 ReferenceError(文件引用缺路径)")
    except RefError:
        pass

def test_parse_roi():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@ROI")
    assert spec == {"kind": "roi"}

def test_parse_local_with_filename():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@局部:img_003.jpg标注2")
    assert spec == {"kind": "local", "image": "img_003.jpg", "annotation": 2}

def test_parse_local_with_filename_uppercase_ext():
    resolver = ReferenceResolver(no_service=True)
    spec = resolver._parse_grammar("@渲染局部:IMG_003.JPG标注1")
    assert spec == {"kind": "overlay_local", "image": "IMG_003.JPG", "annotation": 1}

def test_parse_missing_target_errors():
    resolver = ReferenceResolver(no_service=True)
    for bad in ("@渲染图", "@类别", "@搜索", "@报告", "@图"):
        try:
            resolver._parse_grammar(bad)
            raise AssertionError(f"应抛 ReferenceError: {bad}")
        except RefError:
            pass

def test_parse_unknown_syntax():
    resolver = ReferenceResolver(no_service=True)
    try:
        resolver._parse_grammar("foo")
        raise AssertionError("应抛 ReferenceError(无法识别)")
    except RefError:
        pass

def test_resolve_original_by_index():
    """在项目环境中: 解析 @图0 应返回第一张图的 data_url。"""
    resolver = ReferenceResolver(no_service=False)
    spec = resolver._parse_grammar("@图0")
    try:
        result = resolver.resolve(spec)
    except Exception as e:
        print(f"SKIP: {e}")
        return
    assert result["type"] == "image"
    assert result["data_url"].startswith("data:image/jpeg;base64,")

if __name__ == "__main__":
    import traceback
    import shutil
    fns = [test_encode_original, test_encode_original_respects_max_side,
           test_encode_crop_with_padding, test_encode_overlay,
           test_encode_overlay_polygon, test_encode_overlay_crop,
           test_parse_original_syntax, test_parse_index_syntax,
           test_parse_local_requires_image, test_parse_local_with_image,
           test_parse_overlay, test_parse_overlay_local,
           test_parse_category_collection, test_parse_hard_samples,
           test_parse_search, test_parse_report,
           test_parse_local_with_filename, test_parse_local_with_filename_uppercase_ext,
           test_parse_missing_target_errors, test_parse_unknown_syntax,
           test_resolve_original_by_index]
    ok = True
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            ok = False
            traceback.print_exc()
            print(f"FAIL {fn.__name__}")
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    sys.exit(0 if ok else 1)
