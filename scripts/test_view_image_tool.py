"""view_image 工具: 返回文本标记,不直接编码图片"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent.tools.view_image_tool import ViewImageTool

def test_name_and_schema():
    t = ViewImageTool()
    assert t.name == "view_image"
    assert "mode" in t.parameters["properties"]
    assert t.parameters["properties"]["mode"]["enum"] == \
        ["original", "local", "overlay", "overlay_local", "roi"]

def test_execute_returns_marker():
    t = ViewImageTool()
    result = t.execute(path="test.jpg", mode="overlay")
    assert result.startswith("[IMG_VIEW:"), result
    assert "test.jpg" in result and "overlay" in result

def test_execute_missing_path():
    t = ViewImageTool()
    result = t.execute(mode="original")
    assert result.startswith("错误"), result

def test_execute_invalid_mode():
    t = ViewImageTool()
    result = t.execute(path="x.jpg", mode="bad")
    assert result.startswith("错误"), result

def test_execute_roi_uses_placeholder_path():
    t = ViewImageTool()
    result = t.execute(mode="roi")
    assert result.startswith("[IMG_VIEW: ROI"), result
    assert "mode=roi" in result, result

def test_execute_roi_with_path_keeps_path():
    t = ViewImageTool()
    result = t.execute(path="img.jpg", mode="roi")
    assert result.startswith("[IMG_VIEW: img.jpg"), result
    assert "mode=roi" in result, result

if __name__ == "__main__":
    import traceback
    fns = [test_name_and_schema, test_execute_returns_marker,
           test_execute_missing_path, test_execute_invalid_mode,
           test_execute_roi_uses_placeholder_path,
           test_execute_roi_with_path_keeps_path]
    ok = True
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            ok = False
            traceback.print_exc()
            print(f"FAIL {fn.__name__}")
    sys.exit(0 if ok else 1)
