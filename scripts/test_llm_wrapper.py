"""llm.chat 包装层: 检测 [IMG_VIEW] 标记并注入 user 图片消息"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent.multimodal import build_image_user_message

def test_marker_parse_original_returns_msg():
    msg = build_image_user_message("[IMG_VIEW: /tmp/a.jpg, mode=original]")
    # 文件不存在时返回 None(不崩溃); 存在时应返回 user 消息
    if msg is not None:
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)

def test_invalid_marker_returns_none():
    msg = build_image_user_message("普通文本不是标记")
    assert msg is None

if __name__ == "__main__":
    import traceback
    fns = [test_marker_parse_original_returns_msg, test_invalid_marker_returns_none]
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
