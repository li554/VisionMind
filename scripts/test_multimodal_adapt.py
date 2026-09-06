"""压缩器与凭据掩码对多模态 content 的适配验证"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent.message_compressor import compress_messages

IMG_MSG = {
    "role": "user",
    "content": [
        {"type": "text", "text": "这是说明"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "A" * 5000}},
    ],
}

def test_compress_replaces_image_with_summary():
    messages = [IMG_MSG]
    result = compress_messages(messages, keep_recent_n=0)
    assert result, "compress_messages 返回空"
    s = str(result)
    assert "data:image/jpeg;base64," not in s, "压缩结果不应包含 base64 巨串"
    assert "图片" in s, "压缩结果应含图片摘要标记"

if __name__ == "__main__":
    import traceback
    try:
        test_compress_replaces_image_with_summary()
        print("PASS")
    except Exception:
        traceback.print_exc()
        print("FAIL")
        sys.exit(1)
    sys.exit(0)
