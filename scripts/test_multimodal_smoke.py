"""冒烟验证: agnes-2.0-flash 是否接受 OpenAI 标准多模态 content list"""
import sys, os, base64, struct, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.corecoder import LLM


def _png_1px():
    """生成 1x1 红色像素 PNG bytes"""
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def main():
    api_key = os.environ.get("AI_API_KEY", "")
    base_url = os.environ.get("AI_BASE_URL", "https://apihub.agnes-ai.com/v1/chat/completions")
    model = os.environ.get("AI_MODEL", "agnes-2.0-flash")
    if not api_key:
        print("SKIP: 未设置 AI_API_KEY 环境变量")
        return 0

    b64 = base64.b64encode(_png_1px()).decode()
    llm = LLM(model=model, api_key=api_key, base_url=base_url)
    resp = llm.chat(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "这张图片的主色调是什么?只回答颜色名。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }])
    print(f"模型回复: {resp.content}")
    assert "红" in (resp.content or "") or "red" in (resp.content or "").lower()
    print("PASS: 多模态消息格式可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
