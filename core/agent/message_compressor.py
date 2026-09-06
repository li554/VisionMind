"""
消息历史压缩 — 在 token 预算超阈值时压缩旧消息

策略：
- 保留近期滑动窗口内的完整消息（含 tool_calls 和 tool 结果）
- 将窗口外的消息通过 LLM 生成摘要
- 摘要以 system 消息注入
- 压缩失败时回退为简单截断
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _image_description_map(messages: List[Dict[str, Any]]) -> Dict[int, str]:
    """映射图片 user 消息 → LLM 首次读到该图后给出的描述。

    view_image 把图片注成 ``{role: user, content: [text, image_url]}``，随后的
    assistant 消息通常就是 LLM 基于该图的解读。此处把这条 assistant 文本作为
    ``[图片: 描述]`` 的来源，使压缩后仍能保留图片的语义信息而非 base64 或路径。

    - 对每条图片 user 消息，向下找最近一条含文本的 assistant 消息
    - 多条待描述图片共享同一条后续 assistant 文本（两者都会用）
    - 返回 ``{图片消息在 messages 中的索引: 描述文本}``
    """
    out: Dict[int, str] = {}
    pending: List[int] = []
    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "user" and isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        ):
            pending.append(idx)
        elif role == "assistant":
            text = content if isinstance(content, str) else ""
            if pending and text.strip():
                for i in pending:
                    out[i] = text.strip()
                pending = []
    return out


def _format_modal_content(
    content: Any, max_chars: int = 200, image_caption: Optional[str] = None
) -> str:
    """把多模态 content 序列化为可读文本，图片部件转为 ``[图片: 描述]``。

    不 dump 底层 base64 data_url；text 部件原样保留，image_url 部件用
    ``image_caption``（LLM 首次读到该图给出的描述，见 _image_description_map）
    作为描述；无描述时退化为 ``[图片]``。返回结果截断到 max_chars。
    """
    if not isinstance(content, list):
        try:
            return json.dumps(content, ensure_ascii=False)[:max_chars]
        except (TypeError, ValueError):
            return str(content)[:max_chars]
    if image_caption and len(image_caption) > 160:
        image_caption = image_caption[:160] + "..."
    parts: List[str] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type")
        if ptype == "text":
            parts.append(str(p.get("text", "")))
        elif ptype == "image_url":
            parts.append(f"[图片: {image_caption}]" if image_caption else "[图片]")
        else:
            try:
                parts.append(json.dumps(p, ensure_ascii=False))
            except (TypeError, ValueError):
                parts.append(str(p))
    joined = " ".join(parts)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "[已截断]"
    return joined


def _format_messages_for_summary(messages: List[Dict[str, Any]]) -> str:
    """将消息列表格式化为可读文本，供 LLM 摘要使用。

    格式：``[role] content前200字符\\n``
    处理 tool_calls：序列化为 JSON 字符串追加到 content 之后。
    多模态图片改为 ``[图片: LLM描述]``（来自其后的 assistant 解读）。
    """
    lines: List[str] = []
    caption_map = _image_description_map(messages)
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            lines.append(f"[unknown] {str(msg)[:200]}\n")
            continue

        role = msg.get("role", "unknown")
        content = msg.get("content")

        # 处理 content：字符串直接使用，其余(多模态列表)转为可读文本
        if content is None:
            content_text = ""
        elif isinstance(content, str):
            content_text = content
        else:
            content_text = _format_modal_content(
                content, max_chars=200, image_caption=caption_map.get(i)
            )

        # 处理 tool_calls：序列化为 JSON 追加
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            try:
                content_text += "\n[tool_calls] " + json.dumps(
                    tool_calls, ensure_ascii=False
                )
            except (TypeError, ValueError):
                content_text += "\n[tool_calls] " + str(tool_calls)

        # 截断到 200 字符
        if len(content_text) > 200:
            content_text = content_text[:200]

        lines.append(f"[{role}] {content_text}\n")

    return "".join(lines)


def _call_llm(llm: Any, prompt: str) -> Optional[str]:
    """尝试多种调用约定获取 LLM 文本响应。

    CoreCoder LLM 约定：``llm.chat(messages=[{role, content}])`` 返回
    ``LLMResponse``（含 ``.content`` 字段）。依次尝试：

    1. CoreCoder 约定（messages 列表 + .content 取值）
    2. llm.complete(prompt)  （其他 LLM 实现的字符串约定）
    3. llm.chat(prompt)       （字符串入参兼容）
    4. llm(prompt)            （可调用对象约定）

    任意一种成功且返回非空文本即返回；全部失败返回 None。
    """
    # 1. CoreCoder 约定：chat(messages=[...]) → LLMResponse.content
    try:
        result = llm.chat(messages=[{"role": "user", "content": prompt}])
        if result is not None:
            content = getattr(result, "content", result)
            text = content if isinstance(content, str) else str(content)
            if text.strip():
                return text
    except Exception:
        pass

    # 2. llm.complete(prompt)
    try:
        result = llm.complete(prompt)
        if result is not None:
            text = result if isinstance(result, str) else str(result)
            if text.strip():
                return text
    except Exception:
        pass

    # 3. llm.chat(prompt)
    try:
        result = llm.chat(prompt)
        if result is not None:
            text = result if isinstance(result, str) else str(result)
            if text.strip():
                return text
    except Exception:
        pass

    # 4. llm(prompt)
    try:
        result = llm(prompt)
        if result is not None:
            text = result if isinstance(result, str) else str(result)
            if text.strip():
                return text
    except Exception:
        pass

    return None


def build_operation_summary(messages: List[Dict[str, Any]], max_items: int = 12) -> str:
    """从消息历史中提取「已完成操作摘要」，供压缩后注入上下文。

    扫描 assistant 消息的 tool_calls 与对应的 tool 结果，保留已成功执行
    的操作（名称 + 关键参数）。返回形如：

        ## 已完成操作（压缩前）
        - manual.add_category(category_name=巡检缺陷, color=#FF0000)
        - manual.save_current()

    压缩后注入该摘要，可防止 Agent 因上下文丢失而重复执行已完成步骤。
    无成功操作时返回空字符串。
    """
    tool_calls = []
    pending = {}
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {})
                    idx = len(tool_calls)
                    tool_calls.append({
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "") or "",
                        "result": "",
                        "ok": None,
                    })
                    pending[tc.get("id")] = idx
        elif role == "tool":
            tid = m.get("tool_call_id")
            if tid in pending:
                idx = pending[tid]
                result = m.get("content") or ""
                tool_calls[idx]["result"] = result
                tool_calls[idx]["ok"] = not (result.lstrip().startswith("❌") or result.lstrip().startswith("Error"))

    done = [tc for tc in tool_calls if tc.get("ok") is True]
    if not done:
        return ""
    lines = ["## 已完成操作（压缩前）"]
    for tc in done[-max_items:]:
        args = tc["arguments"].replace("\n", " ")
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict) and parsed:
                args = ", ".join(f"{k}={v}" for k, v in parsed.items())
        except Exception:
            pass
        if len(args) > 120:
            args = args[:120] + "..."
        lines.append(f"- {tc['name']}({args})")
    return "\n".join(lines)


def compress_messages(
    messages: List[Dict[str, Any]],
    keep_recent_n: int = 10,
    llm: Any = None,
) -> List[Dict[str, Any]]:
    """压缩消息历史。

    - 保留后 ``keep_recent_n`` 条完整消息（滑动窗口）
    - 窗口外的旧消息：有 ``llm`` 时生成摘要并以 system 消息注入；
      无 ``llm`` 或调用失败时回退为简单截断（content 前 200 字符 + ``[已截断]``）
    - 不修改传入的 messages 列表与其中 dict
    """
    if not messages:
        return messages

    if len(messages) <= keep_recent_n:
        return messages

    split_idx = len(messages) - keep_recent_n
    old_messages = messages[:split_idx]
    recent_messages = messages[split_idx:]
    op_summary = build_operation_summary(old_messages)

    # 尝试 LLM 摘要
    if llm is not None:
        prompt = (
            "请总结以下对话，保留关键决策、工具调用记录、用户核心需求、重要结果，"
            "输出不超过 500 字：\n\n"
            + _format_messages_for_summary(old_messages)
        )
        summary = _call_llm(llm, prompt)
        if summary:
            summary_message = {
                "role": "system",
                "content": f"以下是之前对话的摘要：\n{summary}",
            }
            result = [summary_message] + list(recent_messages)
            if op_summary:
                result.insert(1, {"role": "system", "content": op_summary})
            return result

    # 回退：简单截断 old_messages
    truncated_old: List[Dict[str, Any]] = []
    # 图片描述映射基于完整 messages 构建，其后的 assistant 解读可能是
    # recent 窗口内的消息，同样能关联到 old 里的图片。
    fallback_caption_map = _image_description_map(messages)
    for idx, msg in enumerate(old_messages):
        new_msg = dict(msg)  # 浅拷贝，避免修改原 dict
        content = new_msg.get("content")
        if isinstance(content, str):
            if len(content) > 200:
                new_msg["content"] = content[:200] + "[已截断]"
        elif content is not None and isinstance(content, list):
            # 非字符串 content（多模态列表）：text 保留、图片转 [图片: 描述]，
            # 避免 dump base64。描述取自 LLM 首次读到该图的 assistant 解读。
            summary = _format_modal_content(
                content, max_chars=200,
                image_caption=fallback_caption_map.get(idx),
            )
            if len(summary) > 200:
                summary = summary[:200] + "[已截断]"
            new_msg["content"] = summary
            truncated_old.append(new_msg)
            continue
        # content 为 None 或未超长时保持原样
        truncated_old.append(new_msg)

    result = truncated_old + list(recent_messages)
    if op_summary:
        result.insert(0, {"role": "system", "content": op_summary})
    return result


def truncate_tool_results(
    messages: List[Dict[str, Any]],
    max_chars: int = 200,
    keep_recent_n: int = 10,
) -> List[Dict[str, Any]]:
    """截断早期 tool 消息的 content。

    扫描最近 ``keep_recent_n`` 条窗口之前的消息，对 ``role == "tool"`` 且
    content 超过 ``max_chars`` 的消息，截断为前 ``max_chars`` 字符 +
    ``\\n[已截断，完整结果见之前对话]``。返回新列表，不修改原列表中的 dict。
    """
    if not messages:
        return messages

    new_messages: List[Dict[str, Any]] = []
    threshold = max(0, len(messages) - keep_recent_n)

    for i, msg in enumerate(messages):
        if (
            i < threshold
            and isinstance(msg, dict)
            and msg.get("role") == "tool"
        ):
            new_msg = dict(msg)  # 浅拷贝
            content = new_msg.get("content")
            if isinstance(content, str) and len(content) > max_chars:
                new_msg["content"] = (
                    content[:max_chars] + "\n[已截断，完整结果见之前对话]"
                )
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)

    return new_messages


def emergency_truncate(
    messages: List[Dict[str, Any]],
    keep_recent_n: int = 5,
) -> List[Dict[str, Any]]:
    """紧急截断：保留最后 ``keep_recent_n`` 条，之前合并为一条 system 摘要。

    每条旧消息提取 role 与 content 前 100 字符，合并后注入为
    ``{"role": "system", "content": "以下是之前对话的摘要（紧急截断）：\\n..."}``。
    不修改传入的 messages 列表与其中 dict。
    """
    if not messages:
        return messages

    if len(messages) <= keep_recent_n:
        return messages

    split_idx = len(messages) - keep_recent_n
    old_messages = messages[:split_idx]
    recent_messages = messages[split_idx:]

    # 合并旧消息为摘要文本
    parts: List[str] = []
    for msg in old_messages:
        if not isinstance(msg, dict):
            parts.append(f"[unknown] {str(msg)[:100]}")
            continue

        role = msg.get("role", "unknown")
        content = msg.get("content")

        if content is None:
            content_text = ""
        elif isinstance(content, str):
            content_text = content
        else:
            try:
                content_text = json.dumps(content, ensure_ascii=False)
            except (TypeError, ValueError):
                content_text = str(content)

        if len(content_text) > 100:
            content_text = content_text[:100]

        parts.append(f"[{role}] {content_text}")

    merged_content = "\n".join(parts)
    summary_message = {
        "role": "system",
        "content": "以下是之前对话的摘要（紧急截断）：\n" + merged_content,
    }

    result = [summary_message] + list(recent_messages)
    op_summary = build_operation_summary(old_messages)
    if op_summary:
        result.insert(1, {"role": "system", "content": op_summary})
    return result
