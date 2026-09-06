"""ask_user 工具 — 向用户提问并阻塞等待回答(对齐 DSH tool-ask-user 语义)

问题以卡片形式展示在对话侧栏;用户点选选项或输入自定义回答后,
回答作为普通 tool result 回灌 agent loop。用户停止执行时返回未回答。
"""

import json

from core.corecoder.tools.base import Tool


class AskUserTool(Tool):
    read_only = True
    name = "ask_user"
    description = (
        "向用户提问并暂停等待回答(用户在对话侧栏的提问卡上作答后才继续)。"
        "适用场景:需要用户确认或选择时(如自动标注前置资源缺失让用户选方式)、"
        "任务有歧义需要澄清时、执行写入类操作前需要确认时。"
        "问题必须简明;提供选项时把推荐项放第一个并在标签后加「(推荐)」。"
        "一次最多问 3 个问题,严禁用于闲聊。"
        "返回 JSON:{\"answers\":[{\"id\",\"selected\":[选项],\"custom\":自定义回答}]}。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": "要问的问题列表(通常 1 个,最多 3 个)",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "问题稳定 id(回答中回显),如 'mode'",
                        },
                        "question": {"type": "string", "description": "问题内容"},
                        "header": {
                            "type": "string",
                            "description": "可选短标题,如「确认」「选择标注方式」",
                        },
                        "options": {
                            "type": "array",
                            "description": "可选选项列表;推荐项放第一位并加「(推荐)」",
                            "items": {"type": "string"},
                        },
                        "multi_select": {
                            "type": "boolean",
                            "description": "是否允许多选,默认 false",
                        },
                    },
                    "required": ["id", "question"],
                },
            },
        },
        "required": ["questions"],
    }

    def __init__(self, requester):
        # requester(qlist) -> answers list | None(用户未回答/已停止)
        self._requester = requester

    def execute(self, questions=None, **_) -> str:
        qlist = questions if isinstance(questions, list) else []
        if isinstance(questions, dict):
            qlist = [questions]
        qlist = [
            q for q in qlist
            if isinstance(q, dict) and (q.get("question") or q.get("id"))
        ]
        if not qlist:
            return "错误: questions 不能为空,至少提供一个含 question 的问题"
        qlist = qlist[:3]
        answers = self._requester(qlist)
        if answers is None:
            return (
                "用户未回答(执行已停止或请求已取消)。"
                "请基于现有信息继续,或向用户说明无法继续的原因。"
            )
        return json.dumps({"answers": answers}, ensure_ascii=False)
