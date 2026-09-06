"""
CommonService — 跨插件共享的纯业务逻辑层

封装示例选择管理、提示词管理等通用数据操作，不依赖任何 Qt UI 组件。
各插件的 Service 层可直接使用本模块提供的类。
"""

from typing import List, Dict, Any, Set


class ExampleSelectionManager:
    """通用示例选择管理器

    维护选中示例的名称列表，提供名称校验、选择状态管理等通用逻辑。
    不关心示例数据的具体存储方式（内存列表 / 文件系统），由调用方提供可用名称集合。
    """

    def __init__(self):
        self._selected_examples: List[str] = []

    # ---- 选择状态管理 ----

    def set_selected_examples(self, names: List[str], available_names: Set[str] = None) -> dict:
        """设置当前选中的示例（按名称）

        Args:
            names: 要选中的示例名称列表
            available_names: 当前可用的示例名称集合（可选，不传则直接设置）

        Returns:
            dict: {"status": "success"/"error", "message": ..., "count": ..., "selected": [...], "invalid": [...]}
        """
        if available_names is not None:
            valid, invalid = self._validate_names(names, available_names)
            if invalid:
                return {
                    "status": "error",
                    "message": f"无效的示例名称: {invalid}",
                    "valid": valid,
                    "invalid": invalid
                }
            self._selected_examples = list(valid)
        else:
            self._selected_examples = list(names)
        return {
            "status": "success",
            "message": f"已选择 {len(self._selected_examples)} 个示例",
            "count": len(self._selected_examples),
            "selected": list(self._selected_examples)
        }

    def get_selected_examples(self) -> List[str]:
        """获取当前选中的示例名称列表"""
        return list(self._selected_examples)

    def clear_selected_examples(self):
        """清空示例选择状态"""
        self._selected_examples = []

    def remove_from_selection(self, name: str):
        """从选中列表中移除指定名称"""
        self._selected_examples = [n for n in self._selected_examples if n != name]

    # ---- 名称校验 ----

    @staticmethod
    def _validate_names(names: List[str], available_names: Set[str]) -> tuple:
        """验证示例名称是否有效，返回 (valid_names, invalid_names)"""
        valid = [n for n in names if n in available_names]
        invalid = [n for n in names if n not in available_names]
        return valid, invalid

    @staticmethod
    def validate_example_names(names: List[str], available_names: Set[str]) -> tuple:
        """静态校验示例名称，返回 (valid_names, invalid_names)"""
        return ExampleSelectionManager._validate_names(names, available_names)

    # ---- 数据查询 ----

    @staticmethod
    def get_examples_by_names(names: List[str], all_items: List[dict],
                              name_key: str = 'name') -> List[dict]:
        """根据名称列表从完整数据集中取出对应的条目

        Args:
            names: 要查找的名称列表
            all_items: 完整的数据条目列表（每个条目应包含 name_key 字段）
            name_key: 条目中名称对应的字段名

        Returns:
            list: 匹配的条目列表
        """
        name_map = {item.get(name_key): item for item in all_items if item.get(name_key)}
        return [name_map[n] for n in names if n in name_map]

    @staticmethod
    def get_available_names(all_items: List[dict], name_key: str = 'name') -> Set[str]:
        """从数据条目列表中提取所有可用名称集合"""
        return {item.get(name_key) for item in all_items if item.get(name_key)}


class PromptManager:
    """通用提示词管理器

    提供提示词库的增删查改等纯数据操作。
    prompt_library 格式: [{"text": "prompt_text", "checked": True/False}, ...]
    """

    @staticmethod
    def add_prompt(prompt_library: List[dict], text: str) -> List[dict]:
        """添加提示词到库，自动去重（以逗号分隔多个提示词）"""
        if not text:
            return prompt_library
        prompts = [p.strip() for p in text.split(',') if p.strip()]
        result = list(prompt_library)
        existing_texts = {item.get('text') for item in result}
        for p in prompts:
            if p not in existing_texts:
                result.append({'text': p, 'checked': True})
                existing_texts.add(p)
        return result

    @staticmethod
    def list_prompts_text(prompt_library: List[dict]) -> str:
        """以文本形式列出提示词库"""
        if not prompt_library:
            return "提示词库为空"
        summary = f"提示词库共 {len(prompt_library)} 个:\n"
        for i, item in enumerate(prompt_library):
            checked = "✓" if item.get('checked', True) else " "
            text = item.get('text', '')
            summary += f"  [{i}] [{checked}] {text}\n"
        return summary.strip()

    @staticmethod
    def delete_prompt(prompt_library: List[dict], index: int) -> tuple:
        """删除指定索引的提示词，返回 (new_library, message)"""
        if not prompt_library or index < 0 or index >= len(prompt_library):
            return prompt_library, f"错误: 索引 {index} 无效"
        new_lib = list(prompt_library)
        removed = new_lib.pop(index)
        return new_lib, f"已删除提示词: {removed.get('text', '')}"

    @staticmethod
    def clear_prompts(prompt_library: List[dict]) -> tuple:
        """清空所有提示词，返回 (new_library, message)"""
        count = len(prompt_library)
        return [], f"已清空 {count} 个提示词"

    @staticmethod
    def get_selected_prompts(prompt_library: List[dict]) -> List[str]:
        """获取所有选中的提示词文本"""
        return [item.get('text', '') for item in prompt_library if item.get('checked', True)]

    @staticmethod
    def update_check_state(prompt_library: List[dict], index: int, checked: bool) -> List[dict]:
        """更新指定索引提示词的勾选状态"""
        if 0 <= index < len(prompt_library):
            new_lib = list(prompt_library)
            new_lib[index] = dict(new_lib[index])
            new_lib[index]['checked'] = checked
            return new_lib
        return prompt_library

    @staticmethod
    def load_prompt_library(settings_obj) -> List[dict]:
        """从 settings 对象加载提示词库"""
        try:
            saved = settings_obj.get("prompt_library", [])
            if saved:
                return list(saved)
        except Exception as e:
            print(f"[PromptManager] 加载提示词库失败: {e}")
        return []

    @staticmethod
    def save_prompt_library(prompt_library: List[dict], settings_obj):
        """保存提示词库到 settings 对象"""
        try:
            settings_obj.set("prompt_library", list(prompt_library))
        except Exception as e:
            print(f"[PromptManager] 保存提示词库失败: {e}")
