"""配置 schema 韧性回归 —— core.json 丢元数据不得让程序启动即崩。

被测缺陷（用户现场）：`core/core.json` 是**受版本控制**的配置 schema（每项含
`key`/`title`/`type`/`value`…）。一旦该文件的 schema 元数据被外部改写丢掉，
`settings.save()` 会把动态键写成 `{"key":..., "value":...}`（没有 title/type），
而 `ConfigInterface._add_schema_section()` 直接取 `item['title']`
-> 启动时 `KeyError: 'title'`，程序在构造系统配置界面时崩掉，之后**每次启动都崩**
（每次启动又会再追加一个无 title 的动态键，形成死亡螺旋）。

断言：
  S1 动态键落盘必须自带 title/type（无 title 的条目是启动崩溃的直接原因）
  S2 已有条目的 schema 元数据在 set()/save() 后必须原样保留（只更新 value）
  S3 真实复现启动路径：schema 全部缺 title 时 ConfigInterface 仍能构造（降级显示）
  S4 ShortcutManager 经 settings 持久化：schema 元数据保留 + 自定义键位可读回
  S5 静态：shortcut_manager.py 不得再直接 open(core.json,'w') 整文件重写
  S6 真实 core/core.json 不得被本测试改动（状态隔离）

退出码即断言：0 通过 / 1 断言失败 / 其他异常退出。
用法: python -u tests/annotation/test_settings_schema_resilience.py
"""
import json
import os
import shutil
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import isolate_state  # noqa: E402

_REAL_CORE = os.path.join(_ROOT, "core", "core.json")
_SHORTCUT_SRC = os.path.join(_ROOT, "core", "common", "shortcut_manager.py")


def code_only(text):
    """去掉注释与字符串字面量的粗略版本（够用于静态检查）。"""
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        out.append(line)
    return "\n".join(out)


def read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_schema_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    real_mtime_before = os.path.getmtime(_REAL_CORE)

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    print("== 配置 schema 韧性回归 ==")

    # ---------------- S1: 动态键必须自带 title/type ----------------
    print("-- S1 动态键落盘带 title/type --")
    settings = None
    cs = None
    from core.common.settings import settings as _settings, CoreSettingsManager
    settings = _settings
    cs = CoreSettingsManager
    settings.set("brand_new_dynamic_key", "hello")
    settings.set("brand_new_dynamic_mapping", {"a": 1})
    items = {it.get("key"): it for it in settings.get_schema()}
    dyn = items.get("brand_new_dynamic_key")
    dyn_map = items.get("brand_new_dynamic_mapping")
    print("   动态标量: %s" % json.dumps(dyn, ensure_ascii=False))
    print("   动态映射: %s" % json.dumps(dyn_map, ensure_ascii=False))
    check(dyn and dyn.get("title"), "S1 动态标量键缺少 title（会导致启动 KeyError）")
    check(dyn and dyn.get("type"), "S1 动态标量键缺少 type")
    check(dyn_map and dyn_map.get("title"), "S1 动态映射键缺少 title")
    check(dyn_map and dyn_map.get("type") == "hidden",
          "S1 dict/list 动态键应为 hidden（否则渲染成错误控件）")
    on_disk = read_json(settings.filepath)
    check(all(isinstance(it, dict) and it.get("title") for it in on_disk),
          "S1 落盘文件里存在无 title 的条目")

    # ---------------- S2: 已有 schema 元数据不得被覆盖 ----------------
    print("-- S2 set()/save() 保留已有 schema 元数据 --")
    settings._schema_data = [
        {"key": "theme", "title": "主题模式", "type": "combo",
         "options": ["dark", "light"], "value": "dark",
         "description": "界面明暗主题"},
    ]
    settings._data = {"theme": "dark"}
    settings.save()
    settings.set("theme", "light")
    item = read_json(settings.filepath)[0]
    print("   theme 条目: %s" % json.dumps(item, ensure_ascii=False))
    check(item.get("title") == "主题模式", "S2 title 被覆盖/丢失")
    check(item.get("type") == "combo", "S2 type 被覆盖/丢失")
    check(item.get("options") == ["dark", "light"], "S2 options 被覆盖/丢失")
    check(item.get("description") == "界面明暗主题", "S2 description 被覆盖/丢失")
    check(item.get("value") == "light", "S2 value 未被更新")

    # ---------------- S3: 降级 schema 下 ConfigInterface 必须能构造 ----------------
    print("-- S3 真实启动路径：schema 全缺 title 时构造 ConfigInterface --")
    degraded = [
        {"key": "theme", "value": "dark"},                 # 无 title/type
        {"key": "outputs_root", "value": os.path.join(state_dir, "projects")},
    ]
    with open(settings.filepath, "w", encoding="utf-8") as f:
        json.dump(degraded, f, indent=2, ensure_ascii=False)
    settings._schema_data = []
    settings._data = {}
    settings.load()
    check(len(settings.get_schema()) == 2, "S3 前置条件：降级 schema 未加载")

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    built = None
    try:
        from core.view.config_interface import ConfigInterface
        built = ConfigInterface()
    except KeyError as exc:
        failures.append("S3 降级 schema 下 ConfigInterface 启动即崩: KeyError %s" % exc)
    except Exception as exc:
        failures.append("S3 降级 schema 下 ConfigInterface 构造异常: %r" % exc)
    check(built is not None, "S3 ConfigInterface 未能构造")
    if built is not None:
        print("   ConfigInterface 构造成功，页面数=%d" % len(getattr(built, "_pages", {})))
        check("theme" in getattr(built, "_all_widgets", {}),
              "S3 缺 title 的 theme 项未降级渲染为控件")

    # ---------------- S4: ShortcutManager 经 settings 持久化 ----------------
    print("-- S4 ShortcutManager 持久化不破坏 schema --")
    settings._schema_data = [
        {"key": "shortcuts", "title": "快捷键配置", "type": "object",
         "default": {}, "value": {}, "description": "自定义快捷键映射"},
    ]
    settings._data = {"shortcuts": {}}
    settings.save()
    from core.common.shortcut_manager import ShortcutManager
    ShortcutManager.reset()
    sm = ShortcutManager.instance()
    sm.set_key("example_annotate", "Shift+G")
    time.sleep(0.05)
    data = read_json(settings.filepath)
    sc = [it for it in data if it.get("key") == "shortcuts"]
    print("   shortcuts 条目: %s" % json.dumps(sc, ensure_ascii=False))
    check(len(sc) == 1, "S4 shortcuts 条目丢失或重复")
    if sc:
        check(sc[0].get("title") == "快捷键配置", "S4 shortcuts 条目 schema 元数据被破坏")
        check(sc[0].get("type") == "object", "S4 shortcuts 条目 type 被破坏")
        check(sc[0].get("value", {}).get("example_annotate") == "Shift+G",
              "S4 自定义快捷键未写入配置")
    ShortcutManager.reset()
    sm2 = ShortcutManager.instance()
    check(sm2.get_key("example_annotate") == "Shift+G", "S4 重新加载后自定义快捷键丢失")

    # ---------------- S5: 静态不变量（不得回到整文件重写） ----------------
    print("-- S5 静态：shortcut_manager 不再直接整文件重写 core.json --")
    with open(_SHORTCUT_SRC, encoding="utf-8", errors="replace") as f:
        src = code_only(f.read())
    check("_config_path" not in src, "S5 shortcut_manager 仍持有 core.json 直接写路径")
    check("json.dump" not in src, "S5 shortcut_manager 仍直接整文件重写配置")
    check("settings" in src, "S5 shortcut_manager 未改走 settings 持久化")

    # ---------------- S6: 真实 core.json 未被改动 ----------------
    real_mtime_after = os.path.getmtime(_REAL_CORE)
    print("-- S6 真实 core/core.json mtime 未变 --")
    check(real_mtime_before == real_mtime_after,
          "S6 测试改动了用户的 core/core.json（状态隔离失效）")

    if failures:
        print("   FAIL:")
        for msg in failures[:12]:
            print("     - %s" % msg)
        code = 1
    else:
        print("   PASS: S1~S6 全部通过")
        code = 0
    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
