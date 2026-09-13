"""运行时状态不得写进受版本控制的配置文件 —— D30 回归。

被测缺陷（审计 D30）：`core/core.json` 是**受版本控制**的配置文件，而 `settings.set()`
会整文件重写它。切图路径每次调用 `save_last_index` -> 每次切图一次落盘，且该文件永久
dirty（会话状态混进版本控制，git diff 全是噪声，还可能被误提交）。另外 `save()` 原先
是裸 `open('w')`，既不原子也不互斥 —— 而 `settings.set()` 会从工作线程被调用，并发
保存会互相截断。

断言：
  R1 会话态键（last_idx_*/last_opened_dir）写入后**存在于内存**（跨切图行为不变）
  R2 会话态键**不落盘**
  R3 普通配置键仍然正常落盘（不能因为加过滤而废掉配置持久化）
  R4 反复写会话态不会改动文件（切图热路径不再产生写盘）
  R5 历史遗留：文件里已有的会话态键在下次 save 时被清理
  R6 保存是串行化 + 原子的（并发 save 后文件仍是合法 JSON）
  R7 静态：annotation 层的 last_idx_/last_opened_dir 调用点必须走 set_session

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_settings_session_state.py
"""
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import isolate_state  # noqa: E402


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from core.common.settings import settings, CoreSettingsManager

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    fp = settings.filepath

    def read_raw():
        return open(fp, encoding="utf-8-sig").read() if os.path.exists(fp) else ""

    print("== 运行时状态隔离回归（D30）==")
    print("   配置文件: %s" % fp)

    # ---------------- R1/R2: 会话态可读但不落盘 ----------------
    settings.set_session("last_idx_demo", 7)
    settings.set_session("last_opened_dir", r"D:\some\dir")
    in_memory_idx = settings.get_session("last_idx_demo")
    in_memory_dir = settings.get_session("last_opened_dir")
    raw = read_raw()
    print("   R1 内存读回: idx=%r dir=%r" % (in_memory_idx, in_memory_dir))
    check(in_memory_idx == 7, "R1 会话态索引未在内存中生效: %r" % in_memory_idx)
    check(in_memory_dir == r"D:\some\dir", "R1 会话态目录未在内存中生效: %r" % in_memory_dir)
    leaked = [k for k in ("last_idx_demo", "last_opened_dir") if k in raw]
    print("   R2 落盘泄露的会话态键: %s（必须为空）" % leaked)
    check(not leaked, "R2 会话态键被写进了受版本控制的配置: %s" % leaked)

    # ---------------- R3: 普通键仍要落盘 ----------------
    settings.set("plain_model", "demo-model-xyz")
    raw2 = read_raw()
    print("   R3 普通键是否落盘: %s" % ("demo-model-xyz" in raw2))
    check("demo-model-xyz" in raw2, "R3 普通配置键没有被持久化（回归）")

    # ---------------- R4: 反复写会话态不改动文件 ----------------
    for i in range(20):
        settings.set_session("last_idx_demo", i)
        settings.set_session("last_opened_dir", r"D:\dir\%d" % i)
    raw3 = read_raw()
    print("   R4 20 次会话态写入后文件是否未变: %s" % (raw2 == raw3))
    check(raw2 == raw3, "R4 写会话态仍会改写配置文件（切图热路径仍每次落盘）")
    check(settings.get_session("last_idx_demo") == 19, "R4 会话态最终值不正确")

    # ---------------- R5: 历史遗留的会话态键会被清理 ----------------
    # 手工把会话态键塞回文件，模拟旧版本留下的脏数据
    data = json.loads(raw3)
    if isinstance(data, list):
        data.append({"key": "last_idx_legacy", "value": 3})
        data.append({"key": "last_opened_dir", "value": r"D:\legacy"})
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        settings.load()
        check(settings.get_session("last_idx_legacy") == 3,
              "R5 前置条件：未能读回历史遗留的会话态键")
        settings.set("plain_model", "demo-model-abc")   # 触发一次 save
        raw4 = read_raw()
        still = [k for k in ("last_idx_legacy", "last_opened_dir") if k in raw4]
        print("   R5 清理后仍残留的会话态键: %s（必须为空）" % still)
        check(not still, "R5 历史遗留的会话态键未被清理: %s" % still)
        check("demo-model-abc" in raw4, "R5 清理动作误删了普通键")
    else:
        failures.append("R5 配置文件格式不是 schema 数组，无法构造遗留场景")

    # ---------------- R6: 并发保存后文件仍是合法 JSON ----------------
    errors = []

    def writer(n):
        try:
            for i in range(10):
                settings.set("concurrent_key_%d" % n, i)
        except Exception as exc:
            errors.append("%s: %s" % (type(exc).__name__, exc))

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    raw5 = read_raw()
    parse_ok = False
    try:
        json.loads(raw5)
        parse_ok = True
    except Exception as exc:
        failures.append("R6 并发保存后配置文件不是合法 JSON: %s" % exc)
    print("   R6 并发保存：异常=%s 文件可解析=%s" % (errors[:1], parse_ok))
    check(not errors, "R6 并发保存抛异常: %s" % errors[:2])
    check(parse_ok, "R6 并发保存破坏了配置文件")

    # ---------------- R7: 静态检查调用点 ----------------
    offenders = []
    for pattern in (os.path.join(_ROOT, "plugins", "annotation", "*", "*.py"),):
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    code = line.split("#", 1)[0]
                    if re.search(r"settings\.set\(\s*f?[\"']last_idx_", code) or \
                       re.search(r"settings\.set\(\s*[\"']last_opened_dir[\"']", code):
                        offenders.append("%s:%d: %s"
                                         % (os.path.basename(path), lineno, line.strip()))
    print("   R7 直接 settings.set 写会话态的调用点: %s（必须为空）" % offenders)
    check(not offenders, "R7 仍有调用点用 settings.set 写运行时状态: %s" % offenders[:3])

    print("   会话态前缀: %s" % (CoreSettingsManager.is_session_key("last_idx_x"),
                                ))
    check(CoreSettingsManager.is_session_key("last_idx_x"), "R2 前缀判定失效")
    check(CoreSettingsManager.is_session_key("last_opened_dir"), "R2 前缀判定失效")
    check(not CoreSettingsManager.is_session_key("last_trained_model_path"),
          "R2 用户选择类键被误判为会话态（last_trained_model_path 应落盘）")
    check(not CoreSettingsManager.is_session_key("plain_model"), "R2 普通键被误判为会话态")

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: R1~R7 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
