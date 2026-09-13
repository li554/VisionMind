"""标注界面回归测试总入口 — 一次跑完并汇总退出码。

为什么要这个入口：本项目的被测缺陷里有 `qFatal -> abort()`，**测试无法自报失败**，
所以约定「退出码即断言」。这个脚本把每个测试作为子进程运行（输出重定向到文件，
不走管道），逐个判读退出码：

    0           = 通过
    1           = 断言失败
    -1073740791 = 0xC0000409，进程被 abort
    其他非零     = 异常退出

同时校验 `core/core.json` 的 mtime 未被改动 —— 测试必须通过 `isolate_state()`
把运行时状态重定向到临时目录，污染用户数据即视为失败。

用法: python -u tests/run_all.py
"""
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_ROOT, "logs")

# (测试文件, 参数, 说明)
TESTS = [
    ("tests/annotation/test_switch_stress.py", ["40", "20"],
     "切图不 abort / 并发有界 / 状态一致 / 最终图不丢 / 可排空"),
    ("tests/annotation/test_switch_stress.py", ["80", "5"],
     "同上，更极端强度（5ms 间隔）"),
    ("tests/annotation/test_label_list_leak.py", ["30", "20"],
     "列表重建不泄漏子控件"),
    ("tests/annotation/test_shutdown.py", ["5", "120"],
     "关闭排空 + 排空期间不死锁 + 幂等"),
    ("tests/annotation/test_thumbnail_pending.py", [],
     "缩略图在途集合必定收敛、失败路径可重试"),
    ("tests/annotation/test_atomic_write.py", [],
     "原子写：序列化中途崩溃不得破坏已有标注文件"),
    ("tests/annotation/test_delete_image.py", [],
     "删图：专属标注随删、共享文件保留、解析失败不半删、确认门"),
    ("tests/annotation/test_no_processevents.py", [],
     "无 processEvents 重入 + 批量期间 GUI 线程空闲 + 取消/并发"),
    ("tests/annotation/test_single_state_owner.py", ["6"],
     "唯一状态所有者 + 原子提交（只读真值 / 窗口内不提交 / 过期提交被拒）"),
    ("tests/annotation/test_annotation_io_async.py", [],
     "标注装载后台 I/O + 代次校验 + 最新优先 + 去重复刷新"),
    ("tests/annotation/test_enhance_async.py", [],
     "增强图后台生成 + 结果校验 + 代次守卫（D17）"),
    ("tests/annotation/test_model_load_thread.py", [],
     "模型加载不再用 QThread 子类 + 拒绝并发 + 代次跟踪（D12）"),
    ("tests/annotation/test_save_async.py", [],
     "标注后台写入 + 同路径最新优先 + 读等待写 + 失败上报"),
    ("tests/annotation/test_search_async.py", [],
     "搜索去抖 + 后台扫描 + 代次校验（D15）"),
    ("tests/annotation/test_cross_image_writes.py", [],
     "跨图写入路径校验（D2 批量写回 / D3 跨图撤销）"),
    ("tests/annotation/test_non_project_dirs.py", [],
     "非项目模式目录隔离（D4）+ AI 全量转换可用（D9）"),
    ("tests/annotation/test_stop_and_wait.py", [],
     "stop_and_wait/run_on_main 互等 + 放弃后不留残任务（D14）"),
    ("tests/annotation/test_settings_session_state.py", [],
     "运行时状态不落盘受版本控制的配置 + 保存原子化（D30）"),
    ("tests/annotation/test_settings_schema_resilience.py", [],
     "core.json 丢 schema 元数据不得让设置界面启动即崩"),
    ("tests/annotation/test_refresh_signature.py", [],
     "刷新去重签名覆盖几何 + 真值替换时缓存作废（D33）"),
    ("tests/annotation/test_undo_visibility.py", [],
     "撤销/重做与可见性同帧回滚 + 空列表也能写回（D21）"),
    ("tests/annotation/test_format_write_paths.py", [],
     "各标注格式写入路径正确 + 写读闭环（D35/D36）"),
    ("tests/annotation/test_batch_cancel.py", [],
     "AI 全量转换取消真正生效 + 已落盘数据完整（D9 余项）"),
    ("tests/annotation/test_navigation_single_source.py", [],
     "导航可见集唯一真值来源，筛选后不跳到隐藏图（阶段 3e）"),
    ("tests/annotation/test_canvas_invariants.py", [],
     "损坏 ROI / 畸形标注不得让画布停止重绘（D18）"),
    ("tests/annotation/test_list_rebuild_safety.py", [],
     "列表重建屏蔽信号 + 发射栈内延后重建（D16）"),
    ("tests/annotation/test_edit_state_owner.py", [],
     "编辑态（撤销栈/可见性/副标注）并入唯一状态所有者"),
    ("tests/annotation/test_long_switch_soak.py", [],
     "长跑+连发切图：无资源增长、无冻结、无 abort（原始症状形态）"),
]

_ABORT = -1073740791  # 0xC0000409


def run_one(python, script, args):
    name = os.path.basename(script).replace(".py", "") + ("_" + "_".join(args) if args else "")
    log_path = os.path.join(_LOG_DIR, "run_%s.txt" % name)
    os.makedirs(_LOG_DIR, exist_ok=True)
    cmd = [python, "-u", os.path.join(_ROOT, script)] + args
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", errors="replace") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=_ROOT)
    elapsed = time.perf_counter() - t0
    return name, proc.returncode, elapsed, log_path


def main():
    python = sys.executable
    settings_path = os.path.join(_ROOT, "core", "core.json")
    before = os.path.getmtime(settings_path) if os.path.exists(settings_path) else None

    print("=== 标注界面回归套件 ===")
    print("   python = %s" % python)
    violations = []
    results = []
    for script, args, desc in TESTS:
        name, code, elapsed, log_path = run_one(python, script, args)
        if code == 0:
            verdict = "PASS"
        elif code == _ABORT:
            verdict = "ABORT(qFatal)"
        elif code == 1:
            verdict = "FAIL(断言)"
        else:
            verdict = "ERROR(%d)" % code
        results.append((name, verdict, elapsed, log_path, desc))
        print("  %-34s %-14s %6.1fs  %s" % (name, verdict, elapsed, desc))
        if code != 0:
            violations.append(name)

    after = os.path.getmtime(settings_path) if os.path.exists(settings_path) else None
    if before != after:
        print("\n  !! core/core.json 被测试改动（状态隔离失效）")
        violations.append("state-isolation")

    print("\n=== 汇总 ===")
    for name, verdict, elapsed, log_path, _desc in results:
        print("  %-34s %s" % (name, verdict))
    print("  日志目录: %s" % _LOG_DIR)

    if violations:
        print("\n  未通过: %s" % ", ".join(violations))
        print("  退出码 1")
        return 1
    print("\n  全部通过；退出码 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
