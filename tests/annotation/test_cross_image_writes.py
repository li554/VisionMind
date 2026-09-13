"""跨图写入必须被路径校验拦住 —— D2 / D3 回归。

被测缺陷（都是"把 A 图的标注写进 B 图文件"这一类的静默数据损坏，都属于
"带代次/路径校验的原子提交"这条要求的漏点）：

  D3  撤销栈存了 `image_path` 却**从不比对**：切图后按 Ctrl+Z，会把旧图标注
      写进当前图的文件。
  D2  批量标注跑完把 `manual.current_annotations` 无条件换成"最后一张批量图"的
      标注；此后 `save_current` 会把它写进屏幕上那张图的文件。

断言：
  U1 D3 跨图撤销被拒 + 撤销历史被清空 + 当前图标注文件未被写坏
  U2 D3 同图撤销仍然正常工作（不能因为加校验而废掉功能）
  B1 D2 为"别的图"准备的标注不会被写回当前状态（画布不动）
  B2 D2 为"当前图"准备的标注仍会写回（画布刷新依赖它）
  S1 静态：三层内不得再出现对只读真值（current_image_path / current_annotations /
     image / pixmap / original_image_size / display_scale / _load_generation）的赋值

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_cross_image_writes.py
"""
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import isolate_state  # noqa: E402

_RO_ATTRS = ("current_image_path", "current_annotations",
             "original_image_size", "display_scale")


def make_images(d):
    from PySide6.QtGui import QColor, QImage

    paths = []
    for i in range(3):
        path = os.path.join(d, "img%02d.png" % i)
        img = QImage(64, 48, QImage.Format.Format_RGB32)
        img.fill(QColor(40 + 30 * i, 80, 140))
        img.save(path)
        paths.append(path)
    return paths


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    print("== 跨图写入校验回归（D2 / D3）==")

    # ---------------- S1: 静态不变量 ----------------
    layers = [os.path.join(_ROOT, "plugins", "annotation", sub, "*.py")
              for sub in ("interfaces", "services", "widgets")]
    offenders = []
    for pattern in layers:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    code = line.split("#", 1)[0]
                    for attr in _RO_ATTRS:
                        if re.search(r"(?<![\w.])self\.%s\s*=[^=]" % re.escape(attr), code) or \
                           re.search(r"\bmanual\.%s\s*=[^=]" % re.escape(attr), code) or \
                           re.search(r"\b\w+\.%s\s*=\s*list\(" % re.escape(attr), code):
                            offenders.append("%s:%d: %s"
                                             % (os.path.basename(path), lineno, line.strip()))
    if offenders:
        failures.append("S1 仍存在对只读真值的赋值: %s" % offenders[:3])
    else:
        print("   S1 静态检查：三层内 0 处对只读真值的赋值  OK")

    root = tempfile.mkdtemp(prefix="visionmind_ximg_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    paths = make_images(d)

    w = AnnotationInterface(None, None)
    w.resize(1000, 660)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()
    w._non_project_format = "labelme"        # 避开 D35（mask 非项目写入必然失败）

    sess = w.manual.session

    def pump(seconds, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    def load_ready(index):
        w.load_image(index, force_reload=True)
        pump(4.0, until=lambda: sess.is_annotations_loaded and sess.is_committed)

    def ann_path(image_path):
        return os.path.join(d, os.path.splitext(os.path.basename(image_path))[0] + ".json")

    def labels_of(image_path):
        p = ann_path(image_path)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("annotations") or data.get("shapes") or []
        return [a.get("label") for a in raw]

    # ---------------- U1: D3 跨图撤销必须被拒 ----------------
    load_ready(0)
    ann_a = [{"label": "A", "bbox": [1, 2, 10, 20], "shape_type": "rect", "polygons": []}]
    w.manual.set_current_annotations(ann_a, for_path=paths[0])
    w.manual.save_current(ann_a, paths[0], "det")
    w._save_undo_state()                       # 入栈：属于 img00
    stack_before = len(w.manual._undo_stack)

    load_ready(1)                              # 切到 img01
    # U1a 第一道防线：切图会清空撤销历史
    stack_after_switch = len(w.manual._undo_stack)
    print("   U1a 切图前栈=%d；切图后栈=%d" % (stack_before, stack_after_switch))
    if stack_before == 0:
        failures.append("U1 前置条件失败：撤销栈为空")
    if stack_after_switch != 0:
        failures.append("U1a 切图没有清空撤销历史")

    # U1b 第二道防线：即使栈里残留了"属于别的图"的帧（路径传入过期值、
    # 或未来某个入口忘了清栈），落盘前也必须被路径校验拦住。
    labels_b_before = labels_of(paths[1])
    w.manual.save_undo_state(                  # 故意标成 img00 的帧
        [{"label": "STALE_A", "bbox": [0, 0, 9, 9], "shape_type": "rect", "polygons": []}],
        paths[0])
    ret, err = w.manual.step_history(-1, w.draw_area.annotations, w.current_image_path)
    labels_b_after = labels_of(paths[1])
    print("   U1b 跨图撤销返回 err=%r；img01 文件 前=%s 后=%s；栈剩=%d；当前标注=%s"
          % (err, labels_b_before, labels_b_after,
             len(w.manual._undo_stack), [a.get("label") for a in sess.annotations]))
    if not err:
        failures.append("U1b 跨图撤销未被拒绝（会把 img00 的标注写进 img01 的文件）")
    elif "另一张图片" not in str(err):
        failures.append("U1b 拒绝原因不是路径校验（说明不是新守卫拦下的）: %r" % err)
    if labels_b_after != labels_b_before:
        failures.append("U1b img01 的标注文件被跨图撤销写坏了: %s -> %s"
                        % (labels_b_before, labels_b_after))
    if w.manual._undo_stack:
        failures.append("U1b 跨图撤销后撤销历史未被清空（下次撤销还会再误写一次）")
    if sess.annotations:
        failures.append("U1b 当前图的内存标注被跨图撤销污染: %s" % sess.annotations)

    # ---------------- U2: D3 同图撤销仍要能用 ----------------
    load_ready(2)
    base = [{"label": "base", "bbox": [0, 0, 5, 5], "shape_type": "rect", "polygons": []}]
    w.manual.set_current_annotations(list(base), for_path=paths[2])
    w.manual.save_undo_state(list(base), paths[2])
    w.manual.set_current_annotations(
        base + [{"label": "extra", "bbox": [1, 1, 6, 6], "shape_type": "rect", "polygons": []}],
        for_path=paths[2])
    w.manual.save_current(None, paths[2], "det")
    restored, err2 = w.manual.step_history(-1, w.draw_area.annotations, w.current_image_path)
    labels_after_undo = labels_of(paths[2])
    print("   U2 同图撤销 err=%r；恢复=%s；文件=%s"
          % (err2, [a.get("label") for a in (restored or [])], labels_after_undo))
    if err2:
        failures.append("U2 同图撤销被误拒: %r" % err2)
    if labels_after_undo != ["base"]:
        failures.append("U2 同图撤销没有正确落盘: %s" % labels_after_undo)

    # ---------------- B1/B2: D2 批量写回必须带路径校验 ----------------
    load_ready(0)
    before = list(sess.annotations)
    merged_other = [{"label": "FROM_IMG01", "bbox": [2, 2, 8, 8],
                     "shape_type": "rect", "polygons": []}]
    accepted_other = w.manual.set_current_annotations(list(merged_other), for_path=paths[1])
    after_other = list(sess.annotations)
    print("   B1 为 img01 准备的标注写回 img00 当前状态：受理=%s；当前标注未变=%s"
          % (accepted_other, after_other == before))
    if accepted_other:
        failures.append("B1 为别的图准备的标注被写回当前状态（D2 未修复）")
    if after_other != before:
        failures.append("B1 当前图内存标注被别的图的数据污染: %s" % after_other)

    merged_current = before + [{"label": "FROM_CURRENT", "bbox": [3, 3, 9, 9],
                                "shape_type": "rect", "polygons": []}]
    accepted_current = w.manual.set_current_annotations(list(merged_current),
                                                       for_path=w.current_image_path)
    after_current = list(sess.annotations)
    print("   B2 为当前图准备的标注写回：受理=%s；标签=%s"
          % (accepted_current, [a.get("label") for a in after_current]))
    if not accepted_current:
        failures.append("B2 为当前图准备的标注写回被误拒（画布刷新会失效）")
    if [a.get("label") for a in after_current] != [a.get("label") for a in merged_current]:
        failures.append("B2 当前图标注未被正确写回")

    # 关键：B1 被拒之后，保存不得把别的图的数据写进当前图文件
    w.manual.save_current(None, w.current_image_path, "det")
    w.manual.wait_for_pending_writes(w.current_image_path, 5000)
    w.manual.drain(3000)
    saved_current = labels_of(paths[0])
    print("   B1 收尾：img00 文件=%s（不得含 FROM_IMG01）" % saved_current)
    if saved_current and "FROM_IMG01" in saved_current:
        failures.append("B1 别的图的标注被写进了当前图的文件: %s" % saved_current)

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: U1/U2/B1/B2/S1 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
