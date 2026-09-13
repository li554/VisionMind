"""非项目模式：目录串号（D4）与"AI 全量转换"不可用（D9）—— 回归。

被测缺陷：

  D4  `enter_non_project_mode` 用 `if not self.context._non_project_save_dir` 判断"是否
      首次进入"，于是**先开目录 A、再开目录 B** 时 `output_dir` 仍是 A —— B 的标注被
      写进 A 里，同名 basename 直接**覆盖别人的标注文件**。
  D9  `auto_annotation_service.batch_ai_convert` 以 `self.` 调用
      `find_and_load_annotations` / `save_image_annotations`，而这两个方法只存在于
      `ManualAnnotationService` 上 -> 第一次迭代必 AttributeError，
      "AI 全量转换" 100% 不可用。

断言：
  D4-1 依次进入 A、B：非项目保存目录 = B（不是 A）
  D4-2 在 B 里保存标注：文件出现在 B，**A 的标注文件不被触碰**
  D4-3 再回到 A：保存目录重新指向 A（目录→保存路径的映射始终是当前目录）
  D9-1 静态：auto_annotation_service 内不得再出现 self.<manual 专有方法>( 调用
  D9-2 动态：batch_ai_convert 不抛 AttributeError，能读源标注并写出转换结果

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_non_project_dirs.py
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


def make_images(d, names):
    from PySide6.QtGui import QColor, QImage

    out = []
    for i, name in enumerate(names):
        path = os.path.join(d, name)
        img = QImage(48, 36, QImage.Format.Format_RGB32)
        img.fill(QColor(60 + 20 * i, 100, 160))
        img.save(path)
        out.append(path)
    return out


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

    print("== 非项目模式目录隔离回归（D4 / D9）==")

    # ---------------- D9-1: 静态不变量 ----------------
    p = os.path.join(_ROOT, "plugins", "annotation", "services",
                     "auto_annotation_service.py")
    manual_only = ("find_and_load_annotations", "save_image_annotations")
    offenders = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            code = line.split("#", 1)[0]
            for name in manual_only:
                if re.search(r"(?<![\w.])self\.%s\s*\(" % re.escape(name), code):
                    offenders.append("%d: %s" % (lineno, line.strip()))
    if offenders:
        failures.append("D9-1 auto_annotation_service 仍在 self. 调用 manual 专有方法: %s"
                        % offenders[:3])
    else:
        print("   D9-1 静态检查：auto_annotation_service 内 0 处 self.<manual 专有方法>  OK")

    root = tempfile.mkdtemp(prefix="visionmind_np_")
    dir_a = os.path.join(root, "dataset_A")
    dir_b = os.path.join(root, "dataset_B")
    os.makedirs(dir_a)
    os.makedirs(dir_b)
    paths_a = make_images(dir_a, ["same.png"])
    paths_b = make_images(dir_b, ["same.png"])   # 故意同名：basename 冲突最容易暴露串号

    w = AnnotationInterface(None, None)
    w.resize(1000, 660)
    w.show()
    app.processEvents()

    def pump(seconds, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    def open_and_wait(path):
        w.open_directory(path)
        pump(4.0, until=lambda: w.manual.session.is_annotations_loaded
             and not w._annotation_runner.busy())
        w._non_project_format = "labelme"        # 避开 D35（mask 非项目写入必然失败）

    def ann_file(d, stem="same"):
        return os.path.join(d, stem + ".json")

    def labels(path):
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("annotations") or data.get("shapes") or []
        return [a.get("label") for a in raw]

    # ---------------- D4-1 / D4-2 ----------------
    open_and_wait(dir_a)
    ctx = w.manual.context
    save_dir_a = ctx._non_project_save_dir
    w.manual.set_current_annotations(
        [{"label": "FROM_A", "bbox": [1, 1, 10, 10], "shape_type": "rect", "polygons": []}],
        for_path=paths_a[0])
    w.manual.save_current(None, paths_a[0], "det")
    w.manual.wait_for_pending_writes(paths_a[0], 5000)

    open_and_wait(dir_b)
    save_dir_b = ctx._non_project_save_dir
    print("   D4-1 A 的保存目录=%s；进入 B 后的保存目录=%s"
          % (os.path.basename(save_dir_a or ""), os.path.basename(save_dir_b or "")))
    if not save_dir_b or os.path.normcase(os.path.abspath(save_dir_b)) != \
            os.path.normcase(os.path.abspath(dir_b)):
        failures.append("D4-1 进入 B 后保存目录仍指向 %r（应为 B）" % save_dir_b)

    labels_a_before = labels(ann_file(dir_a))
    w.manual.set_current_annotations(
        [{"label": "FROM_B", "bbox": [2, 2, 11, 11], "shape_type": "rect", "polygons": []}],
        for_path=paths_b[0])
    w.manual.save_current(None, paths_b[0], "det")
    w.manual.wait_for_pending_writes(paths_b[0], 5000)
    labels_a_after = labels(ann_file(dir_a))
    labels_b_after = labels(ann_file(dir_b))
    print("   D4-2 B 保存后：B 文件=%s；A 文件 前=%s 后=%s"
          % (labels_b_after, labels_a_before, labels_a_after))
    if labels_b_after != ["FROM_B"]:
        failures.append("D4-2 B 的标注没有写进 B: %s" % labels_b_after)
    if labels_a_after != labels_a_before:
        failures.append("D4-2 A 的标注文件被 B 的保存写坏了: %s -> %s"
                        % (labels_a_before, labels_a_after))
    if labels_a_after and "FROM_B" in labels_a_after:
        failures.append("D4-2 B 的标注被写进了 A（同名 basename 覆盖）")

    # ---------------- D4-3: 回到 A ----------------
    open_and_wait(dir_a)
    save_dir_back = ctx._non_project_save_dir
    print("   D4-3 回到 A 后的保存目录=%s" % os.path.basename(save_dir_back or ""))
    if os.path.normcase(os.path.abspath(save_dir_back or "")) != \
            os.path.normcase(os.path.abspath(dir_a)):
        failures.append("D4-3 回到 A 后保存目录没有指回 A: %r" % save_dir_back)

    # ---------------- D9-2: batch_ai_convert 真正可用 ----------------
    # 造一份 yolo 源标注，让转换有东西可读
    yolo_dir = os.path.join(root, "yolo_src")
    os.makedirs(yolo_dir, exist_ok=True)
    src_img = os.path.join(yolo_dir, "src.png")
    from PySide6.QtGui import QColor, QImage
    im = QImage(64, 48, QImage.Format.Format_RGB32)
    im.fill(QColor(70, 110, 170))
    im.save(src_img)
    with open(os.path.join(yolo_dir, "src.txt"), "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.25 0.25\n")
    with open(os.path.join(yolo_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("thing\n")

    auto = w.auto
    # (a) 委托方法必须真的能经由 manual 服务读到源标注
    loaded = auto._load_annotations_for_convert(src_img, 'det', 'yolo', yolo_dir)
    n_loaded = len(loaded.get('annotations') or [])
    print("   D9-2a 委托读路径：status=%s 读到 %d 条标注" % (loaded.get('status'), n_loaded))
    if loaded.get('status') != 'success' or n_loaded < 1:
        failures.append("D9-2a 委托读路径拿不到源标注: %s" % loaded)

    # (b) 端到端不得再崩（修复前第一次迭代就 AttributeError）
    errors_seen = []
    try:
        res = auto.batch_ai_convert([src_img], source_mode='det', target_mode='seg',
                                    output_dir=yolo_dir, categories={"thing": 0})
    except AttributeError as exc:
        res = None
        errors_seen.append(repr(exc))
    except Exception as exc:
        res = None
        errors_seen.append("%s: %s" % (type(exc).__name__, exc))
    print("   D9-2b batch_ai_convert 结果=%s；异常=%s"
          % ({k: v for k, v in (res or {}).items() if k != 'errors'} if res else None,
             errors_seen[:1]))
    if errors_seen:
        failures.append("D9-2b batch_ai_convert 抛出异常（应为可诊断的失败）: %s" % errors_seen[0])
    if res is None:
        failures.append("D9-2b batch_ai_convert 没有返回结果 dict")
    else:
        # 转换本身需要真实分割模型（本环境没有 -> 'No objects found' 属正常降级），
        # 因此这里只要求：不崩、返回结构完整、且**计数自洽**（读到了标注才会有空结果计数）
        for key in ("converted_count", "empty_results_count", "total", "errors"):
            if key not in res:
                failures.append("D9-2b 结果缺少字段 %s" % key)
        if res.get("total") != 1:
            failures.append("D9-2b total 不正确: %s" % res.get("total"))
        # 说明确实走到了"读标注 + 逐条转换"的环节（没有模型 -> 全部空结果）
        if res.get("converted_count", 0) == 0 and res.get("empty_results_count", 0) == 0:
            failures.append("D9-2b 既没有转换也没有空结果 —— 说明根本没读到源标注: %s" % res)

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: D4-1/D4-2/D4-3/D9-1/D9-2a/D9-2b 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
