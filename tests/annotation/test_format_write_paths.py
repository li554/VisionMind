"""非项目模式下各标注格式必须真的能写入（D35/D36）—— 回归。

被测缺陷：
  D35  `save_image_annotations` 的兜底分支把输出路径**硬编码为 `<base>.json`**，
       而 `MaskFormat.save()` 写的是**图像**、扩展名直接取自 output_path 交给 OpenCV：
           imwrite_unicode(<...>.json.atomic-1-2>, ext='.json')
           -> OpenCV: could not find encoder for the specified extension
       于是非项目模式下 mask/sa1b 保存 **100% 失败**，异常还被本函数的 except
       吞成 {"status": "error"}（阶段 9 加的失败上报才让它显形）。
  D36  目录里没有任何标注文件时 `_non_project_format` 会被解析成 `mask`，
       与 D35 组合 = "新目录第一次保存必然失败"。

  另外：掩码若按 `<dir>/<base>.png` 落盘会与**源图同名同目录**，直接覆盖别人的图。
  因此必须落到 `masks/` 子目录（与 `MaskFormat.scan_categories` 的约定一致）。

断言：
  W1 每个格式的 (目录, 文件名) 解析正确（mask -> masks/<base>.png；sa1b -> <base>.json）
  W2 mask 端到端写入成功，且落盘是**图像**、不是 .json；源图未被覆盖
  W3 sa1b 端到端写入成功
  W4 labelme/coco/yolo 等原有格式不受影响（回归）
  W5 新目录首次保存成功（D36 + D35 组合场景）
  W6 落盘结果能被**读回**（写-读闭环，避免只测"没报错"）

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_format_write_paths.py
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


def make_image(path, w=96, h=64):
    from PySide6.QtGui import QColor, QImage

    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(60, 110, 160))
    img.save(path)


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

    root = tempfile.mkdtemp(prefix="visionmind_fmt_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    img_path = os.path.join(d, "img00.png")
    make_image(img_path)
    src_size_before = os.path.getsize(img_path)

    w = AnnotationInterface(None, None)
    w.resize(900, 600)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()

    def pump(seconds=1.0, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    manual = w.manual
    manual._non_project_format = "labelme"
    w.load_image(0, force_reload=True)
    pump(4.0, until=lambda: manual.session.is_annotations_loaded
         and manual.session.is_committed)
    manual.wait_for_pending_writes(img_path, 5000)

    anns = [
        {"label": "cat", "bbox": [10, 8, 40, 30],
         "polygons": [[[10, 8], [50, 8], [50, 38], [10, 38]]],
         "shape_type": "polygon"},
        {"label": "dog", "bbox": [55, 20, 20, 20], "polygons": [], "shape_type": "rect"},
    ]

    print("== 标注格式写入路径回归（D35 / D36）==")

    # ---------------- W1: 路径解析 ----------------
    expect = {
        "mask": (os.path.join(d, "masks"), "img00.png"),
        "sa1b": (d, "img00.json"),
        "labelme": (d, "img00.json"),
    }
    for fmt, (want_dir, want_name) in expect.items():
        got_dir, got_name = manual._resolve_annotation_output(fmt, d, "img00")
        ok = (os.path.normcase(got_dir) == os.path.normcase(want_dir)
              and got_name == want_name)
        print("   W1 %-8s -> %s%s  %s" % (fmt, os.path.basename(got_dir) or ".",
                                          os.sep + got_name, "OK" if ok else "错"))
        check(ok, "W1 %s 解析错误: %s / %s（期望 %s / %s）"
              % (fmt, got_dir, got_name, want_dir, want_name))

    # ---------------- W2: mask 端到端 ----------------
    res = manual.save_image_annotations(img_path, anns, fmt="mask",
                                        output_dir=d, categories={"cat": 0, "dog": 1},
                                        project_mode=False)
    mask_path = os.path.join(d, "masks", "img00.png")
    print("   W2 mask 保存 status=%s file=%s 存在=%s"
          % (res.get("status"), res.get("file_path"), os.path.exists(mask_path)))
    check(res.get("status") == "success",
          "W2 mask 保存失败（D35 未修复）: %s" % res.get("message"))
    check(os.path.exists(mask_path), "W2 掩码没有落到 masks/img00.png")
    check(not os.path.exists(os.path.join(d, "img00.json")),
          "W2 掩码被误写成 .json")
    check(os.path.getsize(img_path) == src_size_before,
          "W2 源图被掩码写入覆盖了（同名同目录）")
    if os.path.exists(mask_path):
        from PySide6.QtGui import QImage
        mi = QImage(mask_path)
        print("     掩码是图像: %s (%dx%d)" % (not mi.isNull(), mi.width(), mi.height()))
        check(not mi.isNull(), "W2 落盘文件不是可读图像")
        check((mi.width(), mi.height()) == (96, 64), "W2 掩码尺寸不对")

    # ---------------- W3: sa1b 端到端 ----------------
    sa1b_path = os.path.join(d, "img00.json")
    if os.path.exists(sa1b_path):
        os.remove(sa1b_path)
    res_s = manual.save_image_annotations(img_path, anns, fmt="sa1b",
                                          output_dir=d, categories={"cat": 0},
                                          project_mode=False)
    ok_s = res_s.get("status") == "success" and os.path.exists(sa1b_path)
    print("   W3 sa1b 保存 status=%s 文件存在=%s" % (res_s.get("status"), os.path.exists(sa1b_path)))
    check(ok_s, "W3 sa1b 保存失败: %s" % res_s.get("message"))

    # ---------------- W4: 原有格式回归 ----------------
    if os.path.exists(sa1b_path):
        os.remove(sa1b_path)
    res_l = manual.save_image_annotations(img_path, anns, fmt="labelme",
                                          output_dir=d, categories={"cat": 0},
                                          project_mode=False)
    labelme_ok = res_l.get("status") == "success" and os.path.exists(sa1b_path)
    print("   W4 labelme 保存 status=%s 文件存在=%s"
          % (res_l.get("status"), os.path.exists(sa1b_path)))
    check(labelme_ok, "W4 labelme 保存回归失败: %s" % res_l.get("message"))
    if os.path.exists(sa1b_path):
        with open(sa1b_path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("annotations") or data.get("shapes") or []
        check(len(raw) == 2, "W4 labelme 落盘标注数不对: %d" % len(raw))

    # ---------------- W6: 掩码可读回（写-读闭环）----------------
    # MaskFormat 的语义是"像素值 = class_id"，其中 0 表示背景（无标注）。
    # 因此读回测试必须用**非 0** 类别编号，否则写出来的就是一张纯背景掩码
    # （这不是缺陷，是格式本身的规定）。
    if os.path.exists(os.path.join(d, "masks", "img00.png")):
        os.remove(os.path.join(d, "masks", "img00.png"))
    manual.save_image_annotations(img_path, anns, fmt="mask", output_dir=d,
                                  categories={"cat": 1, "dog": 2}, project_mode=False)
    read_back = manual.load_image_annotations(img_path, output_dir=d, fmt="mask")
    n_read = len(read_back.get("annotations") or [])
    labels_read = sorted({a.get("label") for a in (read_back.get("annotations") or [])})
    print("   W6 掩码读回标注数=%d 标签=%s（status=%s）"
          % (n_read, labels_read, read_back.get("status")))
    check(n_read >= 1, "W6 写入的掩码无法读回（写-读闭环失败）")

    # ---------------- W5: 新目录首次保存（D36 组合场景）----------------
    new_dir = os.path.join(root, "fresh")
    os.makedirs(new_dir)
    fresh_img = os.path.join(new_dir, "a.png")
    make_image(fresh_img)
    # 模拟 D36：目录里没有任何标注文件 -> 自动识别为 mask
    detected = manual.detect_and_set_export_format(new_dir)
    w._non_project_format = "mask"
    res_f = manual.save_image_annotations(fresh_img, anns, fmt="mask",
                                          output_dir=new_dir,
                                          categories={"cat": 0}, project_mode=False)
    fresh_mask = os.path.join(new_dir, "masks", "a.png")
    print("   W5 新目录首次 mask 保存 status=%s 存在=%s（D36 组合场景）"
          % (res_f.get("status"), os.path.exists(fresh_mask)))
    check(res_f.get("status") == "success",
          "W5 新目录首次保存仍失败（D35+D36 未修复）: %s" % res_f.get("message"))

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: W1~W6 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
