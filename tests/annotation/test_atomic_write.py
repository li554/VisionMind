"""原子写入回归 — 崩溃不得毁掉已有标注文件。

被测缺陷（审计 D5）：`core/backend/formats/*` 与 `manual_annotation_service.py` 的
**全部**标注/划分/类别写入原本都是原地 `open(path, 'w')`，而 `open(..., 'w')` 会
**立即截断**原文件。本应用平均每天崩多次（`logs/crash_log_*.txt`），于是"崩溃瞬间
正好有一次保存在途"就是真实数据丢失：那张图的标注 JSON / 掩码 PNG 被清空或半写。

本测试用**真实的 save_annotations 路径**注入一次"序列化中途失败"，断言原文件
字节不变 —— 这是修复前必然失败、修复后必然通过的判别性断言。

退出码即断言：0 通过 / 1 断言失败 / 其他异常。
用法: python -u tests/annotation/test_atomic_write.py
"""
import glob
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.common.atomic_io import (atomic_open, atomic_write_json,  # noqa: E402
                                   atomic_write_text, atomic_writer_path, temp_path_for)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def leftovers(directory):
    return sorted(glob.glob(os.path.join(directory, "*atomic-*")))


def main():
    tmp = tempfile.mkdtemp(prefix="visionmind_atomic_")
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    print("== 原子写入回归 ==")
    print("   临时目录: %s" % tmp)

    # ---------- A1: atomic_open 块内异常 -> 原文件不变、无残留 ----------
    p1 = os.path.join(tmp, "a1.json")
    atomic_write_text(p1, '{"v": 1}')
    before = read_bytes(p1)
    try:
        with atomic_open(p1, "w", encoding="utf-8") as f:
            f.write('{"v": 2')           # 制造"半写"
            raise RuntimeError("boom")   # 模拟崩溃/异常
    except RuntimeError:
        pass
    check(read_bytes(p1) == before, "A1 atomic_open 异常后原文件被改动")
    check(not leftovers(tmp), "A1 残留临时文件: %s" % leftovers(tmp))
    print("   A1 atomic_open 块内异常: 原文件未变, 无残留  OK")

    # ---------- A2: atomic_open 正常路径 -> 内容生效 ----------
    with atomic_open(p1, "w", encoding="utf-8") as f:
        json.dump({"v": 2}, f)
    check(json.loads(read_bytes(p1).decode("utf-8"))["v"] == 2, "A2 正常写入未生效")
    print("   A2 atomic_open 正常路径: 内容生效  OK")

    # ---------- A3: atomic_writer_path 失败 -> 原文件不变 ----------
    p3 = os.path.join(tmp, "a3.xml")
    atomic_write_text(p3, "<root><old/></root>")
    before3 = read_bytes(p3)
    try:
        with atomic_writer_path(p3) as t3:
            with open(t3, "w", encoding="utf-8") as f:
                f.write("<root><new/></root>")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check(read_bytes(p3) == before3, "A3 atomic_writer_path 异常后原文件被改动")
    check(not leftovers(tmp), "A3 残留临时文件: %s" % leftovers(tmp))
    print("   A3 atomic_writer_path 失败: 原文件未变, 无残留  OK")

    # ---------- A4: 临时路径不得保留原扩展名结尾 ----------
    for target, ext in ((os.path.join(tmp, "x.json"), ".json"),
                        (os.path.join(tmp, "y.png"), ".png")):
        tp = temp_path_for(target)
        check(not tp.lower().endswith(ext),
              "A4 临时路径仍以 %s 结尾（会被按扩展名的扫描当成真实数据）: %s" % (ext, tp))
    print("   A4 临时路径不保留原扩展名结尾  OK")

    # ---------- A5: 真实格式 round-trip（原子写不得改坏 writer）----------
    ann = {
        "label": "cat",
        "bbox": [10, 20, 30, 40],
        "polygons": [[[10, 20], [40, 20], [40, 60], [10, 60]]],
        "shape_type": "polygon",
    }
    data = {
        "annotations": [dict(ann)],
        "image_info": {"filename": "img.png", "width": 100, "height": 80},
        "categories": [{"id": 0, "name": "cat"}],
    }
    cases = [
        ("labelme", "out.json", lambda p: json.load(open(p, encoding="utf-8"))),
        ("coco", "out.coco.json", lambda p: json.load(open(p, encoding="utf-8"))),
        ("voc", "out.xml", lambda p: ET.parse(p).getroot()),
        ("yolo", "out.txt", lambda p: open(p, encoding="utf-8").read()),
    ]
    from core.backend.formats import save_annotations

    for fmt, name, loader in cases:
        target = os.path.join(tmp, name)
        try:
            save_annotations(data, target, fmt)
        except Exception as exc:
            failures.append("A5 %s.save 抛异常: %r" % (fmt, exc))
            continue
        if not check(os.path.exists(target), "A5 %s 未生成 %s" % (fmt, name)):
            continue
        check(os.path.getsize(target) > 0, "A5 %s 生成空文件" % fmt)
        try:
            loader(target)
        except Exception as exc:
            failures.append("A5 %s 产物无法解析: %r" % (fmt, exc))
        print("   A5 %-8s -> %-14s %4d bytes 可解析" % (fmt, name, os.path.getsize(target)))
    check(not leftovers(tmp), "A5 残留临时文件: %s" % leftovers(tmp))

    # ---------- A6: 真实 save_annotations 中途崩溃 -> 原标注文件必须完好 ----------
    target = os.path.join(tmp, "victim.json")
    save_annotations(data, target, "labelme")
    original = read_bytes(target)
    check(len(original) > 10, "A6 初始文件异常")

    real_dump = json.dump

    def exploding_dump(obj, fp, **kwargs):
        fp.write('{"broken": ')          # 先写一半
        raise RuntimeError("simulated crash during serialization")

    json.dump = exploding_dump
    try:
        try:
            save_annotations(data, target, "labelme")
        except Exception:
            pass
    finally:
        json.dump = real_dump

    after = read_bytes(target)
    if not check(after == original,
                 "A6 序列化中途失败后原标注文件被破坏（%d -> %d 字节）"
                 % (len(original), len(after))):
        print("   原文: %r" % original[:120])
        print("   现状: %r" % after[:120])
    check(not leftovers(tmp), "A6 残留临时文件: %s" % leftovers(tmp))
    print("   A6 真实 save_annotations 中途崩溃: 原标注文件完好  OK")

    # ---------- A7: 掩码写入失败不得产出损坏文件 ----------
    try:
        import cv2
        import numpy as np
        mpath = os.path.join(tmp, "mask.png")
        atomic_write_text(mpath, "OLD")
        old_mask = read_bytes(mpath)
        from core.backend.utils import imwrite_unicode
        with atomic_writer_path(mpath) as mt:
            ok = imwrite_unicode(mt, None, ext=".png")   # 故意传 None -> 编码失败
            if not ok:
                raise IOError("写入掩码失败")
        failures.append("A7 掩码写入失败却未抛异常")
    except IOError:
        if read_bytes(mpath) == old_mask and not leftovers(tmp):
            print("   A7 掩码写入失败: 原文件未变, 无残留  OK")
        else:
            failures.append("A7 掩码写入失败后原文件被改动或残留临时文件")
    except Exception as exc:
        failures.append("A7 未预期的异常: %r" % exc)

    if failures:
        print("   FAIL:")
        for f in failures:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: A1~A7 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
