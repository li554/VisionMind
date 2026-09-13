"""删图不能删错、不能半删 — D1 / D10 回归。

被测缺陷：
  D1 `ManualAnnotationService.delete_image` 里
     `from core.backend.formats import find_label_file` **必定 ImportError**
     （该符号只从 `core.backend.utils` 导出），而它在 `os.remove(image_path)`
     **之后** -> 图片已被删除、标注留在原地、调用方拿到 success=False
     并直接 return error，列表里残留失效条目。
  D1b 修好 import 之后又暴露第二个风险：`find_label_file` 可能返回**共享**标注
     文件（`_annotations.coco.json`），直接 `os.remove` 会毁掉整个数据集的标注。
  D10 `interface.delete_image_file` 的 `skip_confirm` 是死参数、从不确认，
     而 action 描述写着"物理删除不可恢复"。

断言：
  B1 专属标注（img1.json）+ 缓存文件随图片一起删除，success=True 且无 errors
  B2 邻图 img2.png / img2.json 完全不受影响
  B3 共享 COCO 文件：图片被删、**共享文件保留**、给出 warning
  B4 解析阶段失败 -> 一个文件都不删（杜绝"图已删、标注留下"）
  B5 确认门：拒绝确认时文件必须还在；确认后才删

退出码即断言：0 通过 / 1 断言失败 / 其他异常。
用法: python -u tests/annotation/test_delete_image.py
"""
import json
import os
import shutil
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import isolate_state  # noqa: E402

_IMG = os.path.join(_ROOT, "resources", "img.png")


def build_dir(root):
    """造一个含图片/专属标注/缓存/共享COCO的目录。"""
    d = os.path.join(root, "ds")
    os.makedirs(os.path.join(d, "coco_annotations"), exist_ok=True)

    for name in ("img1.png", "img2.png", "img3.png"):
        shutil.copy(_IMG, os.path.join(d, name))

    # 专属标注（labelme 风格，与图片同名）——应当随图删除
    for name in ("img1.json", "img2.json"):
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            json.dump({"version": "5.0", "shapes": [], "imagePath": name,
                       "imageData": None}, f)

    # 缓存文件——应当随图删除
    for name in ("img1_cache.json", "img1_cache.npy"):
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"cache")

    # 共享 COCO 标注——**绝不能**被删
    with open(os.path.join(d, "coco_annotations", "_annotations.coco.json"),
              "w", encoding="utf-8") as f:
        json.dump({"images": [{"id": 1, "file_name": "img3.png"}],
                   "annotations": [], "categories": []}, f)
    return d


class _FakeBox:
    """替掉 MessageBox：避免 headless 测试里模态对话框阻塞。"""
    answer = False

    def __init__(self, *args, **kwargs):
        pass

    def exec(self):
        return _FakeBox.answer


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    import plugins.annotation.interfaces.annotation_interface as iface_mod
    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    root = tempfile.mkdtemp(prefix="visionmind_delete_")
    d = build_dir(root)

    w = AnnotationInterface(None, None)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()

    failures = []
    p1 = os.path.join(d, "img1.png")
    p2 = os.path.join(d, "img2.png")
    p3 = os.path.join(d, "img3.png")

    print("== 删图回归 ==")
    print("   目录: %s" % d)

    # ---------- B1/B2: **刚打开目录就删图**（WinError 32 回归）+ 专属标注/缓存 ----------
    # 这里刻意不等任何东西：open_directory 会为所有可见行排入缩略图解码，
    # 此刻文件句柄被占用，删除必须由界面层的 wait_idle 兜住。
    real_box = iface_mod.MessageBox
    iface_mod.MessageBox = _FakeBox
    _FakeBox.answer = False
    try:
        msg = w.delete_image_file(p1, skip_confirm=True)
    finally:
        iface_mod.MessageBox = real_box
    print("   B1 刚打开目录即删 img1.png -> %r" % msg)
    for name in ("img1.png", "img1.json", "img1_cache.json", "img1_cache.npy"):
        if os.path.exists(os.path.join(d, name)):
            failures.append("B1 %s 未被删除（msg=%r）" % (name, msg))
    if os.path.exists(os.path.join(d, "img2.png")) is False:
        failures.append("B2 img2.png 被误删")
    if not os.path.exists(os.path.join(d, "img2.json")):
        failures.append("B2 img2.json 被误删")
    print("   B2 邻图 img2.png / img2.json 未受影响  OK")

    # ---------- B3: 共享 COCO 文件必须保留 ----------
    # 先等在途解码结束（界面层会自动做，这里直接调服务所以显式等），
    # 让本断言只检验"共享文件不删"这一条，不掺杂句柄占用。
    w._thumb_manager.wait_idle(3000)
    w.draw_area.wait_io_idle(3000)
    coco = os.path.join(d, "coco_annotations", "_annotations.coco.json")
    r3 = w.manual.delete_image(p3)
    print("   B3 deleted=%s warnings=%s"
          % ([os.path.basename(x) for x in r3.get("deleted_files", [])],
             r3.get("warnings")))
    if os.path.exists(p3):
        failures.append("B3 图片本身未被删除: %s" % r3.get("errors"))
    if not os.path.exists(coco):
        failures.append("B3 **共享 COCO 标注被删除**，整个数据集的标注都会丢失")
    if not r3.get("warnings"):
        failures.append("B3 未给出共享文件警告")

    # ---------- B4: 解析阶段失败 -> 一个文件都不删 ----------
    import core.backend.utils as backend_utils
    real_find = backend_utils.find_label_file

    def boom(*args, **kwargs):
        raise RuntimeError("simulated resolution failure")

    backend_utils.find_label_file = boom
    try:
        r4 = w.manual.delete_image(p2)
    finally:
        backend_utils.find_label_file = real_find
    print("   B4 success=%s deleted=%s errors=%s"
          % (r4.get("success"), r4.get("deleted_files"), r4.get("errors")))
    if r4.get("success") or r4.get("deleted_files"):
        failures.append("B4 解析失败却仍然删了文件: %s" % r4.get("deleted_files"))
    if not os.path.exists(p2):
        failures.append("B4 解析失败后图片被删除（原缺陷正是：图已删、标注留下）")

    # ---------- B5: 确认门 ----------
    real_box = iface_mod.MessageBox
    iface_mod.MessageBox = _FakeBox
    try:
        _FakeBox.answer = False
        msg = w.delete_image_file(p2, skip_confirm=False)
        print("   B5 拒绝确认 -> %r  文件仍在=%s" % (msg, os.path.exists(p2)))
        if not os.path.exists(p2):
            failures.append("B5 拒绝确认后文件仍被删除")
        if msg != "已取消删除":
            failures.append("B5 拒绝确认未返回取消: %r" % msg)

        _FakeBox.answer = True
        msg = w.delete_image_file(p2, skip_confirm=False)
        print("   B5 同意确认 -> %r  文件已删=%s" % (msg, not os.path.exists(p2)))
        if os.path.exists(p2):
            failures.append("B5 同意确认后文件未被删除")
        if os.path.exists(os.path.join(d, "img2.json")):
            failures.append("B5 img2.json 未随图删除（目录内容: %s）"
                            % sorted(os.listdir(d)))

        # skip_confirm=True 不应弹框（用一张新图验证：img3 已在 B3 被删，
        # 否则这里会因为"文件不存在"而变成一个永远通过的空断言）
        _FakeBox.answer = False
        p5 = os.path.join(d, "img4.png")
        shutil.copy(_IMG, p5)
        r5 = w.delete_image_file(p5, skip_confirm=True)
        print("   B5 skip_confirm=True -> %r  file_gone=%s" % (r5, not os.path.exists(p5)))
        if os.path.exists(p5):
            failures.append("B5 skip_confirm=True 未删除文件: %r" % r5)
    finally:
        iface_mod.MessageBox = real_box

    if failures:
        print("   FAIL:")
        for f in failures:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: B1/B2/B3/B4/B5 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
