"""唯一状态所有者 + 原子提交回归（审计 W1/W2 / D2 / D3 / D7 / D8）。

被测缺陷：同一件事原本有三份状态 ——
    interface.current_image_path / service.current_image_path /
    widget._current_image_path，再叠加 service.current_annotations 的别名。
于是"当前图片"与"当前标注"可由三处分别改写，中间出现

    (图 N-1 的 pixels / original_image_size / zoom / pan)
  + (图 N 的 path / annotations)

这种既不自洽、又会被自动保存写进磁盘的状态；而任何一条自动保存路径都会把它
落盘成"把 A 图标注写进 B 图文件"。

断言：
  O1 三份真值不得再被直接赋值（只读 property -> AttributeError）
  O2 原子性：load_image(i) 返回后、事件循环跑之前，画布**必须未提交**，
     度量必须已清空、坐标换算必须被拒、标注必须已属于图 i（绝不能是 i-1）
  O3 过期提交被拒：旧代次的 commit_pixels 返回 False 且不改动已提交状态；
     set_qimage 必须换代次，使在途解码无法覆盖
  O4 未提交时交互被闸门拦住（坐标换算 / SAM 打点 / 多边形顶点）
  O5 为**别的图**准备的标注被拒绝装载（D2/D3 的直接拦截点）
  O6 三处投影在快速切换全程都必须等于 session 真值

退出码即断言：0 通过 / 1 断言失败 / 其他异常。
用法: python -u tests/annotation/test_single_state_owner.py [rounds]
"""
import os
import shutil
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import isolate_state  # noqa: E402

_IMAGE_COUNT = 5


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def make_images(d):
    """生成尺寸各不相同的图片：这样"度量与图的对应关系"才可判别。"""
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QColor, QImage

    sizes = {}
    for i in range(_IMAGE_COUNT):
        w_, h_ = 64 + i * 32, 48 + i * 24
        img = QImage(w_, h_, QImage.Format.Format_RGB32)
        img.fill(QColor((20 * i + 30) % 255, 80, 140))
        path = os.path.join(d, "img%02d.png" % i)
        if not img.save(path):
            raise RuntimeError("无法写出测试图片: %s" % path)
        sizes[norm(path)] = QSize(w_, h_)
    return sizes


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtCore import QPoint, QSize
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    root = tempfile.mkdtemp(prefix="visionmind_session_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    true_sizes = make_images(d)

    w = AnnotationInterface(None, None)
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()

    sess = w.manual.session
    paths = [os.path.join(d, "img%02d.png" % i) for i in range(_IMAGE_COUNT)]

    # 用假 loader 让每张图的标注自带"属于哪张图"的标签，从而可判别归属
    def fake_load(image_path, mode, strict=False, **kw):
        tag = "tag:" + os.path.basename(image_path)
        return {"status": "success", "annotations": [
            {"label": tag, "bbox": [1, 2, 30, 40],
             "polygons": [[[1, 2], [31, 2], [31, 42], [1, 42]]],
             "shape_type": "polygon"}]}, "labelme"

    w.manual.load_annotations_for_image = fake_load

    def tags_of(annotations):
        return {a.get("label") for a in annotations if isinstance(a, dict)}

    print("== 唯一状态所有者 + 原子提交回归 ==")

    # ---------------- O1: 不得再有第二份真值 ----------------
    def try_assign(obj, attr, value):
        try:
            setattr(obj, attr, value)
            return False
        except AttributeError:
            return True

    for label, obj, attr in (
            ("manual.current_image_path", w.manual, "current_image_path"),
            ("manual.current_annotations", w.manual, "current_annotations"),
            ("interface.current_image_path", w, "current_image_path"),
            ("draw_area.image", w.draw_area, "image"),
            ("draw_area.pixmap", w.draw_area, "pixmap"),
            ("draw_area.original_image_size", w.draw_area, "original_image_size"),
            ("draw_area.display_scale", w.draw_area, "display_scale"),
    ):
        if not try_assign(obj, attr, None):
            failures.append("O1 %s 仍可被直接赋值 —— 说明还存在第二份真值" % label)
    if not failures:
        print("   O1 三份真值均已只读（7 个属性赋值全部 AttributeError）  OK")

    # ---------------- O2: 原子提交（确定性断言）----------------
    o2_ok = True
    for i in range(_IMAGE_COUNT):
        w.load_image(i)
        # 此刻事件循环还没跑，解码结果（队列化信号）不可能已经交付
        if w.draw_area.is_committed:
            failures.append("O2 load_image(%d) 返回后画布已提交 —— 提交不是原子的" % i)
            o2_ok = False
        if w.draw_area.pixmap is not None:
            failures.append("O2 未提交却已有 pixmap")
            o2_ok = False
        if w.draw_area.original_image_size.width() != 0 or \
                w.draw_area.original_image_size.height() != 0:
            failures.append("O2 未提交却残留度量 %s（会与旧坐标系混用）"
                            % w.draw_area.original_image_size)
            o2_ok = False
        if w.draw_area.to_image_coords(QPoint(5, 5)) is not None:
            failures.append("O2 未提交却允许坐标换算")
            o2_ok = False
        if sess.path != paths[i]:
            failures.append("O2 session.path=%r 期望 %r" % (sess.path, paths[i]))
            o2_ok = False
        # 标注必须已经是新图的（绝不能还是上一张的）
        got = tags_of(w.draw_area.annotations)
        expect = "tag:" + os.path.basename(paths[i])
        if got and got != {expect}:
            failures.append("O2 图 %d 的窗口内标注属于 %s，期望 %s（图 N-1 标注 + 图 N 路径）"
                            % (i, got, expect))
            o2_ok = False
        # 放行事件循环，等解码提交
        deadline = time.time() + 3
        while not w.draw_area.is_committed and time.time() < deadline:
            app.processEvents()
            time.sleep(0.002)
        if not w.draw_area.is_committed:
            failures.append("O2 图 %d 在 3s 内未提交像素" % i)
            o2_ok = False
        # 提交后度量必须与**这张图**的真实尺寸一致
        want = true_sizes[norm(paths[i])]
        got_size = w.draw_area.original_image_size
        if (got_size.width(), got_size.height()) != (want.width(), want.height()):
            failures.append("O2 图 %d 度量 %sx%s != 真实 %sx%s（度量与图不匹配）"
                            % (i, got_size.width(), got_size.height(),
                               want.width(), want.height()))
            o2_ok = False
    if o2_ok:
        print("   O2 原子提交：%d 次切换的窗口内均未提交、度量已清空、"
              "标注已属新图、坐标换算被拒  OK" % _IMAGE_COUNT)

    # ---------------- O3: 过期提交被拒 ----------------
    gen = sess.generation
    pix_before = w.draw_area.pixmap
    size_before = w.draw_area.original_image_size
    stale_img = QImage(8, 8, QImage.Format.Format_RGB32)
    stale_img.fill(QColor(255, 0, 0))
    from PySide6.QtGui import QPixmap
    accepted = sess.commit_pixels(gen - 1, stale_img, QPixmap.fromImage(stale_img),
                                 QSize(8, 8), 1.0)
    if accepted:
        failures.append("O3 旧代次 commit_pixels 被接受")
    if w.draw_area.pixmap is not pix_before or w.draw_area.original_image_size != size_before:
        failures.append("O3 旧代次提交改动了已提交状态")
    # set_qimage 必须换代次
    gen2 = sess.generation
    fresh = QImage(32, 16, QImage.Format.Format_RGB32)
    fresh.fill(QColor(0, 255, 0))
    w.draw_area.set_qimage(fresh)
    if sess.generation != gen2 + 1:
        failures.append("O3 set_qimage 未换代次（%d -> %d），在途解码会覆盖它"
                        % (gen2, sess.generation))
    if (w.draw_area.original_image_size.width(),
            w.draw_area.original_image_size.height()) != (32, 16):
        failures.append("O3 set_qimage 后度量未更新为 32x16: %s" % w.draw_area.original_image_size)
    if sess.commit_pixels(gen2, stale_img, QPixmap.fromImage(stale_img), QSize(99, 99), 1.0):
        failures.append("O3 set_qimage 之后旧代次仍能提交")
    if (w.draw_area.original_image_size.width(),
            w.draw_area.original_image_size.height()) != (32, 16):
        failures.append("O3 旧代次提交污染了 set_qimage 的结果")
    if not failures:
        print("   O3 过期提交被拒 + set_qimage 换代次  OK")

    # ---------------- O4: 未提交时交互被闸门拦住 ----------------
    w.draw_area.set_image(paths[1])
    o4_ok = not w.draw_area.is_committed
    if not o4_ok:
        failures.append("O4 前置条件失败：set_image 后竟然已提交")
    n_sam = len(w.draw_area.sam_points)
    r = w.draw_area.add_sam_point(5.0, 5.0)
    if not (isinstance(r, str) and r.startswith("错误")):
        failures.append("O4 未提交时 add_sam_point 未被拒绝: %r" % r)
    if len(w.draw_area.sam_points) != n_sam:
        failures.append("O4 未提交时 add_sam_point 仍改动了状态")
    n_poly = len(w.draw_area.current_poly)
    r2 = w.draw_area.add_polygon_vertex(5.0, 5.0)
    if not (isinstance(r2, str) and r2.startswith("错误")):
        failures.append("O4 未提交时 add_polygon_vertex 未被拒绝: %r" % r2)
    if len(w.draw_area.current_poly) != n_poly:
        failures.append("O4 未提交时 add_polygon_vertex 仍改动了状态")
    if w.draw_area.to_image_coords(QPoint(1, 1)) is not None:
        failures.append("O4 未提交时坐标换算未被拒")
    if o4_ok and not any(f.startswith("O4") for f in failures):
        print("   O4 未提交时交互被闸门拦住（坐标换算/SAM/多边形）  OK")

    # ---------------- O5: 给别的图准备的标注被拒绝 ----------------
    settled = time.time() + 3
    while not w.draw_area.is_committed and time.time() < settled:
        app.processEvents()
        time.sleep(0.002)
    before = list(w.draw_area.annotations)
    ok = w.manual.set_current_annotations([{"label": "tag:OTHER"}], for_path=paths[4])
    if ok:
        failures.append("O5 为别的图准备的标注被接受装载（D2/D3 拦截点失效）")
    if list(w.draw_area.annotations) != before:
        failures.append("O5 被拒的装载仍改动了当前标注")
    if not any(f.startswith("O5") for f in failures):
        print("   O5 为其它图准备的标注被拒绝装载  OK")

    # ---------------- O7: 装载失败必须禁止保存（D8）----------------
    def pump_until_committed(limit=3.0):
        end = time.time() + limit
        while not w.draw_area.is_committed and time.time() < end:
            app.processEvents()
            time.sleep(0.002)

    def failing_load(image_path, mode, strict=False, **kw):
        return {"status": "error", "message": "simulated corrupt annotation file"}, None

    w.load_image(2)
    pump_until_committed()
    files_before = set(os.listdir(d))
    w.manual.load_annotations_for_image = failing_load
    # force_reload=True 是必需的：load_image 对"同一张图"默认提前返回，
    # 不会重新装载标注，也就测不到装载失败路径
    w.load_image(2, force_reload=True)
    pump_until_committed()
    if sess.is_save_allowed:
        failures.append("O7 装载失败后 is_save_allowed 仍为 True")
    res = w.manual.save_current()
    files_after = set(os.listdir(d))
    if res.get("status") != "error":
        failures.append("O7 装载失败后 save_current 未拒绝: %r" % res)
    if files_after != files_before:
        failures.append("O7 装载失败后仍写出了文件（把空标注覆盖到坏文件上）: %s"
                        % sorted(files_after - files_before))
    # 正向对照：装载成功时必须允许保存，否则本闸门就过宽了
    w.manual.load_annotations_for_image = fake_load
    w.load_image(2, force_reload=True)
    # 标注装载现在在后台完成，必须等到"已装载"（只等像素提交会读到尚未装载的窗口）
    _end = time.time() + 3
    while not sess.is_annotations_loaded and time.time() < _end:
        app.processEvents()
        time.sleep(0.002)
    if not sess.is_save_allowed:
        failures.append("O7 装载成功后 is_save_allowed 仍为 False（闸门过宽）")
    res_ok = w.manual.save_current()
    if res_ok.get("status") != "success":
        failures.append("O7 装载成功后 save_current 失败: %r" % res_ok)
    elif set(os.listdir(d)) == files_after:
        failures.append("O7 装载成功后 save_current 未写出任何文件")
    if not any(f.startswith("O7") for f in failures):
        print("   O7 装载失败禁止保存 / 装载成功正常保存  OK")

    # ---------------- O6: 快速切换全程三处投影一致 ----------------
    o6_samples = 0
    o6_ok = True

    def sample(stage):
        nonlocal o6_samples, o6_ok
        o6_samples += 1
        sp = sess.path
        for name, val in (("service", w.manual.current_image_path),
                          ("interface", w.current_image_path),
                          ("widget", w.draw_area.current_image_path)):
            if val != sp:
                failures.append("O6 %s 投影 %r != session 真值 %r" % (name, val, sp))
                o6_ok = False
        if sess.is_committed and sp:
            want = true_sizes.get(norm(sp))
            got = w.draw_area.original_image_size
            if want and (got.width(), got.height()) != (want.width(), want.height()):
                failures.append("O6 已提交但度量 %sx%s != %s 的真实 %sx%s"
                                % (got.width(), got.height(), os.path.basename(sp),
                                   want.width(), want.height()))
                o6_ok = False
            tg = tags_of(sess.annotations)
            if tg and tg != {"tag:" + os.path.basename(sp)}:
                failures.append("O6 已提交但标注 %s 不属于 %s"
                                % (tg, os.path.basename(sp)))
                o6_ok = False

    t0 = time.perf_counter()
    for k in range(rounds):
        w.load_image(k % _IMAGE_COUNT)
        sample("after-load")
        # 每轮 pump 若干次并在每次事件循环后采样
        for _ in range(3):
            app.processEvents()
            sample("pump")
            time.sleep(0.003)
    # 静置到提交
    end = time.time() + 3
    while not w.draw_area.is_committed and time.time() < end:
        app.processEvents()
        sample("settle")
        time.sleep(0.002)
    elapsed = (time.perf_counter() - t0) * 1000.0
    if o6_ok:
        print("   O6 三处投影与 session 全程一致（%d 次采样，%.0f ms）  OK"
              % (o6_samples, elapsed))

    if failures:
        print("   FAIL:")
        for f in failures[:20]:
            print("     - %s" % f)
        if len(failures) > 20:
            print("     ... 共 %d 条" % len(failures))
        code = 1
    else:
        print("   PASS: O1~O7 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
