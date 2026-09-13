"""原子文件写入 — 临时文件 + `os.replace`。

为什么必须有它：审计确认 `core/backend/formats/*` 与 `manual_annotation_service.py`
里**所有**标注/划分/类别写入都是原地 `open(path, 'w')`，而 `open(..., 'w')` 会
**立即截断**原文件。这个应用平均每天崩好几次（`logs/crash_log_*.txt`），
于是"崩溃瞬间正好有一次保存在途"不是理论风险 —— 那一刻这张图的标注 JSON /
掩码 PNG 就被清空或半写，**用户的标注数据直接丢失**。
全仓唯一原本就安全的写入是 `core/service/project_service.py` 的 `_write_json_safely`
（temp + `os.replace`），本模块就是把那个做法抽出来给全仓用。

用法（三种，覆盖全部现有写入形态）::

    # 1) 文本/JSON：直接把 open() 换成 atomic_open()，函数体不用改
    with atomic_open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 2) 需要"先写临时文件再改名"的第三方 API（ElementTree.write / cv2.imwrite）
    with atomic_writer_path(path) as tmp:
        tree.write(tmp, encoding='utf-8', xml_declaration=True)

    # 3) 一次性写入
    atomic_write_json(path, data)

设计要点：

* 临时文件与目标**同目录** —— `os.replace` 只有同卷才是原子的；放到系统临时目录
  会跨卷退化成"拷贝+删除"，既非原子又可能在 Windows 上直接失败。
* 临时文件名**改变扩展名结尾**（`foo.json` -> `foo.json.atomic-<pid>-<n>`），
  这样按扩展名过滤的扫描/遍历（`scan_image_files`、`os.walk` 找图、
  `endswith('.json')` 的类别重写循环）不会把临时文件当成真实数据。
  代价是调用方不能用路径推断格式，因此 `atomic_writer_path` 的调用方必须显式
  指定格式（例如 `imwrite_unicode(tmp, img, ext='.png')`）。
* 失败时删除临时文件并**保留原文件**；异常继续向上抛（不吞）。
* 默认 `fsync=False`：本模块要防的是**进程崩溃**（本项目是 `qFatal -> abort`），
  此时数据已在 OS page cache、`close()` 之后 `os.replace` 的元数据操作即已完成；
  防掉电才需要 fsync，可按需开启。
"""
import itertools
import json
import os
import tempfile
from contextlib import contextmanager

_counter = itertools.count(1)


def temp_path_for(path: str) -> str:
    """返回同目录、扩展名结尾与目标不同的临时路径（保证不会被按扩展名扫描命中）。"""
    return "%s.atomic-%d-%d" % (path, os.getpid(), next(_counter))


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def atomic_write_bytes(path: str, data: bytes, fsync: bool = False) -> None:
    """原子写入字节串。"""
    _ensure_parent(path)
    tmp = temp_path_for(path)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        _discard(tmp)
        raise


def atomic_write_text(path: str, text: str, encoding: str = "utf-8",
                      fsync: bool = False) -> None:
    """原子写入文本。"""
    _ensure_parent(path)
    tmp = temp_path_for(path)
    try:
        with open(tmp, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        _discard(tmp)
        raise


def atomic_write_json(path: str, obj, indent: int = 2, ensure_ascii: bool = False,
                      fsync: bool = False) -> None:
    """原子写入 JSON（与全仓现有 json.dump(path, indent=2, ensure_ascii=False) 对齐）。"""
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii),
                      fsync=fsync)


@contextmanager
def atomic_open(path: str, mode: str = "w", encoding: str = "utf-8", fsync: bool = False):
    """`open()` 的原子替代（只支持写入模式）。

    用法与 `open()` 完全一致，退出 with 块时才把临时文件改名为目标文件；
    块内抛异常则删除临时文件、原文件保持不变。
    """
    if "w" not in mode and "a" not in mode and "x" not in mode:
        raise ValueError("atomic_open 只用于写入模式，当前 mode=%r" % mode)
    _ensure_parent(path)
    tmp = temp_path_for(path)
    text_mode = "b" not in mode
    f = open(tmp, mode, encoding=encoding) if text_mode else open(tmp, mode)
    try:
        yield f
        f.flush()
        if fsync:
            os.fsync(f.fileno())
        f.close()
        os.replace(tmp, path)
    except BaseException:
        try:
            f.close()
        except Exception:
            pass
        _discard(tmp)
        raise


@contextmanager
def atomic_writer_path(path: str):
    """给出一个临时路径，交给第三方 writer 写完后原子替换到 `path`。

    临时路径的**扩展名结尾与目标不同**，所以 writer 不能用路径推断格式：
    必须显式指定（如 `imwrite_unicode(tmp, img, ext='.png')`）。
    """
    _ensure_parent(path)
    tmp = temp_path_for(path)
    try:
        yield tmp
        if not os.path.exists(tmp):
            raise FileNotFoundError("writer 未生成临时文件: %s" % tmp)
        os.replace(tmp, path)
    except BaseException:
        _discard(tmp)
        raise


def _discard(tmp: str) -> None:
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError as exc:
        print(f"[atomic_io] 清理临时文件失败 {tmp}: {exc}")


__all__ = [
    "atomic_open",
    "atomic_writer_path",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "temp_path_for",
]
