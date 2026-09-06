"""
Anomalib 修补模块 - 持久化修补机制

此模块提供 Anomalib 及相关库的运行时修补，解决 Windows 兼容性问题。
修补只会执行一次，通过全局状态标记避免重复修补。
"""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# 全局状态标记：记录是否已经修补过
_PATCHED = False


def is_patched() -> bool:
    """检查是否已经修补过"""
    global _PATCHED
    return _PATCHED


def set_patched():
    """设置修补完成状态"""
    global _PATCHED
    _PATCHED = True


def patch_symlink_functions():
    """
    修补符号链接函数以避免 Windows 权限问题。
    此函数可以安全地多次调用，但只有第一次会实际执行修补。
    """
    if is_patched():
        return

    print("[INFO] 开始执行 Anomalib 兼容性修补...")

    # 1. 修补 anomalib
    try:
        import anomalib.utils.path as path_module

        def patched_create_versioned_dir(root_dir: Path) -> Path:
            """创建版本化目录但不创建符号链接"""
            root_dir = Path(root_dir)
            root_dir.mkdir(parents=True, exist_ok=True)

            existing_versions = [
                int(d.name[1:]) for d in root_dir.iterdir()
                if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
            ]

            highest_version = max(existing_versions) if existing_versions else -1
            new_version_number = highest_version + 1
            new_version_dir = root_dir / f"v{new_version_number}"
            new_version_dir.mkdir()

            return new_version_dir

        path_module.create_versioned_dir = patched_create_versioned_dir
        print("[INFO] 已修补 anomalib 符号链接函数")
    except Exception as e:
        print(f"[WARNING] 修补 anomalib 符号链接函数失败: {e}")

    # 2. 修补 huggingface_hub
    try:
        import huggingface_hub.file_download as hf_download

        def patched_create_symlink(src: str, dst: str, new_blob: bool = False) -> None:
            """使用复制代替符号链接"""
            import shutil
            src_path = Path(src)
            dst_path = Path(dst)

            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()

            if src_path.exists():
                shutil.copy2(str(src_path), str(dst_path))
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                dst_path.touch()

        hf_download._create_symlink = patched_create_symlink
        print("[INFO] 已修补 huggingface_hub 符号链接函数")
    except Exception as e:
        print(f"[WARNING] 修补 huggingface_hub 符号链接函数失败: {e}")

    # 3. 设置环境变量禁用符号链接
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # 4. 修补 WinClip 的 setup 方法
    try:
        from anomalib.models.image.winclip import lightning_model as winclip_module
        from anomalib.data import PredictDataset
        import logging

        logger = logging.getLogger(__name__)

        class BatchWrapper:
            def __init__(self, image_tensor):
                self.image = image_tensor

        def patched_setup(self, stage: str) -> None:
            """Patched setup method for WinClip to support new PreProcessor API."""
            del stage
            if self.is_setup:
                return

            self.class_name = self._get_class_name()
            ref_images = None

            if self.k_shot:
                if self.few_shot_source:
                    logger.info("Loading reference images from %s", self.few_shot_source)
                    transform = None
                    if self.pre_processor:
                        transform = getattr(self.pre_processor, 'transform', None)

                    reference_dataset = PredictDataset(
                        self.few_shot_source,
                        transform=transform,
                    )

                    def image_item_collate(batch):
                        images = []
                        for item in batch:
                            if hasattr(item, 'image'):
                                images.append(item.image)
                            elif isinstance(item, dict) and 'image' in item:
                                images.append(item['image'])
                            elif torch.is_tensor(item):
                                images.append(item)
                        if images:
                            return BatchWrapper(torch.stack(images))
                        return batch

                    dataloader = DataLoader(
                        reference_dataset,
                        batch_size=1,
                        shuffle=False,
                        pin_memory=True,
                        collate_fn=image_item_collate
                    )
                else:
                    logger.info("Collecting reference images from training dataset")
                    dataloader = self.trainer.datamodule.train_dataloader()

                ref_images = self.collect_reference_images(dataloader)

            self.model.setup(self.class_name, ref_images)
            self.is_setup = True

        winclip_module.WinClip.setup = patched_setup
        print("[INFO] 已修补 WinClip setup 方法")
    except Exception as e:
        print(f"[WARNING] 修补 WinClip setup 方法失败: {e}")

    # 5. 修补 EfficientAD 的 prepare_pretrained_model 方法
    try:
        from anomalib.models.image.efficient_ad import lightning_model as efficientad_module
        from anomalib.data.utils import download_and_extract

        def patched_prepare_pretrained_model(self) -> None:
            """Patched prepare_pretrained_model to support custom pretrained_models_dir."""
            custom_dir = getattr(self, 'pretrained_models_dir', None)
            if custom_dir:
                pretrained_models_dir = Path(custom_dir)
            else:
                pretrained_models_dir = Path("./pre_trained/")

            pretrained_models_dir.mkdir(parents=True, exist_ok=True)

            model_size_str = self.model_size.value if hasattr(self.model_size, 'value') else str(self.model_size).lower()
            weights_dir = pretrained_models_dir / "efficientad_pretrained_weights"
            teacher_path = weights_dir / f"pretrained_teacher_{model_size_str}.pth"

            if not teacher_path.exists():
                download_and_extract(pretrained_models_dir, efficientad_module.WEIGHTS_DOWNLOAD_INFO)

            if teacher_path.exists():
                self.model.teacher.load_state_dict(
                    torch.load(teacher_path, map_location=torch.device(self.device), weights_only=True),
                )
                for param in self.model.teacher.parameters():
                    param.requires_grad = False
                self.model.teacher.eval()
            else:
                raise RuntimeError(f"Failed to load pretrained teacher model from {teacher_path}")

        efficientad_module.EfficientAd.prepare_pretrained_model = patched_prepare_pretrained_model
        print("[INFO] 已修补 EfficientAD prepare_pretrained_model 方法")
    except Exception as e:
        print(f"[WARNING] 修补 EfficientAD prepare_pretrained_model 方法失败: {e}")

    set_patched()
    print("[INFO] Anomalib 兼容性修补完成")


# 为了向后兼容，保留旧函数名
_patch_symlink_functions = patch_symlink_functions
