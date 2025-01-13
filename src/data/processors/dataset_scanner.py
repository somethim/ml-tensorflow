from pathlib import Path
from typing import Dict, List, Tuple

from src.settings import logger


class DatasetScanner:
    """Handles scanning dataset directory and creating class mappings."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def get_class_mapping(self) -> Dict[str, int]:
        """Get mapping of class names to indices."""
        class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        return {cls_name: i for i, cls_name in enumerate(class_names)}

    def scan_dataset(self) -> Tuple[List[Path], List[int], Dict[str, int]]:
        """Scan dataset directory and collect image paths and labels.

        Returns:
            Tuple of (image_paths, labels, class_mapping)
        """
        class_to_idx = self.get_class_mapping()
        image_paths: List[Path] = []
        labels: List[int] = []

        logger.info("Scanning dataset directory structure...")
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_idx = class_to_idx[class_dir.name]
            class_images = list(class_dir.glob("*"))
            class_count = len([f for f in class_images if f.is_file()])
            logger.info(f"Found {class_count} images in class '{class_dir.name}'")
            for img_path in class_images:
                if img_path.is_file():
                    image_paths.append(img_path)
                    labels.append(class_idx)

        if not image_paths:
            raise ValueError(f"No images found in {self.data_dir}")

        return image_paths, labels, class_to_idx
