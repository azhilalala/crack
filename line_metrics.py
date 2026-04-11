"""
线宽线距计算工具类 (Line Width and Spacing Metrics Calculator)

基于裂纹图像数据集，计算 PCB 相关的线宽/线距指标：
- 最小线宽 (Minimum line width)
- 最小线距 (Minimum line spacing)
- 内层最小线宽 (Inner layer minimum line width)
- 内层最小线距 (Inner layer minimum line spacing)
- 外层最小线宽 (Outer layer minimum line width)
- 外层最小线距 (Outer layer minimum line spacing)

算法说明：
1. 线宽计算：对前景（线条区域）做距离变换，取骨架点处距离值的最小值 × 2
2. 线距计算：对背景（间隙区域）做距离变换，取骨架点处距离值的最小值 × 2
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class LineMetrics:
    """单张图像的线宽线距计算结果"""
    min_line_width: Optional[float] = None   # 最小线宽（像素）
    min_line_spacing: Optional[float] = None  # 最小线距（像素）
    file_path: str = ""


@dataclass
class LayerMetrics:
    """分层汇总结果"""
    min_line_width: Optional[float] = None    # 最小线宽
    min_line_spacing: Optional[float] = None  # 最小线距
    image_count: int = 0


@dataclass
class AllMetrics:
    """全部指标汇总"""
    min_line_width: Optional[float] = None        # 最小线宽
    min_line_spacing: Optional[float] = None      # 最小线距
    inner_min_line_width: Optional[float] = None  # 内层最小线宽
    inner_min_line_spacing: Optional[float] = None  # 内层最小线距
    outer_min_line_width: Optional[float] = None  # 外层最小线宽
    outer_min_line_spacing: Optional[float] = None  # 外层最小线距


class LineMetricsCalculator:
    """
    线宽线距计算工具类

    通过图像处理技术（二值化、距离变换、骨架提取）分析图像中的
    线条宽度和线条间距，支持按内层/外层分别统计。

    Parameters
    ----------
    threshold : int
        二值化阈值，默认 128。灰度值高于此值的像素视为前景（线条）。
    use_otsu : bool
        是否使用 Otsu 自适应阈值，默认 True。启用时忽略 threshold 参数。
    pixel_scale : float
        像素到实际尺寸的比例因子（例如 mil/pixel 或 mm/pixel），默认 1.0。
    """

    def __init__(
        self,
        threshold: int = 128,
        use_otsu: bool = True,
        pixel_scale: float = 1.0,
    ):
        self.threshold = threshold
        self.use_otsu = use_otsu
        self.pixel_scale = pixel_scale

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """将图像转为二值图（前景=1，背景=0）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if self.use_otsu:
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                gray, self.threshold, 255, cv2.THRESH_BINARY
            )

        return (binary > 0).astype(np.uint8)

    @staticmethod
    def _compute_min_width(mask: np.ndarray) -> Optional[float]:
        """
        计算 mask 中区域的最小宽度。

        对 mask 区域做距离变换，提取骨架，取骨架上距离值的最小值 × 2
        即为该区域的最小宽度。
        """
        if mask.sum() == 0:
            return None

        dist = ndimage.distance_transform_edt(mask)

        skeleton = skeletonize(mask.astype(bool))
        skeleton_distances = dist[skeleton]

        if skeleton_distances.size == 0:
            return None

        # 骨架点处的距离值代表到最近边界的距离（半宽度）
        # 最小线宽 = 2 × 最小半宽度
        min_half_width = skeleton_distances.min()
        if min_half_width == 0:
            # Single-pixel-wide line: the distance transform is 0 at the
            # boundary but skeletonize still produces a skeleton point there.
            # Treat this as a line of width 1 pixel.
            return 1.0
        return float(min_half_width * 2)

    def compute_single_image(self, image_path: str) -> LineMetrics:
        """
        计算单张图像的线宽和线距。

        Parameters
        ----------
        image_path : str
            图像文件路径

        Returns
        -------
        LineMetrics
            包含最小线宽、最小线距的结果
        """
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        binary = self._binarize(image)

        # 前景区域（线条）的最小宽度 → 最小线宽
        foreground = binary
        min_line_width = self._compute_min_width(foreground)

        # 背景区域（间隙）的最小宽度 → 最小线距
        background = 1 - binary
        min_line_spacing = self._compute_min_width(background)

        # 应用像素比例因子
        if min_line_width is not None:
            min_line_width *= self.pixel_scale
        if min_line_spacing is not None:
            min_line_spacing *= self.pixel_scale

        return LineMetrics(
            min_line_width=min_line_width,
            min_line_spacing=min_line_spacing,
            file_path=image_path,
        )

    def compute_batch(self, image_paths: List[str]) -> LayerMetrics:
        """
        批量计算一组图像的最小线宽和最小线距。

        对每张图像分别计算，取所有图像中的全局最小值。

        Parameters
        ----------
        image_paths : list of str
            图像文件路径列表

        Returns
        -------
        LayerMetrics
            该组图像的汇总结果
        """
        all_widths: List[float] = []
        all_spacings: List[float] = []

        for path in image_paths:
            try:
                metrics = self.compute_single_image(path)
                if metrics.min_line_width is not None:
                    all_widths.append(metrics.min_line_width)
                if metrics.min_line_spacing is not None:
                    all_spacings.append(metrics.min_line_spacing)
            except (FileNotFoundError, cv2.error) as exc:
                logger.warning("跳过图像 %s: %s", path, exc)
                continue

        return LayerMetrics(
            min_line_width=min(all_widths) if all_widths else None,
            min_line_spacing=min(all_spacings) if all_spacings else None,
            image_count=len(image_paths),
        )

    def compute_all(
        self,
        inner_layer_paths: List[str],
        outer_layer_paths: List[str],
    ) -> AllMetrics:
        """
        分别计算内层和外层图像的线宽线距，并汇总全局最小值。

        Parameters
        ----------
        inner_layer_paths : list of str
            内层图像文件路径列表
        outer_layer_paths : list of str
            外层图像文件路径列表

        Returns
        -------
        AllMetrics
            包含六项指标的完整结果：
            - min_line_width        最小线宽
            - min_line_spacing      最小线距
            - inner_min_line_width  内层最小线宽
            - inner_min_line_spacing 内层最小线距
            - outer_min_line_width  外层最小线宽
            - outer_min_line_spacing 外层最小线距
        """
        inner_metrics = self.compute_batch(inner_layer_paths)
        outer_metrics = self.compute_batch(outer_layer_paths)

        # 全局最小值 = min(内层, 外层)
        global_widths = [
            v
            for v in [inner_metrics.min_line_width, outer_metrics.min_line_width]
            if v is not None
        ]
        global_spacings = [
            v
            for v in [
                inner_metrics.min_line_spacing,
                outer_metrics.min_line_spacing,
            ]
            if v is not None
        ]

        return AllMetrics(
            min_line_width=min(global_widths) if global_widths else None,
            min_line_spacing=min(global_spacings) if global_spacings else None,
            inner_min_line_width=inner_metrics.min_line_width,
            inner_min_line_spacing=inner_metrics.min_line_spacing,
            outer_min_line_width=outer_metrics.min_line_width,
            outer_min_line_spacing=outer_metrics.min_line_spacing,
        )

    @staticmethod
    def load_image_paths(directory: str, extensions: tuple = _IMAGE_EXTENSIONS) -> List[str]:
        """
        从目录加载图像文件路径列表。

        Parameters
        ----------
        directory : str
            图像所在目录
        extensions : tuple of str
            要匹配的文件扩展名，默认匹配常见图像格式

        Returns
        -------
        list of str
            匹配到的文件路径列表（排序后）
        """
        paths = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(extensions)
        )
        return paths


def main():
    """演示用法：使用项目中的裂纹图像计算线宽线距指标"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    calc = LineMetricsCalculator(use_otsu=True, pixel_scale=1.0)

    # 加载训练集裂纹图像作为外层样本（示例：取前 20 张）
    outer_dir = os.path.join(base_dir, "train", "t_c")
    outer_paths = calc.load_image_paths(outer_dir)[:20]

    # 加载验证集裂纹图像作为内层样本（示例：取前 20 张）
    inner_dir = os.path.join(base_dir, "val", "v_c")
    inner_paths = calc.load_image_paths(inner_dir)[:20]

    if not outer_paths and not inner_paths:
        print("未找到图像文件，请检查路径。")
        return

    # 计算全部指标
    result = calc.compute_all(
        inner_layer_paths=inner_paths,
        outer_layer_paths=outer_paths,
    )

    print("=" * 50)
    print("线宽线距计算结果")
    print("=" * 50)
    print(f"最小线宽:       {result.min_line_width}")
    print(f"最小线距:       {result.min_line_spacing}")
    print(f"内层最小线宽:   {result.inner_min_line_width}")
    print(f"内层最小线距:   {result.inner_min_line_spacing}")
    print(f"外层最小线宽:   {result.outer_min_line_width}")
    print(f"外层最小线距:   {result.outer_min_line_spacing}")
    print("=" * 50)


if __name__ == "__main__":
    main()
