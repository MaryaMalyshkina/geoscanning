"""
Модуль для кластеризации географических точек.

Содержит классы для группировки точек по менеджерам и выполнения
иерархической кластеризации на основе матрицы расстояний.
"""
import logging
from typing import List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .point import Point
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointClusterer:
    """
    Базовый класс для кластеризации точек.

    Attributes:
        max_points_per_cluster (int): Максимальное количество точек в кластере.
    """

    def __init__(self, max_points_per_cluster: int = config.MAX_VISITS_PER_DAY):
        """
        Инициализация базового кластеризатора.

        Args:
            max_points_per_cluster: Максимальное количество точек в кластере.
        """
        self.max_points_per_cluster = max_points_per_cluster

    def cluster_by_manager(self, points: List[Point]) -> List[Point]:
        """
        Группирует точки по менеджерам и кластеризует каждую группу.

        Args:
            points: Список объектов Point для кластеризации.

        Returns:
            List[Point]: Список точек с присвоенными кластерами.
        """
        manager_groups = defaultdict(list)
        for p in points:
            manager_groups[p.manager].append(p)

        all_clustered = []
        for manager_id, group in manager_groups.items():
            logger.info(f"Кластеризация менеджера {manager_id}: {len(group)} точек")
            clustered = self._cluster_points(group)
            all_clustered.extend(clustered)
        return all_clustered

    def _cluster_points(self, points: List[Point]) -> List[Point]:
        """
        Метод кластеризации, переопределяемый в наследниках.

        Args:
            points: Список точек для кластеризации.

        Raises:
            NotImplementedError: Если метод не переопределён в наследнике.
        """
        raise NotImplementedError


class ImprovedClusterer(PointClusterer):
    """
    Улучшенный кластеризатор с использованием иерархической кластеризации.

    Использует предварительно загруженную матрицу расстояний для кластеризации
    точек с порогом eps_meters.

    Attributes:
        eps_meters (float): Пороговое расстояние для кластеризации в метрах.
        _dist_matrix (pd.DataFrame): Кешированная матрица расстояний.
    """
    def __init__(
            self, 
            max_points_per_cluster: int = config.MAX_VISITS_PER_DAY, 
            eps_meters: float = config.EPS_METERS, 
            random_seed: int = config.RANDOM_SEED):
        """
        Инициализация улучшенного кластеризатора.

        Args:
            max_points_per_cluster: Максимальное количество точек в кластере.
            eps_meters: Пороговое расстояние для кластеризации в метрах.
            random_seed: Seed для воспроизводимости результатов.
        """
        super().__init__(max_points_per_cluster)
        self.eps_meters = eps_meters
        self._dist_matrix = None   # кеш полной матрицы
        np.random.seed(random_seed)


    def _cluster_points(
        self,
        points: List[Point],
        distance_matrix_path: Optional[str] = None,
        eps_meters: float = config.EPS_METERS
    ) -> List[Point]:
        """
        Выполняет иерархическую кластеризацию точек на основе подматрицы расстояний.

        Args:
            points: Список точек для кластеризации.
            distance_matrix_path: Путь к файлу с матрицей расстояний.
            eps_meters: Пороговое расстояние для кластеризации в метрах.

        Returns:
            List[Point]: Список точек с присвоенными кластерами.

        Raises:
            ValueError: Если индексы точек выходят за пределы матрицы.
        """
        if not points:
            return points

        n = len(points)
        if n == 1:
            points[0].cluster = 0
            return points

        # Загружаем полную матрицу (кешируем)
        if distance_matrix_path is None:
            distance_matrix_path = config.DISTANCE_MATRIX_PATH

        # Загружаем и кешируем матрицу
        if self._dist_matrix is None:
            self._dist_matrix = pd.read_csv(distance_matrix_path, index_col=0)
            # Приводим индексы и колонки к int (на случай, если они строки)
            self._dist_matrix.index = self._dist_matrix.index.astype(int)
            self._dist_matrix.columns = self._dist_matrix.columns.astype(int)


        # Получаем индексы точек (orig_idx)
        indices = [p.orig_idx for p in points]

        # Проверяем, что все индексы присутствуют
        max_idx = self._dist_matrix.index.max()
        if any(i > max_idx for i in indices):
            raise ValueError(f"Некоторые индексы {indices} выходят за пределы матрицы (макс {max_idx})")

        # Извлекаем подматрицу
        sub_df = self._dist_matrix.loc[indices, indices]
        sub_matrix = sub_df.values

        if sub_matrix.shape != (n, n):
            raise ValueError(f"Размер подматрицы {sub_matrix.shape} не совпадает с числом точек {n}")

        # Кластеризация
        condensed = squareform(sub_matrix, checks=False)
        Z = linkage(condensed, method='average', metric='precomputed')
        labels = fcluster(Z, t=eps_meters, criterion='distance')

        for point, label in zip(points, labels):
            point.cluster = int(label) - 1

        logger.info(
            f"Создано {len(set(labels))} кластеров (порог {eps_meters} м) "
            f"для {n} точек менеджера {points[0].manager if points else '?'}"
        )

        return points
    