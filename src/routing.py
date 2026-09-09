"""
Модуль для построения маршрутов.

Содержит класс RouteBuilder для построения оптимальных маршрутов
на основе кластеризованных точек с учётом ограничений по дням и визитам.
"""
import random
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from .point import Point
import config

logger = logging.getLogger(__name__)

class RouteBuilder:
    """
    Построение маршрутов: фильтрация → балансировка кластеров → назначение дней → TSP.

    Attributes:
        dist_matrix_df (pd.DataFrame): Матрица расстояний между точками.
        nn_dist_map (Dict[str, float]): Расстояния от базы до каждой точки.
        days (int): Количество дней для планирования маршрутов.
        capacity (int): Максимальное количество визитов в день.
        max_dist_km (float): Максимальное расстояние от базы в км.
        nn_lat (float): Широта базы (Нижний Новгород).
        nn_lon (float): Долгота базы (Нижний Новгород).
        random_seed (int): Seed для воспроизводимости.
    """

    def __init__(
        self,
        dist_matrix_df: pd.DataFrame,
        nn_dist_map: Optional[Dict[str, float]] = None,
        days: int = config.DAYS,
        capacity: int = config.MAX_VISITS_PER_DAY,
        max_dist_km: float = config.MAX_DISTANCE_FROM_BASE_KM,
        nn_lat: float = config.NN_LAT,
        nn_lon: float = config.NN_LON,
        random_seed: int = config.RANDOM_SEED
    ):
        """
        Инициализация построителя маршрутов.

        Args:
            dist_matrix_df: Матрица расстояний между точками.
            nn_dist_map: Расстояния от базы до каждой точки.
            days: Количество дней для планирования маршрутов.
            capacity: Максимальное количество визитов в день.
            max_dist_km: Максимальное расстояние от базы в км.
            nn_lat: Широта базы.
            nn_lon: Долгота базы.
            random_seed: Seed для воспроизводимости.
        """         
        self.dist_matrix_df = dist_matrix_df
        self.nn_dist_map = nn_dist_map or {}
        self.days = days
        self.capacity = capacity
        self.max_dist_km = max_dist_km
        self.nn_lat = nn_lat
        self.nn_lon = nn_lon
        self.random_seed = random_seed

        # Устанавливаем seed для воспроизводимости
        random.seed(random_seed)
        np.random.seed(random_seed)

    def build(
            self, 
            points: List[Point], 
            manager_id: int = 0
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Основной пайплайн построения маршрутов для заданного менеджера.

        Args:
            points: Список всех точек.
            manager_id: ID менеджера, для которого строятся маршруты.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - DataFrame с маршрутами (point_id, visit_day, cluster_id, order_in_route)
                - DataFrame со статистикой по дням (day, n_points, total_km и т.д.)
        """
        # Выбираем только точки нужного менеджера 
        manager_points = [p for p in points if p.manager == manager_id]
        if not manager_points:
            logger.warning(f"Нет точек для менеджера {manager_id}")
            return pd.DataFrame(), pd.DataFrame()

        df = self._points_to_df(manager_points)
  
        # Расстояния от базы (если не переданы — считаем геодезически)
        if not self.nn_dist_map:
            self.nn_dist_map = self._calc_nn_distances(df)
        df['dist_from_nn_km'] = df['point_id'].map(self.nn_dist_map)

        # Фильтр кластеров по 150 км
        df = self._filter_by_distance(df)

        # Удаление самых дальних кластеров до лимита визитов
        target = self.days * self.capacity
        df = self._remove_farthest_clusters(df, target)

        # Разбиение крупных кластеров (по сумме n_visits)
        df = self._split_clusters_by_visits(df)

        # Подматрица расстояний для оставшихся точек
        remaining_indices = df['orig_idx'].tolist()
        dist_matrix = self.dist_matrix_df.iloc[remaining_indices, remaining_indices].values
        pid_to_local = {pid: i for i, pid in enumerate(df['point_id'])}

        visits_df = self._build_visits_df(df)

        # Назначение дней
        schedule, day_load = self._assign_days(visits_df, manager_id=manager_id)

        if not any(schedule):
            logger.error(f"Не удалось распределить визиты по дням для менеджера {manager_id}")
            return pd.DataFrame(), pd.DataFrame()

        # TSP (жадный)
        result_df = self._build_tsp_routes(schedule, dist_matrix, pid_to_local)

        # Статистика пробега (только если есть маршруты)
        if not result_df.empty:
            stats_df, total_km = self._calc_stats(result_df, dist_matrix, pid_to_local)
        else:
            stats_df = pd.DataFrame(columns=['day', 'n_points', 'from_base_km', 'inter_point_km', 'total_km'])
            total_km = 0.0
            logger.warning(f"Для менеджера {manager_id} не построено ни одного маршрута")

        logger.info(
            f"Маршрут готов: {len(result_df)} визитов, "
            f"{self.days - day_load.count(0)} рабочих дней, пробег {total_km:.1f} км"
        )
        return result_df, stats_df

    def _points_to_df(self, points: List[Point]) -> pd.DataFrame:
        """
        Преобразует список точек в DataFrame.

        Args:
            points: Список объектов Point.

        Returns:
            pd.DataFrame: DataFrame с колонками point_id, manager, lat, lon,
                n_visits, cluster, orig_idx.
        """
        rows = []
        for p in points:
            rows.append({
                'point_id': p.point_id,
                'manager': p.manager,
                'lat': p.lat,
                'lon': p.lon,
                'n_visits': p.n_visits,
                'cluster': p.cluster,
                'orig_idx': p.orig_idx
            })
        return pd.DataFrame(rows)

    def _calc_nn_distances(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Вычисляет геодезическое расстояние от базы до каждой точки (если нет матрицы расстояний).

        Args:
            df: DataFrame с колонками point_id, lat, lon.

        Returns:
            Dict[str, float]: Словарь {point_id: расстояние_в_км}.
        """
        nn = (self.nn_lat, self.nn_lon)
        return {
            row['point_id']: geodesic(nn, (row['lat'], row['lon'])).km
            for _, row in df.iterrows()
        }

    def _filter_by_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Фильтрует кластеры по максимальному расстоянию от базы.

        Args:
            df: DataFrame с колонками manager, cluster, dist_from_nn_km.

        Returns:
            pd.DataFrame: Отфильтрованный DataFrame.
        """
        cluster_dist = df.groupby(['manager', 'cluster'])['dist_from_nn_km'].mean().reset_index()
        valid_clusters = cluster_dist[cluster_dist['dist_from_nn_km'] <= self.max_dist_km][['manager', 'cluster']]
        # Приводим типы для слияния
        df[['manager', 'cluster']] = df[['manager', 'cluster']].astype('int64')
        valid_clusters[['manager', 'cluster']] = valid_clusters[['manager', 'cluster']].astype('int64')
        # Сливаем только по ключам, оставляя все колонки из df (включая dist_from_nn_km)
        return df.merge(valid_clusters, on=['manager', 'cluster'])

    def _remove_farthest_clusters(self, df: pd.DataFrame, target: int) -> pd.DataFrame:
        """
        Удаляет самые дальние кластеры до достижения целевого числа визитов.

        Args:
            df: DataFrame с колонками cluster, dist_from_nn_km, n_visits.
            target: Целевое количество визитов.

        Returns:
            pd.DataFrame: DataFrame с оставшимися кластерами, перенумерованными.
        """
        df['day_cluster'] = df['cluster'].astype(int)

        cluster_stats = df.groupby('day_cluster').agg(
            dist_from_nn=('dist_from_nn_km', 'mean'),
            total_visits=('n_visits', 'sum')
        ).reset_index().sort_values('dist_from_nn', ascending=False)

        current_visits = int(cluster_stats['total_visits'].sum())
        to_remove = []
        for idx, row in cluster_stats.iterrows():
            # Останавливаемся, если достигли target или остался только один кластер
            if current_visits <= target or len(cluster_stats) - len(to_remove) <= 1:
                break
            to_remove.append(int(row['day_cluster']))
            current_visits -= int(row['total_visits'])

        if to_remove:
            df = df[~df['day_cluster'].isin(to_remove)].copy()

        # Если DataFrame стал пустым (крайний случай), оставляем самый ближний кластер
        if df.empty:
            closest_cluster = cluster_stats.iloc[-1]['day_cluster']
            df = df[df['day_cluster'] == closest_cluster].copy()
            df['day_cluster'] = 0

        # Перенумеровываем кластеры заново
        if not df.empty:
            unique = sorted(df['day_cluster'].unique())
            mapping = {old: new for new, old in enumerate(unique)}
            df['day_cluster'] = df['day_cluster'].map(mapping)

        return df

    def _split_clusters_by_visits(
            self, df: pd.DataFrame, 
            max_visits: Optional[int] = None
            ) -> pd.DataFrame:
        """
        Разбивает крупные кластеры на подкластеры по лимиту визитов.

        Args:
            df: DataFrame с колонками day_cluster, n_visits.
            max_visits: Максимальное количество визитов в подкластере.

        Returns:
            pd.DataFrame: DataFrame с разбитыми кластерами.
        """
        max_visits = max_visits or self.capacity
        result = []
        new_cl_id = 0

        for cl, group in df.groupby('day_cluster'):
            group_sorted = group.sort_values('n_visits', ascending=False)
            current_sub = []
            current_visits = 0

            for _, row in group_sorted.iterrows():
                if current_visits + row['n_visits'] > max_visits and len(current_sub) > 0:
                    g = pd.DataFrame(current_sub)
                    g['day_cluster'] = new_cl_id
                    result.append(g)
                    new_cl_id += 1
                    current_sub = []
                    current_visits = 0

                current_sub.append(row)
                current_visits += row['n_visits']

            if current_sub:
                g = pd.DataFrame(current_sub)
                g['day_cluster'] = new_cl_id
                result.append(g)
                new_cl_id += 1

        return pd.concat(result, ignore_index=True)

    def _build_visits_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создаёт таблицу визитов, размножая точки по числу визитов.

        Args:
            df: DataFrame с колонками point_id, manager, lat, lon, day_cluster, n_visits.

        Returns:
            pd.DataFrame: DataFrame с визитами (каждая строка - один визит).
        """
        visits = []

        for _, row in df.iterrows():
            for v in range(1, row['n_visits'] + 1):
                visits.append({
                    'point_id': row['point_id'],
                    'manager': row['manager'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'day_cluster': row['day_cluster'],
                    'visit_num': v
                })
        result = pd.DataFrame(visits)
        return result

    def _assign_days(
            self, 
            visits_df: pd.DataFrame, 
            manager_id: int = 0
            )-> Tuple[List[List[dict]], List[int]]:
        """
        Назначает визиты по дням с учётом ограничений.

        Args:
            visits_df: DataFrame с визитами.
            manager_id: ID менеджера.

        Returns:
            Tuple[List[List[dict]], List[int]]:
                - schedule: список дней, каждый день содержит список визитов.
                - day_load: загрузка каждого дня.
        """
        manager_id = int(manager_id)
       
        # Приводим к int
        visits_df['manager'] = visits_df['manager'].astype(int)
        mdf = visits_df[visits_df['manager'] == manager_id].copy()
      
        schedule = [[] for _ in range(self.days)]
        day_load = [0] * self.days
        point_days = {}

        groups = mdf.groupby(['day_cluster', 'visit_num'])
        group_info = groups.size().reset_index(name='size').sort_values('size', ascending=False)
     
        for _, g_row in group_info.iterrows():
            cl, vn = int(g_row['day_cluster']), int(g_row['visit_num'])
            gdf = mdf[(mdf['day_cluster'] == cl) & (mdf['visit_num'] == vn)]
          
            # Попытка 1: целиком в один день
            best_day = None
            best_score = float('inf')
            for day in range(self.days):
                if day_load[day] + len(gdf) > self.capacity:
                    continue
                conflict = any(
                    day in point_days.get(r['point_id'], set())
                    for _, r in gdf.iterrows()
                )
                if conflict:
                    continue
                same_cluster = sum(1 for p in schedule[day] if p.get('day_cluster') == cl)
                score = day_load[day] - same_cluster * 10
                if score < best_score:
                    best_score = score
                    best_day = day

            if best_day is not None:
                for _, r in gdf.iterrows():
                    schedule[best_day].append(r.to_dict())
                    point_days.setdefault(r['point_id'], set()).add(best_day)
                day_load[best_day] += len(gdf)
                continue

            # Попытка 2: поштучно с возможностью перегрузки 
            for _, r in gdf.iterrows():
                pid = r['point_id']
                placed = False
                # Сначала пытаемся найти день с учётом capacity
                for day in range(self.days):
                    if day_load[day] + 1 > self.capacity:
                        continue
                    if day in point_days.get(pid, set()):
                        continue
                    schedule[day].append(r.to_dict())
                    point_days.setdefault(pid, set()).add(day)
                    day_load[day] += 1
                    placed = True
                    break
                if not placed:
                    raise ValueError(f"Критическая нехватка места для {pid}")

        return schedule, day_load

    def _build_tsp_routes(
            self, 
            schedule: List[List[dict]], 
            dist_matrix: np.ndarray, 
            pid_to_local: Dict[str, int]
            )-> pd.DataFrame:
        """
        Строит маршруты TSP (жадный алгоритм) для каждого дня.

        Args:
            schedule: Список дней с визитами.
            dist_matrix: Матрица расстояний между точками.
            pid_to_local: Словарь для преобразования point_id в индекс матрицы.

        Returns:
            pd.DataFrame: DataFrame с маршрутами.
        """
        results = []
        for day_idx, day_visits in enumerate(schedule):
            if not day_visits:
                continue

            day_pids = [v['point_id'] for v in day_visits]
            unvisited = set(day_pids)
            route_pids = []
            current_pid = None

            while unvisited:
                if current_pid is None:
                    # Первая точка - ближайшая к базе
                    nearest = min(unvisited, key=lambda pid: self.nn_dist_map.get(pid, float('inf')))  
                else:
                    # Проверяем, что current_pid есть в словаре
                    current_idx = pid_to_local.get(current_pid)
                    if current_idx is None:
                        # Если current_pid не найден, берём ближайшую непосещенную от текущей
                        nearest = next(iter(unvisited))
                    else:
                        nearest = min(
                            unvisited,
                            key=lambda pid: dist_matrix[current_idx][pid_to_local.get(pid, 0)]
                        )
                route_pids.append(nearest)
                unvisited.remove(nearest)
                current_pid = nearest

            for order, pid in enumerate(route_pids, 1):
                results.append({
                    'point_id': pid,
                    'visit_day': day_idx + 1,
                    'cluster_id': next(v['day_cluster'] for v in day_visits if v['point_id'] == pid),
                    'order_in_route': order
                })
        if not results:
            return pd.DataFrame(columns=['point_id', 'visit_day', 'cluster_id', 'order_in_route'])
        return pd.DataFrame(results)

    def _calc_stats(
            self,
            result_df: pd.DataFrame,
            dist_matrix: np.ndarray,
            pid_to_local: Dict[str, int]
            ) -> Tuple[pd.DataFrame, float]:
        """
        Вычисляет статистику маршрута.

        Args:
            result_df: DataFrame с маршрутами.
            dist_matrix: Матрица расстояний между точками.
            pid_to_local: Словарь для преобразования point_id в индекс матрицы.

        Returns:
            Tuple[pd.DataFrame, float]:
                - DataFrame со статистикой по дням.
                - Общий пробег в км.
        """
        stats = []
        for day in sorted(result_df['visit_day'].unique()):
            day_df = result_df[result_df['visit_day'] == day].sort_values('order_in_route')
            pids = day_df['point_id'].tolist()
            if not pids:
                continue

            from_base = self.nn_dist_map.get(pids[0], 0)
            inter = 0.0
            for i in range(1, len(pids)):
                d = dist_matrix[pid_to_local[pids[i - 1]]][pid_to_local[pids[i]]]
                inter += d / 1000.0

            total = from_base + inter
            stats.append({
                'day': int(day),
                'n_points': len(pids),
                'from_base_km': round(from_base, 2),
                'inter_point_km': round(inter, 2),
                'total_km': round(total, 2)
            })

        stats_df = pd.DataFrame(stats)
        total_km = stats_df['total_km'].sum() if not stats_df.empty else 0.0
        return stats_df, total_km