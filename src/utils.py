"""
Модуль с утилитами для работы с таблицами Qt.

Содержит функции для заполнения и обновления таблиц с точками и маршрутами.
"""
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from typing import List

from .point import Point
import pandas as pd

def update_clusters_in_table(table: QTableWidget, points: List[Point]) -> None:
    """
    Обновляет только колонку с кластерами в таблице после расчёта кластеров.

    Args:
        table: QTableWidget для обновления.
        points: Список объектов Point с присвоенными кластерами.
    """
    if not points or table.rowCount() != len(points):
        return
    
    # Сохраняем сортировку
    was_sorted = table.isSortingEnabled()
    table.setSortingEnabled(False)
     
    for row, point in enumerate(points):
        if point.cluster == -1:
            item = QTableWidgetItem("не назначен")
        else:
            item = QTableWidgetItem(str(point.cluster))
        table.setItem(row, 5, item)
    
    # Восстанавливаем сортировку
    table.setSortingEnabled(was_sorted)

def fill_points_table(table: QTableWidget, points: List[Point]) -> None:
    """
    Заполняет таблицу точками с полной информацией.

    Args:
        table: QTableWidget для заполнения.
        points: Список объектов Point.
    """
    # Очистка и настройка заголовков
    table.clear()
    table.setRowCount(0)
    table.setColumnCount(6)
    headers = ["ID точки", "Менеджер", "Широта", "Долгота", "Визитов", "Кластер"]
    table.setHorizontalHeaderLabels(headers)
    vertical_header = table.verticalHeader()
    if vertical_header is not None:
        vertical_header.setVisible(False)

    if not points:
        return

    # Устанавливаем количество строк
    table.setRowCount(len(points))

    # Заполняем данными 
    for row, point in enumerate(points):
        table.setItem(row, 0, QTableWidgetItem(point.point_id))
        table.setItem(row, 1, QTableWidgetItem(str(point.manager)))
        table.setItem(row, 2, QTableWidgetItem(f"{point.lat:.6f}"))
        table.setItem(row, 3, QTableWidgetItem(f"{point.lon:.6f}"))
        table.setItem(row, 4, QTableWidgetItem(str(point.n_visits)))

        if point.cluster == -1:
            item_cluster = QTableWidgetItem("не назначен")
        else:
            item_cluster = QTableWidgetItem(str(point.cluster))
        table.setItem(row, 5, item_cluster)

    header = table.horizontalHeader()
    if header is not None:
        header.setStretchLastSection(False)  

        # Настройка ширины колонок
        header.resizeSection(0, 80)   
        header.resizeSection(1, 90)
        header.resizeSection(2, 90)  
        header.resizeSection(3, 90)  
        header.resizeSection(4, 80)  
        header.resizeSection(5, 105)

    viewport = table.viewport()
    if viewport is not None:
        viewport.update()

def fill_routes_table(table: QTableWidget, routes_df: pd.DataFrame) -> None:
    """
    Заполняет таблицу маршрутами.

    Args:
        table: QTableWidget для заполнения.
        routes_df: DataFrame с колонками:
            manager, visit_day, cluster_id, order_in_route, point_id.
    """
    # Очистка и настройка заголовков
    table.clear()
    table.setRowCount(0)
    table.setColumnCount(5)
    headers = ["Менеджер", "День", "Кластер", "Порядок", "ID точки"]
    table.setHorizontalHeaderLabels(headers)
    
    vertical_header = table.verticalHeader()
    if vertical_header is not None:
        vertical_header.setVisible(False)

    if routes_df.empty:
        return

   # Сбрасываем индекс для безопасного доступа по индексу
    routes_df_reset = routes_df.reset_index(drop=True)
    table.setRowCount(len(routes_df_reset))

    for row_idx in range(len(routes_df_reset)):
        row = routes_df_reset.iloc[row_idx]
        table.setItem(row_idx, 0, QTableWidgetItem(str(row.get('manager', 'N/A'))))
        table.setItem(row_idx, 1, QTableWidgetItem(str(row.get('visit_day', 'N/A'))))
        table.setItem(row_idx, 2, QTableWidgetItem(str(row.get('cluster_id', 'N/A'))))  
        table.setItem(row_idx, 3, QTableWidgetItem(str(row.get('order_in_route', 'N/A'))))
        table.setItem(row_idx, 4, QTableWidgetItem(str(row.get('point_id', 'N/A'))))

    # Настройка ширины колонок
    header = table.horizontalHeader()
    if header is not None:
        header.setStretchLastSection(False)
        header.resizeSection(0, 105)   # Менеджер
        header.resizeSection(1, 105)   # День
        header.resizeSection(2, 100)   # Кластер
        header.resizeSection(3, 100)   # Порядок
        header.resizeSection(4, 120)  # ID точки

    viewport = table.viewport()
    if viewport is not None:
        viewport.update()

