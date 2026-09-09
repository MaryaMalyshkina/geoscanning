"""
Модуль для генерации интерактивных карт с маршрутами.

Содержит класс MapGenerator для создания HTML-карт с точками, кластерами
и маршрутами, отображёнными с помощью библиотеки folium.
"""
import os
from typing import Dict, Optional, List

import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .point import Point
import config

class MapGenerator:
    """
    Генератор интерактивных карт с использованием folium.

    Attributes:
        map (folium.Map): Объект карты.
        html_path (str): Путь к сохранённому HTML-файлу.
    """
    
    def __init__(self):
        """Инициализация генератора карт."""
        self.map = None
        self.html_path = None
        
    def create_map_from_csv(
            self,
            csv_path: str,
            lat_col: str = 'lat',
            lon_col: str = 'lon',
            name_col: str = 'point_id',
            group_col: str = 'manager',
            cluster_col: str = 'cluster',
            selected_manager: Optional[int] = None
        )-> Optional[str]:
        """
        Создаёт карту с точками из CSV-файла, раскрашенными по кластерам.

        Args:
            csv_path: Путь к CSV-файлу с данными.
            lat_col: Название колонки с широтой.
            lon_col: Название колонки с долготой.
            name_col: Название колонки с идентификатором точки.
            group_col: Название колонки с менеджером.
            cluster_col: Название колонки с кластером.
            selected_manager: ID менеджера для фильтрации.

        Returns:
            Optional[str]: Путь к сохранённому HTML-файлу или None при ошибке.
        """
        try:
            df = pd.read_csv(csv_path)
            print(f"Прочитано {len(df)} точек")

            # Фильтр по менеджеру
            if selected_manager is not None:
                df = df[df[group_col] == selected_manager]
                if df.empty:
                    print(f"Нет данных для менеджера {selected_manager}")
                    return None
                print(f"Отфильтровано до {len(df)} точек для менеджера {selected_manager}")

            # Если нет колонки с кластерами — используем менеджеров
            if cluster_col not in df.columns:
                print(f"Колонка '{cluster_col}' не найдена. Используем менеджеров для раскраски.")
                return self._create_map_by_manager(df, lat_col, lon_col, name_col, group_col)

            # Определяем уникальные кластеры
            clusters = df[df[cluster_col] >= 0][cluster_col].unique()
            clusters = sorted(clusters)
            print(f"Найдено кластеров: {len(clusters)}")

            # Динамическая генерация цветов для кластеров
            unique_clusters = df[cluster_col].unique()
            # Расширенная палитра (можно добавить ещё)
            base_colors = [
                'red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
                'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink',
                'lightblue', 'lightgreen', 'gray', 'black', 'lightgray', 'white'
            ]

            cluster_color_map = {cl: base_colors[i % len(base_colors)] for i, cl in enumerate(sorted(unique_clusters))}

            # Центр карты
            center_lat = df[lat_col].mean()
            center_lon = df[lon_col].mean()

            # Создаём карту
            self.map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                control_scale=True
            )
            folium.TileLayer(
                'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            ).add_to(self.map)

            # Добавляем точки через CircleMarker с явным цветом
            for idx, row in df.iterrows():
                cluster = row.get(cluster_col, -1)
                if cluster >= 0 and cluster in cluster_color_map:
                    color = cluster_color_map[cluster]
                    cluster_text = str(cluster)
                else:
                    color = 'gray'
                    cluster_text = "не назначен"

                # Попап с информацией
                popup_text = f"""
                <b>ID:</b> {row[name_col]}<br>
                <b>Менеджер:</b> {row[group_col]}<br>
                <b>Кластер:</b> {cluster_text}<br>
                <b>Посещений:</b> {row.get('n_visits', 1)}<br>
                <b>Координаты:</b><br>
                {row[lat_col]:.6f}, {row[lon_col]:.6f}
                """
                
                folium.Marker(
                    location=[row[lat_col], row[lon_col]],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"ID: {row[name_col]}",
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(self.map)

            # Добавляем легенду
            for cluster_id, color in cluster_color_map.items():
                # Создаём отдельный слой для каждого кластера
                fg = folium.FeatureGroup(name=f'Кластер {cluster_id}')
                folium.Marker(
                    location=[center_lat, center_lon],
                    icon=folium.Icon(color=color, icon='info-sign'),
                    popup=f"Кластер {cluster_id}"
                ).add_to(fg)

                fg.add_to(self.map)

            # Добавляем управление слоями
            folium.LayerControl().add_to(self.map)

            # Сохраняем
            self.html_path = os.path.join(config.MAPS_DIR, 'generated_map.html')
            self.map.save(self.html_path)
            print(f"Карта сохранена: {self.html_path}")
            return self.html_path

        except Exception as e:
            print(f"ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _create_map_by_manager(
            self,
            df: pd.DataFrame,
            lat_col: str,
            lon_col: str,
            name_col: str,
            group_col: str
    ) -> Optional[str]:
        """
        Создаёт карту с раскраской по менеджерам (если нет кластеров).

        Args:
            df: DataFrame с данными.
            lat_col: Название колонки с широтой.
            lon_col: Название колонки с долготой.
            name_col: Название колонки с идентификатором точки.
            group_col: Название колонки с менеджером.

        Returns:
            Optional[str]: Путь к сохранённому HTML-файлу или None при ошибке.
        """
        try:
            # Определяем уникальных менеджеров
            managers = df[group_col].unique()
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                    'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                    'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 
                    'black', 'lightgray']
            
            manager_colors = {}
            for i, manager in enumerate(managers):
                manager_colors[manager] = colors[i % len(colors)]
            
            print(f"Найдено менеджеров: {len(managers)}")
            
            # Вычисляем центр карты
            center_lat = df[lat_col].mean()
            center_lon = df[lon_col].mean()
            
            # Создаём карту
            self.map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                control_scale=True
            )
            
            folium.TileLayer(
                'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            ).add_to(self.map)
            
            # Добавляем точки
            for idx, row in df.iterrows():
                lat = row[lat_col]
                lon = row[lon_col]
                manager = row[group_col]
                point_id = row[name_col]
                visits = row.get('n_visits', 1)
                
                color = manager_colors[manager]
                
                popup_text = f"""
                <b>ID:</b> {point_id}<br>
                <b>Менеджер:</b> {manager}<br>
                <b>Посещений:</b> {visits}<br>
                <b>Координаты:</b><br>
                {lat:.6f}, {lon:.6f}
                """
                
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"ID: {point_id}",
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(self.map)
            
            # Легенда по менеджерам
            for manager, color in manager_colors.items():
                fg = folium.FeatureGroup(name=f'Менеджер {manager}')
                for idx, row in df[df[group_col] == manager].iterrows():
                    folium.Marker(
                        location=[row[lat_col], row[lon_col]],
                        tooltip=f"ID: {row[name_col]}",
                        icon=folium.Icon(color=color, icon='info-sign')
                    ).add_to(fg)
                fg.add_to(self.map)
            
            folium.LayerControl().add_to(self.map)
            
            self.html_path = os.path.join(config.MAPS_DIR, 'generated_map.html')
            self.map.save(self.html_path)
            print(f"Карта сохранена: {self.html_path}")
            
            return self.html_path
            
        except Exception as e:
            print(f"ОШИБКА в _create_map_by_manager: {e}")
            import traceback
            traceback.print_exc()
            return None
   
    def draw_map_by_days(
        self,
        routes_df: pd.DataFrame,
        points: List[Point],
        nn_lat: float = config.NN_LAT,
        nn_lon: float = config.NN_LON,
        days: int = config.DAYS,
        show_routes: bool = True,
        zoom_start: int = 10,
        title: Optional[str] = None,
        save_path: Optional[str] = None
) -> Optional[str]:
        """
        Отображает маршруты на карте, раскрашенные по дням визита.

        Args:
            routes_df: DataFrame с маршрутами.
            points: Список объектов Point.
            nn_lat: Широта базы.
            nn_lon: Долгота базы.
            days: Количество дней.
            show_routes: Показывать линии маршрутов.
            zoom_start: Начальный масштаб.
            title: Заголовок карты.
            save_path: Путь для сохранения.

        Returns:
            Optional[str]: Путь к сохранённому HTML-файлу или None при ошибке.
        """
        try:
            # Создаём DataFrame с координатами из points
            coords_data = []
            for p in points:
                coords_data.append({
                    'point_id': p.point_id,
                    'lat': p.lat,
                    'lon': p.lon
                })
            df_coords = pd.DataFrame(coords_data)
            
            # Подтягиваем координаты
            plot_df = routes_df.merge(
                df_coords[['point_id', 'lat', 'lon']],
                on='point_id',
                how='left'
            ).dropna(subset=['lat', 'lon'])
            
            if plot_df.empty:
                print("Нет данных для отображения на карте")
                return None
            
            # Центр карты
            center_lat = plot_df['lat'].mean()
            center_lon = plot_df['lon'].mean()
            
            # Создаём карту
            self.map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_start,
                control_scale=True
            )
            
            # Добавляем базовый слой OpenStreetMap
            folium.TileLayer(
                'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            ).add_to(self.map)
            
            # Цвета для дней
            cmap = plt.colormaps['hsv']
            day_colors = {
                day: mcolors.to_hex(cmap(i / max(1, days)))
                for i, day in enumerate(range(1, days + 1))
            }
            
            # База (Нижний Новгород)
            folium.Marker(
                location=[nn_lat, nn_lon],
                popup='<b>🏠 База: Нижний Новгород</b>',
                tooltip='База',
                icon=folium.Icon(color='black', icon='home', prefix='fa')
            ).add_to(self.map)
            
            # Для каждого дня создаём свой слой
            for day in sorted(plot_df['visit_day'].unique()):
                day_df = plot_df[plot_df['visit_day'] == day].sort_values('order_in_route')
                color = day_colors[int(day)]
                day_int = int(day)
                
                # Создаём FeatureGroup для управления видимостью слоя
                fg = folium.FeatureGroup(name=f'День {day_int}')
                
                # Линия маршрута
                if show_routes and len(day_df) > 0:
                    route_coords = [[nn_lat, nn_lon]]
                    for _, row in day_df.iterrows():
                        route_coords.append([row['lat'], row['lon']])
                    
                    folium.PolyLine(
                        locations=route_coords,
                        color=color,
                        weight=2.5,
                        opacity=0.7,
                        popup=f'Маршрут дня {day_int}'
                    ).add_to(fg)
                
                # Точки
                for _, row in day_df.iterrows():
                    popup_text = f"""
                    <b>ID точки:</b> {row['point_id']}<br>
                    <b>День:</b> {day_int}<br>
                    <b>Порядок:</b> {int(row['order_in_route'])}<br>
                    <b>Кластер:</b> {int(row['cluster_id'])}<br>
                    <b>Координаты:</b><br>
                    {row['lat']:.6f}, {row['lon']:.6f}
                    """
                    
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=8,
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"День {day_int}, #{int(row['order_in_route'])}",
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.8,
                        weight=2
                    ).add_to(fg)
                    
                    # Номер порядка поверх точки
                    folium.Marker(
                        location=[row['lat'], row['lon']],
                        icon=folium.DivIcon(
                            html=(
                                f'<div style="font-size: 10px; color: white; font-weight: bold; '
                                f'text-align: center; background-color: rgba(0,0,0,0.5); '
                                f'border-radius: 10px; padding: 0px 4px; '
                                f'margin-top: -5px;">{int(row["order_in_route"])}</div>'
                            )
                        )
                    ).add_to(fg)
                
                fg.add_to(self.map)
            
            # Добавляем управление слоями
            folium.LayerControl().add_to(self.map)

            # Легенда
            legend_html = '''
            <div style="position: fixed; bottom: 30px; left: 30px; width: 150px;
                        background-color: white; border: 2px solid grey; z-index: 9999;
                        font-size: 11px; padding: 10px; max-height: 90%; overflow-y: auto;
                        border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
            <b>📅 Дни маршрута</b><br><br>
            '''
            for day in range(1, days + 1):
                c = day_colors[day]
                legend_html += (
                    f'<span style="background-color:{c}; padding: 0px 12px; '
                    f'margin-right: 8px; border: 1px solid #555; border-radius: 3px;">'
                    f'&nbsp;</span> День {day}<br>'
                )
            legend_html += '</div>'

            # Сохраняем карту
            if save_path is None:
                # save_path = os.path.join(os.path.dirname(__file__), 'routes_map.html')
                save_path = os.path.join(config.MAPS_DIR, 'routes_map.html')

            self.map.save(save_path)

            # Вшиваем легенду прямо в <body> — самый надёжный способ для QWebEngineView
            with open(save_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            html_content = html_content.replace('</body>', legend_html + '\n</body>')
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"Карта маршрутов сохранена: {save_path}")
            return save_path

            
        except Exception as e:
            print(f"ОШИБКА при создании карты маршрутов: {e}")
            import traceback
            traceback.print_exc()
            return None

    def draw_map_single_day(
        self,
        routes_df: pd.DataFrame,
        points: List[Point],
        day_number: int,
        dist_matrix_df: pd.DataFrame,
        nn_dist_map: Optional[Dict[str, float]] = None,
        nn_lat: float = config.NN_LAT,
        nn_lon: float = config.NN_LON,
        days: int = config.DAYS,
        show_routes: bool = True,
        zoom_start: int = 10,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Отображает маршрут одного выбранного дня на карте.

        Args:
            routes_df: DataFrame с маршрутами.
            points: Список объектов Point.
            day_number: Номер дня.
            dist_matrix_df: Матрица расстояний.
            nn_dist_map: Расстояния от базы.
            nn_lat: Широта базы.
            nn_lon: Долгота базы.
            days: Общее количество дней.
            show_routes: Показывать линии маршрута.
            zoom_start: Начальный масштаб.
            title: Заголовок карты.
            save_path: Путь для сохранения.

        Returns:
            Optional[str]: Путь к сохранённому HTML-файлу или None при ошибке.
        """
        try:
            # Создаём DataFrame с координатами из points
            coords_data = []
            for p in points:
                coords_data.append({
                    'point_id': p.point_id,
                    'lat': p.lat,
                    'lon': p.lon
                })
            df_coords = pd.DataFrame(coords_data)

            # Фильтруем нужный день и подтягиваем координаты
            day_df = routes_df[routes_df['visit_day'] == day_number].merge(
                df_coords[['point_id', 'lat', 'lon']],
                on='point_id',
                how='left'
            ).dropna(subset=['lat', 'lon']).sort_values('order_in_route')

            if len(day_df) == 0:
                print(f"День {day_number} пустой!")
                return None
            # Собираем словарь для быстрого доступа к координатам точек
            point_coords = {row['point_id']: (row['lat'], row['lon']) for _, row in day_df.iterrows()}
            
            # Создаём pid_to_local для этого дня
            day_pids = day_df['point_id'].tolist()
            pid_to_local = {pid: i for i, pid in enumerate(day_pids)}
            
            # Берём подматрицу расстояний для точек этого дня
            # Находим индексы точек в полной матрице
            day_indices = []
            for pid in day_pids:
                # Ищем точку в self.points по point_id
                for p in points:
                    if p.point_id == pid:
                        day_indices.append(p.orig_idx)
                        break
            
            # Извлекаем подматрицу
            if day_indices:
                sub_matrix = dist_matrix_df.iloc[day_indices, day_indices].values
            else:
                sub_matrix = None
            
            # Рассчитываем расстояние по дорогам
            if nn_dist_map and sub_matrix is not None:
                total_distance = self.estimate_day_distance_exact(
                    day_df, sub_matrix, pid_to_local, nn_dist_map
                )
            else:
                # Fallback: считаем геодезически
                from geopy.distance import geodesic
                total_distance = 0.0
                pids = ['BASE'] + day_df.sort_values('order_in_route')['point_id'].tolist()
                for i in range(len(pids) - 1):
                    if pids[i] == 'BASE':
                        p1 = (nn_lat, nn_lon)
                    else:
                        p1 = point_coords.get(pids[i], (0, 0))
                    p2 = point_coords.get(pids[i+1], (0, 0))
                    if p1 != (0, 0) and p2 != (0, 0):
                        total_distance += geodesic(p1, p2).km


            # Центр карты — по точкам дня (с учётом базы)
            all_lats = list(day_df['lat']) + [nn_lat]
            all_lons = list(day_df['lon']) + [nn_lon]
            center_lat = sum(all_lats) / len(all_lats)
            center_lon = sum(all_lons) / len(all_lons)

            # Создаём карту
            self.map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_start,
                control_scale=True
            )

            folium.TileLayer(
                'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            ).add_to(self.map)

            # Цвет для выбранного дня
            cmap = plt.colormaps['hsv']
            color = mcolors.to_hex(cmap((day_number - 1) / max(1, days)))

            # База (Нижний Новгород)
            folium.Marker(
                location=[nn_lat, nn_lon],
                popup='<b> База: Нижний Новгород</b>',
                tooltip='База',
                icon=folium.Icon(color='black', icon='home', prefix='fa')
            ).add_to(self.map)

            # Заголовок карты
            title_text = title if title else f'Маршрут дня {day_number}'
            title_html = (
                f'<div style="position: fixed; top: 15px; left: 50%; transform: translateX(-50%);'
                f' background-color: white; border: 2px solid grey; z-index: 9999;'
                f' font-size: 14px; padding: 6px 16px; border-radius: 8px;'
                f' box-shadow: 0 2px 8px rgba(0,0,0,0.2);">'
                f'<b>{title_text}</b></div>'
            )
             # Заголовок с информацией о длине маршрута
            title_text = title if title else f'Маршрут дня {day_number}'
            title_html = f'''
            <div style="position: fixed; top: 15px; left: 50%; transform: translateX(-50%);
                        background-color: white; border: 2px solid grey; z-index: 9999;
                        font-size: 14px; padding: 10px 20px; border-radius: 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2); text-align: center;">
                <b>{title_text}</b><br>
                <span style="font-size: 12px; color: #555;">
                     Общая длина: <b>{total_distance:.1f} км</b> 
                    ({len(day_df)} точек)
                </span>
            </div>
            '''
            # Линия маршрута
            if show_routes:
                route_coords = [[nn_lat, nn_lon]]
                for _, row in day_df.iterrows():
                    route_coords.append([row['lat'], row['lon']])

                folium.PolyLine(
                    locations=route_coords,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f'Маршрут дня {day_number}'
                ).add_to(self.map)

            # Точки
            for _, row in day_df.iterrows():
                popup_text = f"""
                <b>ID точки:</b> {row['point_id']}<br>
                <b>День:</b> {day_number}<br>
                <b>Порядок:</b> {int(row['order_in_route'])}<br>
                <b>Кластер:</b> {int(row['cluster_id'])}<br>
                <b>Координаты:</b><br>
                {row['lat']:.6f}, {row['lon']:.6f}
                """

                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"#{int(row['order_in_route'])}",
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    weight=2
                ).add_to(self.map)

                # Номер порядка поверх точки
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size: 10px; color: white; font-weight: bold; '
                            f'text-align: center; background-color: rgba(0,0,0,0.5); '
                            f'border-radius: 10px; padding: 0px 4px; '
                            f'margin-top: -5px;">{int(row["order_in_route"])}</div>'
                        )
                    )
                ).add_to(self.map)

            # Информационная панель
            # Формируем последовательность точек для отображения
            order_sequence = ' → '.join(map(str, day_df['order_in_route'].tolist()))
            
            info_panel_html = f'''
            <div style="position: fixed; bottom: 30px; right: 30px; width: 200px;
                        background-color: white; border: 2px solid {color}; z-index: 9999;
                        font-size: 11px; padding: 12px; border-radius: 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
                <b>📊 День {day_number}</b><br>
                <hr style="margin: 5px 0;">
                <b>Точек:</b> {len(day_df)}<br>
                <b>Расстояние:</b> {total_distance:.1f} км<br>
                <hr style="margin: 5px 0;">
                <b>Порядок:</b><br>
                <span style="font-size: 10px;">База → {order_sequence}</span>
            </div>
            '''
            # Сохраняем
            if save_path is None:
                save_path = os.path.join(config.MAPS_DIR, f'routes_day_{day_number}.html')

            self.map.save(save_path)

            # Вшиваем заголовок прямо в <body>
            with open(save_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            html_content = html_content.replace('</body>', title_html + '\n</body>')
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"Карта дня {day_number} сохранена: {save_path}")
            return save_path

        except Exception as e:
            print(f"ОШИБКА при создании карты дня {day_number}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Функция для точного расчёта расстояния
    @staticmethod
    def estimate_day_distance_exact(
        day_df: pd.DataFrame,
        dist_matrix: Optional[np.ndarray],
        pid_to_local: Dict[str, int],
        nn_dist_map: Dict[str, float]
    ) -> float:
        """
        Точная оценка маршрута по матрице расстояний.

        Args:
            day_df: DataFrame с данными дня.
            dist_matrix: Матрица расстояний.
            pid_to_local: Словарь для преобразования point_id в индекс.
            nn_dist_map: Расстояния от базы.

        Returns:
            float: Общая длина маршрута в км.
        """
        pids = ['BASE'] + day_df.sort_values('order_in_route')['point_id'].tolist()
        total = 0
        for i in range(len(pids) - 1):
            if pids[i] == 'BASE':
                total += nn_dist_map.get(pids[i+1], 0)
            else:
                if dist_matrix is not None:
                    total += dist_matrix[pid_to_local[pids[i]]][pid_to_local[pids[i+1]]]
        return total / 1000  # в км