"""
Главный модуль приложения для кластеризации и построения маршрутов.

Содержит класс MainWindow - главное окно приложения с логикой работы
с данными, кластеризацией, построением маршрутов и визуализацией на карте.
"""
import sys
import os
import random
import logging
from typing import Optional

import pandas as pd
import numpy as np
import folium

from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMessageBox,
    QTableWidgetItem
)
from PyQt5.QtCore import QUrl

from ui.ui_geoscan import Ui_Dialog
from src.point import Point
from src.clustering import ImprovedClusterer
from src.utils import (
    fill_points_table,
    update_clusters_in_table,
    fill_routes_table
)
from src.routing import RouteBuilder
from src.visualize import MapGenerator
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Пути для выходных файлов
OUTPUT_DIR = "output"
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")
ROUTES_DIR = os.path.join(OUTPUT_DIR, "routes")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")

# Создаём папки при запуске
for dir_path in [OUTPUT_DIR, MAPS_DIR, ROUTES_DIR, STATS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Устанавливаем seed для воспроизводимости
def set_random_seed(seed: int = config.RANDOM_SEED)-> None:
    """
    Устанавливает seed для воспроизводимости результатов.

    Args:
        seed: Число для инициализации генератора случайных чисел.
    """
    random.seed(seed)
    np.random.seed(seed)

set_random_seed()

class MainWindow(QDialog):
    """
    Главное окно приложения.

    Управляет загрузкой данных, кластеризацией точек, построением маршрутов
    и их визуализацией на карте.

    Attributes:
        points (List[Point]): Список загруженных точек.
        clustered_csv_path (str): Путь к сохранённому CSV с кластерами.
        clusterer (ImprovedClusterer): Объект для кластеризации.
        dist_matrix (pd.DataFrame): Матрица расстояний между точками.
        nn_dist_map (Dict[str, float]): Расстояния от базы до каждой точки.
        routes_data (Dict[int, pd.DataFrame]): Маршруты по менеджерам.
        routes_stats (Dict[int, pd.DataFrame]): Статистика маршрутов по менеджерам.
        map_gen (MapGenerator): Генератор карт.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.points = []  
        self.clustered_csv_path = None  
        self.clusterer = ImprovedClusterer(max_points_per_cluster=12)

        self.dist_matrix = pd.read_csv(config.DISTANCE_MATRIX_PATH, index_col=0)
        self.dist_matrix.index = self.dist_matrix.index.astype(int)
        self.dist_matrix.columns = self.dist_matrix.columns.astype(int)

        # Храним расстояния от НН
        self.nn_dist_map = None

        # Хранилище маршрутов для каждого менеджера
        self.routes_data = {}   # {manager_id: DataFrame с маршрутами}
        self.routes_stats = {}  # {manager_id: DataFrame со статистикой}

        self.map_gen = MapGenerator()

        # Подключаем кнопки
        self.ui.btnSelectFile.clicked.connect(self.on_select_file)
        self.ui.btnCount.clicked.connect(self.on_calculate_and_cluster)
        self.ui.btnBuildRoute.clicked.connect(self.on_build_routes)
        self.ui.btnDays.clicked.connect(self.on_view_day_routes)
        self.ui.cboxDays.currentIndexChanged.connect(self.on_day_changed)
        
        # Изначально комбобокс дня неактивен
        self.ui.cboxDays.setEnabled(False)
        self.ui.cboxDays.clear()

        self.ui.comboBoxManager.currentIndexChanged.connect(self.on_manager_changed)

        # Загружаем пустую карту
        self.load_empty_map()

    def load_empty_map(self) -> None:
        """Загружает пустую карту OpenStreetMap."""
        m = folium.Map(location=[55.75, 37.61], zoom_start=10)
        html_path = os.path.join(config.MAPS_DIR, 'empty_map.html')
        m.save(html_path)
        self.ui.widget.load(QUrl.fromLocalFile(os.path.abspath(html_path)))

    def load_map_in_widget(self, html_path) -> None:
        """
        Загружает HTML-карту в виджет.

        Args:
            html_path: Путь к HTML-файлу с картой.
        """
        if html_path and os.path.exists(html_path):
            self.ui.widget.load(QUrl.fromLocalFile(os.path.abspath(html_path)))

    def on_select_file(self) -> None:
        """Обработчик выбора CSV-файла с данными."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV-файл", "", "CSV files (*.csv);;All files (*.*)"
        )
        if file_path:
            self.ui.txtInputFile.setText(file_path)
            self.clustered_csv_path = None
            self.ui.btnBuildRoute.setEnabled(False)
            self.routes_data.clear()
            self.routes_stats.clear()

    def on_calculate_and_cluster(self) -> None:
        """
        Обработчик кнопки "Расчет кластеров".

        Выполняет загрузку данных и кластеризацию.
        После успешной кластеризации активирует кнопку построения маршрутов.
        """
        self.on_calculate()
        if self.points:
            # После кластеризации активируем кнопку построения маршрутов
            self.ui.btnBuildRoute.setEnabled(True)

    def on_calculate(self) -> None:
        """
        Обработка данных: загрузка, кластеризация, сохранение результатов.

        Загружает точки из CSV, выполняет кластеризацию, сохраняет результаты
        и обновляет интерфейс.

        Raises:
            Exception: При ошибках загрузки или обработки данных.
        """
        file_path = self.ui.txtInputFile.toPlainText().strip()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите CSV-файл!")
            return

        try:
            self.points = Point.from_csv(file_path)
            for idx, p in enumerate(self.points):
                p.orig_idx = idx

            # Загружаем расстояния от НН
            if os.path.exists(config.NN_DISTANCES_PATH):
                nn_dists_km = np.load(config.NN_DISTANCES_PATH, allow_pickle=True)
                self.nn_dist_map = {
                    self.points[i].point_id: float(nn_dists_km[i])
                    for i in range(len(self.points)) if i < len(nn_dists_km)
                }
            else:
                self.nn_dist_map = None

            fill_points_table(self.ui.tblPoints, self.points)

            # Заполняем комбобокс только менеджерами (без "Все")
            managers = sorted(set(p.manager for p in self.points))
            self.ui.comboBoxManager.clear()
            for m in managers:
                self.ui.comboBoxManager.addItem(str(m))

            # Кластеризация
            self.points = self.clusterer.cluster_by_manager(self.points)
            update_clusters_in_table(self.ui.tblPoints, self.points)

            # Сохраняем CSV с кластерами
            self.clustered_csv_path = self._save_points_with_clusters(file_path)

            # Показываем карту с кластерами для первого менеджера (если есть)
            if self.clustered_csv_path:
                self.update_map()  # покажет кластеры для выбранного (по умолчанию первого)
                # После загрузки очищаем старые маршруты
                self.routes_data.clear()
                self.routes_stats.clear()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось обработать файл:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def update_map(self) -> None:
        """
        Обновляет карту, показывая кластеры для выбранного менеджера.

        Использует текущий выбор в комбобоксе менеджеров для фильтрации.
        """
        if not self.clustered_csv_path:
            return
        selected = self.ui.comboBoxManager.currentText()
        if not selected:
            return
        try:
            selected_manager = int(selected)
        except ValueError:
            return

        html_path = self.map_gen.create_map_from_csv(
            self.clustered_csv_path,
            lat_col='lat',
            lon_col='lon',
            name_col='point_id',
            group_col='manager',
            cluster_col='cluster',
            selected_manager=selected_manager
        )
        if html_path:
            self.load_map_in_widget(html_path)

    def on_manager_changed(self) -> None:
        """
        Обработчик изменения выбранного менеджера.

        При выборе менеджера показывает его маршрут (если построен)
        или кластеры (если маршрут ещё не построен).
        """
        if not self.points or not self.clustered_csv_path:
            return

        selected = self.ui.comboBoxManager.currentText()
        if not selected:
            return
        try:
            mgr_id = int(selected)
        except ValueError:
            return
        # Сбрасываем режим просмотра по дням
        self.ui.cboxDays.setEnabled(False)
        self.ui.cboxDays.setCurrentIndex(0)  # "Все дни"

        # Если есть маршрут для этого менеджера – показываем его
        if mgr_id in self.routes_data:
            self.show_routes_on_map(self.routes_data[mgr_id])
        else:
            # Иначе показываем кластеры
            self.update_map()

    def _save_points_with_clusters(self, original_file_path) -> Optional[str]:
        """
        Сохраняет точки с присвоенными кластерами в CSV-файл.

        Args:
            original_file_path: Путь к исходному CSV-файлу.

        Returns:
            str: Путь к сохранённому файлу или None при ошибке.
        """
        if not self.points:
            return None
        try:
            data = [{
                'point_id': p.point_id,
                'manager': p.manager,
                'lat': p.lat,
                'lon': p.lon,
                'n_visits': p.n_visits,
                'cluster': p.cluster
            } for p in self.points]
            df = pd.DataFrame(data)
            base_name = os.path.splitext(os.path.basename(original_file_path))[0]
            new_path = os.path.join(config.ROUTES_DIR, f"{base_name}_clustered.csv")
            df.to_csv(new_path, index=False, encoding='utf-8')
            return new_path
        except Exception as e:
            print(f"Ошибка сохранения CSV с кластерами: {e}")
            return None

    def on_build_routes(self) -> None:
        """
        Обработчик кнопки "Построить маршруты".

        Строит маршруты для всех менеджеров, сохраняет результаты
        и обновляет интерфейс (таблицу, карту, комбобокс дней).
        """
        if not self.points:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите данные!")
            return
        if not any(p.cluster >= 0 for p in self.points):
            QMessageBox.warning(self, "Ошибка", "Выполните кластеризацию (нажмите 'Расчет')!")
            return

        managers = sorted(set(p.manager for p in self.points))
        builder = RouteBuilder(
            dist_matrix_df=self.dist_matrix,
            nn_dist_map=self.nn_dist_map,
            days=config.DAYS,
            capacity=config.MAX_VISITS_PER_DAY,
            max_dist_km=config.MAX_DISTANCE_FROM_BASE_KM,
            nn_lat=config.NN_LAT,
            nn_lon=config.NN_LON,
            random_seed=config.RANDOM_SEED 
        )

        self.routes_data.clear()
        self.routes_stats.clear()
        all_stats = []
        all_routes_list = []  # для объединения всех маршрутов в одну таблицу

        for mgr in managers:
            result_df, stats_df = builder.build(self.points, manager_id=mgr)
            if not result_df.empty:
                self.routes_data[mgr] = result_df
                self.routes_stats[mgr] = stats_df
                base_name = os.path.splitext(os.path.basename(self.clustered_csv_path or "routes"))[0]
                routes_path = os.path.join(config.ROUTES_DIR, f"{base_name}_routes_{mgr}.csv")
                stats_path = os.path.join(config.STATS_DIR, f"{base_name}_stats_{mgr}.csv")
                result_df.to_csv(routes_path, index=False, encoding='utf-8')
                stats_df.to_csv(stats_path, index=False, encoding='utf-8')
                all_stats.append(stats_df)
                all_routes_list.append(result_df)
                logger.info(f"Менеджер {mgr}: {len(result_df)} визитов, {len(stats_df)} дней")
            else:
                logger.warning(f"Менеджер {mgr}: маршрут не построен")

        if not self.routes_data:
            QMessageBox.information(self, "Информация", "Нет маршрутов для построения.")
            return

        if all_routes_list:
            # Объединяем все маршруты в один DataFrame
            combined_routes = pd.concat(all_routes_list, ignore_index=True)
            
            # Добавляем колонку с менеджером (если её нет)
            if 'manager' not in combined_routes.columns:
                # Добавляем менеджера из self.points
                manager_map = {p.point_id: p.manager for p in self.points}
                combined_routes['manager'] = combined_routes['point_id'].map(manager_map)
            
            # Переупорядочиваем колонки для удобства
            cols = ['manager', 'visit_day', 'cluster_id', 'order_in_route', 'point_id']
            combined_routes = combined_routes[cols]
         
            fill_routes_table(self.ui.tblPoints, combined_routes)
            QMessageBox.information(self, "Ура!", "Построение маршрутов прошло успешно!")

            # Показать маршрут для текущего выбранного менеджера (или первого)
            current_selected = self.ui.comboBoxManager.currentText()
            if current_selected:
                try:
                    mgr_id = int(current_selected)
                    if mgr_id in self.routes_data:
                        self.show_routes_on_map(self.routes_data[mgr_id])
                    else:
                        # Если для выбранного нет маршрута – показываем первого
                        first_mgr = next(iter(self.routes_data))
                        self.ui.comboBoxManager.setCurrentText(str(first_mgr))
                        self.show_routes_on_map(self.routes_data[first_mgr])
                except:
                    first_mgr = next(iter(self.routes_data))
                    self.ui.comboBoxManager.setCurrentText(str(first_mgr))
                    self.show_routes_on_map(self.routes_data[first_mgr])
            else:
                first_mgr = next(iter(self.routes_data))
                self.ui.comboBoxManager.setCurrentText(str(first_mgr))
                self.show_routes_on_map(self.routes_data[first_mgr])
            if self.routes_data:
            # После построения маршрутов заполняем комбобокс днями
                self._update_day_combo()

    def show_routes_on_map(self, routes_df) -> None:
        """
        Отображает маршруты на карте с цветовой схемой по дням.

        Args:
            routes_df: DataFrame с маршрутами.
        """
        # Определяем название для заголовка
        current_selected = self.ui.comboBoxManager.currentText()
        title = f"Маршруты для менеджера {current_selected}" if current_selected else "Маршруты"
        
        # Создаём карту с помощью MapGenerator
        html_path = self.map_gen.draw_map_by_days(
            routes_df=routes_df,
            points=self.points,
            nn_lat=config.NN_LAT,
            nn_lon=config.NN_LON,
            days=config.DAYS,
            show_routes=True,
            zoom_start=8,
            title=title
        )
        
        if html_path:
            self.load_map_in_widget(html_path)
        else:
            print("Не удалось создать карту маршрутов")

    def _update_day_combo(self) -> None:
        """
        Обновляет комбобокс с днями на основе построенных маршрутов.

        Добавляет все уникальные дни из маршрутов всех менеджеров.
        """
        self.ui.cboxDays.clear()
        self.ui.cboxDays.addItem("Все дни")
        
        # Собираем все дни из всех маршрутов
        all_days = set()
        for mgr, routes_df in self.routes_data.items():
            days = routes_df['visit_day'].unique()
            all_days.update(days)
        
        # Добавляем дни в комбобокс
        for day in sorted(all_days):
            self.ui.cboxDays.addItem(str(day))
        
        # Активируем комбобокс, только если есть дни
        self.ui.cboxDays.setEnabled(len(all_days) > 0)

    def on_view_day_routes(self) -> None:
        """
        Обработчик кнопки "Просмотр по дням".

        Включает/выключает режим просмотра маршрутов по отдельным дням.
        """
        if not self.routes_data:
            QMessageBox.warning(self, "Ошибка", "Сначала постройте маршруты!")
            return
        
        # Переключаем состояние комбобокса
        current_state = self.ui.cboxDays.isEnabled()
        self.ui.cboxDays.setEnabled(not current_state)
        
        if not current_state:
            # Включаем режим - показываем первый день или все дни
            self.on_day_changed()
        else:
            # Выключаем режим - возвращаемся к обычному отображению
            current_selected = self.ui.comboBoxManager.currentText()
            if current_selected:
                try:
                    mgr_id = int(current_selected)
                    if mgr_id in self.routes_data:
                        self.show_routes_on_map(self.routes_data[mgr_id])
                except:
                    pass

    def on_day_changed(self) -> None:
        """
        Обработчик изменения выбранного дня.

        Показывает маршрут для выбранного дня.
        """
        if not self.routes_data:
            return
        
        selected_day_text = self.ui.cboxDays.currentText()
        
        if selected_day_text == "Все дни" or not selected_day_text:
            # Показываем все маршруты для текущего менеджера
            current_selected = self.ui.comboBoxManager.currentText()
            if current_selected:
                try:
                    mgr_id = int(current_selected)
                    if mgr_id in self.routes_data:
                        self.show_routes_on_map(self.routes_data[mgr_id])
                except:
                    pass
            return
        
        try:
            selected_day = int(selected_day_text)
            current_selected = self.ui.comboBoxManager.currentText()
            
            # Определяем, для какого менеджера показывать
            if current_selected:
                try:
                    mgr_id = int(current_selected)
                    if mgr_id in self.routes_data:
                        # Фильтруем маршруты только для этого менеджера и дня
                        day_routes = self.routes_data[mgr_id]
                        day_df = day_routes[day_routes['visit_day'] == selected_day]
                        
                        if day_df.empty:
                            QMessageBox.information(self, "Информация", 
                                f"Нет маршрутов для менеджера {mgr_id} в день {selected_day}")
                            return
                        
                        # Создаём карту для одного дня через MapGenerator
                        html_path = self.map_gen.draw_map_single_day(
                            routes_df=day_df,
                            points=self.points,
                            day_number=selected_day,
                            dist_matrix_df=self.dist_matrix, 
                            nn_dist_map=self.nn_dist_map or {},   
                            nn_lat=config.NN_LAT,
                            nn_lon=config.NN_LON,
                            days=config.DAYS,
                            show_routes=True,
                            zoom_start=8,
                            title=f"Менеджер {mgr_id}, день {selected_day}"
                        )
                        
                        if html_path:
                            self.load_map_in_widget(html_path)
                        else:
                            QMessageBox.warning(self, "Ошибка", 
                                f"Не удалось создать карту для дня {selected_day}")
                        return
                except ValueError:
                    pass
            
            # Если не выбран конкретный менеджер - показываем все маршруты за этот день
            day_routes_list = []
            for mgr, routes_df in self.routes_data.items():
                day_df = routes_df[routes_df['visit_day'] == selected_day]
                if not day_df.empty:
                    day_routes_list.append(day_df)
            
            if not day_routes_list:
                QMessageBox.information(self, "Информация", 
                    f"Нет маршрутов для дня {selected_day}")
                return
            
            combined_day_routes = pd.concat(day_routes_list, ignore_index=True)
            
            # Создаём карту для одного дня через MapGenerator
            html_path = self.map_gen.draw_map_single_day(
                routes_df=combined_day_routes,
                points=self.points,
                day_number=selected_day,
                nn_lat=config.NN_LAT,
                nn_lon=config.NN_LON,
                days=config.DAYS,
                show_routes=True,
                zoom_start=8,
                title=f"Все менеджеры, день {selected_day}"
            )
            
            if html_path:
                self.load_map_in_widget(html_path)
            else:
                QMessageBox.warning(self, "Ошибка", 
                    f"Не удалось создать карту для дня {selected_day}")
                    
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Неверный номер дня")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())