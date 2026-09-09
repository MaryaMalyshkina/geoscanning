import csv
import math
from typing import List, Optional, Union

class Point:
    """Класс для представления точки с геоданными и кластером"""
    
    def __init__(self, point_id: str, manager: int, lat: float, lon: float, n_visits: int, cluster: int = -1, orig_idx: Optional[int] = None):
        """
        Инициализация точки
        
        Args:
            point_id: Уникальный идентификатор точки
            manager: ID менеджера
            lat: Широта
            lon: Долгота
            n_visits: Количество визитов
            cluster: Номер кластера (по умолчанию -1, если не рассчитан)
            orig_idx: Индекс в матрице расстояний (позиция в глобальном списке точек)
        """
        self.point_id = point_id
        self.manager = manager
        self.lat = lat
        self.lon = lon
        self.n_visits = n_visits
        self.cluster = cluster  
        self.orig_idx = orig_idx
    
    @classmethod
    def from_csv(cls, file_path: str, has_header: bool = True, delimiter: str = ',') -> List['Point']:
        """
        Создание списка точек из CSV-файла
        
        Args:
            file_path: Путь к CSV-файлу
            has_header: Есть ли заголовок в файле
            delimiter: Разделитель полей
            
        Returns:
            List[Point]: Список созданных точек
            
        Raises:
            ValueError: При некорректных данных
            FileNotFoundError: Если файл не найден
        """
        points = []
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=delimiter)
                
                # Пропускаем заголовок, если он есть
                if has_header:
                    header = next(reader, None)
                    if header:
                        # Проверяем наличие необходимых колонок
                        expected_cols = ['point_id', 'manager', 'lat', 'lon', 'n_visits']
                        if not all(col in header for col in expected_cols):
                            print(f"Предупреждение: заголовок не совпадает с ожидаемым. "
                                  f"Ожидается: {expected_cols}, получено: {header}")
                
                # Читаем строки с данными
                for row_num, row in enumerate(reader, start=2 if has_header else 1):
                    try:
                        # Проверка на пустую строку
                        if not row or all(cell.strip() == '' for cell in row):
                            continue
                        
                        # Проверка количества полей
                        if len(row) < 5:
                            errors.append(f"Строка {row_num}: недостаточно полей ({len(row)} вместо 5)")
                            continue
                        
                        # Извлекаем и очищаем данные
                        point_id = row[0].strip()
                        manager_str = row[1].strip()
                        lat_str = row[2].strip()
                        lon_str = row[3].strip()
                        n_visits_str = row[4].strip()
                        
                        # Проверка на пустые значения
                        if not point_id:
                            errors.append(f"Строка {row_num}: пустой point_id")
                            continue
                        
                        if not manager_str:
                            errors.append(f"Строка {row_num}: пустой manager для ID={point_id}")
                            continue
                            
                        if not lat_str:
                            errors.append(f"Строка {row_num}: пустая широта для ID={point_id}")
                            continue
                            
                        if not lon_str:
                            errors.append(f"Строка {row_num}: пустая долгота для ID={point_id}")
                            continue
                            
                        if not n_visits_str:
                            errors.append(f"Строка {row_num}: пустое n_visits для ID={point_id}")
                            continue
                        
                        # Преобразование типов с проверкой
                        try:
                            manager = int(manager_str)
                        except ValueError:
                            errors.append(f"Строка {row_num}: некорректный manager '{manager_str}' для ID={point_id}")
                            continue
                        
                        try:
                            lat = float(lat_str)
                        except ValueError:
                            errors.append(f"Строка {row_num}: некорректная широта '{lat_str}' для ID={point_id}")
                            continue
                        
                        try:
                            lon = float(lon_str)
                        except ValueError:
                            errors.append(f"Строка {row_num}: некорректная долгота '{lon_str}' для ID={point_id}")
                            continue
                        
                        try:
                            n_visits = int(n_visits_str)
                        except ValueError:
                            errors.append(f"Строка {row_num}: некорректное n_visits '{n_visits_str}' для ID={point_id}")
                            continue
                        
                        # Проверка диапазонов координат
                        if not (-90 <= lat <= 90):
                            errors.append(f"Строка {row_num}: широта {lat} вне допустимого диапазона [-90, 90] для ID={point_id}")
                            continue
                        
                        if not (-180 <= lon <= 180):
                            errors.append(f"Строка {row_num}: долгота {lon} вне допустимого диапазона [-180, 180] для ID={point_id}")
                            continue
                        
                        # Проверка положительности n_visits
                        if n_visits < 0:
                            errors.append(f"Строка {row_num}: n_visits={n_visits} не может быть отрицательным для ID={point_id}")
                            continue
                        
                        # Создаем точку с кластером по умолчанию -1
                        point = cls(point_id, manager, lat, lon, n_visits, cluster=-1)
                        points.append(point)
                        
                    except Exception as e:
                        errors.append(f"Строка {row_num}: неизвестная ошибка - {str(e)}")
                        continue
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {file_path} не найден")
        except Exception as e:
            raise Exception(f"Ошибка при чтении файла: {str(e)}")
        
        # Выводим все ошибки, если они есть
        if errors:
            print("\n".join(errors))
            print(f"\nВсего ошибок: {len(errors)}")
        
        if not points:
            raise ValueError("Не удалось загрузить ни одной корректной точки из файла")
        
        print(f"Загружено точек: {len(points)}")
        return points
    
    def __repr__(self) -> str:
        """Строковое представление точки"""
        return (f"Point(ID={self.point_id}, manager={self.manager}, "
                f"lat={self.lat:.6f}, lon={self.lon:.6f}, "
                f"n_visits={self.n_visits}, cluster={self.cluster})")
    
    def to_dict(self) -> dict:
        """Преобразование точки в словарь"""
        return {
            'point_id': self.point_id,
            'manager': self.manager,
            'lat': self.lat,
            'lon': self.lon,
            'n_visits': self.n_visits,
            'cluster': self.cluster
        }
    


