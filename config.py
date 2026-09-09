import os

# Ограничения
DAYS = 22
MAX_VISITS_PER_DAY = 12
MAX_DISTANCE_FROM_BASE_KM = 150

# База
NN_LAT = 56.326887
NN_LON = 44.005986

# Кластеризация
EPS_METERS = 4000  # порог расстояния в метрах для иерархической кластеризации

# Пути 
DISTANCE_MATRIX_PATH = 'data/distance_matrix_undirected.csv'
NN_DISTANCES_PATH = 'data/nn_dists_km.npy'

# Базовые пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Выходные данные
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")
ROUTES_DIR = os.path.join(OUTPUT_DIR, "routes")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")

# Воспроизводимость
RANDOM_SEED = 42 