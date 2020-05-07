"""
Этот метод построения заточен на следующую эвристику:
Для построения кластеров мы откинем ребра меньшие по весу чем 0.75,
так как они практически не учавствуют в построении минимального остовного графа.
(Это верно лишь при достаточно большом количестве имен: ~10^4)
Но малые "далекие" кластеры будут все равно учтены при построении.
Возможны потери "одиночек", но у них и так нет соседей, на практике < 0.5% """

from tqdm import tqdm
from pyjarowinkler import distance
from collections import deque
import pandas as pd 
import numpy as np
import concurrent.futures
import functools


class Batch:

    def __init__(self, from_vertix: int, name: str, length: int) -> None:
        """Инициализация"""
        self.vertix = from_vertix
        self.name = name
        self.edges = np.zeros((length, 1), dtype=np.uint16)
        self.dist = np.zeros((length, 1), np.float32)


    def fit(self, fit_data: 'DataFrame') -> None:
        """Считаем значения на вход подавая индексы и названия"""
        counter = 0
        for idx, row in fit_data.iterrows():
            # Переворот расстояния, так как алгоритм будет минимизировать
            word_distance = 1 - distance.get_jaro_distance(self.name, row['name'])
            if word_distance < 0.25:
                self.edges[counter] = row['index']
                self.dist[counter] = word_distance
                counter += 1
        self.true_len = counter


def iteration(index, data, rows):
    cur_name = data.loc[index, 'name']
    length = rows - index
    to_fit = data.iloc[index+1: ]

    cur_batch = Batch(index, cur_name, length)
    cur_batch.fit(to_fit)

    return cur_batch


if __name__ == '__main__':
    # Read data & Iniziate
    excel = pd.read_excel('mdm_brands_tree_full_fincode.xlsx')
    brends = excel.loc[:,'name'].reset_index()
    brends['name'] = brends['name'].astype(str).str.strip().str.lower()
    total_rows = brends.shape[0] - 1 # Последний ни с чем не надо сравнивать
    indexes = np.arange(total_rows)
    results = deque()

    # Calculate weights
    print('Начали считать веса')
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for ind, obj in tqdm(zip(indexes, executor.map(functools.partial(iteration, data=brends, rows=total_rows), indexes)), total=len(indexes)):
            results.append(obj)

    # Calculate cumulative row count
    print('Считаем количество ребер в графе')
    cumul_count = np.zeros((total_rows + 1, 1), dtype=np.int32) # Выделим память
    accum = 0
    for idx in tqdm(range(total_rows)):
        assert idx == results[idx].vertix # Вершины должны быть упорядочены
        cumul_count[idx] = accum
        accum += results[idx].true_len
    cumul_count[-1] = accum # Последняя запись без исходящих вершин

    # Concatenate results
    print('Собираем результаты в одну структуру')
    total_edges = np.zeros((accum, 2), dtype=np.uint16)
    total_weights = np.zeros((accum, 1), dtype=np.float32)

    my_iter = 0
    for idx in tqdm(range(total_rows)):
        obj = results.popleft()
        n_rows = obj.true_len
        if n_rows > 0:
            total_edges[my_iter: my_iter + n_rows, 0] = obj.vertix
            total_edges[my_iter: my_iter + n_rows, 1] = obj.edges[:n_rows].ravel()
            total_weights[my_iter: my_iter + n_rows] = obj.dist[:n_rows]
            my_iter += n_rows

    # Saving results
    print('Сохраняем результаты')
    pd.DataFrame(total_edges, columns=['from', 'to']).to_csv('graph_vert.csv', index=None)
    pd.DataFrame(total_weights, columns=['dist']).to_csv('graph_dist.csv', index=None)
    pd.DataFrame(cumul_count, columns=['count_cumul']).to_csv('graph_counts.csv', index=None)