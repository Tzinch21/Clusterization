from tqdm import tqdm
import pandas as pd 
import numpy as np
import concurrent.futures
import functools
from collections import deque
from datetime import datetime

class Cluster:
    
    def __init__(self, max_vertix_count, shape):
        self.curr_idx = 0
        self.ostov_edges = np.zeros((max_vertix_count, 2), dtype=np.uint16) # Храним число вершин
        self.ostov_dist = np.zeros((max_vertix_count, 1), dtype=np.float32) # Храним расстояние между ними
        self.vert_added = deque() # Вершины, что уже добавили
        self.mask = np.zeros(shape[0]).astype(np.bool) # Маска доступных элементов
        self.inv_mask_shutdown = np.ones(shape[0]).astype(np.bool) # Маска выключателей (циклов графа)

    
    def blinking(self, forw_counter, backw_counter, mapper_idx, vertix, limits, values):
        # Включаем новые вершины
        switch_on = extract_indexes(vertix, forw_counter, backw_counter, mapper_idx, limits)
        self.mask[switch_on] = True
        # Исключаем циклы
        switch_off_total = calculate_cycles(self.vert_added, vertix, forw_counter, backw_counter, mapper_idx, limits, values)
        switch_off_true = [i for i in switch_off_total if i is not None]
        if len(switch_off_true) > 0:
            self.inv_mask_shutdown[switch_off_true] = False
        # Мы добавили вершину
        self.vert_added.append(vertix)
        # Обновляем маску доступных элементов
        self.mask = self.mask & self.inv_mask_shutdown

    
    def add_first_edge(self, vert_avail, dist_avail, edges_counter, reversed_edges_counter, reversed_indexes, limits):
        step, weight = get_minimal_edge(vert_avail, dist_avail)
        self.ostov_edges[self.curr_idx, :] = step
        self.ostov_dist[self.curr_idx] = weight
        self.curr_idx += 1
        self.blinking(edges_counter, reversed_edges_counter, reversed_indexes, step[0], limits, vert_avail)
        self.blinking(edges_counter, reversed_edges_counter, reversed_indexes, step[1], limits, vert_avail)


    def add_edge(self, vert, dist, edges_counter, reversed_edges_counter, reversed_indexes, limits):
        vert_avail = vert[self.mask, :]
        dist_avail = dist[self.mask]
        step, weight = get_minimal_edge(vert_avail, dist_avail)
        self.ostov_edges[self.curr_idx, :] = step
        self.ostov_dist[self.curr_idx] = weight
        self.curr_idx += 1
        new_vert = step[1] if step[0] in self.vert_added else step[0]
        self.blinking(edges_counter, reversed_edges_counter, reversed_indexes, new_vert, limits, vert)


def get_minimal_edge(vert, distat):
    if vert.shape[0] != distat.shape[0]:
        raise IndexError('Число ребер и весов к ним должно совпадать')
    min_index = distat.argmin()
    min_val = distat.min()
    return vert[min_index, :], min_val


def calc_forward_indexes(number, forward_counter, supremum):
    """Расчет индексов, по которым находятся ребра графа, где from == number"""
    if number < supremum:
        forward_start = forward_counter[number,0]
        forward_end = forward_counter[number+1,0]
        forward_indexes = np.arange(forward_start, forward_end, dtype=np.int32)
    else:
        forward_indexes = np.array([], dtype=np.int32)
    return forward_indexes


def calc_backward_indexes(number, backward_counter, mapper_back_to_front, infimum):
    """Расчет индексов, по которым находятся ребра графа, где to == number"""
    if number > infimum:
        backward_start = backward_counter[number,0]
        backward_end = backward_counter[number-1,0]
        backward_indexes = np.arange(backward_start, backward_end, dtype=np.int32)
        mapped_indexes = mapper_back_to_front[backward_indexes]
    else:
        mapped_indexes = np.array([], dtype=np.int32)
    return mapped_indexes


def extract_indexes(number, forward_counter, backward_counter, mapper_back_to_front, limits):
    """Расчет индексов, по которым находятся ребра графа, содержащие number"""
    forward_indexes = calc_forward_indexes(number, forward_counter, limits[1])
    backward_indexes = calc_backward_indexes(number, backward_counter, mapper_back_to_front, limits[0])
    return np.hstack([forward_indexes, backward_indexes])
     

def find_pos(val_arr, idx_arr, column_flag, val):
    column = int(column_flag)
    to_check = list(val_arr[idx_arr, column])
    try:
        fake_idx = to_check.index(val)
    except:
        pos = None
    else:
        pos = idx_arr[fake_idx]
    return pos


def calculate_index(n1, n2, forward_counter, backward_counter, mapper_back_to_front, limits, values):
    """Вычисляем позицию для ребра со значениями индексов n1 и n2"""
    if n1 == n2:
        raise Exception('Ребра без петлей')
    elif n1 > n2:
        n1_is_forward_direction = False
        idx_n1 = calc_backward_indexes(n1, backward_counter, mapper_back_to_front, limits[0])
        idx_n2 = calc_forward_indexes(n2, forward_counter, limits[1])
    else:
        n1_is_forward_direction = True
        idx_n1 = calc_forward_indexes(n1, forward_counter, limits[1])
        idx_n2 = calc_backward_indexes(n2, backward_counter, mapper_back_to_front, limits[0])
    
    if len(idx_n1) > len(idx_n2):
        position = find_pos(values, idx_n2, not n1_is_forward_direction, n1)
    else:
        position = find_pos(values, idx_n1, n1_is_forward_direction, n2)
    return position


def calculate_cycles(arr, n, forw_counter, back_counter, mapper_idx, limits, values):
    """Вычисляем для массива индексов все позиции пересечения с новым индексом
    
    n - индекс вершины, которую мы только что добавили в граф
    max_idx - последний индекс вершин из датасета"""
    return [calculate_index(i, n, forw_counter, back_counter, mapper_idx, limits, values) for i in arr]


if __name__ == '__main__':
    # Загружаем данные из предыдущего шага
    edges = pd.read_csv('graph_vert.csv').values.astype(np.uint16)
    dist = pd.read_csv('graph_dist.csv').values.astype(np.float32)
    edges_counter = pd.read_csv('graph_counts.csv').values.astype(np.int32)

    #edges = np.array([[0, 1], [0, 2], [0,3], [1, 2], [1, 3], [4,5], [4,6], [5,6], [7, 8]], dtype=np.uint16)
    #dist = np.array([0.3, 0.1, 0.4, 0.05, 2.4, 2, 6, 1, 0.4], dtype=np.float32).reshape(-1, 1)
    #edges_counter = np.array([0, 3, 5, 5, 5, 7, 8, 8, 9, 9], dtype=np.int32).reshape(-1, 1)

    # Создаем копию-разворот для ускорения вычислений
        # Потратим в 2 раза больше памяти, но выйграем по скорости
    reversed_indexes = np.lexsort((edges[:, 0], edges[:, 1]))[::-1]
    reversed_edges = edges[reversed_indexes, ::-1]

        # Агрегируем, получая количество вершин
    count_it = []
    obj = reversed_edges[0, 0]
    counter = 0
    for row in reversed_edges:
        cur_obj = row[0]
        if cur_obj != obj:
            count_it.append([obj, counter])
            obj = cur_obj
            counter = 0
        counter += 1
    count_it.append([obj, counter])

        # Создаем кумулятивный подсчет сумм реверсированных вершин
    reversed_edges_counter = np.zeros((edges_counter.shape[0]+1, 1))
    summary = 0
    iterator = 0
    for i in range(edges_counter.shape[0], -1, -1):
        reversed_edges_counter[i] = summary
        if iterator < len(count_it):
            if count_it[iterator][0] == i:
                summary += count_it[iterator][1]
                iterator += 1
    
    # Количество вершин
    total_vertix_count = len(set(edges[:, 0]) | set(edges[:, 1]))
    limits = (edges[0][0], edges_counter.shape[0]-1)


    # На одну меньше, так как хоть и без повторений в остове но в первой строке будут две новые вершины
    current_cluster = Cluster(total_vertix_count - 1, edges.shape) 
    current_cluster.add_first_edge(edges, dist, edges_counter, 
                                    reversed_edges_counter, reversed_indexes, limits)


    # Наивное предположение о том, что меж всеми элементами можно провести минимальный остов 
    for i in tqdm(range(37840)):
        try:
            current_cluster.add_edge(edges, dist, edges_counter, 
                                  reversed_edges_counter, reversed_indexes, limits)
        except:
            break

    idx = current_cluster.curr_idx

    pd.DataFrame(current_cluster.ostov_edges[:idx, :], columns=['from', 'to']).to_csv('ostov_vert1.csv', index=None)
    pd.DataFrame(current_cluster.ostov_dist[:idx], columns=['dist']).to_csv('ostov_dist1.csv', index=None)