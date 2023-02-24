import numpy as np
import pandas as pd
"""

1. Реализовать метрики Recall@k и MoneyRecall@k

"""


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    # TODO: Ваш код здесь

    flags = np.isin(bought_list, recommended_list)

    return flags.sum() / len(bought_list)

def recall_at_k(recommmended_list, bought_list, k=5):

    return recall(recommmended_list[:k], bought_list)

recommmended_list = [205, 458, 122, 78, 981, 135, 337, 789, 546, 666]
prices_recommended = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
bought_list = [205, 122, 300, 546]
prices_bought = [100, 200, 300, 400]

result_1 = recall(recommmended_list, bought_list)
print(result_1)
result_2 = recall_at_k(recommmended_list, bought_list)
print(result_2)

def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    prices_bought = np.array(prices_bought)

    # TODO: Ваш код здесь

    flags = np.isin(recommended_list, bought_list)
    recall = np.dot(flags, prices_recommended).sum() / prices_bought.sum()
    return  recall # Добавьте сюда результат расчета метрики

result_3 = money_recall_at_k(recommmended_list, bought_list, prices_recommended, prices_bought, k=5)
print(result_3)

"""

2. Реализовать метрикy MRR@k

"""

def reciprocal_rank(recommended_list, bought_list, k=1):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))
    if len(relevant_indexes) == 0:
        return 0
    reciprocal_rank = 1 / (relevant_indexes[0] + 1)
    return reciprocal_rank

def mrr_at_k(recommended_lists, bought_lists, k=1):
    reciprocal_rank_list = []
    for i in range(len(recommended_lists)):
        bought_list = np.array(bought_lists[i][0])
        recommended_list = np.array(recommended_lists[i])[:k]
        reciprocal_rank_list.append(reciprocal_rank(recommended_list, bought_list, k))
    return np.mean(reciprocal_rank_list)


recommmended_lists = [[122, 123, 432, 433, 567, 568, 789, 788, 902, 901], [125, 126, 437, 438, 562, 563, 790, 797, 910, 920], [156, 157, 478, 479, 543, 533, 756, 757, 984, 974]]
bought_lists = [[432, 135, 126, 433], [433, 902, 790, 157], [344, 157]]
result_5 = reciprocal_rank(recommmended_list, bought_list, k=5)
#print(result_5)
result_4 = mrr_at_k(recommmended_lists, bought_lists, k=5)
#print(result_4)

Q = len(recommmended_lists)
cumulative_reciprlocal = 0
for i in range(Q):
    first_result = recommmended_lists[i][0]
    second_result = bought_lists[i][0]
    relevant_indexes = np.nonzero(np.isin(first_result, second_result, 5))
    reciprocal = 1 / (relevant_indexes[0] + 1)
    cumulative_reciprlocal += reciprocal

mrr = 1 / Q * cumulative_reciprlocal
print(mrr)