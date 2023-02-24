# popular recommender
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, coo_matrix
from implicit.nearest_neighbours import ItemItemRecommender, CosineRecommender, TFIDFRecommender, BM25Recommender


data = pd.read_csv('retail_train.csv')
#print(pd.DataFrame(data.head(10)))
#print(data['week_no'].nunique())
users, items, interactions = data.user_id.nunique(), data.item_id.nunique(), data.shape[0]
# на 2500 тысячи пользователей и на 89000 товаров в таблице показано почти 2,5 миллиона взаимодействий
# print('users', users)
# print('items', items)
# print('interactions ', interactions)

# данные по характеристикам товаров и пользователей

item_features = pd.read_csv('product.csv')
#print(pd.DataFrame(item_features.head(10)))

# описание покупателей, которые делали покупки
user_features = pd.read_csv('hh_demographic.csv')
#print(pd.DataFrame(user_features.head(2)))

"""
TRAIN-test split 
В рекомендательных системах корректнее использовать TRAIN-test split по времени, а не случайно
Возьмем последние 3 недели в качестве теста
"""
test_size_weeks = 3
data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

#print(data_train.shape[0], data_test.shape[0])

"""
Baselines 
"""
# Создадим датафрейм с покупками пользователей на тестовом датасете (последние 3 недели)
result = data_test.groupby('user_id')['item_id'].unique().reset_index()
result.columns = ['user_id', 'actual'] # actual - это items из теста, те которые пользователь действительно купил
#print(result.head(5))
test_users = result.shape[0]
new_test_users = len(set(data_test['user_id']) - set(data_train['user_id']))
print(f'В тестовом датасете {test_users}  пользователей')
print(f'В тестовом датасете {new_test_users}  новых пользователей')

"""
1. Random recommendation
"""

def random_recommendation(items, n = 5):
    """Случайные рекоммендации"""
    items = np.array(items)
    recs = np.random.choice(items, size=n, replace=False)
    return recs.tolist()

"""
1. Реализуем Weighted Random Recommendation 
"""
#функция для получения весов товаров в зависимости от объёма продаж в денежном эквиваленте
def get_items_weights(df):
    total_sales=df['sales_value'].sum()
    items_weights=df.groupby('item_id').agg({'sales_value':'sum'}).reset_index().rename(columns={'sales_value':'weight'})
    items_weights['weight']=items_weights['weight'].apply(lambda x: x/total_sales)
    return items_weights

def weighted_random_recommendation(items_weights, n=5):
    # Подсказка: необходимо модифицировать функцию random_recommendation()
    items = np.array(items_weights['item_id'])
    weights = np.array(items_weights['weight'])
    recs = np.random.choice(items, size=n, p=weights,
                            replace=False)  # используем параметр p метода choice в который передадим веса товаров
    return recs.tolist()

items_weights=get_items_weights(data_train)

result['weighted_random_recommendation'] = result['user_id'].apply(lambda x: weighted_random_recommendation(items_weights, n=5))

print(result.head(2))

"""
2. Улучшение бейзлайнов и Item - Item
"""
#функция для получения Топ 5000 товаров по количеству проданного(можно варьировать какой Топ получать и по какому показателю)
def get_top(df, column='quantity', top=5000):
    top_df=df.groupby('item_id').agg({f'{column}':'sum'}).reset_index().sort_values(column, ascending=False).head(5000).item_id.tolist()
    return top_df

items = get_top(data_train)

result['top5000_random_recommendation'] = result['user_id'].apply(lambda x: random_recommendation(items, n=5))

print(result.head(2))

# Посчитаем качество случайной рекомендации на товарах из Топ-5000 по количеству проданного товара и по сумме продаж

items = get_top(data_train, column='sales_value')

result['top5000_sales_random_recommendation'] = result['user_id'].apply(lambda x: random_recommendation(items, n=5))

print(result.head(2))

def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)

for name_col in result.columns[1:]:
    print(f"{round(result.apply(lambda row: precision_at_k(row[name_col], row['actual']), axis=1).mean(),4)}:{name_col}")

top_5000= get_top(data_train)
data_train.loc[ ~ data_train['item_id'].isin(top_5000), 'item_id'] = 6666
print(data_train.head(100))

user_item_matrix = pd.pivot_table(data_train,
                                  index='user_id', columns='item_id',
                                  values='quantity',
                                  aggfunc='count',
                                  fill_value=0
                                 )

user_item_matrix[user_item_matrix > 0] = 1 # так как в итоге хотим предсказать

user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit

# переведем в формат sparse matrix
sparse_user_item = csr_matrix(user_item_matrix).tocsr()

# создаем словари мапинга между id бизнеса к строчному id матрицы

userids = user_item_matrix.index.values
itemids = user_item_matrix.columns.values

matrix_userids = np.arange(len(userids))
matrix_itemids = np.arange(len(itemids))

id_to_itemid = dict(zip(matrix_itemids, itemids))
id_to_userid = dict(zip(matrix_userids, userids))

itemid_to_id = dict(zip(itemids, matrix_itemids))
userid_to_id = dict(zip(userids, matrix_userids))

neighbors = np.arange(1, 21)
fltrs = [None, [itemid_to_id[6666]]]
max_score, m_neighbor, m_fltr = 0, 0, 0
res_dict = {'no_fltr': [[], []], 'fltr': [[], []]}
for neighbor in neighbors:
    for fltr in fltrs:
        current_key = 'fltr' if fltr else 'no_fltr'
        model = ItemItemRecommender(K=neighbor, num_threads=4)
        model.fit(csr_matrix(user_item_matrix).T.tocsr(),  # На вход item-user matrix
                  show_progress=False)

        result['itemitem'] = result['user_id'].apply(lambda user_id: [
            id_to_itemid[rec[0]] for rec in model.recommend(userid=userid_to_id[user_id],
                                                            user_items=sparse_user_item,  # на вход user-item matrix
                                                            N=5,
                                                            filter_already_liked_items=False,
                                                            filter_items=fltr,
                                                            recalculate_user=True)
        ])
        score = round(result.apply(lambda row: precision_at_k(row['itemitem'], row['actual']), axis=1).mean(), 4)
        res_dict[current_key][0].append(score)
        res_dict[current_key][1].append(neighbor)

        if score > max_score:
            max_score, m_neighbor, m_fltr = score, neighbor, 'filtered by 6666' if fltr else 'non filtered'
print(f'Лучший скор: {max_score}, K_neighbors: {m_neighbor}, условие фильтрации: {m_fltr}')

plt.figure(figsize=(16, 8))
plt.plot(res_dict['fltr'][1], res_dict['fltr'][0], label='filtered by 6666')
plt.plot(res_dict['no_fltr'][1], res_dict['no_fltr'][0], label='non filtered')
plt.xlabel('N-neighbor')
plt.ylabel('precision_at_k')
plt.legend()
plt.show()