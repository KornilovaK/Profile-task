import pandas as pd
import numpy as np
import os

from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from catboost import CatBoost, Pool
from dataclasses import dataclass
from lightgbm import LGBMRanker, Booster


@dataclass
class Configs:
    models_path = os.environ.get('MODELS_PATH')
    data_path = os.environ.get('DATA_PATH')

    train_df = pd.read_csv(os.path.join(data_path, 'train_df.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_df.csv'))
    
    del_cols = ['feature_3', 'feature_0', 'feature_73',
                'feature_74', 'feature_75'
                ]
    catboost_path = os.path.join(models_path, 'catboost.cbm')
    lgbm_path = os.path.join(models_path, 'lgbm.txt')
    res_path = os.path.join(data_path, 'result.txt')


def init():
    configs = Configs()
    
    train_df, test_df = preprocess_data(configs)
    train_catboost(configs, train_df)
    train_lgbm(configs, train_df)
    test(configs, test_df)


def norm(column, eps=1e-6):
    mean = column.mean()
    std = column.std()
    column = (column-mean)/(std+eps)

    return column


def preprocess_data(configs):
    train_df, test_df, del_cols = configs.train_df, configs.test_df, configs.del_cols

    for col in train_df.columns.values[1:-1]:
        train_df[col] = norm(train_df[col])
        test_df[col] = norm(test_df[col])

    train_df = train_df.drop(del_cols, axis=1)
    test_df = test_df.drop(del_cols, axis=1)

    return train_df, test_df


def catboost_data(X, y):
    X_train, x_val, y_train, y_val = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

    train = X_train
    train['target'] = y_train
    train = train.sort_values(by=['search_id'])

    qids_train = train.search_id.values
    X_train = train.drop(["search_id", "target"], axis=1)
    y_train = train['target']

    val = x_val
    val['target'] = y_val
    val = val.sort_values(by=['search_id'])

    qids_val = val.search_id.values
    x_val = val.drop(["search_id", "target"], axis=1)
    y_val = val['target']

    catboost_pool_train = Pool(X_train, y_train, group_id=qids_train)
    catboost_pool_test = Pool(x_val, y_val, group_id=qids_val)

    return catboost_pool_train, catboost_pool_test


def train_args(train_df):
    X = train_df.drop(["target"], axis=1)
    y = train_df["target"]

    count = y.value_counts()
    weight_0 = count[0]
    weight_1 = count[1]
    class_weights = [weight_1/weight_0, weight_0/weight_0]

    return X, y, class_weights


def train_catboost(configs, train_df):
    X, y, class_weights = train_args(train_df)

    catboost_pool_train, catboost_pool_test = catboost_data(X, y)

    catboost_model = CatBoost({
                        'loss_function':'Logloss',
                        'learning_rate': 0.01,
                        'iterations': 1000,
                        'early_stopping_rounds': 300,
                        'eval_metric': 'NDCG',
                        'class_weights':class_weights,
                        })
    catboost_model.fit(catboost_pool_train,
                        eval_set=catboost_pool_test,
                        use_best_model=True
                        )

    catboost_model.save_model(configs.catboost_path)


def lgbm_data(X, y):
    X_train, x_val, y_train, y_val = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

    qids_train = X_train.groupby("search_id")["search_id"].count().to_numpy()
    X_train = X_train.drop(["search_id"], axis=1)

    qids_validation = x_val.groupby("search_id")["search_id"].count().to_numpy()
    x_val = x_val.drop(["search_id"], axis=1)

    return X_train, y_train, qids_train, x_val, y_val, qids_validation


def train_lgbm(configs, train_df):
    X, y, class_weights = train_args(train_df)
    X_train, y_train, qids_train, x_val, y_val, qids_validation = lgbm_data(X, y)

    model = LGBMRanker(metric="ndcg", n_estimators=1000, learning_rate=0.01, class_weights=class_weights)

    model.fit(
            X=X_train,
            y=y_train,
            group=qids_train,
            eval_set=[(x_val, y_val)],
            eval_group=[qids_validation],
            eval_at=[1, 2, 5, 10],
            )
    
    model.booster_.save_model(configs.lgbm_path)


def test(configs, test_df):
    X_test = test_df.drop(["target", "search_id"], axis=1)
    y_test = test_df["target"]
    
    catboost = CatBoost()
    catboost.load_model(configs.catboost_path)
    cat_pred = catboost.predict(X_test)
    ndcg_cat = ndcg_score([y_test], [cat_pred])

    lgbm = Booster(model_file=configs.lgbm_path)
    lgmb_pred = lgbm.predict(X_test)
    ndcg_lgbm = ndcg_score([y_test], [lgmb_pred])

    text = f'NDCG CatBoost score: {ndcg_cat}\nNDCG LightGBM score: {ndcg_lgbm}\n'
    file = open(configs.res_path, 'w')
    file.write(text)
    file.close()

if __name__ == '__main__':
	init()