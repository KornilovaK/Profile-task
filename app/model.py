import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, ndcg_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

sns.set()

train_df = pd.read_csv('/content/train_df.csv')
test_df = pd.read_csv('/content/test_df.csv')

del_cols = ['feature_0', 'feature_73', 'feature_74', 'feature_75']
train_df = train_df.drop(del_cols, axis=1)

X = train_df.drop(["target"], axis=1)
y = train_df["target"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, x_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42)

count = y.value_counts()
weight_0 = count[0]
weight_1 = count[1]

class_weights = [weight_1/weight_0, weight_0/weight_0]

model = CatBoostClassifier(class_weights=class_weights,)
                           #task_type="GPU")

model.fit(X, y, use_best_model=True, eval_set=(x_val, y_val), plot=True)

test_df = test_df.drop(del_cols, axis=1)

X_test = test_df.drop(["target"], axis=1)
y_test = test_df["target"]

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

y_pred = model.predict(X_test)
ndcg_score([y_test], [y_pred])
model.save_model('last_model.cbm')


if __name__ == '__main__':
	dp.run_polling(bot)