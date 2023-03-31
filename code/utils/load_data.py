import pandas as pd
import torch

'''
title : 项目标题，如titanic
label : 标签列名，如Survived
id : id列名，如PassengerId
drop_features : 需要删除的特征，如Name
fill_na : 需要填充缺失值的特征，如Age
one_hot : 需要one-hot编码的特征，如Sex
val_size : 验证集比例，默认0.2
'''


def load_data(title, label, id=None, drop_features=None, fill_na=None, one_hot=None, val_size=0.2):
    train = pd.read_csv("../../data/raw/" + title + "/train.csv")
    test = pd.read_csv("../../data/raw/" + title + "/test.csv")

    # 删除无用的特征
    train.drop(drop_features, axis=1, inplace=True)
    test.drop(drop_features, axis=1, inplace=True)

    # 填充缺失值
    for feature in fill_na:
        if (train[feature].dtype == "object"):
            train[feature].fillna(train[feature].mode()[0], inplace=True)
            test[feature].fillna(test[feature].mode()[0], inplace=True)
        else:
            train[feature].fillna(train[feature].mean(), inplace=True)
            test[feature].fillna(test[feature].mean(), inplace=True)

    # one-hot编码
    train = pd.get_dummies(train, columns=one_hot, drop_first=True)
    test = pd.get_dummies(test, columns=one_hot, drop_first=True)


    # 分离特征和标签
    X_train = torch.tensor(train.drop(
        [id, label], axis=1).values, dtype=torch.float32)
    y_train = torch.tensor(train[label].values, dtype=torch.float32)
    X_test = torch.tensor(test.drop(id, axis=1).values, dtype=torch.float32)

    # 分离验证集
    if val_size > 0 and val_size < 1:
        val_size = int(val_size * len(X_train))
        X_train, X_val = X_train[:-val_size], X_train[-val_size:]
        y_train, y_val = y_train[:-val_size], y_train[-val_size:]

    return X_train, y_train, X_val, y_val, X_test
