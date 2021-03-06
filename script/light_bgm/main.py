# coding=utf-8
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt


def main():
    # Breat Cancerデータセットの読み込み
    breate_cancer_dataset = datasets.load_breast_cancer()
    data, label = breate_cancer_dataset.data, breate_cancer_dataset.target

    data_train, data_test, label_train, label_test = train_test_split(data, label)

    # データセットを生成する
    lgb_train = lgb.Dataset(data_train, label_train)
    lgb_eval = lgb.Dataset(data_test, label_test, reference=lgb_train)

    weight_column = []  # 変数の重み
    lgbm_params = {
        'objective': 'binary',  # 二値分類問題
        'learning_rate': 0.1,   # default 0.1
        'num_iterations': 100,  # default 100
        'num_leaves': 31,       # default 31
        'max_depth': -1,        # default -1(制限無し)
        'verbosity': -1,        # 学習経過の表示フラグ
        'metric': 'auc',        # AUC の最大化を目指す
        'weight_column': weight_column
    }

    # 上記のパラメータでモデルを学習する
    model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)

    # 保存
    model.save_model('model.txt')

    # テストデータを予測する
    y_pred = model.predict(data_test, num_iteration=model.best_iteration)

    # 保存したモデルを使う場合はこんな感じ
    # bst = lgb.Booster(model_file='model.txt')
    # ypred = bst.predict(X_test, num_iteration=bst.best_iteration)

    # AUC (Area Under the Curve) を計算する
    fpr, tpr, thresholds = metrics.roc_curve(label_test, y_pred)
    print(label_test[:10])
    print(y_pred[:10])
    auc = metrics.auc(fpr, tpr)
    print(auc)

    # ROC曲線
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
