import pandas as pd
import numpy as np
import os
import dill
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split  # Split the dataframe
import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Обертка для модели CatBoostClassifier, чтобы она могла использоваться в пайплайне
class CatBoostWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        train_data = Pool(X, y)
        self.model.fit(train_data)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                    num_parts_to_read: int = 2, columns=None, verbose=False) -> pd.DataFrame:
    """
    читает num_parts_to_read партиций, преобразовывает их к pd.DataFrame и возвращает
    :param verbose:
    :param path_to_dataset: путь до директории с партициями
    :param start_from: номер партиции, с которой нужно начать чтение
    :param num_parts_to_read: количество партиций, которые требуется прочитать
    :param columns: список колонок, которые нужно прочитать из партиции
    :return: pd.DataFrame
    """

    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                            if filename.startswith('train')])
    print(dataset_paths)

    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm.tqdm(chunks, desc="Reading dataset with pandas"):
        print('chunk_path', chunk_path)
        chunk = pd.read_parquet(chunk_path, columns=columns)
        res.append(chunk)

    return pd.concat(res).reset_index(drop=True)


def agg_feat(features, prefixes):
    feats = []
    for col in features:
        for prefix in prefixes:
            if col.startswith(prefix):
                feats.append(col)

    return feats


def select_features_():
    selected_features = [
        'rn', 'pre_since_opened', 'pre_pterm', 'pre_fterm', 'pre_till_fclose', 'pre_loans_credit_limit',
        'pre_loans_next_pay_summ', 'pre_loans_outstanding',
        'pre_loans_total_overdue', 'pre_loans_max_overdue_sum', 'pre_loans_credit_cost_rate', 'pre_loans5',
        'pre_loans530', 'pre_loans3060', 'pre_loans6090',
        'pre_loans90', 'is_zero_loans5', 'is_zero_loans530', 'pre_util', 'pre_over2limit', 'pre_maxover2limit',
        'is_zero_util', 'is_zero_over2limit',
        'is_zero_maxover2limit', 'enc_paym_0', 'enc_paym_1', 'enc_paym_4', 'enc_paym_6', 'enc_paym_7', 'enc_paym_8',
        'enc_paym_9', 'enc_paym_10', 'enc_paym_11',
        'enc_paym_12', 'enc_paym_13', 'enc_paym_14', 'enc_paym_15', 'enc_paym_16', 'enc_paym_17', 'enc_paym_19',
        'enc_paym_20', 'enc_paym_21', 'enc_paym_22',
        'enc_paym_23', 'enc_paym_24', 'enc_loans_account_holder_type', 'enc_loans_credit_status',
        'enc_loans_credit_type', 'enc_loans_account_cur', 'fclose_flag',
        'count_active_loan', 'perc_debt_repayment', 'diff_pre_loans', 'avg_stat_pay', 'use_loans_credit_limit',
        'loan_overdue', 'weights', 'cat_late_payment',
        'pre_loans530_3', 'enc_paym_13_1', 'pre_util_2', 'enc_paym_19_0', 'pre_maxover2limit_18',
        'pre_since_confirmed_3', 'enc_loans_account_cur_2',
        'pre_loans530_10', 'pre_maxover2limit_6', 'enc_paym_21_0', 'pre_over2limit_9', 'enc_paym_0_1', 'enc_paym_5_3',
        'is_zero_loans530_1', 'pre_pterm_4',
        'pre_over2limit_1', 'pre_pterm_6', 'pre_loans530_16', 'pre_till_fclose_13', 'pre_loans5_13',
        'pre_loans_next_pay_summ_5', 'pre_till_pclose_14',
        'pre_loans530_18', 'enc_paym_8_0', 'enc_loans_account_cur_3', 'enc_paym_9_1', 'pre_maxover2limit_12',
        'is_zero_loans530_0', 'pre_pterm_2', 'pre_pterm_5',
        'enc_loans_account_holder_type_4', 'pre_over2limit_18', 'pre_till_pclose_3', 'pre_loans_credit_limit_13',
        'pre_maxover2limit_17', 'pre_since_opened_3',
        'pre_loans530_4', 'pre_till_fclose_0', 'pre_loans6090_4', 'pre_util_3', 'enc_paym_14_0',
        'pre_since_confirmed_13', 'pre_loans_credit_limit_11',
        'pre_till_pclose_16', 'pre_fterm_4', 'pre_fterm_8', 'pre_util_10', 'enc_paym_9_0', 'pclose_flag_0',
        'enc_paym_1_2', 'pre_util_14', 'is_zero_loans6090_1',
        'pre_till_fclose_7', 'pre_over2limit_3', 'pre_fterm_9', 'pre_loans530_0', 'enc_paym_3_1', 'pre_pterm_17',
        'pre_loans90_14', 'pre_till_fclose_2',
        'enc_paym_19_2', 'enc_paym_1_1', 'pre_maxover2limit_3', 'enc_loans_credit_status_3', 'pre_over2limit_6',
        'enc_paym_15_2', 'enc_paym_23_2',
        'enc_paym_18_3', 'enc_loans_account_holder_type_2', 'enc_paym_10_2', 'pre_over2limit_19', 'is_zero_loans5_0',
        'pre_loans_credit_cost_rate_4',
        'pre_util_17', 'enc_paym_18_2', 'enc_paym_16_1', 'pre_since_confirmed_5', 'enc_loans_credit_type_5',
        'pre_loans6090_2', 'pre_util_16', 'pre_fterm_3',
        'enc_paym_22_1', 'pre_util_15', 'pre_util_6', 'enc_loans_account_holder_type_5', 'pre_since_opened_5',
        'pre_loans_credit_limit_2', 'pre_over2limit_12',
        'enc_paym_21_2', 'pre_over2limit_11', 'pre_util_5', 'pre_till_fclose_6', 'pre_loans_credit_limit_1',
        'enc_paym_8_3', 'pre_loans3060_7', 'enc_paym_7_0',
        'pre_loans6090_1', 'pre_loans_credit_cost_rate_5', 'pre_loans90_2', 'pre_util_0', 'pre_util_8',
        'enc_loans_credit_type_0', 'pre_maxover2limit_4',
        'enc_paym_11_4', 'pre_since_confirmed_12', 'pre_maxover2limit_5', 'pre_loans5_2', 'pre_maxover2limit_7',
        'pre_maxover2limit_14',
        'pre_loans_credit_limit_4', 'pre_maxover2limit_11', 'pre_till_pclose_13', 'pre_maxover2limit_16', 'pre_util_19',
        'enc_paym_6_0',
        'enc_loans_credit_type_3', 'pre_util_12', 'pre_loans_credit_limit_16', 'pre_since_opened_8', 'pre_loans530_14',
        'enc_paym_2_1', 'enc_paym_13_0',
        'pre_maxover2limit_19', 'pre_since_confirmed_15', 'enc_paym_14_2', 'enc_paym_13_2', 'pre_loans_credit_limit_18',
        'pre_since_confirmed_10',
        'is_zero_loans6090_0', 'enc_paym_5_1', 'enc_paym_12_2', 'pre_pterm_0', 'is_zero_loans90_0', 'pre_util_1',
        'pre_since_confirmed_2',
        'pre_loans_max_overdue_sum_1', 'pre_loans530_15', 'enc_paym_13_3', 'enc_paym_22_2', 'pre_pterm_15',
        'enc_paym_0_3', 'pre_over2limit_4',
        'pre_pterm_16', 'enc_paym_14_3', 'pre_loans_next_pay_summ_1', 'enc_paym_12_1', 'enc_paym_12_3', 'pre_pterm_10',
        'enc_paym_4_2', 'pre_loans530_12',
        'pre_loans_credit_cost_rate_8', 'pre_over2limit_16', 'enc_loans_account_holder_type_0', 'is_zero_loans3060_1',
        'enc_paym_23_0', 'pre_till_pclose_2',
        'pre_loans_total_overdue_0', 'pre_util_13', 'pre_loans_credit_cost_rate_13', 'enc_paym_18_1',
        'pre_loans_credit_limit_8', 'pre_loans_credit_cost_rate_6',
        'enc_loans_credit_status_2', 'enc_paym_11_1', 'pre_fterm_11', 'pre_till_fclose_10', 'pre_util_4', 'pre_util_9',
        'enc_loans_credit_status_0',
        'pre_since_opened_14', 'pre_till_pclose_10', 'enc_loans_account_cur_1', 'enc_paym_10_3', 'enc_paym_16_2',
        'pre_till_fclose_9', 'pre_maxover2limit_1',
        'pre_loans_outstanding_5', 'pre_loans_next_pay_summ_0', 'pre_loans530_7', 'enc_paym_6_2', 'pre_since_opened_19',
        'pre_loans3060_5',
        'pre_loans_credit_cost_rate_9', 'is_zero_loans3060_0', 'pre_since_opened_2', 'enc_paym_15_1', 'pre_loans3060_9',
        'pre_over2limit_8',
        'enc_loans_account_cur_0', 'is_zero_util_0', 'pre_loans3060_8', 'enc_paym_9_2', 'pre_maxover2limit_15',
        'pre_since_opened_1', 'pre_since_opened_7',
        'pre_loans5_16', 'enc_loans_credit_status_6', 'enc_paym_24_4', 'pre_till_fclose_11', 'pre_since_opened_9',
        'enc_paym_17_2', 'pre_loans5_3',
        'is_zero_over2limit_1', 'enc_paym_7_1', 'enc_loans_credit_type_4', 'pre_maxover2limit_9',
        'pre_loans_credit_limit_15', 'pre_maxover2limit_13',
        'pre_loans5_5', 'pre_loans530_11', 'pre_since_confirmed_0', 'pre_util_11', 'enc_paym_5_2', 'pre_loans530_1',
        'enc_paym_3_0', 'enc_paym_20_3',
        'pre_maxover2limit_8', 'enc_paym_24_3', 'enc_paym_3_2', 'pre_till_pclose_6', 'enc_paym_24_2', 'enc_paym_20_1',
        'pre_since_opened_12',
        'pre_since_confirmed_4', 'pre_since_confirmed_9', 'pre_pterm_8', 'enc_paym_10_1', 'enc_paym_19_1',
        'pre_loans5_6', 'pre_loans_credit_limit_14',
        'pre_fterm_10', 'pre_loans3060_1', 'pre_loans_credit_limit_10', 'pre_loans530_2', 'pre_loans5_1',
        'pre_till_fclose_15', 'pre_loans_outstanding_2',
        'enc_loans_account_holder_type_1', 'pre_loans90_19', 'pre_loans_next_pay_summ_3', 'enc_loans_credit_type_1',
        'pre_fterm_16', 'fclose_flag_1',
        'enc_paym_7_2', 'pre_pterm_13', 'enc_paym_22_0', 'pre_since_opened_11', 'pre_loans_credit_limit_5',
        'pre_maxover2limit_10', 'pre_loans6090_3',
        'pre_over2limit_13', 'pre_loans_credit_cost_rate_0', 'pre_since_opened_16', 'pre_maxover2limit_0',
        'pre_since_confirmed_6', 'enc_loans_credit_status_5',
        'pre_loans5_7', 'pre_loans_credit_cost_rate_12', 'pre_loans530_19', 'pre_till_pclose_7', 'pre_over2limit_10',
        'pre_loans_credit_cost_rate_11',
        'pre_since_opened_13', 'pre_till_fclose_3', 'enc_paym_18_0', 'pre_loans_next_pay_summ_4', 'enc_paym_16_0',
        'pre_since_confirmed_7',
        'pre_loans_outstanding_4', 'enc_paym_21_3', 'enc_paym_19_3', 'pre_util_18', 'pre_maxover2limit_2',
        'enc_paym_8_1', 'pre_fterm_1',
        'pre_since_confirmed_11', 'pre_over2limit_0', 'enc_loans_credit_status_1', 'pre_over2limit_7',
        'pre_over2limit_14', 'pre_loans90_13',
        'enc_paym_0_2', 'enc_paym_8_2', 'pre_loans_credit_cost_rate_2', 'pre_loans3060_2', 'enc_paym_2_2',
        'pre_fterm_7', 'pre_over2limit_15',
        'pre_since_opened_4', 'pre_since_confirmed_1', 'pre_since_opened_0'
    ]

    return selected_features


def feat_gen(df) -> pd.DataFrame:
    print('Генерация признаков', str(datetime.now()))
    # Количество активных кредитов
    pre_loans_columns = ['pre_loans5', 'pre_loans530', 'pre_loans3060', 'pre_loans6090', 'pre_loans90']
    df['count_active_loan'] = df[pre_loans_columns].sum(axis=1)

    # Процент погашения задолженности от общей суммы кредита
    df['perc_debt_repayment'] = df.apply(
        lambda x: x.pre_loans_outstanding / x.pre_loans_credit_limit if x.pre_loans_credit_limit > 0 else 0, axis=1)

    # Периодичность просрочек
    df['diff_pre_loans'] = df[pre_loans_columns].max(axis=1) - df[pre_loans_columns].min(axis=1)

    # Средний статус платежей
    enc_paym_columns = [col for col in df.columns.tolist() if col.startswith("enc_paym_")]
    df['avg_stat_pay'] = df[enc_paym_columns].mean(axis=1)

    # Сумма переплаты
    df['Overpayment_amount'] = df['pre_loans_credit_cost_rate'] - df['pre_loans_credit_limit']

    # Процент использования кредитного лимита
    df['use_loans_credit_limit'] = df.apply(
        lambda x: x.pre_loans_outstanding / x.pre_loans_credit_limit if x.pre_loans_credit_limit > 0 else 0, axis=1)
    # флаги просрочек
    is_zero_loans = [
        "is_zero_loans5",
        "is_zero_loans530",
        "is_zero_loans3060",
        "is_zero_loans6090",
        "is_zero_loans90"
    ]
    df[is_zero_loans] = df[is_zero_loans].replace([0, 1], [1, 0])

    df["loan_overdue"] = df[is_zero_loans].any(axis=1)

    # веса для просрочек
    def set_weights(x):
        return x.pre_loans5 + x.pre_loans530 * 2 + x.pre_loans3060 * 3 + x.pre_loans6090 * 4 + x.pre_loans90 * 5

    df['weights'] = df.apply(set_weights, axis=1)
    q25 = df.weights.quantile(0.25)
    q75 = df.weights.quantile(0.75)
    df['cat_late_payment'] = df['weights'].apply(lambda x: 0.0 if x < q25 else (1.0 if x > q75 else 0.5))
    print('Генерация признаков завершена', str(datetime.now()))
    select_feat = select_features_()
    return df[select_feat]


def read_credit_history() -> pd.DataFrame:
    print('Чтение исходного датасета', str(datetime.now()))
    path = 'c:/Model_credit_risk/train_data/'
    df = read_parquet_dataset_from_local(path)
    target = pd.read_csv('c:/Model_credit_risk/train_target.csv')
    print('Чтение исходного датасета завершено', str(datetime.now()))
    print('Группировка датасета', str(datetime.now()))
    features = list(set(df.columns) - set(['id', 'rn']))
    dummies = pd.get_dummies(df[features], columns=features)
    prepared_df = pd.concat([df[['id', 'rn']], dummies], axis=1)
    agg_d = {f: 'sum' for f in dummies.columns}
    prepared_df = prepared_df.groupby('id').agg(agg_d).astype(int).reset_index(drop=False)
    last_rec_df = df.loc[df.groupby('id')['rn'].idxmax()]
    df = last_rec_df.merge(prepared_df, on='id').merge(target, on='id')
    print('Группировка датасета завершена', str(datetime.now()))
    
    return df


def train_test_split_(X, y):
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True,
                                                        stratify=y)
    train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, test_size=0.3, random_state=42,
                                                      shuffle=True, stratify=y_train)
    train_pool = Pool(train_X, train_y)
    val_pool = Pool(val_X, val_y)
    test_pool = Pool(X_test, y_test)
    return train_pool, val_pool, test_pool


def main():
    print('Default State Prediction Pipeline', str(datetime.now()))
    
    df = read_credit_history()
    df = df.astype('int8')
    # Разделение данных на признаки и целевую переменную
    X = df.drop(['id', 'flag'], axis=1)
    y = df['flag']

    X = feat_gen(X)
    print(X)
    print(X.info(max_cols=5, memory_usage='deep'))

if __name__ == '__main__':
    main()
