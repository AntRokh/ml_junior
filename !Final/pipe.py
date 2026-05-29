import copy
import numpy as np
from sklearn.base import clone
import pandas as pd
import os
import dill
from datetime import datetime
from catboost import CatBoostClassifier, EFeaturesSelectionAlgorithm, EShapCalcType, Pool
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline
from sklearn.model_selection import train_test_split  # Split the dataframe
import tqdm
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0, #Чтение из parquet файла
                                    num_parts_to_read: int = 11, columns=None, verbose=False) -> pd.DataFrame:
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
    feats = []  # Создание пустого списка для хранения отфильтрованных признаков
    for col in features:  # Итерация по каждому признаку в списке features
        for prefix in prefixes:  # Итерация по каждому префиксу в списке prefixes
            if col.startswith(prefix):  # Проверка, начинается ли имя признака с текущего префикса
                feats.append(col)  # Если да, добавляем признак в список feats
          
    return feats  # Возвращаем список отфильтрованных признаков


def feat_gen(df) -> pd.DataFrame: # Генерация признаков
    
    print('Генерация признаков')
    # Количество активных кредитов
    pre_loans_columns = ['pre_loans5', 'pre_loans530', 'pre_loans3060', 'pre_loans6090', 'pre_loans90']
    df['count_active_loan'] = df[pre_loans_columns].sum(axis=1)

    # Процент погашения задолженности от общей суммы кредита
    df['perc_debt_repayment'] = df.apply(
        lambda x: x['pre_loans_outstanding'] / x['pre_loans_credit_limit'] if x['pre_loans_credit_limit'] > 0 else 0, axis=1)

    # Периодичность просрочек
    df['diff_pre_loans'] = df[pre_loans_columns].max(axis=1) - df[pre_loans_columns].min(axis=1)

    # Средний статус платежей
    enc_paym_columns = [col for col in df.columns.tolist() if col.startswith("enc_paym_")]
    df['avg_stat_pay'] = df[enc_paym_columns].mean(axis=1)

    # Сумма переплаты
    df['Overpayment_amount'] = df['pre_loans_credit_cost_rate'] - df['pre_loans_credit_limit']

    # Процент использования кредитного лимита
    df['use_loans_credit_limit'] = df.apply(
        lambda x: x['pre_loans_outstanding'] / x['pre_loans_credit_limit'] if x['pre_loans_credit_limit'] > 0 else 0, axis=1)
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
        return x['pre_loans5'] + x['pre_loans530'] * 2 + x['pre_loans3060'] * 3 + x['pre_loans6090'] * 4 + x['pre_loans90'] * 5

    df['weights'] = df.apply(set_weights, axis=1)
    q25 = df['weights'].quantile(0.25)
    q75 = df['weights'].quantile(0.75)
    df['cat_late_payment'] = df['weights'].apply(lambda x: 0.0 if x < q25 else (1.0 if x > q75 else 0.5))
    
    # Чтение выбранных признаков из файла
    with open('selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f]

    # Добавление отсутствующих признаков
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0

    # Удаление лишних признаков
    for feature in df.columns:
        if feature not in selected_features:
            df = df.drop(columns=[feature])

    print('Генерация признаков завершена')
    return df


def read_credit_history(start_from: int = 0, num_parts_to_read: int = 11) -> pd.DataFrame: #Группировка и кодирование исходных данных
    print('Чтение исходного датасета', str(datetime.now()))
    path = 'c:/Model_credit_risk/train_data/'
    df = read_parquet_dataset_from_local(path, start_from, num_parts_to_read)
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

def select_features_(X,y) -> pd.DataFrame: #Выбор признаков имеющих важность
    print('Определение важности фичей')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True,
                                                        stratify=y)
    feature_names = list(X_train.columns)
    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    val_pool = Pool(X_val, y_val, feature_names=feature_names)
    model = CatBoostClassifier(random_seed=63,
                               task_type='GPU',
                               early_stopping_rounds=20,
                               eval_metric='AUC:hints=skip_train~false',
                               auto_class_weights='Balanced',
                               )
    summary = model.select_features(
        train_pool,
        eval_set=val_pool,
        features_for_select=X.columns,
        num_features_to_select=400,
        steps=9,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Exact,
        train_final_model=False,
        logging_level='Verbose',
        plot=False
    )


    selected_features = summary['selected_features_names'] + ['count_active_loan', 'perc_debt_repayment', 'diff_pre_loans', 
                                                              'avg_stat_pay', 'Overpayment_amount', 'use_loans_credit_limit', 
                                                              'loan_overdue', 'weights''cat_late_payment']
    print('Определение важности фичей Завершено')
    
    #Сохраняем список признаков в файл
    with open('selected_features.txt', 'w') as f:
        for item in selected_features:
            f.write("%s\n" % item)

def get_threshold(y_test, y_score):
    t = pd.DataFrame({'y_true': y_test, 
                  'y_score': y_score})
    (fpr, tpr, thresholds) = roc_curve(t.y_true, t.y_score)
    # Вычисляем статистику Youden's J 
    youdenJ = tpr - fpr

    # Ищем оптимальный threshold
    index = np.argmax(youdenJ)
    thr = round(thresholds[index], ndigits = 4)
    return thr
def main():
    print('Default State Prediction Pipeline', str(datetime.now()))


    df = read_credit_history(0,11)
    df.to_csv('dataset.csv', index=False)
    
    
    df = pd.read_csv('dataset.csv')
    print('dataset is readed')
    df = df.astype('int8')
   

    # Разделение данных на признаки и целевую переменную
    X = df.drop(['id', 'flag'], axis=1)
    y = df['flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True,
                                                        stratify=y)

    #определяем важность признаков
    select_features_(X_train, y_train)  
 

    print('Запуск pipeline', str(datetime.now()))
    
    preprocessor = Pipeline(steps=[
        ('feature_generator', FunctionTransformer(feat_gen, validate=False)),  # генерация новых признаков
    ])
    # Создание списка моделей для голосования
    # Создание пайплайна для логистической регрессии с предварительной обработкой StandardScaler
    logistic_regression_pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight='balanced', max_iter=1500, random_state=63, verbose=1)
    )

    models = [
        ('catboost_gpu', CatBoostClassifier(random_seed=63,
                        task_type='GPU',
                        iterations=4000,
                        early_stopping_rounds=10,
                        eval_metric='AUC:hints=skip_train~false',
                        learning_rate=0.011842098798901503,
                        depth=6,
                        l2_leaf_reg=4.877764869966795,
                        auto_class_weights='Balanced',
                        )),
        ('catboost_cpu', CatBoostClassifier(random_seed=63,
                        early_stopping_rounds=10,
                        iterations=4000,
                        eval_metric='AUC:hints=skip_train~false',
                        learning_rate=0.013825994516283503,
                        depth=6,
                        l2_leaf_reg=8.045849262672586,
                        auto_class_weights='SqrtBalanced',
                        )),
        ('logistic_regression', logistic_regression_pipeline)
    ]

    # Создание голосующего классификатора
    voting_clf = Pipeline(steps=[
        ('voting', VotingClassifier(estimators=models, voting='soft'))
    ])

    # Обучение моделей на тренировочной выборке
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('voting_clf', voting_clf)
    ])

    pipe.fit(X_train, y_train)
    # Замер ROC AUC на тестовой выборке
    print("Замер ROC AUC на тестовой выборке:", str(datetime.now()))
    y_pred = pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC на тестовой выборке:", roc_auc, str(datetime.now()))
    
    thr = get_threshold(y_test=y_test, y_score=y_pred)
    print('Порог принятия решения= ', thr)
    print('Сохранение модели', str(datetime.now()))
    with open('model_credit_risk.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,  # Use clone instead of deepcopy
            'metadata': {
                'name': 'Default State Prediction Pipeline',
                'author': 'Anton Rokhmistrov',
                'version': 1,
                'ROC AUC': roc_auc,
                'feature_names': X_train.columns.tolist(),
                'thr': thr
            }
        }, file)
    print('Сохранение модели завершено', str(datetime.now()))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


