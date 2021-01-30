import base64
from typing import List

import craft_ml as cml

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from category_encoders.cat_boost import CatBoostEncoder


@st.cache
def load_data(uploaded_file: str) -> pd.DataFrame:
    data = pd.read_csv(uploaded_file)
    return data


def find_categorical_features(X: pd.DataFrame) -> List[str]:
    categorical = X.select_dtypes(
        include=["object", "category"]
    )
    return categorical.columns.tolist()


def find_best_threshold(y_true: np.array, y_pred: np.array) -> float:
    thresholds = np.arange(0, 1, 0.01)
    scores = []

    for p in thresholds:
        y_labels = np.where(
            y_pred >= p, 1, 0
        )
        score = f1_score(y_true, y_labels)
        scores.append(score)

    best_threshold = thresholds[np.argmax(scores)]
    return thresholds, scores, float(best_threshold)


def get_pipeline() -> cml.Pipeline:
    blocks_str = json.dumps(cml.default_pipeline())
    pipeline = cml.Pipeline(blocks_str)
    return pipeline


@st.cache
def get_predictions(train_file: str, test_file: str, pipeline: cml.Pipeline) -> np.ndarray:
    y_pred = pipeline.run_pipeline(dict(
        train_path=train_file,
        test_path=test_file
    ))
    return y_pred[:, 1]


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


st.sidebar.title('Обучающая выборка')
train_data = st.sidebar.file_uploader("Выбрать данные для обучения модели")

st.sidebar.title('Тестовая выборка')
test_data = st.sidebar.file_uploader("Выбрать данные для применения модели")

try:
    train = load_data(train_data)
    if st.sidebar.checkbox('Показать сырые данные'):
        st.write(train.head())

    st.sidebar.title('Целевая переменная')
    target_name = st.sidebar.selectbox(
        'Выбрать столбец с целевой переменной',
        train.columns
    )
    target, train = train[target_name], train.drop(target_name, axis=1)
    if st.sidebar.checkbox('Показать распределение целевой переменной'):
        st.title("Распределение целевой переменной")
        hist_values = target.value_counts()
        st.bar_chart(hist_values)

    st.sidebar.title("Служебные переменные")
    msg = (
        "Служебные столбцы не будут участвовать в обучении модели (ID-записи, даты,...)."
    )
    drop_columns = st.sidebar.multiselect(
        msg, train.columns,
    )
    submit_columns = st.sidebar.multiselect(
        "Столбцы для формирования файла с прогнозами", train.columns,
    )
    # if drop_columns:
    #     train = train.drop(drop_columns, axis=1)
    #
    # categorical_features = find_categorical_features(train)
    # if categorical_features:
    #     encoder = CatBoostEncoder()
    #     encoded_features = encoder.fit_transform(
    #         train[categorical_features], target
    #     )
    #     st.table(encoded_features.head())
    #     train = train.drop(categorical_features, axis=1)
    #     used_features = train.columns.tolist()
    #
    # st.title("Категориальные признаки")
    # st.text(categorical_features)

    st.title("Обучение модели")
    # train_size = st.slider(
    #     label="Доля наблюдений для обучения модели", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    # train_, valid_ = train_test_split(
    #     train, train_size=train_size, random_state=27, shuffle=True
    # )
    # train_target, valid_target = train_test_split(
    #     target, train_size=train_size, random_state=27, shuffle=True
    # )
    if st.checkbox('Обучить модель'):
        # model = fit_model(train, target)
        pipeline = get_pipeline()
        predictions = get_predictions(train_data, test_data, pipeline)
        st.text("Модель обучена!!!")

        # y_valid_pred = create_predictions(model, valid_)
        # y_train_pred = create_predictions(model, train_)
        #
        # valid_score = roc_auc_score(valid_target, y_valid_pred)
        # train_score = roc_auc_score(train_target, y_train_pred)
        #
        # fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        # fpr, tpr, _ = roc_curve(valid_target, y_valid_pred)
        # axes[0].plot(fpr, tpr, linewidth=3, label=f"Valid score = {round(valid_score, 4)}")
        # fpr, tpr, _ = roc_curve(train_target, y_train_pred)
        # axes[0].plot(fpr, tpr, linewidth=3, label=f"Train score = {round(train_score, 4)}")
        # axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", label="baseline", alpha=0.25)
        # axes[0].set_xlabel("False Positive Rate", size=15)
        # axes[0].set_ylabel("True Positive Rate", size=15)
        # axes[0].set_title("ROC-Curve", size=15)
        # axes[0].legend(loc="best")
        # axes[0].set_xlim(0, 1)
        # axes[0].set_ylim(0, 1)
        #
        # valid_score = average_precision_score(valid_target, y_valid_pred)
        # train_score = average_precision_score(train_target, y_train_pred)
        # precision, recall, _ = precision_recall_curve(valid_target, y_valid_pred)
        # axes[1].plot(recall, precision, linewidth=3, label=f"Valid score = {round(valid_score, 4)}")
        # fpr, tpr = [0, 1], [np.mean(valid_target), np.mean(valid_target)]
        # axes[1].plot(fpr, tpr, linestyle="--", color="black", alpha=0.25)
        # precision, recall, _ = precision_recall_curve(train_target, y_train_pred)
        # axes[1].plot(recall, precision, linewidth=3, label=f"Train score = {round(valid_score, 4)}")
        # fpr, tpr = [0, 1], [np.mean(train_target), np.mean(train_target)]
        # axes[1].plot(fpr, tpr, linestyle="--", color="black", alpha=0.25, label="baseline")
        # axes[1].set_title("Precision-Recall-Curve", size=15)
        # axes[1].set_ylabel("Precision", size=15)
        # axes[1].set_xlabel("Recall", size=15)
        # axes[1].legend(loc="best")
        # axes[1].set_xlim(0, 1)
        # axes[1].set_ylim(0, 1)
        #
        # st.pyplot(fig)

        if st.button('Сохранить CSV-файл с прогнозами для тестовой выборки') and test_data:
            test = load_data(test_data)
            if submit_columns:
                submit = test[submit_columns]
            else:
                submit = pd.DataFrame()

            # test = test[train_.columns]
            # prediction = create_predictions(model, test)
            submit[target_name] = predictions

            #if st.checkbox('Использовать метки классов (0/1), а не вероятности:'):
            # thresholds, scores, best_threshold = find_best_threshold(valid_target, y_valid_pred)

            # fig, axes = plt.subplots(1, 1, figsize=(15, 7))
            # axes.plot(thresholds, scores, linewidth=3)
            # axes.set_xlabel("thresholds", size=15)
            # axes.set_ylabel("F1-score", size=15)
            # axes.set_xlim(0, 1)
            # st.pyplot(fig)

            # msg = (
            #     "Значение вероятности, при котором объект относится к классу 1, "
            #     f"для данной задачи мы рекомендуем значение ({best_threshold})."
            # )
            # threshold = st.slider(
            #     label=msg, min_value=0.0, max_value=1.0, value=best_threshold, step=0.01
            # )
            # submit[target_name] = np.where(
            #     prediction >= threshold, 1, 0
            # )
            st.table(submit.head())
            tmp_download_link = download_link(submit, 'prediction.csv', 'Скачать прогнозы')
            st.markdown(tmp_download_link, unsafe_allow_html=True)

except ValueError:
    pass
