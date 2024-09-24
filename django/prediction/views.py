import matplotlib

matplotlib.use("Agg")  # Agg 백엔드 사용

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import joblib
from django.shortcuts import render
from django.conf import settings
from sklearn.pipeline import Pipeline
import numpy as np


# 데이터 로드 및 전처리 함수
def load_data(num):
    df = pd.read_csv(
        os.path.join(settings.BASE_DIR, "static/data/teleco-customer-churn.csv")
    )
    if num == 1:
        df2 = pd.read_csv(
            os.path.join(
                settings.BASE_DIR,
                "static/data/transformed_modified_telco_customer_data_1.csv",
            )
        )
    else:
        df2 = pd.read_csv(
            os.path.join(
                settings.BASE_DIR, "static/data/modified_telco_customer_data_2.csv"
            )
        )
    # 데이터 전처리 로직 추가
    return df, df2


def main_view(request):

    # 데이터 로드 및 전처리
    df, df2 = load_data(1)

    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    # 모델 로드
    model_path = os.path.join(settings.BASE_DIR, "static", "pkl", "CatBoost.pkl")
    model = joblib.load(model_path)

    # 예측 값 생성
    y_pred = model.predict(df2)
    y_No = len(y[y == 0])  # y에서 0인 경우
    y_Yes = len(y[y == 1])  # y에서 1인 경우

    y_pred_No = len(y_pred[y_pred == 0])  # y_pred에서 0인 경우
    y_pred_Yes = len(y_pred[y_pred == 1])  # y_pred에서 1인 경우

    ## 원본 데이터 도넛 차트
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor("white")
    ax = fig.add_subplot()
    pie = ax.pie(
        [y_No, y_Yes],  # <- 위 변수로 수정
        startangle=90,
        counterclock=False,
        autopct=lambda p: "{:.2f}%".format(p),
        wedgeprops=dict(width=0.7),
    )
    plt.legend(pie[0], ["Churn : No", "Churn : Yes"])  ## 범례 표시
    plt.title("original Data 1")
    Donut_image_path = os.path.join(settings.MEDIA_ROOT, "Donut_chart_origin.png")
    plt.savefig(Donut_image_path)
    plt.close()

    ## 첫 번째 수정 데이터 도넛 차트
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor("white")
    ax = fig.add_subplot()
    pie = ax.pie(
        [y_pred_No, y_pred_Yes],  # <- 위 변수로 수정
        startangle=90,
        counterclock=False,
        autopct=lambda p: "{:.2f}%".format(p),
        wedgeprops=dict(width=0.7),
    )
    plt.legend(pie[0], ["Churn : No", "Churn : Yes"])  ## 범례 표시
    plt.title("Edited Data 1")
    Donut_image_path = os.path.join(settings.MEDIA_ROOT, "Donut_chart_1.png")
    plt.savefig(Donut_image_path)
    plt.close()

    Donut_image_url_origin = os.path.join(settings.MEDIA_URL, "Donut_chart_origin.png")
    Donut_image_url = os.path.join(settings.MEDIA_URL, "Donut_chart_1.png")

    return render(
        request,
        "prediction.html",
        {
            "Donut_image_url": Donut_image_url,
            "Donut_image_url_origin": Donut_image_url_origin,
        },
    )


def more_view(request):

    # 데이터 로드 및 전처리
    df, df2 = load_data(2)

    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    # 모델 로드
    model_path = os.path.join(
        settings.BASE_DIR, "static", "pkl", "HistGradientBoostingClassifier.pkl"
    )
    model = joblib.load(model_path)

    # 예측 값 생성
    y_pred = model.predict(df2)
    y_No = len(y[y == 0])  # y에서 0인 경우
    y_Yes = len(y[y == 1])  # y에서 1인 경우

    y_pred_No = len(y_pred[y_pred == 0])  # y_pred에서 0인 경우
    y_pred_Yes = len(y_pred[y_pred == 1])  # y_pred에서 1인 경우

    ## 원본 데이터 도넛 차트
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor("white")
    ax = fig.add_subplot()
    pie = ax.pie(
        [y_No, y_Yes],  # <- 위 변수로 수정
        startangle=90,
        counterclock=False,
        autopct=lambda p: "{:.2f}%".format(p),
        wedgeprops=dict(width=0.7),
    )
    plt.legend(pie[0], ["Churn : No", "Churn : Yes"])  ## 범례 표시
    plt.title("original Data 1")
    Donut_image_path = os.path.join(settings.MEDIA_ROOT, "Donut_chart_origin.png")
    plt.savefig(Donut_image_path)
    plt.close()

    ## 첫 번째 수정 데이터 도넛 차트
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor("white")
    ax = fig.add_subplot()
    pie = ax.pie(
        [y_pred_No, y_pred_Yes],  # <- 위 변수로 수정
        startangle=90,
        counterclock=False,
        autopct=lambda p: "{:.2f}%".format(p),
        wedgeprops=dict(width=0.7),
    )
    plt.legend(pie[0], ["Churn : No", "Churn : Yes"])  ## 범례 표시
    plt.title("Edited Data 2")
    Donut_image_path = os.path.join(settings.MEDIA_ROOT, "Donut_chart_2.png")
    plt.savefig(Donut_image_path)
    plt.close()

    Donut_image_url_origin = os.path.join(settings.MEDIA_URL, "Donut_chart_origin.png")
    Donut_image_url_2 = os.path.join(settings.MEDIA_URL, "Donut_chart_2.png")

    return render(
        request,
        "prediction_2.html",
        {
            "Donut_image_url_2": Donut_image_url_2,
            "Donut_image_url_origin": Donut_image_url_origin,
        },
    )
