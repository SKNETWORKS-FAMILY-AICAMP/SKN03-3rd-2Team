import matplotlib
matplotlib.use('Agg')  # Agg 백엔드 사용

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import joblib
from django.shortcuts import render
from django.conf import settings
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from main.transformers import DataCleaning, FeatureEngineering, ScaleAndTransform  # 임포트 추가
import numpy as np

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data():
    df = pd.read_csv(os.path.join(settings.BASE_DIR, 'static/data/teleco-customer-churn.csv'))
    # 데이터 전처리 로직 추가
    return df

def main_view(request):
    # 필요한 클래스를 미리 정의
    globals().update({
        'DataCleaning': DataCleaning,
        'FeatureEngineering': FeatureEngineering,
        'ScaleAndTransform': ScaleAndTransform
    })

    # 데이터 로드 및 전처리
    df = load_and_preprocess_data()
    pipeline = Pipeline([
        ('cleaning', DataCleaning()),
        ('engineering', FeatureEngineering()),
        ('scaling', ScaleAndTransform(degree=2))
    ])
    
    # 특성과 레이블 분리
    X = df.drop(columns=['Churn'])
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 파이프라인을 사용하여 데이터 전처리 및 피처 엔지니어링
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)

    # 모델 로드
    model_path = os.path.join(settings.BASE_DIR, 'static', 'pkl', 'HistGradientBoostingClassifier.pkl')
    model = joblib.load(model_path)

    # 예측 값 생성
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

    # 혼동 행렬 계산
    cm = confusion_matrix(y_test, y_pred)

    # ROC AUC 계산
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # F1 스코어 계산
    f1 = f1_score(y_test, y_pred)

    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    confusion_matrix_image_path = os.path.join(settings.MEDIA_ROOT, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_image_path)
    plt.close()

    # ROC 곡선 시각화
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    roc_curve_image_path = os.path.join(settings.MEDIA_ROOT, 'roc_curve.png')
    plt.savefig(roc_curve_image_path)
    plt.close()

    # 이미지 URL 생성
    confusion_matrix_image_url = os.path.join(settings.MEDIA_URL, 'confusion_matrix.png')
    roc_curve_image_url = os.path.join(settings.MEDIA_URL, 'roc_curve.png')

    return render(request, 'main.html', {
        'confusion_matrix_image_url': confusion_matrix_image_url,
        'roc_curve_image_url': roc_curve_image_url,
        'roc_auc': roc_auc,
        'f1_score': f1
    })