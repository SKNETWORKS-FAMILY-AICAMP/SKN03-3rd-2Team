import matplotlib
matplotlib.use('Agg')  # Agg 백엔드 사용
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import joblib
from django.shortcuts import render
from django.conf import settings
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc  # 임포트 추가
import sys
from prediction.transformers import (
    DataCleaning,
    FeatureEngineering,
    ScaleAndTransform,
    FeatureSelection,
    CorrelationFilter,
)

def main_view(request):
    # 모델 로드
    sys.modules["__main__"].DataCleaning = DataCleaning
    sys.modules["__main__"].FeatureEngineering = FeatureEngineering
    sys.modules["__main__"].ScaleAndTransform = ScaleAndTransform
    sys.modules["__main__"].FeatureSelection = FeatureSelection
    sys.modules["__main__"].CorrelationFilter = CorrelationFilter
    model_path = os.path.join(settings.BASE_DIR, 'static', 'pkl', 'CatBoost.pkl')
    model = joblib.load(model_path)


    def convert_churn_to_binary(df):
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        return df
    

    # 데이터 로드
    df = pd.read_csv(
        os.path.join(settings.BASE_DIR, "static/data/teleco-customer-churn.csv")
    )
        
    df = convert_churn_to_binary(df)
    X = df.drop(columns=['Churn'])
    y = df['Churn']  # 타겟 변수는 Churn


    # 예측 값 생성
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # 혼동 행렬 계산
    cm = confusion_matrix(y, y_pred)

    # 정규화된 혼동 행렬 계산 및 시각화 추가
    norm_conf_mx = confusion_matrix(y, y_pred, normalize="true")
    plt.figure(figsize=(7, 5))
    sns.heatmap(norm_conf_mx, annot=True, cmap="coolwarm", linewidth=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    norm_confusion_matrix_image_path = os.path.join(settings.MEDIA_ROOT, 'norm_confusion_matrix.png')
    plt.savefig(norm_confusion_matrix_image_path)
    plt.close()

    # ROC AUC 계산
    roc_auc = roc_auc_score(y, y_pred_proba)

    # F1 스코어 계산
    f1 = f1_score(y, y_pred)

    # 추가 평가지표 계산
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    # Precision-Recall 곡선 및 AUC 계산
    precision_vals, recall_vals, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)

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
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
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

    # Precision-Recall 곡선 시각화
    plt.figure(figsize=(10, 7))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    pr_curve_image_path = os.path.join(settings.MEDIA_ROOT, 'pr_curve.png')
    plt.savefig(pr_curve_image_path)
    plt.close()

    # 소프트맥스 함수(Softmax Function)
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    softmax_probs = softmax(y_pred_proba)

    plt.figure(figsize=(10, 7))
    plt.hist(softmax_probs, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Softmax Probability')
    plt.ylabel('Frequency')
    plt.title('Softmax Probability Distribution')
    softmax_image_path = os.path.join(settings.MEDIA_ROOT, 'softmax_distribution.png')
    plt.savefig(softmax_image_path)
    plt.close()


    # 이미지 URL 생성
    confusion_matrix_image_url = os.path.join(settings.MEDIA_URL, 'confusion_matrix.png')
    norm_confusion_matrix_image_url = os.path.join(settings.MEDIA_URL, 'norm_confusion_matrix.png')
    roc_curve_image_url = os.path.join(settings.MEDIA_URL, 'roc_curve.png')
    pr_curve_image_url = os.path.join(settings.MEDIA_URL, 'pr_curve.png')
    softmax_image_url = os.path.join(settings.MEDIA_URL, 'softmax_distribution.png')  

    return render(request, 'main.html', {
        'confusion_matrix_image_url': confusion_matrix_image_url,
        'norm_confusion_matrix_image_url': norm_confusion_matrix_image_url,
        'roc_curve_image_url': roc_curve_image_url,
        'pr_curve_image_url': pr_curve_image_url,
        'softmax_image_url': softmax_image_url,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc
    })