import matplotlib
matplotlib.use('Agg')  # Agg 백엔드 사용

import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from django.shortcuts import render
from django.conf import settings
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc  # 임포트 추가

def main_view(request):
    # 모델 로드
    model_path = os.path.join(settings.BASE_DIR, 'static', 'pkl', 'HistGradientBoostingClassifier.pkl')
    model = joblib.load(model_path)

    # 데이터 로드
    data_path = os.path.join(settings.BASE_DIR, 'static', 'pkl', 'data.pkl')
    X_test_transformed, y_test = joblib.load(data_path)

    # 예측 값 생성
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

    # 혼동 행렬 계산
    cm = confusion_matrix(y_test, y_pred)

    # ROC AUC 계산
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # F1 스코어 계산
    f1 = f1_score(y_test, y_pred)

    # 추가 평가지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Precision-Recall 곡선 및 AUC 계산
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
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

    # 이미지 URL 생성
    confusion_matrix_image_url = os.path.join(settings.MEDIA_URL, 'confusion_matrix.png')
    roc_curve_image_url = os.path.join(settings.MEDIA_URL, 'roc_curve.png')
    pr_curve_image_url = os.path.join(settings.MEDIA_URL, 'pr_curve.png')

    return render(request, 'main.html', {
        'confusion_matrix_image_url': confusion_matrix_image_url,
        'roc_curve_image_url': roc_curve_image_url,
        'pr_curve_image_url': pr_curve_image_url,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc
    })







# import matplotlib
# matplotlib.use('Agg')  # Agg 백엔드 사용

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import os
# import joblib
# from django.shortcuts import render
# from django.conf import settings
# from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from main.transformers import DataCleaning, FeatureEngineering, ScaleAndTransform
# from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc  # 임포트 추가
# import numpy as np
# from main.train_and_save_model import model, X_test_transformed, y_test

# # # 데이터 로드
# # def load_data():
# #     df = pd.read_csv(os.path.join(settings.BASE_DIR, 'static/data/teleco-customer-churn.csv'))

# #     return df

# # def main_view(request):
# #     # 필요한 클래스를 미리 정의
# #     globals().update({
# #         'DataCleaning': DataCleaning,
# #         'FeatureEngineering': FeatureEngineering,
# #         'ScaleAndTransform': ScaleAndTransform
# #     })

# #     # 데이터 로드 
# #     df = load_data()
    
# #     # 특성과 레이블 분리
# #     X = df.drop(columns=['Churn'])
# #     y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# #     # 학습 데이터와 테스트 데이터 분리
# #     _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #     # 모델 로드
# #     model_path = os.path.join(settings.BASE_DIR, 'static', 'pkl', 'HistGradientBoostingClassifier.pkl')
# #     model = joblib.load(model_path)

#     # 예측 값 생성

#     y_pred = model.predict(X_test_transformed)
#     y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

#     # 혼동 행렬 계산
#     cm = confusion_matrix(y_test, y_pred)

#     # ROC AUC 계산
#     roc_auc = roc_auc_score(y_test, y_pred_proba)

#     # F1 스코어 계산
#     f1 = f1_score(y_test, y_pred)

#     # 추가 평가지표 계산
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)

#     # Precision-Recall 곡선 및 AUC 계산
#     precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
#     pr_auc = auc(recall_vals, precision_vals)


#     # 혼동 행렬 시각화
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     confusion_matrix_image_path = os.path.join(settings.MEDIA_ROOT, 'confusion_matrix.png')
#     plt.savefig(confusion_matrix_image_path)
#     plt.close()

#     # ROC 곡선 시각화
#     fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#     plt.figure(figsize=(10, 7))
#     plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     roc_curve_image_path = os.path.join(settings.MEDIA_ROOT, 'roc_curve.png')
#     plt.savefig(roc_curve_image_path)
#     plt.close()

#     # Precision-Recall 곡선 시각화
#     plt.figure(figsize=(10, 7))
#     plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc="lower left")
#     pr_curve_image_path = os.path.join(settings.MEDIA_ROOT, 'pr_curve.png')
#     plt.savefig(pr_curve_image_path)
#     plt.close()

#     # 이미지 URL 생성
#     confusion_matrix_image_url = os.path.join(settings.MEDIA_URL, 'confusion_matrix.png')
#     roc_curve_image_url = os.path.join(settings.MEDIA_URL, 'roc_curve.png')
#     pr_curve_image_url = os.path.join(settings.MEDIA_URL, 'pr_curve.png')

#     # return render(request, 'main.html', {
#     #     'confusion_matrix_image_url': confusion_matrix_image_url,
#     #     'roc_curve_image_url': roc_curve_image_url,
#     #     'pr_curve_image_url': pr_curve_image_url,
#     #     'roc_auc': roc_auc,
#     #     'f1_score': f1,
#     #     'accuracy': accuracy,
#     #     'precision': precision,
#     #     'recall': recall,
#     #     'pr_auc': pr_auc
#     # })