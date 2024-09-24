import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from main.transformers import DataCleaning, FeatureEngineering, ScaleAndTransform  # 임포트 추가

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # 데이터 전처리 로직 추가
    return df

# 데이터 로드 및 전처리
file_path = os.path.join('static', 'data', 'teleco-customer-churn.csv')
df = load_and_preprocess_data(file_path)
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
pipeline.fit(X_train, y_train)
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# 모델 로드
model_path = os.path.join('static', 'pkl', 'HistGradientBoostingClassifier.pkl')
model = HistGradientBoostingClassifier(random_state=42)

# 모델 학습
model.fit(X_train_transformed, y_train)

# 모델 및 파이프라인 저장
joblib.dump(model, model_path)
pipeline_path = os.path.join('static', 'pkl', 'pipeline.pkl')
joblib.dump(pipeline, pipeline_path)

# 파일이 생성되었는지 확인
if os.path.exists(model_path):
    print(f"Model file {model_path} has been created successfully.")
else:
    print(f"Failed to create model file {model_path}.")

if os.path.exists(pipeline_path):
    print(f"Pipeline file {pipeline_path} has been created successfully.")
else:
    print(f"Failed to create pipeline file {pipeline_path}.")
# import pandas as pd
# import os
# import joblib
# import sys
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.pipeline import Pipeline
# import numpy as np
# from .views import load_data

# # 상위 디렉토리를 모듈 경로에 추가
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from main.transformers import DataCleaning, FeatureEngineering, ScaleAndTransform  # 임포트 추가

# # 데이터 로드 및 전처리 함수
# def load_and_preprocess_data(file_path):
#     df = pd.read_csv(file_path)
#     # 데이터 전처리 로직 추가
#     return df

# # 데이터 로드 및 전처리
# file_path = os.path.join('..', 'static', 'data', 'teleco-customer-churn.csv')
# df = load_and_preprocess_data(file_path)
# pipeline = Pipeline([
#     ('cleaning', DataCleaning()),
#     ('engineering', FeatureEngineering()),
#     ('scaling', ScaleAndTransform(degree=2))
# ])

# # 특성과 레이블 분리
# X = df.drop(columns=['Churn'])
# y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# # 학습 데이터와 테스트 데이터 분리
# _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 파이프라인을 사용하여 데이터 전처리 및 피처 엔지니어링
# X_train_transformed = pipeline.fit_transform(_, _)
# X_test_transformed = pipeline.transform(X_test)

# model = load_data()

# # 모델 학습
# model = load_data()(random_state=42)
# model.fit(X_train_transformed, _)

# # 모델 저장
# model_path = os.path.join('..', 'static', 'pkl', 'HistGradientBoostingClassifier.pkl')
# joblib.dump(model, model_path)