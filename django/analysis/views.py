import matplotlib
matplotlib.use('Agg')  # Agg 백엔드 사용

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from django.shortcuts import render
from django.conf import settings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .models import CustomerChurn

import os
import numpy as np
import random
import torch

def reset_seeds(seed=52):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)    # 파이썬 환경변수 시드 고정
  np.random.seed(seed)
  torch.manual_seed(seed) # cpu 연산 무작위 고정
  torch.cuda.manual_seed(seed) # gpu 연산 무작위 고정
  torch.backends.cudnn.deterministic = True  # cuda 라이브러리에서 Deterministic(결정론적)으로 예측하기 (예측에 대한 불확실성 제거 )


def main(request):
        # MySQL에서 데이터 가져오기
        # CSV 파일 경로 설정
    csv_file_path = os.path.join(settings.BASE_DIR, 'static/data/teleco-customer-churn.csv')

    # CSV 데이터 읽기
    df = pd.read_csv(csv_file_path)
    # QuerySet을 DataFrame으로 변환
    # df = pd.DataFrame(list(CustomerChurn.objects.all().values()))

    # 데이터 출력
    print(df.head())

    df['Churn_numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    
    # TotalCharges를 숫자로 변환하고 결측값 처리
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    mean_totalcharges = df['TotalCharges'].mean()
    df['TotalCharges'].fillna(mean_totalcharges, inplace=True)
    # 수치형 데이터프레임 생성
    numerical_df = df.select_dtypes(include=['number'])

    # 범주형으로 변환할 컬럼 리스트
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'SeniorCitizen', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]

    # 각 컬럼을 카테고리형으로 변환
    df[categorical_columns] = df[categorical_columns].astype('category')

    # 상관관계 함수 호출
    image_url = analysis(df)
    corellation_url = Corellaction(df, numerical_df)
    standard_compare_url = standard_compare(df, numerical_df)
    customer_char_url = customer(df)
    service_url = service(df)
    additinal_service_url = additinal_service(df)
    contract_url =  contract(df)
    important_url = important(df)
    elbow_url = cluster(df)
    cluster_bar_url = cluster_bargraph(df)
    cluster_pca_url = cluster_pca(df)

    return render(request, 'analysis.html', 
                {'image_url': image_url, 
                'corellation_url': corellation_url, 
                'standard_compare_url': standard_compare_url,
                'customer_char_url':customer_char_url,
                'service_url':service_url,
                'additinal_service_url':additinal_service_url,
                'contract_url': contract_url,
                'important_url':important_url,
                'elbow_url':elbow_url,
                'cluster_bar_url': cluster_bar_url,
                'cluster_pca_url':cluster_pca_url
                })

def analysis(df):

    # 이미지 생성
    plt.figure(figsize=(18, 6))

    # 1. tenure vs Churn
    plt.subplot(1, 3, 1)
    sns.boxplot(x='Churn', y='tenure', data=df)
    plt.title('Tenure vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Tenure (Months)')
    plt.grid(True)

    # 2. MonthlyCharges vs Churn
    plt.subplot(1, 3, 2)
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
    plt.title('MonthlyCharges vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Monthly Charges')
    plt.grid(True)

    # 3. TotalCharges vs Churn
    plt.subplot(1, 3, 3)
    sns.boxplot(x='Churn', y='TotalCharges', data=df)
    plt.title('TotalCharges vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Total Charges')
    plt.grid(True)

    plt.tight_layout()

    # 이미지 파일 저장
    image_path = os.path.join(settings.MEDIA_ROOT, 'chart.png')
    plt.savefig(image_path)
    plt.close()

    # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'chart.png')
    return image_url


def Corellaction(df, numerical_df):
    # Churn을 숫자로 변환
    df['Churn_numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 수치형 데이터프레임에 Churn_numeric 추가
    numerical_df['Churn_numeric'] = df['Churn_numeric']
    correlation_matrix = numerical_df.corr()

    # 상관관계 히트맵 생성
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix[['Churn_numeric']].sort_values(by='Churn_numeric', ascending=False), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation between Churn and Numerical Features')

    # 이미지 파일 저장 (선택 사항, 필요시)
    correlation_image_path = os.path.join(settings.MEDIA_ROOT, 'correlation_chart.png')
    plt.savefig(correlation_image_path)
    plt.close()

        # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'correlation_chart.png')
    return image_url


def standard_compare(df, numerical_df):
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    churn_stats = df.groupby('Churn')[numerical_columns].agg(['mean', 'std'])

    print(churn_stats)

    plt.figure(figsize=(18, 12))

    for i, column in enumerate(numerical_columns, 1):
        plt.subplot(2, 3, i)
        sns.barplot(x=churn_stats.index, y=churn_stats[column]['mean'])
        plt.title(f'{column} Mean by Churn')
        plt.xlabel('Churn')
        plt.ylabel(f'{column} Mean')

    for i, column in enumerate(numerical_columns, 1):
        plt.subplot(2, 3, i + 3)
        sns.barplot(x=churn_stats.index, y=churn_stats[column]['std'], color = 'orange')
        plt.title(f'{column} Std by Churn')
        plt.xlabel('Churn')
        plt.ylabel(f'{column} Std')

    plt.tight_layout()
    plt.show()

        # 이미지 파일 저장 (선택 사항, 필요시)
    standard_compare_image_path = os.path.join(settings.MEDIA_ROOT, 'standard_compare.png')
    plt.savefig(standard_compare_image_path)
    plt.close()

        # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'standard_compare.png')
    return image_url


def customer(df):
    # 고객 특성 관련 변수
    customer_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

    plt.figure(figsize=(15, 10))

    for i, column in enumerate(customer_columns, 1):
        plt.subplot(2, 2, i)
        sns.countplot(x=column, hue='Churn', data=df)
        plt.title(f'{column} vs Churn')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title='Churn')
        plt.grid(True)

    plt.tight_layout()
    plt.show()
            # 이미지 파일 저장 (선택 사항, 필요시)
    standard_compare_image_path = os.path.join(settings.MEDIA_ROOT, 'customer_characteristic.png')
    plt.savefig(standard_compare_image_path)
    plt.close()

        # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'customer_characteristic.png')
    return image_url


def service(df):
    # 전화 및 인터넷 서비스 관련 변수
    service_columns = ['PhoneService', 'MultipleLines', 'InternetService']

    plt.figure(figsize=(15, 6))

    for i, column in enumerate(service_columns, 1):
        plt.subplot(1, 3, i)
        sns.countplot(x=column, hue='Churn', data=df)
        plt.title(f'{column} vs Churn')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title='Churn')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

                # 이미지 파일 저장 (선택 사항, 필요시)
    standard_compare_image_path = os.path.join(settings.MEDIA_ROOT, 'services_url.png')
    plt.savefig(standard_compare_image_path)
    plt.close()

        # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'services_url.png')
    return image_url

def additinal_service(df):
    # 온라인 및 기타 부가 서비스 관련 변수
    online_service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies']

    plt.figure(figsize=(15, 12))

    for i, column in enumerate(online_service_columns, 1):
        plt.subplot(3, 2, i)
        sns.countplot(x=column, hue='Churn', data=df)
        plt.title(f'{column} vs Churn')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title='Churn')
        plt.grid(True)

    plt.tight_layout()
    plt.show()
                    # 이미지 파일 저장 (선택 사항, 필요시)
    standard_compare_image_path = os.path.join(settings.MEDIA_ROOT, 'additional_service.png')
    plt.savefig(standard_compare_image_path)
    plt.close()

        # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'additional_service.png')
    return image_url


def contract(df):
    # 계약 및 결제 관련 변수
    contract_columns = ['Contract', 'PaperlessBilling', 'PaymentMethod']

    plt.figure(figsize=(15, 8))

    for i, column in enumerate(contract_columns, 1):
        plt.subplot(1, 3, i)
        sns.countplot(x=column, hue='Churn', data=df)
        plt.title(f'{column} vs Churn')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title='Churn')
        plt.grid(True)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
                        # 이미지 파일 저장 (선택 사항, 필요시)
    standard_compare_image_path = os.path.join(settings.MEDIA_ROOT, 'contract.png')
    plt.savefig(standard_compare_image_path)
    plt.close()

        # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'contract.png')
    return image_url

def important(df):
    # 상관관계가 높은 범주형 변수 목록
    important_category_cols = ['Contract', 'OnlineSecurity', 'TechSupport', 'InternetService', 'PaymentMethod']

    plt.figure(figsize=(15, 12))

    for i, column in enumerate(important_category_cols, 1):
        plt.subplot(3, 2, i)
        sns.countplot(x=column, hue='Churn', data=df)
        plt.title(f'{column} vs Churn')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title='Churn')
        plt.grid(True)
        if column == 'PaymentMethod':
            plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    standard_compare_image_path = os.path.join(settings.MEDIA_ROOT, 'important.png')
    plt.savefig(standard_compare_image_path)
    plt.close()

        # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'important.png')
    return image_url


def cluster(df):
    reset_seeds()
    cluster_df = df.apply(lambda x: pd.factorize(x)[0]).drop(['customerID', 'Churn', 'Churn_numeric'], axis=1)
    cluster_df.head()


    
    scale = StandardScaler()
    scaled_df = pd.DataFrame(scale.fit_transform(cluster_df), columns=cluster_df.columns)


# 정규화된 데이터 출력
    wcss = []
    K_range = range(1, 11)

    # K 값에 따른 WCSS 계산
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=52)
        kmeans.fit(scaled_df)
        wcss.append(kmeans.inertia_)  # WCSS 값 저장

    # 엘보우 그래프 그리기
    plt.plot(K_range, wcss, marker='o')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS')
    plt.show()
    
    elbow_image_path = os.path.join(settings.MEDIA_ROOT, 'elbow_graph.png')
    plt.savefig(elbow_image_path)
    plt.close()

        # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'elbow_graph.png')
    return image_url

def cluster_bargraph(df):
    reset_seeds()
    df.info()
    cluster_df = df.apply(lambda x: pd.factorize(x)[0]).drop(['customerID', 'Churn', 'Churn_numeric'], axis=1)

    scale = StandardScaler()
    scaled_df = pd.DataFrame(scale.fit_transform(cluster_df), columns=cluster_df.columns)
    # K = 3으로 K-means 모델 생성
    kmeans = KMeans(n_clusters=3, random_state=52)

    # 데이터를 K-means로 클러스터링
    kmeans.fit(scaled_df)

    # 각 데이터 포인트의 클러스터 할당 (레이블)
    cluster_labels = kmeans.labels_

    # 클러스터 레이블을 원본 데이터에 추가 (선택 사항)
    scaled_df['Cluster'] = cluster_labels

    # scaled_df에 원본 데이터의 'Churn' 컬럼을 추가
    scaled_df['Churn'] = df['Churn']

    # 클러스터별 Churn 비율 계산
    churn_rate_by_cluster = scaled_df.groupby('Cluster')['Churn'].value_counts(normalize=True).unstack()


    # 결과 출력
    print('churn:',churn_rate_by_cluster)

    # 클러스터별 Churn 비율을 바차트로 시각화
    ax = churn_rate_by_cluster.plot(kind='bar', stacked=True, color=['skyblue', 'red'])

    plt.title('Churn Rate by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')

    # 이탈한 고객 (Churn = 1, 빨간색 막대)
    for i, container in enumerate(ax.containers):
        if i == 1:
            ax.bar_label(container, fmt='%.2f', label_type='center', color='white')

    plt.show()
    cluster_bar_image_path = os.path.join(settings.MEDIA_ROOT, 'cluster_bar_graph.png')
    plt.savefig(cluster_bar_image_path)
    plt.close()

        # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'cluster_bar_graph.png')
    return image_url


def cluster_pca(df):
    reset_seeds()
    cluster_df = df.apply(lambda x: pd.factorize(x)[0]).drop(['customerID', 'Churn', 'Churn_numeric'], axis=1)
    scale = StandardScaler()
    scaled_df = pd.DataFrame(scale.fit_transform(cluster_df), columns=cluster_df.columns)
        
    # K = 3으로 K-means 모델 생성
    kmeans = KMeans(n_clusters=3, random_state=52)

    # 데이터를 K-means로 클러스터링
    kmeans.fit(scaled_df)

    # 각 데이터 포인트의 클러스터 할당 (레이블)
    cluster_labels = kmeans.labels_

    # 클러스터 레이블을 원본 데이터에 추가 (선택 사항)
    scaled_df['Cluster'] = cluster_labels


    # PCA를 사용하여 2차원으로 차원 축소
    pca = PCA(n_components=2, random_state=52)
    pca_components = pca.fit_transform(scaled_df.drop(columns=['Cluster']))

    # 차원 축소된 데이터를 이용해 시각화
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=scaled_df['Cluster'], cmap='viridis')
    plt.title('K-Means Clustering with PCA (K=3)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    cluster_pca_image_path = os.path.join(settings.MEDIA_ROOT, 'cluster_pca.png')
    plt.savefig(cluster_pca_image_path)
    plt.show()
    plt.close()

        # 이미지 URL 생성
    image_url = os.path.join(settings.MEDIA_URL, 'cluster_pca.png')
    return image_url