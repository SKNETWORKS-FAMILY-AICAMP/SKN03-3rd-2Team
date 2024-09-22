import matplotlib
matplotlib.use('Agg')  # Agg 백엔드 사용

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from django.shortcuts import render
from django.conf import settings

def analysis(request):
    # CSV 파일 경로 설정
    csv_file_path = os.path.join(settings.BASE_DIR, 'static/data/teleco-customer-churn.csv')

    # CSV 데이터 읽기
    df = pd.read_csv(csv_file_path)

    # TotalCharges를 숫자로 변환하고 결측값 처리
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)  # 결측값을 0으로 대체

    # 수치형 데이터프레임 생성
    numerical_df = df.select_dtypes(include=['number'])

    # 상관관계 함수 호출
    corellation_url = Corellaction(df, numerical_df)
    standard_compare_url = standard_compare(df, numerical_df)
    customer_char_url = customer(df)
    service_url = service(df)
    additinal_service_url = additinal_service(df)
    contract_url =  contract(df)
    important_url = important(df)

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
    
    return render(request, 'analysis.html', 
                  {'image_url': image_url, 'corellation_url': corellation_url, 
                   'standard_compare_url': standard_compare_url,
                   'customer_char_url':customer_char_url,
                   'service_url':service_url,
                   'additinal_service_url':additinal_service_url,
                   'contract_url': contract_url,
                   'important_url':important_url
                   })


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