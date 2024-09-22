from django.shortcuts import render
import os
import pandas as pd

def prediction(request):
    # CSV 파일 경로 설정 (static 디렉토리 또는 다른 위치에서 파일 읽기)
    csv_file_path = os.path.join(os.path.dirname(__file__), 'static/data/teleco-customer-churn.csv')
    
    # CSV 데이터 읽기
    df = pd.read_csv(csv_file_path)
    
    # 필요한 데이터 준비
    churn_data = df.groupby('Churn').agg({
        'tenure': ['min', '25%', '50%', '75%', 'max'],
        'MonthlyCharges': ['min', '25%', '50%', '75%', 'max'],
        'TotalCharges': ['min', '25%', '50%', '75%', 'max']
    }).reset_index()


    
    # boxplot 데이터 변환
    boxplot_data = {
        'tenure': {
            'Yes': churn_data[churn_data['Churn'] == 'Yes']['tenure'].values.tolist(),
            'No': churn_data[churn_data['Churn'] == 'No']['tenure'].values.tolist()
        },
        'MonthlyCharges': {
            'Yes': churn_data[churn_data['Churn'] == 'Yes']['MonthlyCharges'].values.tolist(),
            'No': churn_data[churn_data['Churn'] == 'No']['MonthlyCharges'].values.tolist()
        },
        'TotalCharges': {
            'Yes': churn_data[churn_data['Churn'] == 'Yes']['TotalCharges'].values.tolist(),
            'No': churn_data[churn_data['Churn'] == 'No']['TotalCharges'].values.tolist()
        }
    }

    # 데이터 템플릿에 전달
    return render(request, 'prediction.html', {'chart_data': boxplot_data})
