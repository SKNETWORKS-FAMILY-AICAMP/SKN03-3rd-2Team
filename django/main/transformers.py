import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer

class DataCleaning(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        X['SeniorCitizen'] = X['SeniorCitizen'].astype('category')
        Q1 = X[['MonthlyCharges', 'TotalCharges']].quantile(0.25)
        Q3 = X[['MonthlyCharges', 'TotalCharges']].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X['TotalCharges'].fillna(X['TotalCharges'].median(), inplace=True)
        X['MonthlyCharges'] = X['MonthlyCharges'].clip(lower=lower_bound['MonthlyCharges'], upper=upper_bound['MonthlyCharges'])
        X['TotalCharges'] = X['TotalCharges'].clip(lower=lower_bound['TotalCharges'], upper=upper_bound['TotalCharges'])
        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'])
        return X

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['LongTermCustomer'] = X['tenure'].apply(lambda x: 1 if x >= 12 else 0)
        X['AvgMonthlyCharges'] = X.apply(lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] != 0 else 0, axis=1)
        X['AvgMonthlyCharges'].fillna(0, inplace=True)
        X['MultipleServices'] = ((X['InternetService'] != 'No') & (X['PhoneService'] == 'Yes')).astype(int)
        X['FixedContract'] = X['Contract'].apply(lambda x: 1 if x != 'Month-to-month' else 0)
        X['IsPaperless'] = X['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)
        X['AutoPayment'] = X['PaymentMethod'].apply(lambda x: 1 if 'automatic' in x.lower() else 0)
        bins = [0, 12, 24, 48, 72]
        labels = ['0-12months', '12-24months', '24-48months', '48-72months']
        X['TenureGroup'] = pd.cut(X['tenure'], bins=bins, labels=labels, include_lowest=True)
        X['CustomerType'] = X.apply(lambda row: 'Senior' if row['SeniorCitizen'] == 1 else
                                    ('Partner' if row['Partner'] == 'Yes' else
                                    ('Dependent' if row['Dependents'] == 'Yes' else 'Individual')), axis=1)
        services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        X['TotalServicesUsed'] = X[services].apply(lambda x: sum(x != 'No'), axis=1)
        X['AgeCategory'] = X['SeniorCitizen'].apply(lambda x: 'Senior' if x == 1 else 'Adult')
        X['AboveAverageCharges'] = X['MonthlyCharges'].apply(lambda x: 1 if x > X['MonthlyCharges'].mean() else 0)
        X['IsFiberOptic'] = X['InternetService'].apply(lambda x: 1 if x == 'Fiber optic' else 0)
        X['TechSupportSecurity'] = X.apply(lambda row: 1 if row['TechSupport'] == 'Yes' and row['OnlineSecurity'] == 'Yes' else 0, axis=1)
        X['IsElectronicCheck'] = X['PaymentMethod'].apply(lambda x: 1 if 'Electronic check' in x else 0)
        high_charge_threshold = X['MonthlyCharges'].quantile(0.75)
        X['IsHighMonthlyCharge'] = X['MonthlyCharges'].apply(lambda x: 1 if x > high_charge_threshold else 0)
        X['UsesMultipleServices'] = X.apply(lambda row: 1 if row['InternetService'] != 'No' and row['StreamingTV'] == 'Yes' and row['MultipleLines'] == 'Yes' else 0, axis=1)
        X['ShortTermContract'] = X.apply(lambda row: 1 if row['Contract'] == 'Month-to-month' and row['tenure'] < 12 else 0, axis=1)
        return X

class ScaleAndTransform(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2):
        self.degree = degree
        self.preprocessor = None

    def fit(self, X, y=None):
        X = X.copy()
        continuous_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('poly', PolynomialFeatures(degree=self.degree, interaction_only=False, include_bias=False))
        ])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, continuous_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X_processed = self.preprocessor.transform(X)
        feature_names = self.preprocessor.get_feature_names_out()
        if X_processed.shape[1] != len(feature_names):
            raise ValueError("변환된 데이터의 열 개수와 피처 이름의 개수가 일치하지 않습니다.")
        return pd.DataFrame(X_processed, columns=feature_names)