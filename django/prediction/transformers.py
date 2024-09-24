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
        # 데이터를 수정 가능하게 복사
        X = X.copy()

        # TotalCharges를 숫자형으로 변환
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        # SeniorCitizen을 범주형으로 변환
        X["SeniorCitizen"] = X["SeniorCitizen"].astype("category")

        # IQR 계산 및 이상치 처리
        Q1 = X[["MonthlyCharges", "TotalCharges"]].quantile(0.25)
        Q3 = X[["MonthlyCharges", "TotalCharges"]].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X["TotalCharges"].fillna(X["TotalCharges"].median(), inplace=True)  # => 0으로

        X["MonthlyCharges"] = X["MonthlyCharges"].clip(
            lower=lower_bound["MonthlyCharges"], upper=upper_bound["MonthlyCharges"]
        )
        X["TotalCharges"] = X["TotalCharges"].clip(
            lower=lower_bound["TotalCharges"], upper=upper_bound["TotalCharges"]
        )

        # customerID 열 제거
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])

        return X


# 피처 생성 함수와 연동할 클래스를 만들어서 파이프라인에 추가
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 데이터를 수정 가능하게 복사
        X = X.copy()

        # 1. 고객의 장기/단기 구분 (가입 기간이 12개월 이상이면 장기 고객)
        X["LongTermCustomer"] = X["tenure"].apply(lambda x: 1 if x >= 12 else 0)

        # 2. 월 요금 대비 총 요금 비율 (총 사용 기간 동안의 평균 요금)
        X["AvgMonthlyCharges"] = X.apply(
            lambda row: (
                row["TotalCharges"] / row["tenure"] if row["tenure"] != 0 else 0
            ),
            axis=1,
        )
        X["AvgMonthlyCharges"].fillna(0, inplace=True)  # 결측치 처리

        # 3. 여러 가지 서비스 이용 여부 (Internet + Phone)
        X["MultipleServices"] = (
            (X["InternetService"] != "No") & (X["PhoneService"] == "Yes")
        ).astype(int)

        # 4. 계약 유형에 따른 고정 계약 여부 (Month-to-month를 제외한 장기 계약)
        X["FixedContract"] = X["Contract"].apply(
            lambda x: 1 if x != "Month-to-month" else 0
        )

        # 5. 무서류 청구 여부에 따른 디지털 고객 여부
        X["IsPaperless"] = X["PaperlessBilling"].apply(lambda x: 1 if x == "Yes" else 0)

        # 6. 결제 방법의 자동 여부 (자동 이체 vs 기타 결제 방식)
        X["AutoPayment"] = X["PaymentMethod"].apply(
            lambda x: 1 if "automatic" in x.lower() else 0
        )

        # 7. 가입 기간(tenure)에 따른 그룹화
        bins = [0, 12, 24, 48, 72]
        labels = ["0-12months", "12-24months", "24-48months", "48-72months"]
        X["TenureGroup"] = pd.cut(
            X["tenure"], bins=bins, labels=labels, include_lowest=True
        )

        # 8. 고객 유형 (SeniorCitizen, Partner, Dependents의 조합)
        X["CustomerType"] = X.apply(
            lambda row: (
                "Senior"
                if row["SeniorCitizen"] == 1
                else (
                    "Partner"
                    if row["Partner"] == "Yes"
                    else ("Dependent" if row["Dependents"] == "Yes" else "Individual")
                )
            ),
            axis=1,
        )

        # 9. 사용 중인 서비스의 수 (서비스 != 'No')
        services = [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        X["TotalServicesUsed"] = X[services].apply(lambda x: sum(x != "No"), axis=1)

        # 10. 고객 나이를 범주화 (SeniorCitizen을 기반으로)
        X["AgeCategory"] = X["SeniorCitizen"].apply(
            lambda x: "Senior" if x == 1 else "Adult"
        )

        # 11. 요금이 평균 이상인지 여부 (MonthlyCharges가 평균보다 큰 경우)
        X["AboveAverageCharges"] = X["MonthlyCharges"].apply(
            lambda x: 1 if x > X["MonthlyCharges"].mean() else 0
        )

        # 12. Fiber Optic 서비스 사용 여부
        X["IsFiberOptic"] = X["InternetService"].apply(
            lambda x: 1 if x == "Fiber optic" else 0
        )

        # 13. 기술 지원 및 보안 서비스를 모두 사용하는 고객 여부
        X["TechSupportSecurity"] = X.apply(
            lambda row: (
                1
                if row["TechSupport"] == "Yes" and row["OnlineSecurity"] == "Yes"
                else 0
            ),
            axis=1,
        )

        # 14. 전자 수표 사용 고객 여부
        X["IsElectronicCheck"] = X["PaymentMethod"].apply(
            lambda x: 1 if "Electronic check" in x else 0
        )

        # 15. 고요금 고객 여부 (월 요금을 기준으로 상위 25%)
        X["IsHighMonthlyCharge"] = X["MonthlyCharges"].apply(
            lambda x: 1 if x > X["MonthlyCharges"].quantile(0.75) else 0
        )

        # 16. 여러 서비스를 함께 사용하는 고객 (인터넷, 스트리밍, 다중 회선 사용 여부)
        X["UsesMultipleServices"] = X.apply(
            lambda row: (
                1
                if row["InternetService"] != "No"
                and row["StreamingTV"] == "Yes"
                and row["MultipleLines"] == "Yes"
                else 0
            ),
            axis=1,
        )

        # 17. 계약 유형과 가입 기간에 따른 고객 분류 (월별 계약자 중 가입 기간이 짧은 고객)
        X["ShortTermContract"] = X.apply(
            lambda row: (
                1 if row["Contract"] == "Month-to-month" and row["tenure"] < 12 else 0
            ),
            axis=1,
        )

        # 18. 계약 만료 임박 여부 (Month-to-month 계약을 사용 중이고, tenure가 낮은 경우)
        X["ContractEndingSoon"] = X.apply(
            lambda row: (
                1 if row["Contract"] == "Month-to-month" and row["tenure"] < 6 else 0
            ),
            axis=1,
        )

        # 19. 요금 변동성 (MonthlyCharges가 평균 이상으로 급격히 증가한 고객)
        X["HighChargeVolatility"] = (
            X["MonthlyCharges"]
            .diff()
            .abs()
            .apply(lambda x: 1 if x > X["MonthlyCharges"].std() else 0)
        )

        # 20. 서비스 패턴에 따른 고객 분류 (인터넷만 사용하거나, 전화 서비스만 사용 중인 고객)
        X["SingleServiceCustomer"] = X.apply(
            lambda row: (
                1
                if (row["InternetService"] == "No" and row["PhoneService"] == "Yes")
                or (row["InternetService"] != "No" and row["PhoneService"] == "No")
                else 0
            ),
            axis=1,
        )

        # 21. 서비스 중단 위험 고객 (기술 지원이나 보안 서비스를 사용하지 않으면서 고요금인 고객)
        X["ServiceRisk"] = X.apply(
            lambda row: (
                1
                if row["TechSupport"] == "No"
                and row["OnlineSecurity"] == "No"
                and row["MonthlyCharges"] > X["MonthlyCharges"].mean()
                else 0
            ),
            axis=1,
        )

        # 22. 청구서 전환 여부 (무서류 청구서를 사용하고 자동 결제를 하지 않는 고객)
        X["PaperlessManualBilling"] = X.apply(
            lambda row: (
                1 if row["PaperlessBilling"] == "Yes" and row["AutoPayment"] == 0 else 0
            ),
            axis=1,
        )

        # 23. 계약 갱신 여부 (장기 계약을 하고 계약이 끝난 뒤에도 유지 중인 고객)
        X["RenewedContract"] = X.apply(
            lambda row: (
                1 if row["Contract"] != "Month-to-month" and row["tenure"] >= 12 else 0
            ),
            axis=1,
        )

        # 24. 요금 상승률 (가입 기간에 따른 요금 상승)
        X["ChargeIncreaseRate"] = X.apply(
            lambda row: (row["MonthlyCharges"] - X["MonthlyCharges"].mean())
            / (row["tenure"] + 1),
            axis=1,
        )

        # 25. 장기 고객 보너스 (가입 기간이 길고 요금이 낮은 고객)
        long_term_threshold = X["tenure"].quantile(0.75)
        low_charge_threshold = X["MonthlyCharges"].quantile(0.25)
        X["LongTermLowCharge"] = X.apply(
            lambda row: (
                1
                if row["tenure"] >= long_term_threshold
                and row["MonthlyCharges"] <= low_charge_threshold
                else 0
            ),
            axis=1,
        )

        # 26. 계약 후 추가 서비스 변경 여부 (스트리밍 또는 보안 서비스 도입 여부)
        X["ServiceUpgrade"] = X.apply(
            lambda row: (
                1
                if (row["StreamingTV"] == "Yes" or row["OnlineSecurity"] == "Yes")
                and row["tenure"] >= 6
                else 0
            ),
            axis=1,
        )

        # 27. 고객 충성도 점수 (가입 기간 + 고정 계약 + 여러 서비스 이용 여부 결합)
        X["LoyaltyScore"] = (
            X["tenure"] + (X["FixedContract"] * 5) + (X["MultipleServices"] * 2)
        )

        # 28. 디지털 친화 고객 (자동 결제와 무서류 청구서 사용 여부)
        X["DigitalSavvyCustomer"] = X.apply(
            lambda row: (
                1 if row["AutoPayment"] == 1 and row["PaperlessBilling"] == "Yes" else 0
            ),
            axis=1,
        )

        # 29. 월 요금 대비 가입 기간 (MonthlyCharges를 tenure로 나눈 값)
        X["MonthlyCharges_to_Tenure"] = X["MonthlyCharges"] / X["tenure"].replace(0, 1)

        # 30. SeniorCitizen이면서 Dependents가 있는 경우
        X["IsSeniorAndDependents"] = (X["SeniorCitizen"] == 1) & (
            X["Dependents"] == "Yes"
        )

        # 31. Partner가 Yes일 때 월 요금을 곱한 값
        X["MonthlyCharges_Per_Partner"] = X["MonthlyCharges"] * (
            X["Partner"] == "Yes"
        ).astype(int)

        # 32. StreamingTV 또는 StreamingMovies를 사용하는 경우
        X["Has_Streaming_Service"] = (
            (X["StreamingTV"] == "Yes") | (X["StreamingMovies"] == "Yes")
        ).astype(int)

        # 33. 기술 지원 서비스를 사용하는 경우
        X["Has_TechSupport"] = (X["TechSupport"] == "Yes").astype(int)

        # 34. DeviceProtection을 사용하는 경우
        X["Has_Device_Protection"] = (X["DeviceProtection"] == "Yes").astype(int)

        # 35. InternetService를 사용하지 않는 경우
        X["No_internet_services"] = (X["InternetService"] == "No").astype(int)

        # 36. 가입 기간이 상위 25% 이상인 고객
        X["TopQuartileTenure"] = (X["tenure"] >= X["tenure"].quantile(0.75)).astype(int)

        # 37. TotalCharges가 중간값보다 높은 경우
        X["High_TotalCharges"] = (
            X["TotalCharges"] > X["TotalCharges"].median()
        ).astype(int)

        # 38. Contract의 기간을 수치로 변환
        X["Contract_Length"] = X["Contract"].replace(
            {"Month-to-month": 1, "One year": 12, "Two year": 24}
        )

        # 39. TotalCharges를 MonthlyCharges와 tenure 곱한 값으로 나눈 비율
        X["Total_Monthly_to_TotalCharge"] = X["TotalCharges"] / (
            X["MonthlyCharges"] * X["tenure"].replace(0, 1)
        )

        # 40. OnlineBackup을 사용하는 경우
        X["Has_OnlineBackup"] = (X["OnlineBackup"] == "Yes").astype(int)

        # 41. PhoneService와 InternetService를 모두 사용하는 경우
        X["Phone_Internet_Combined"] = (X["PhoneService"] == "Yes") & (
            X["InternetService"] != "No"
        )

        # 42. 고객 충성도 점수 (가입 기간 * 월 요금 * 사용 중인 서비스 개수)
        X["Customer_Loyalty_Score_Alt"] = (
            X["tenure"] * X["MonthlyCharges"] * X["TotalServicesUsed"]
        )

        # 43. 온라인 서비스 사용량 (OnlineSecurity, OnlineBackup, DeviceProtection 사용 여부 합계)
        X["Total_OnlineServices_Used"] = (
            X[["OnlineSecurity", "OnlineBackup", "DeviceProtection"]]
            .eq("Yes")
            .sum(axis=1)
        )

        # 44. TechSupport와 Streaming 서비스를 동시에 사용하는 경우
        X["TechSupport_Streaming_Combined"] = (X["TechSupport"] == "Yes") & (
            (X["StreamingTV"] == "Yes") | (X["StreamingMovies"] == "Yes")
        )
        return X


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 상관관계 행렬 계산
        corr_matrix = X.corr().abs()

        # 상위 삼각 행렬을 사용해 상관관계가 높은 피처 제거
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column
            for column in upper_tri.columns
            if any(upper_tri[column] > self.threshold)
        ]

        # 선택된 피처 고정
        self.selected_features_ = X.columns.difference(to_drop)

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X[self.selected_features_]


class ScaleAndTransform(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2):
        self.degree = degree
        self.preprocessor = None

    def fit(self, X, y=None):
        X = X.copy()

        continuous_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numerical_pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "poly",
                    PolynomialFeatures(
                        degree=self.degree, interaction_only=False, include_bias=False
                    ),
                ),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, continuous_features),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                    categorical_features,
                ),
            ]
        )

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X_processed = self.preprocessor.transform(X)
        feature_names = self.preprocessor.get_feature_names_out()
        feature_names = [name.replace(" ", "_") for name in feature_names]
        if X_processed.shape[1] != len(feature_names):
            raise ValueError(
                "변환된 데이터의 열 개수와 피처 이름의 개수가 일치하지 않습니다."
            )

        return pd.DataFrame(X_processed, columns=feature_names)


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_select):
        self.features_to_select = features_to_select

    def fit(self, X, y=None):
        missing_features = set(self.features_to_select) - set(X.columns)
        if missing_features:
            print("FeatureSelection fit 단계에서 누락된 피처:", missing_features)
        return self

    def transform(self, X):
        missing_features = set(self.features_to_select) - set(X.columns)
        if missing_features:
            print("FeatureSelection transform 단계에서 누락된 피처:", missing_features)
        return X[self.features_to_select]
