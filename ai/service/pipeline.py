import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
import joblib
import pickle
import optuna

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from sklearn.svm import SVC
from sklearn.ensemble import (
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB

import xgboost as xgb
import catboost as cb

# 상수 정의
RANDOM_STATE = 42
N_JOBS = -1
CV_SPLITS = 10
N_ITER_RANDOM_SEARCH = 10
N_TRIALS_OPTUNA = 50
MODEL_SAVE_PATH = "/content/data/MyDrive/models/"
common_features = [
    "num__AvgMonthlyCharges_FixedContract",
    "num__TopQuartileTenure_Total_Monthly_to_TotalCharge",
    "num__ContractEndingSoon",
    "num__AboveAverageCharges_PaperlessManualBilling",
    "num__FixedContract_TopQuartileTenure",
    "num__LongTermCustomer_Total_Monthly_to_TotalCharge",
    "num__ShortTermContract_PaperlessManualBilling",
    "num__ChargeIncreaseRate_MonthlyCharges_to_Tenure",
    "num__ServiceRisk_MonthlyCharges_to_Tenure",
    "num__ContractEndingSoon_ChargeIncreaseRate",
    "num__AvgMonthlyCharges_IsElectronicCheck",
    "num__ServiceUpgrade_Contract_Length",
    "num__IsFiberOptic_ServiceRisk",
    "num__MonthlyCharges_Contract_Length",
    "num__AvgMonthlyCharges_PaperlessManualBilling",
    "num__FixedContract_SingleServiceCustomer",
    "num__LoyaltyScore_Total_Monthly_to_TotalCharge",
    "num__IsFiberOptic_IsElectronicCheck",
    "num__IsFiberOptic_ChargeIncreaseRate",
    "num__IsPaperless_ShortTermContract",
    "num__IsElectronicCheck_Total_Monthly_to_TotalCharge",
    "num__IsFiberOptic_ShortTermContract",
    "num__ShortTermContract",
    "num__HighChargeVolatility_Contract_Length",
    "num__AvgMonthlyCharges_Contract_Length",
    "num__IsFiberOptic",
    "num__IsPaperless_IsFiberOptic",
    "num__tenure_AutoPayment",
    "num__LoyaltyScore",
    "num__MonthlyCharges_MonthlyCharges_to_Tenure",
    "num__RenewedContract",
    "num__IsElectronicCheck",
    "num__TotalServicesUsed_LoyaltyScore",
    "num__tenure_Contract_Length",
    "num__MultipleServices_IsElectronicCheck",
    "num__AvgMonthlyCharges_IsFiberOptic",
    "num__IsPaperless_ServiceRisk",
    "num__MonthlyCharges_IsElectronicCheck",
    "num__MonthlyCharges_PaperlessManualBilling",
    "num__IsPaperless_IsElectronicCheck",
    "num__TotalServicesUsed_Contract_Length",
    "num__AutoPayment_TotalServicesUsed",
    "num__ShortTermContract_Total_Monthly_to_TotalCharge",
    "num__ShortTermContract_ChargeIncreaseRate",
    "num__PaperlessManualBilling_ChargeIncreaseRate",
    "num__tenure_MonthlyCharges_to_Tenure",
    "num__ServiceRisk_ChargeIncreaseRate",
    "num__ShortTermContract_MonthlyCharges_to_Tenure",
    "num__ContractEndingSoon_Total_Monthly_to_TotalCharge",
    "num__tenure_TotalServicesUsed",
    "num__TopQuartileTenure_Contract_Length",
    "num__AutoPayment_RenewedContract",
    "num__TotalServicesUsed_TopQuartileTenure",
    "num__IsFiberOptic_Total_Monthly_to_TotalCharge",
    "num__tenure_FixedContract",
    "num__SingleServiceCustomer_RenewedContract",
    "num__LongTermCustomer_LoyaltyScore",
    "num__TotalServicesUsed_SingleServiceCustomer",
    "num__MonthlyCharges_ShortTermContract",
    "num__SingleServiceCustomer_LoyaltyScore",
    "num__ChargeIncreaseRate_Contract_Length",
    "num__AvgMonthlyCharges_MonthlyCharges_to_Tenure",
    "num__FixedContract_High_TotalCharges",
    "num__SingleServiceCustomer_Contract_Length",
    "num__TotalServicesUsed^2",
    "num__ServiceRisk_PaperlessManualBilling",
    "num__FixedContract_No_internet_services",
    "num__tenure_SingleServiceCustomer",
    "num__AboveAverageCharges_MonthlyCharges_to_Tenure",
    "num__LongTermCustomer_Contract_Length",
    "num__ChargeIncreaseRate_No_internet_services",
    "num__MultipleServices_ShortTermContract",
    "num__AvgMonthlyCharges_ContractEndingSoon",
    "num__ServiceRisk",
    "num__FixedContract_TotalServicesUsed",
    "num__TopQuartileTenure",
    "num__MonthlyCharges_to_Tenure",
    "num__LoyaltyScore_MonthlyCharges_to_Tenure",
    "num__High_TotalCharges_Contract_Length",
    "num__TotalServicesUsed",
    "cat__TenureGroup_48-72months",
    "num__RenewedContract_Total_Monthly_to_TotalCharge",
    "num__AvgMonthlyCharges_ServiceRisk",
    "num__MonthlyCharges_ContractEndingSoon",
    "num__IsFiberOptic_MonthlyCharges_to_Tenure",
    "num__tenure_LongTermCustomer",
    "num__TotalCharges_FixedContract",
    "num__IsFiberOptic_PaperlessManualBilling",
    "num__TotalServicesUsed_RenewedContract",
    "num__AutoPayment_Contract_Length",
    "num__Contract_Length_Total_OnlineServices_Used",
    "num__tenure_TopQuartileTenure",
    "num__TotalServicesUsed_ChargeIncreaseRate",
    "num__LongTermCustomer_TotalServicesUsed",
    "num__tenure^2",
    "num__IsElectronicCheck_ShortTermContract",
    "num__FixedContract",
    "num__tenure_Total_Monthly_to_TotalCharge",
    "num__ServiceRisk_Total_Monthly_to_TotalCharge",
    "num__AboveAverageCharges_IsElectronicCheck",
    "num__ContractEndingSoon_MonthlyCharges_to_Tenure",
    "num__Contract_Length^2",
    "num__LongTermCustomer_SingleServiceCustomer",
    "num__AboveAverageCharges_ShortTermContract",
    "num__FixedContract_HighChargeVolatility",
    "num__LongTermCustomer",
    "cat__Contract_Two_year",
    "num__AvgMonthlyCharges_RenewedContract",
    "num__MultipleServices_ContractEndingSoon",
    "num__IsElectronicCheck_ChargeIncreaseRate",
    "num__ShortTermContract_ServiceRisk",
    "num__tenure",
    "num__Contract_Length_Customer_Loyalty_Score_Alt",
    "num__TotalCharges_Contract_Length",
    "num__IsPaperless_MonthlyCharges_to_Tenure",
    "num__MonthlyCharges_ServiceRisk",
    "num__IsElectronicCheck_ServiceRisk",
    "num__FixedContract_ChargeIncreaseRate",
    "num__FixedContract_AutoPayment",
    "num__MultipleServices_PaperlessManualBilling",
    "num__AvgMonthlyCharges_ShortTermContract",
    "num__Contract_Length",
    "num__MultipleServices_MonthlyCharges_to_Tenure",
    "num__FixedContract_Total_OnlineServices_Used",
]


# 모델 정의
def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForestClassifier": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(
            random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="logloss"
        ),
        "CatBoost": cb.CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
        "SVC": SVC(random_state=RANDOM_STATE, probability=True),
        "ExtraTreesClassifier": ExtraTreesClassifier(random_state=RANDOM_STATE),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=RANDOM_STATE),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
    }


# 전처리 단계 정의
def get_preprocessing_steps():
    return [
        ("data_cleaning", DataCleaning()),
        ("feature_engineering", FeatureEngineering()),
        ("scaling_transforming", ScaleAndTransform(degree=2)),
        ("correlation_filter", CorrelationFilter(threshold=0.999)),
        ("feature-select", FeatureSelection(features_to_select=common_features)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
    ]


# 파라미터 분포 정의
def get_param_distributions():
    return {
        "LogisticRegression": {
            "classifier__C": np.logspace(-4, 4, 20),
        },
        "RandomForestClassifier": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
        },
        # ... 다른 모델들의 파라미터 분포도 여기에 추가
    }


# Random Search 수행 함수
def perform_random_search(model_name, model, param_distributions, X, y, cv):
    print(f"Starting Random Search for {model_name}")
    full_pipeline_steps = get_preprocessing_steps() + [("classifier", model)]
    pipeline_with_model = Pipeline(full_pipeline_steps)

    random_search = RandomizedSearchCV(
        pipeline_with_model,
        param_distributions=param_distributions,
        n_iter=N_ITER_RANDOM_SEARCH,
        cv=cv,
        scoring="f1",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    random_search.fit(X, y)

    print(
        f"Best parameters for {model_name} from Random Search: {random_search.best_params_}"
    )
    return random_search.best_estimator_, random_search.best_params_


# Optuna 최적화 함수
def perform_optuna_search(model_name, model_class, X, y, cv, best_params_random_search):
    print(
        f"Starting Optuna optimization for {model_name} based on Random Search results"
    )

    def objective(trial):
        params = get_optuna_params(trial, model_name, best_params_random_search)
        full_pipeline_steps = get_preprocessing_steps() + [("classifier", model_class)]
        pipeline_with_model = Pipeline(full_pipeline_steps)
        pipeline_with_model.set_params(**params)
        scores = cross_val_score(
            pipeline_with_model, X, y, cv=cv, scoring="f1", n_jobs=N_JOBS
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA, n_jobs=N_JOBS)

    print(f"Best parameters for {model_name} from Optuna: {study.best_params}")

    best_params = study.best_params
    full_pipeline_steps = get_preprocessing_steps() + [("classifier", model_class)]
    pipeline_with_model = Pipeline(full_pipeline_steps)
    pipeline_with_model.set_params(**best_params)
    pipeline_with_model.fit(X, y)

    return pipeline_with_model, best_params


# Optuna 파라미터 설정 함수
def get_optuna_params(trial, model_name, best_params_random_search):
    params = {}
    if model_name == "LogisticRegression":
        C_best = best_params_random_search.get("classifier__C", 1.0)
        params["classifier__C"] = trial.suggest_float(
            "classifier__C", max(1e-4, C_best * 0.5), C_best * 1.5, log=True
        )
    elif model_name == "RandomForestClassifier":
        n_estimators_best = best_params_random_search.get(
            "classifier__n_estimators", 100
        )
        params["classifier__n_estimators"] = trial.suggest_int(
            "classifier__n_estimators",
            max(50, n_estimators_best - 50),
            n_estimators_best + 50,
        )
        # ... 다른 파라미터들도 여기에 추가
    # ... 다른 모델들의 파라미터 설정도 여기에 추가
    return params


# 모델 저장 함수
def save_model(model_name, model):
    joblib.dump(model, f"{MODEL_SAVE_PATH}{model_name}.joblib")
    with open(f"{MODEL_SAVE_PATH}{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model {model_name} saved as .joblib and .pkl")


# 메인 실행 함수
def main(X, y):
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()
    param_distributions = get_param_distributions()

    for model_name, model in models.items():
        print(f"\nProcessing model: {model_name}")
        params = param_distributions.get(model_name, {})
        if not params:
            print(
                f"No hyperparameters to tune for {model_name}. Proceeding with default parameters."
            )
            full_pipeline_steps = get_preprocessing_steps() + [("classifier", model)]
            pipeline_with_model = Pipeline(full_pipeline_steps)
            pipeline_with_model.fit(X, y)
            save_model(model_name, pipeline_with_model)
            continue

        best_estimator, best_params_random_search = perform_random_search(
            model_name, model, params, X, y, cv
        )
        final_model, optuna_best_params = perform_optuna_search(
            model_name, model.__class__(), X, y, cv, best_params_random_search
        )
        save_model(model_name, final_model)


if __name__ == "__main__":
    # X와 y 데이터 로드
    # X, y = load_data()
    main(X, y)
