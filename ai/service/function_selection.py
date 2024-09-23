pipeline = Pipeline(
    [
        ("data_cleaning", DataCleaning()),
        ("feature_engineering", FeatureEngineering()),
        ("scaling_transforming", ScaleAndTransform(degree=2)),
        ("smote", SMOTE(random_state=42)),
        ("correlation_filter", CorrelationFilter(threshold=0.999)),
        ("select-k-best", SelectKBestWithNames(score_func=f_classif, k=135)),
        ("clf", RandomForestClassifier(random_state=42)),
    ]
)


def collect_selected_features(pipeline, X, y, cv):
    results = cross_validate(
        pipeline, X, y, cv=cv, return_estimator=True, scoring="f1", n_jobs=-1
    )

    selected_feature_sets = []
    for i, est in enumerate(results["estimator"]):
        select_k_best = est.named_steps["select-k-best"]
        selected_features = select_k_best.selected_features_
        selected_feature_sets.append(set(selected_features))

        print(f"Fold {i+1}에서 선택된 피처: {selected_features}")

    return selected_feature_sets


# 모든 폴드에서 공통으로 선택된 피처를 찾는 함수
def get_common_features(selected_feature_sets):
    common_features = set.intersection(*selected_feature_sets)
    return list(common_features)


# 교차 검증을 통해 선택된 피처들을 수집
selected_feature_sets = collect_selected_features(pipeline, X, y, cv=10)

# 모든 폴드에서 공통으로 선택된 피처들만 추출
common_features = get_common_features(selected_feature_sets)
