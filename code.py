import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost is not installed. Falling back without XGBoost.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("shap is not installed. SHAP explanation section will be skipped.")

TARGET = "SalePrice"
RANDOM_STATE = 42

TRAIN_PATH = "/Users/varun/Downloads/house-prices-advanced-regression-techniques/train.csv"
TEST_PATH = "/Users/varun/Downloads/house-prices-advanced-regression-techniques/test.csv"
SCHOOL_PATH = "/Users/varun/Downloads/ISPP_School_Summary.xlsx"

OUTPUT_DIR = "housing_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#LOAD DATA

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
school_summary = pd.read_excel(SCHOOL_PATH)

print("Original train shape:", train_df.shape)
print("Original test shape :", test_df.shape)

test_ids = test_df["Id"].copy()

#BUILD REAL SCHOOL TABLE

ames_schools = school_summary[
    school_summary["District_Name"].astype(str).str.contains("Ames", case=False, na=False)
].copy()

ames_schools["School_Index"] = pd.to_numeric(ames_schools["School_Index"], errors="coerce")
ames_schools["Count_All_Students"] = pd.to_numeric(ames_schools["Count_All_Students"], errors="coerce")

ames_schools = ames_schools.dropna(subset=["School_Index", "Count_All_Students"])

district_avg_school_index = (
    (ames_schools["School_Index"] * ames_schools["Count_All_Students"]).sum()
    / ames_schools["Count_All_Students"].sum()
)

school_score_table = ames_schools[
    ["School_Name", "School_Index", "School_Rating_Category", "Count_All_Students"]
].copy()

min_idx = school_score_table["School_Index"].min()
max_idx = school_score_table["School_Index"].max()

if max_idx == min_idx:
    school_score_table["School_Score_Norm"] = 0.5
    district_avg_norm = 0.5
else:
    school_score_table["School_Score_Norm"] = (
        (school_score_table["School_Index"] - min_idx) / (max_idx - min_idx)
    )
    district_avg_norm = (district_avg_school_index - min_idx) / (max_idx - min_idx)

print("Ames district weighted average School_Index:", round(district_avg_school_index, 2))
print("\nAmes schools used:")
print(school_score_table.to_string(index=False))

#SAFE SCHOOL PROXY MAP

neighborhood_to_school = {
    "Edwards": "Edwards Elementary School",
    "Sawyer": "Sawyer Elementary School",
    "Mitchel": "Mitchell Elementary School",
}

school_index_map = dict(zip(school_score_table["School_Name"], school_score_table["School_Index"]))
school_norm_map = dict(zip(school_score_table["School_Name"], school_score_table["School_Score_Norm"]))
school_rating_map = dict(zip(school_score_table["School_Name"], school_score_table["School_Rating_Category"]))


#FEATURE ENGINEERING

def add_features(df):
    df = df.copy()

    #City-level context
    df["City_Median_Income"] = 58709
    df["City_Median_Age"] = 23.9
    df["City_Poverty_Rate"] = 0.239
    df["City_Levy"] = 10.30

    # B. Real school merge
    df["Matched_School"] = df["Neighborhood"].map(neighborhood_to_school)

    df["School_Index_Real"] = df["Matched_School"].map(school_index_map)
    df["School_Score_Norm"] = df["Matched_School"].map(school_norm_map)
    df["School_Rating_Category_Real"] = df["Matched_School"].map(school_rating_map)

    df["School_Index_Real"] = df["School_Index_Real"].fillna(district_avg_school_index)
    df["School_Score_Norm"] = df["School_Score_Norm"].fillna(district_avg_norm)
    df["School_Rating_Category_Real"] = df["School_Rating_Category_Real"].fillna("District Average Proxy")

    df["School_Merge_Type"] = np.where(
        df["Neighborhood"].isin(neighborhood_to_school.keys()),
        "Exact name-based school proxy",
        "Ames district weighted average"
    )

    #Age features
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    if "GarageYrBlt" in df.columns:
        df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    else:
        df["GarageAge"] = np.nan

    #Total area / utility
    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    df["TotalPorchSF"] = (
        df["OpenPorchSF"].fillna(0)
        + df["EnclosedPorch"].fillna(0)
        + df["3SsnPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
    )

    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    df["OutdoorSF"] = (
        df["WoodDeckSF"].fillna(0)
        + df["OpenPorchSF"].fillna(0)
        + df["EnclosedPorch"].fillna(0)
        + df["3SsnPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
    )

    #Binary indicators
    df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"].fillna(0) > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)
    df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"].fillna(0) > 0).astype(int)


    #Existing interaction features
    df["QualityLivArea"] = df["OverallQual"].fillna(0) * df["GrLivArea"].fillna(0)
    df["BathPerBedroom"] = df["TotalBath"] / (df["BedroomAbvGr"].fillna(0) + 1)
    df["GarageScore"] = df["GarageCars"].fillna(0) * df["GarageArea"].fillna(0)

    df["Quality_x_School"] = df["OverallQual"].fillna(0) * df["School_Score_Norm"].fillna(0)
    df["LivingArea_x_School"] = df["GrLivArea"].fillna(0) * df["School_Score_Norm"].fillna(0)
    df["Age_x_School"] = df["HouseAge"].fillna(0) * df["School_Score_Norm"].fillna(0)

    #NEW engineered features
    df["TotalHomeQuality"] = df["OverallQual"].fillna(0) + df["OverallCond"].fillna(0)
    df["TotalHouseSF"] = df["TotalSF"] + df["GarageArea"].fillna(0)
    df["LivLotRatio"] = df["GrLivArea"].fillna(0) / df["LotArea"].replace(0, np.nan)
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["TotalRooms"] = df["TotRmsAbvGrd"].fillna(0) + df["FullBath"].fillna(0)

    for col in [
        "HouseAge", "RemodAge", "GarageAge", "BathPerBedroom",
        "LivLotRatio", "YearsSinceRemodel"
    ]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

print("\nSample merged school columns:")
print(
    train_df[
        [
            "Neighborhood",
            "Matched_School",
            "School_Index_Real",
            "School_Score_Norm",
            "School_Rating_Category_Real",
            "School_Merge_Type"
        ]
    ].head(10).to_string(index=False)
)

#OUTLIER FILTERING
before_outlier = train_df.shape[0]

train_keep = ~(
    (train_df["GrLivArea"] > 4000) &
    (train_df["SalePrice"] < 300000)
)

train_df = train_df.loc[train_keep].copy()

after_outlier = train_df.shape[0]
print(f"\nOutlier removal: removed {before_outlier - after_outlier} rows")
print("Train shape after outlier removal:", train_df.shape)

#SPLIT FEATURES / TARGET
X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]
y_log = np.log1p(y)
X_test = test_df.copy()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_log, test_size=0.2, random_state=RANDOM_STATE
)

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features    :", len(numeric_features))
print("Categorical features:", len(categorical_features))


#PREPROCESSOR
def build_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", build_onehot())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

#HELPER FUNCTIONS
def evaluate_model(name, model, X_train, y_train, X_valid, y_valid):
    y_train_actual = np.expm1(y_train)
    sample_weight = np.where(y_train_actual > 300000, 2, 1)

    try:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
    except Exception:
        model.fit(X_train, y_train)

    pred_valid_log = model.predict(X_valid)
    pred_valid = np.maximum(np.expm1(pred_valid_log), 0)
    y_valid_actual = np.expm1(y_valid)

    rmse_valid = np.sqrt(mean_squared_error(y_valid_actual, pred_valid))
    mae_valid = mean_absolute_error(y_valid_actual, pred_valid)
    r2_valid = r2_score(y_valid_actual, pred_valid)

    return {
        "Model": name,
        "Valid_RMSE": rmse_valid,
        "Valid_MAE": mae_valid,
        "Valid_R2": r2_valid,
    }

def plot_actual_vs_pred(y_true_log, y_pred_log, title, filename):
    y_true = np.expm1(y_true_log)
    y_pred = np.maximum(np.expm1(y_pred_log), 0)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(title)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")

    plt.show(block=False)
    plt.pause(2)
    plt.close()

def plot_residuals(y_true_log, y_pred_log, filename):
    y_true = np.expm1(y_true_log)
    y_pred = np.maximum(np.expm1(y_pred_log), 0)
    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved residual plot: {save_path}")

    plt.show(block=False)
    plt.pause(2)
    plt.close()

def save_feature_importance_plot(feature_importance_df, model_name, filename):
    plt.figure(figsize=(10, 7))
    top_features = feature_importance_df.head(20).iloc[::-1]
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.title(f"Top 20 Feature Importances - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")

    plt.show(block=False)
    plt.pause(2)
    plt.close()

#BASELINE MODELS
results_list = []
models_dict = {}

lr_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)
results_list.append(evaluate_model("Linear Regression", lr_model, X_train, y_train, X_valid, y_valid))
models_dict["Linear Regression"] = lr_model

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]
)
results_list.append(evaluate_model("Random Forest", rf_model, X_train, y_train, X_valid, y_valid))
models_dict["Random Forest"] = rf_model

et_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]
)
results_list.append(evaluate_model("Extra Trees", et_model, X_train, y_train, X_valid, y_valid))
models_dict["Extra Trees"] = et_model

gb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            random_state=RANDOM_STATE
        ))
    ]
)
results_list.append(evaluate_model("Gradient Boosting", gb_model, X_train, y_train, X_valid, y_valid))
models_dict["Gradient Boosting"] = gb_model

if HAS_XGB:
    xgb_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=1200,
                learning_rate=0.02,
                max_depth=3,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
    )
    results_list.append(evaluate_model("XGBoost", xgb_model, X_train, y_train, X_valid, y_valid))
    models_dict["XGBoost"] = xgb_model

#STACKING + PCA RIDGE
base_estimators = [
    ("rf", RandomForestRegressor(
        n_estimators=400,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )),
    ("et", ExtraTreesRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )),
    ("gb", GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        random_state=RANDOM_STATE
    )),
]

if HAS_XGB:
    base_estimators.append(
        ("xgb", XGBRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    )

stack_model = StackingRegressor(
    estimators=base_estimators,
    final_estimator=Ridge(alpha=10.0),
    passthrough=False,
    n_jobs=-1
)

hybrid_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", stack_model)
    ]
)
results_list.append(
    evaluate_model("Hybrid Tree Stacking", hybrid_tree_model, X_train, y_train, X_valid, y_valid)
)
models_dict["Hybrid Tree Stacking"] = hybrid_tree_model

pca_linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ("model", Ridge(alpha=5.0))
    ]
)
results_list.append(
    evaluate_model("PCA + Ridge", pca_linear_model, X_train, y_train, X_valid, y_valid)
)
models_dict["PCA + Ridge"] = pca_linear_model

#MODEL COMPARISON
results_df = pd.DataFrame(results_list).sort_values(by="Valid_RMSE").reset_index(drop=True)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

results_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
results_df.to_csv(results_path, index=False)
print(f"\nSaved model comparison table: {results_path}")


#CROSS-VALIDATION FOR XGBOOST
if HAS_XGB:
    print("\nRunning 5-Fold Cross-Validation for XGBoost...")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    xgb_cv_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=1200,
                learning_rate=0.02,
                max_depth=3,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
    )

    cv_scores = -cross_val_score(
        xgb_cv_model,
        X,
        y_log,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        n_jobs=1
    )

    cv_df = pd.DataFrame({
        "Fold": np.arange(1, len(cv_scores) + 1),
        "RMSE_Log": cv_scores
    })
    cv_df.loc[len(cv_df)] = ["Mean", cv_scores.mean()]
    cv_df.loc[len(cv_df)] = ["Std", cv_scores.std()]

    cv_path = os.path.join(OUTPUT_DIR, "xgboost_5fold_cv.csv")
    cv_df.to_csv(cv_path, index=False)

    print(cv_df)
    print(f"Saved CV results: {cv_path}")


#BEST MODEL + VALIDATION PLOTS
best_model_name = results_df.iloc[0]["Model"]
best_model = models_dict[best_model_name]

print("\nBest validation model:", best_model_name)

best_model.fit(X_train, y_train)
valid_pred_log = best_model.predict(X_valid)

plot_residuals(y_valid, valid_pred_log, "residual_plot.png")
plot_actual_vs_pred(
    y_valid,
    valid_pred_log,
    f"Actual vs Predicted - {best_model_name}",
    "actual_vs_pred_best_model.png"
)


#XGBOOST EARLY STOPPING
xgb_early_model = None

if HAS_XGB:
    print("\nTraining XGBoost with early stopping...")

    X_train_proc = preprocessor.fit_transform(X_train)
    X_valid_proc = preprocessor.transform(X_valid)

    xgb_early_model = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=100
    )

    xgb_early_model.fit(
        X_train_proc,
        y_train,
        eval_set=[(X_valid_proc, y_valid)],
        verbose=False
    )

    early_pred_log = xgb_early_model.predict(X_valid_proc)
    early_pred = np.maximum(np.expm1(early_pred_log), 0)
    y_valid_actual = np.expm1(y_valid)

    early_rmse = np.sqrt(mean_squared_error(y_valid_actual, early_pred))
    early_mae = mean_absolute_error(y_valid_actual, early_pred)
    early_r2 = r2_score(y_valid_actual, early_pred)

    early_results = pd.DataFrame([{
        "Model": "XGBoost_EarlyStopping",
        "Valid_RMSE": early_rmse,
        "Valid_MAE": early_mae,
        "Valid_R2": early_r2,
        "Best_Iteration": getattr(xgb_early_model, "best_iteration", None)
    }])

    early_results_path = os.path.join(OUTPUT_DIR, "xgboost_early_stopping_results.csv")
    early_results.to_csv(early_results_path, index=False)

    print("\nEarly Stopping Results:")
    print(early_results.to_string(index=False))
    print(f"Saved early stopping results: {early_results_path}")

#SHAP EXPLAINABILITY
shap_ready_model = None
shap_ready_name = None
fitted_preprocessor = None

if xgb_early_model is not None:
    shap_ready_name = "XGBoost Early Stopping"
    shap_ready_model = xgb_early_model
    fitted_preprocessor = preprocessor
elif HAS_XGB:
    shap_ready_name = "XGBoost"
    shap_ready_model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
elif True:
    shap_ready_name = "Extra Trees"
    shap_ready_model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

if HAS_SHAP and shap_ready_model is not None:
    print(f"\nTraining {shap_ready_name} for SHAP explanation...")

    if fitted_preprocessor is None:
        fitted_preprocessor = preprocessor.fit(X_train)

    X_train_transformed = fitted_preprocessor.transform(X_train)
    feature_names = fitted_preprocessor.get_feature_names_out()

    if xgb_early_model is None:
        shap_ready_model.fit(X_train_transformed, y_train)

    sample_size = min(300, X_train_transformed.shape[0])
    X_shap_sample = X_train_transformed[:sample_size]

    try:
        explainer = shap.Explainer(shap_ready_model, X_shap_sample)
        shap_values = explainer(X_shap_sample)

        plt.figure()
        shap.summary_plot(
            shap_values,
            features=X_shap_sample,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        shap_summary_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
        plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {shap_summary_path}")
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        try:
            plt.figure()
            shap.plots.bar(shap_values, max_display=20, show=False)
            plt.tight_layout()
            shap_bar_path = os.path.join(OUTPUT_DIR, "shap_bar.png")
            plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {shap_bar_path}")
            plt.show(block=False)
            plt.pause(2)
            plt.close()
        except Exception as e:
            print("SHAP bar plot failed:", e)

    except Exception as e:
        print("SHAP plot generation failed:", e)

    try:
        feature_importance_values = shap_ready_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importance_values
        }).sort_values(by="Importance", ascending=False)

        print("\nTop 20 Important Features:")
        print(feature_importance_df.head(20).to_string(index=False))

        importance_csv_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
        feature_importance_df.to_csv(importance_csv_path, index=False)
        print(f"Saved feature importance table: {importance_csv_path}")

        save_feature_importance_plot(
            feature_importance_df,
            shap_ready_name,
            "top_20_feature_importances.png"
        )
    except Exception as e:
        print("Feature importance plot failed:", e)
else:
    print("\nSHAP not available. Skipping explainability plots.")

#FINAL MODEL TRAINING
use_early_stopping_for_final = False

if HAS_XGB and xgb_early_model is not None:
    early_rmse_value = early_results.loc[0, "Valid_RMSE"]
    best_rmse_value = results_df.iloc[0]["Valid_RMSE"]

    if early_rmse_value <= best_rmse_value:
        use_early_stopping_for_final = True
        print("\nUsing XGBoost Early Stopping as final model.")
    else:
        print("\nUsing best pipeline model as final model.")

if use_early_stopping_for_final:
    final_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    X_full_proc = final_preprocessor.fit_transform(X)
    X_test_proc = final_preprocessor.transform(X_test)

    X_full_train, X_full_valid, y_full_train, y_full_valid = train_test_split(
        X_full_proc, y_log, test_size=0.1, random_state=RANDOM_STATE
    )

    final_xgb = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=100
    )

    final_xgb.fit(
        X_full_train,
        y_full_train,
        eval_set=[(X_full_valid, y_full_valid)],
        verbose=False
    )

    test_pred_log = final_xgb.predict(X_test_proc)
else:
    final_model = models_dict[best_model_name]
    final_model.fit(X, y_log)
    test_pred_log = final_model.predict(X_test)

test_pred = np.maximum(np.expm1(test_pred_log), 0)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_pred
})

submission_path = os.path.join(OUTPUT_DIR, "submission_real_school_model.csv")
submission.to_csv(submission_path, index=False)
print(f"\nSubmission file saved as: {submission_path}")
print(submission.head())

#FINAL FEATURED DATA
train_featured_path = os.path.join(OUTPUT_DIR, "train_real_school_features.csv")
test_featured_path = os.path.join(OUTPUT_DIR, "test_real_school_features.csv")

train_df.to_csv(train_featured_path, index=False)
test_df.to_csv(test_featured_path, index=False)

print(f"\nFeatured datasets saved as:\n- {train_featured_path}\n- {test_featured_path}")
print("\nAll done.")