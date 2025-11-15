import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import joblib

# 1. LOAD DATA

df = pd.read_csv("../data/climate_change_data.csv")

# Feature engineering: create lag feature
df["CO2_lag_1"] = df["CO2 Emissions"].shift(1)

df_fe = df.dropna(subset=["CO2_lag_1"]).copy()

# 2. DEFINE FEATURES & TARGET

X = df_fe[[
    "Temperature", "Sea Level Rise", "Precipitation",
    "Humidity", "Wind Speed", "CO2_lag_1"
]]
y = df_fe["CO2 Emissions"]

# split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. BASE MODELS (EVALUATION)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nBASE MODEL PERFORMANCE")
for name, model in models.items():
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    rmse = np.sqrt(-scores)
    print(f"{name} → RMSE: {rmse.mean():.4f}")

# 4. RANDOM FOREST TUNING

rf = RandomForestRegressor(random_state=42)

rf_params = {
    "n_estimators": [200, 400, 600],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rf_search = RandomizedSearchCV(
    rf,
    rf_params,
    n_iter=20,
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X, y)

print("\nBEST RANDOM FOREST")
print(rf_search.best_params_)
print("RMSE:", np.sqrt(-rf_search.best_score_))

# 5. XGBOOST TUNING

xgb_model = xgb.XGBRegressor(random_state=42)

xgb_params = {
    "n_estimators": [200, 400, 800],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb_model,
    xgb_params,
    n_iter=20,
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X, y)

print("\nBEST XGBOOST")
print(xgb_search.best_params_)
print("RMSE:", np.sqrt(-xgb_search.best_score_))

# 6. CHOOSE BEST MODEL & SAVE

if np.sqrt(-xgb_search.best_score_) < np.sqrt(-rf_search.best_score_):
    best_model = xgb_search.best_estimator_
    print("\n>>> BEST MODEL = XGBOOST`")
else:
    best_model = rf_search.best_estimator_
    print("\n>>> BEST MODEL = RANDOM FOREST")

joblib.dump(best_model, "../models/best_model.pkl")
print("\nModel saved → ../models/best_model.pkl")