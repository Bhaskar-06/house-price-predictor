# train.py
import json
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = "data/Bengaluru_House_Data.csv"


# ---------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------
def convert_sqft(x):
    """'2100 - 2850' -> 2475.0 ; '1200' -> 1200.0 ; '34.46Sq. Meter' -> NaN"""
    try:
        parts = str(x).split("-")
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        return float(x)
    except ValueError:
        return np.nan


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df[["area_type", "location", "size", "total_sqft", "bath", "price"]].dropna()

    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
    df["location"] = df["location"].str.strip()
    df["area_type"] = df["area_type"].str.strip()
    df = df.dropna(subset=["bhk", "total_sqft"])
    df["bhk"] = df["bhk"].astype(int)
    df["bath"] = df["bath"].astype(int)
    df = df.drop(columns=["size"])

    # Group rare locations into "other"
    loc_counts = df["location"].value_counts()
    rare = loc_counts[loc_counts <= 10].index
    df["location"] = df["location"].apply(lambda x: "other" if x in rare else x)
    return df


# ---------------------------------------------------------------------
# 2. OUTLIER REMOVAL
# ---------------------------------------------------------------------
def remove_outliers(df):
    df = df[df["total_sqft"] / df["bhk"] >= 300]        # min 300 sqft per bedroom
    df = df[df["bath"] <= df["bhk"] + 2]                # unrealistic bathrooms
    df = df[(df["bhk"] <= 10) & (df["total_sqft"] <= 10000)]

    df = df.copy()
    df["pps"] = df["price"] * 100000 / df["total_sqft"]  # price per sqft (cleaning only)

    kept = []
    for _, sub in df.groupby("location"):
        m, s = sub["pps"].mean(), sub["pps"].std()
        kept.append(sub[(sub["pps"] > (m - s)) & (sub["pps"] <= (m + s))])
    df = pd.concat(kept, ignore_index=True)
    return df.drop(columns=["pps"])  # never use as a feature -> would leak the target


# ---------------------------------------------------------------------
# 3. MODEL BUILDING
# ---------------------------------------------------------------------
def make_pipeline(regressor):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["area_type", "location"]),
            ("num", StandardScaler(), ["total_sqft", "bath", "bhk"]),
        ]
    )
    pipe = Pipeline([("prep", pre), ("model", regressor)])
    # Train on log(price), predict back in real price
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def main():
    df = load_and_clean(DATA_PATH)
    print("After cleaning:", df.shape)
    df = remove_outliers(df)
    print("After outlier removal:", df.shape)

    features = ["area_type", "location", "total_sqft", "bath", "bhk"]
    X, y = df[features], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ---- Compare models with cross-validation
    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    scores = {}
    for name, reg in candidates.items():
        cv = cross_val_score(make_pipeline(reg), X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        scores[name] = cv.mean()
        print(f"{name:22s} CV R2 = {cv.mean():.4f} (+/- {cv.std():.4f})")

    # ---- Tune Gradient Boosting (small file size, high accuracy)
    search = RandomizedSearchCV(
        make_pipeline(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        param_distributions={
            "regressor__model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "regressor__model__max_iter": [300, 500, 800],
            "regressor__model__max_depth": [None, 6, 8, 10],
            "regressor__model__min_samples_leaf": [10, 20, 40],
            "regressor__model__l2_regularization": [0.0, 0.1, 1.0],
        },
        n_iter=20, cv=5, scoring="r2", n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print("Best params:", search.best_params_)

    # ---- Evaluate on unseen test data
    pred = best.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mdape = float(np.median(np.abs(pred - y_test) / y_test))
    print(f"\nTEST  R2={r2:.4f} | MAE={mae:.2f} lakh | RMSE={rmse:.2f} lakh | MedianAPE={mdape:.2%}")

    # ---- Retrain on full data, then save
    best.fit(X, y)
    joblib.dump(best, "models/model.joblib", compress=3)

    meta = {
        "locations": sorted(df["location"].unique().tolist()),
        "area_types": sorted(df["area_type"].unique().tolist()),
        "metrics": {
            "r2": round(r2, 3), "mae": round(mae, 2),
            "rmse": round(rmse, 2), "mdape": round(mdape, 3),
            "samples": int(len(df)),
        },
    }
    with open("models/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved models/model.joblib and models/meta.json")


if __name__ == "__main__":
    main()