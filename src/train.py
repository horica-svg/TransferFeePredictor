import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import lightgbm as lgb
import joblib

DATA_PATH = Path("data/processed/dataset_processed.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_market_value_predictor():
    print("1. Loading processed data...")
    df = pd.read_csv(DATA_PATH)

    # Identificatori care nu trebuie folositi in invatare
    drop_cols = ["player_id", "name"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Tratam variabila tinta (y)
    y = df["market_value_in_eur"]
    X = df.drop(columns=["market_value_in_eur"])

    # Aplicam log(1 + y) pentru a stabiliza distributia
    y_log = np.log1p(y)

    # Identificare coloane categorice
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category")

    # 2. Impartire Train / Validation (early stopping) / Test (doar evaluare finala)
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X_train, y_train_log, test_size=0.125, random_state=42
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3. Initializare si antrenare LightGBM Regressor
    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    print("2. Antrenare model LightGBM...")
    model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )

    # 4. Evaluare pe setul de test (revenim la scara reala in EUR)
    preds_log = model.predict(X_test)
    preds_eur = np.expm1(preds_log)
    actual_eur = np.expm1(y_test_log)

    mae = mean_absolute_error(actual_eur, preds_eur)
    rmse = root_mean_squared_error(actual_eur, preds_eur)
    r2 = r2_score(actual_eur, preds_eur)
    rmse_log = root_mean_squared_error(y_test_log, preds_log)

    print("\n" + "="*45)
    print("           RESULTS EVALUATION TEST")
    print("="*45)
    print(f"MAE (Mean Absolute Error): €{mae:,.0f}")
    print(f"RMSE:                        €{rmse:,.0f}")
    print(f"R² Score:                    {r2:.4f}")
    print(f"RMSE (log scale):            {rmse_log:.4f}")
    print("="*45 + "\n")

    # 5. Feature Importance (Top 10 variabile)
    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("Top 10 Decisive Features:")
    print(feature_importances.head(10).to_string(index=False))

    # 6. Salvare model si lista de coloane pentru inferenta
    model_artifact = {
        "model": model,
        "features": list(X.columns),
        "categorical_features": cat_cols,
        "categories": {c: X[c].cat.categories.tolist() for c in cat_cols},  # pentru inferenta consistenta
    }
    output_path = MODEL_DIR / "lgbm_market_value.pkl"
    joblib.dump(model_artifact, output_path)
    print(f"\nModel saved successfully to: {output_path}")

if __name__ == "__main__":
    train_market_value_predictor()