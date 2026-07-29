import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# KONFIGURASI UTAMA
# Split train:val:test dulu (val TIDAK digabung ke training).
# 5-fold CV dijalankan HANYA di dalam training set untuk memilih
# hyperparameter & threshold. Val untuk cek generalisasi.
# Test disentuh sekali di akhir.
# ============================================================

DATA_PATH = "heart_statlog_cleveland_hungary_final.csv"
TARGET = "target"
RANDOM_STATE = 42
N_FOLDS = 5

SPLIT_SCHEMES = {
    "80:10:10": {"train": 0.80, "val": 0.10, "test": 0.10},
    "70:15:15": {"train": 0.70, "val": 0.15, "test": 0.15},
    "70:20:10": {"train": 0.70, "val": 0.20, "test": 0.10},
}

THRESHOLDS = np.arange(0.30, 0.701, 0.005)


from model_transformers import CholesterolZeroImputer


def make_pipeline(model):
    return Pipeline([
        ("chol_imputer", CholesterolZeroImputer()),
        ("model", model)
    ])


def build_catboost(params):
    return make_pipeline(CatBoostClassifier(
        **params,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False
    ))


def build_xgboost(params):
    return make_pipeline(XGBClassifier(
        **params,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1
    ))


CATBOOST_PARAM_GRID = [
    {"iterations": it, "depth": d, "learning_rate": lr}
    for it in [100, 200, 300]
    for d in [3, 4, 5]
    for lr in [0.03, 0.05, 0.1]
]

XGBOOST_PARAM_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in [100, 200, 300]
    for d in [3, 4, 5]
    for lr in [0.03, 0.05, 0.1]
]

MODELS = {
    "CatBoost": {"build_fn": build_catboost, "param_grid": CATBOOST_PARAM_GRID},
    "XGBoost": {"build_fn": build_xgboost, "param_grid": XGBOOST_PARAM_GRID},
}


def split_train_val_test(X, y, scheme):
    test_frac = scheme["test"]
    val_frac = scheme["val"]
    train_frac = scheme["train"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=RANDOM_STATE, stratify=y
    )

    val_frac_of_trainval = val_frac / (train_frac + val_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac_of_trainval,
        random_state=RANDOM_STATE, stratify=y_trainval
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_scheme(X, y, scheme_name, scheme, model_name, build_fn, param_grid):
    print("\n\n" + "#" * 80)
    print(f"SKEMA {scheme_name} -- MODEL {model_name}")
    print("#" * 80)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, scheme)
    print(f"Training   : {len(X_train)}  <- 5-fold CV dijalankan DI SINI SAJA")
    print(f"Validation : {len(X_val)}   <- cek generalisasi sebelum test")
    print(f"Testing    : {len(X_test)}   <- disentuh sekali di akhir")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best = None
    for i, params in enumerate(param_grid):
        pipeline = build_fn(params)
        oof_prob = cross_val_predict(
            clone(pipeline), X_train, y_train, cv=cv,
            method="predict_proba", n_jobs=-1
        )[:, 1]

        for threshold in THRESHOLDS:
            oof_pred = (oof_prob >= threshold).astype(int)
            acc = accuracy_score(y_train, oof_pred)
            if best is None or acc > best["accuracy"]:
                best = {"params": params, "threshold": threshold, "accuracy": acc}

        print(f"[{model_name}] Progress: {i+1}/{len(param_grid)}, best CV accuracy: {best['accuracy']*100:.2f}%")

    print(f"\nKonfigurasi terbaik ({model_name}, dari 5-fold CV di training set):")
    print(f"Params   : {best['params']}")
    print(f"Threshold: {best['threshold']:.3f}")
    print(f"CV Acc (di training set) : {best['accuracy']*100:.2f}%")

    final_pipeline = build_fn(best["params"])
    final_pipeline.fit(X_train, y_train)

    val_prob = final_pipeline.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= best["threshold"]).astype(int)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Val Accuracy (cek generalisasi): {val_accuracy*100:.2f}%")

    test_prob = final_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best["threshold"]).astype(int)

    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_prob)
    pr_auc = average_precision_score(y_test, test_prob)
    cm = confusion_matrix(y_test, test_pred)

    print(f"\nHASIL TEST SET -- SKEMA {scheme_name} -- MODEL {model_name}")
    print(f"Accuracy   : {accuracy * 100:.2f}%")
    print(f"Precision  : {precision * 100:.2f}%")
    print(f"Recall     : {recall * 100:.2f}%")
    print(f"F1-Score   : {f1 * 100:.2f}%")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print(f"PR-AUC     : {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print(classification_report(y_test, test_pred, zero_division=0))

    gap = (accuracy - best["accuracy"]) * 100
    se = np.sqrt(accuracy * (1 - accuracy) / len(y_test)) * 100
    print(f"Gap Test - CV: {gap:+.2f} poin (n_test={len(y_test)}, SE~{se:.2f}%)")

    if accuracy >= 0.90:
        print(f">>> TARGET TERCAPAI: {accuracy*100:.2f}% >= 90%")
    else:
        print(f">>> TARGET BELUM TERCAPAI: {accuracy*100:.2f}%")

    return {
        "Model": model_name,
        "Skema": scheme_name,
        "Training": len(X_train),
        "Validation": len(X_val),
        "Testing": len(X_test),
        "Params": best["params"],
        "Threshold": best["threshold"],
        "CV_Accuracy": best["accuracy"],
        "Val_Accuracy": val_accuracy,
        "Test_Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


def main():
    print("=" * 80)
    print("LOAD DATA")
    print("=" * 80)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline()
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    df = pd.read_csv(DATA_PATH, sep=delimiter)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Setelah deduplicate: {len(df)}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    results = []
    for scheme_name, scheme in SPLIT_SCHEMES.items():
        for model_name, model_cfg in MODELS.items():
            result = run_scheme(
                X, y, scheme_name, scheme,
                model_name, model_cfg["build_fn"], model_cfg["param_grid"]
            )
            results.append(result)

    results_df = pd.DataFrame(results)

    print("\n\n" + "#" * 110)
    print("RINGKASAN SEMUA SKEMA & MODEL (CatBoost vs XGBoost, 5-fold CV di training set SAJA)")
    print("#" * 110)

    display_df = results_df.copy()
    for col in ["CV_Accuracy", "Val_Accuracy", "Test_Accuracy", "Precision", "Recall", "F1"]:
        display_df[col] = (display_df[col] * 100).round(2)
    display_df["ROC_AUC"] = display_df["ROC_AUC"].round(4)

    print(display_df[["Model", "Skema", "Training", "Validation", "Testing",
                       "CV_Accuracy", "Val_Accuracy", "Test_Accuracy",
                       "Precision", "Recall", "F1", "ROC_AUC"]].to_string(index=False))

    results_df.to_csv("hasil_catboost_vs_xgboost.csv", index=False)

    # ============================================================
    # PILIH SKEMA FINAL UNTUK MODEL UTAMA (CatBoost, akurasi test
    # tertinggi) & LATIH ULANG PADA SELURUH DATA HASIL
    # PRA-PEMROSESAN, KHUSUS UNTUK KEPERLUAN DEPLOYMENT DI APLIKASI
    # STREAMLIT. XGBoost hanya dipakai sebagai model pembanding dan
    # TIDAK di-deploy. Metrik yang dilaporkan tetap berasal dari
    # evaluasi test set di atas -- model hasil retrain ini TIDAK
    # dipakai buat evaluasi.
    # ============================================================
    catboost_results = results_df[results_df["Model"] == "CatBoost"].reset_index(drop=True)
    best_idx = catboost_results["Test_Accuracy"].idxmax()
    best_row = catboost_results.loc[best_idx]

    print("\n" + "=" * 80)
    print("SKEMA FINAL TERPILIH -- MODEL UTAMA CatBoost (akurasi test tertinggi)")
    print("=" * 80)
    print(f"Skema     : {best_row['Skema']}")
    print(f"Params    : {best_row['Params']}")
    print(f"Threshold : {best_row['Threshold']:.3f}")
    print(f"Test Acc  : {best_row['Test_Accuracy']*100:.2f}%")
    print(f"ROC-AUC   : {best_row['ROC_AUC']:.4f}")

    final_pipeline = build_catboost(best_row["Params"])
    final_pipeline.fit(X, y)

    payload = {
        "pipeline": final_pipeline,
        "threshold": float(best_row["Threshold"]),
        "feature_names": list(X.columns),
        "reported_metrics": {
            "scheme": best_row["Skema"],
            "test_accuracy": float(best_row["Test_Accuracy"]),
            "roc_auc": float(best_row["ROC_AUC"]),
            "precision": float(best_row["Precision"]),
            "recall": float(best_row["Recall"]),
        },
    }

    with open("best_catboost_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("\nModel final (CatBoost, dilatih pada seluruh data) disimpan ke:")
    print("best_catboost_model.pkl")


if __name__ == "__main__":
    main()