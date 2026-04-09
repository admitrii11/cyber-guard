"""
Обучение Isolation Forest для UEBA (детекция аномального поведения).

Что делает этот скрипт:
1) Загружает синтетические сессии из CSV.
2) Делит данные на train/test с сохранением доли классов.
3) Масштабирует признаки (StandardScaler).
4) Обучает Isolation Forest только на нормальных train-сессиях.
5) Считает метрики качества и дополнительные аналитические артефакты.
6) Сохраняет модель, скейлеры и JSON-файлы для будущего дашборда.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Явно задаем имена признаков, чтобы:
# 1) не ошибиться с порядком колонок,
# 2) использовать единый порядок во всех артефактах.
FEATURE_COLUMNS = [
    "clicks_per_minute",
    "avg_time_between_clicks_sec",
    "unique_pages_visited",
    "sensitive_pages_accessed",
    "session_duration_min",
    "data_downloaded_mb",
]


class IsolationForestBinaryWrapper(ClassifierMixin, BaseEstimator):
    """
    Тонкая обертка над IsolationForest для permutation importance.

    Зачем нужна:
    - Базовый IsolationForest.predict(...) возвращает метки {1, -1},
      где 1 = норма, -1 = аномалия.
    - Для метрики f1 и нашей задачи удобнее формат {0, 1},
      где 0 = норма, 1 = аномалия.
    - Эта обертка конвертирует предсказания в нужный формат.
    """

    def __init__(self, model: IsolationForest) -> None:
        self.model = model
        # Нужен для совместимости со scorer-ами sklearn в permutation_importance.
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IsolationForestBinaryWrapper":
        # fit здесь не используется (модель уже обучена), но метод нужен
        # для совместимости со sklearn API.
        _ = (X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw_pred = self.model.predict(X)  # 1 = норма, -1 = аномалия
        return np.where(raw_pred == -1, 1, 0)  # 1 = аномалия, 0 = норма

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Для permutation importance с scoring='f1' sklearn обычно вызовет scorer,
        # но наличие score делает объект более "полным" как estimator.
        y_pred = self.predict(X)
        return f1_score(y, y_pred, zero_division=0)


def _to_python_number(value: Any) -> Any:
    """
    Приводит numpy-числа к обычным Python-типам для JSON.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _to_float_list(values: np.ndarray) -> list[float]:
    """
    Переводит массив numpy в список float для JSON.
    """
    return [float(v) for v in values]


def _to_threshold_list(values: np.ndarray) -> list[float | None]:
    """
    Порог ROC может содержать +inf. JSON не поддерживает inf,
    поэтому такие значения сохраняем как null (None в Python).
    """
    result: list[float | None] = []
    for v in values:
        if np.isfinite(v):
            result.append(float(v))
        else:
            result.append(None)
    return result


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """
    Считает базовые бинарные метрики для positive class = 1 (аномалия).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train_and_evaluate() -> None:
    """
    Полный цикл обучения и оценки модели UEBA.
    """
    # Определяем путь к корню проекта и к папке data.
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) Загружаем данные из data/synthetic_sessions.csv
    dataset_path = data_dir / "synthetic_sessions.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Файл датасета не найден: {dataset_path}\n"
            "Сначала запустите ueba/data_generator.py для генерации CSV."
        )

    df = pd.read_csv(dataset_path)

    # 2) Разделяем на X (признаки) и y (целевой label)
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int).copy()

    # 3) Делим train/test: 80/20, stratified, random_state=42
    # stratify=y сохраняет пропорции классов в train и test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4) Масштабируем признаки StandardScaler:
    # ВАЖНО: fit только на train, чтобы избежать утечки данных из test.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Обучаем Isolation Forest только на нормальных train-данных (label=0)
    # Создаем маску нормальных объектов.
    normal_mask = y_train == 0
    X_train_normal_scaled = X_train_scaled[normal_mask.values]

    model = IsolationForest(
        contamination=0.06,
        random_state=42,
        n_estimators=100,
    )
    model.fit(X_train_normal_scaled)

    # 6) Предсказываем на X_test и перекодируем:
    # sklearn: 1 = норма, -1 = аномалия
    # нам нужно: 0 = норма, 1 = аномалия
    raw_pred_default = model.predict(X_test_scaled)
    y_pred_default = np.where(raw_pred_default == -1, 1, 0)

    # 7) Работа со score:
    # decision_function возвращает "степень нормальности":
    # - чем БОЛЬШЕ значение, тем объект более "нормальный";
    # - чем МЕНЬШЕ значение, тем объект более подозрительный.
    #
    # Для UEBA нам удобнее "risk score":
    # - чем БОЛЬШЕ значение, тем ВЫШЕ риск.
    # Поэтому мы инвертируем знак:
    # risk_raw = -decision_function(...)
    #
    # После этого нормализуем risk_raw в диапазон [0, 1] через MinMaxScaler.
    # Эти нормализованные значения будут использоваться в дашборде
    # и в пороговой логике (например, 0.3 и 0.7).
    decision_scores = model.decision_function(X_test_scaled)
    risk_scores_raw = -decision_scores

    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    risk_scores_normalized = minmax_scaler.fit_transform(risk_scores_raw.reshape(-1, 1)).ravel()

    # 8) Считаем метрики качества в нескольких режимах принятия решения.
    #
    # ВАЖНОЕ ПОЯСНЕНИЕ:
    # - default-метрики (через model.predict) используют внутренний порог IsolationForest,
    #   который завязан на contamination. Это "технический" порог модели, но не бизнес-логика UEBA.
    # - В production мы принимаем решение по risk score [0, 1] и собственным порогам (0.3 / 0.7).
    #   Поэтому production-метрики честнее отражают реальную работу системы.
    roc_auc = roc_auc_score(y_test, risk_scores_raw)  # по raw score (до нормализации)

    # 8.1) DEFAULT: как делает sklearn через contamination=0.06.
    metrics_default = _compute_binary_metrics(y_test.to_numpy(), y_pred_default)
    metrics_default["roc_auc"] = float(roc_auc)

    # 8.2) TUNED_05: порог на нормализованном risk score = 0.5.
    # score >= 0.5 считаем аномалией (1), иначе нормой (0).
    y_pred_tuned_05 = (risk_scores_normalized >= 0.5).astype(int)
    metrics_tuned_05 = _compute_binary_metrics(y_test.to_numpy(), y_pred_tuned_05)
    metrics_tuned_05["roc_auc"] = float(roc_auc)

    # 8.3) PRODUCTION: трехуровневая политика.
    # - score < 0.3: allowed (предсказанная норма)
    # - 0.3 <= score < 0.7: mfa (доп. проверка, не блок)
    # - score >= 0.7: blocked (предсказанная аномалия)
    #
    # Для бинарных метрик positive class = blocked (score >= 0.7).
    y_pred_production = (risk_scores_normalized >= 0.7).astype(int)
    metrics_production_binary = _compute_binary_metrics(y_test.to_numpy(), y_pred_production)

    # Считаем, как реальные классы распределились по зонам production-логики.
    zone_allowed_mask = risk_scores_normalized < 0.3
    zone_mfa_mask = (risk_scores_normalized >= 0.3) & (risk_scores_normalized < 0.7)
    zone_blocked_mask = risk_scores_normalized >= 0.7

    y_test_np = y_test.to_numpy()
    normal_mask = y_test_np == 0
    anomaly_mask = y_test_np == 1

    zones = {
        "normal_in_allowed": int(np.sum(normal_mask & zone_allowed_mask)),
        "normal_in_mfa": int(np.sum(normal_mask & zone_mfa_mask)),
        "normal_in_blocked": int(np.sum(normal_mask & zone_blocked_mask)),
        "anomaly_in_allowed": int(np.sum(anomaly_mask & zone_allowed_mask)),
        "anomaly_in_mfa": int(np.sum(anomaly_mask & zone_mfa_mask)),
        "anomaly_in_blocked": int(np.sum(anomaly_mask & zone_blocked_mask)),
    }

    metrics_production = {
        "precision": float(metrics_production_binary["precision"]),
        "recall": float(metrics_production_binary["recall"]),
        "f1": float(metrics_production_binary["f1"]),
        "zones": zones,
    }

    # Дополнительно: эксперимент с contamination=0.01 (только для сравнения default-режима).
    # Основная сохраняемая модель остается contamination=0.06.
    model_c001 = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_estimators=100,
    )
    model_c001.fit(X_train_normal_scaled)
    y_pred_default_c001 = np.where(model_c001.predict(X_test_scaled) == -1, 1, 0)
    metrics_default_c001 = _compute_binary_metrics(y_test.to_numpy(), y_pred_default_c001)

    # Печатаем отчет по default-подходу, чтобы сохранить совместимость с предыдущим форматом лога.
    class_report_default = classification_report(
        y_test, y_pred_default, digits=4, zero_division=0
    )
    print("\n=== Classification Report (Default / contamination=0.06) ===")
    print(class_report_default)

    # 9) Сохраняем метрики в metrics.json в новой структуре
    metrics = {
        "default": metrics_default,
        "tuned_05": metrics_tuned_05,
        "production": metrics_production,
    }
    metrics_path = data_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # 10) Считаем permutation importance
    # Здесь используем test-набор (уже масштабированный),
    # scoring='f1' как в требовании.
    wrapped_model = IsolationForestBinaryWrapper(model)
    perm = permutation_importance(
        estimator=wrapped_model,
        X=X_test_scaled,
        y=y_test.to_numpy(),
        n_repeats=10,
        random_state=42,
        scoring="f1",
    )

    feature_importance_items: list[dict[str, Any]] = []
    for idx, feature_name in enumerate(FEATURE_COLUMNS):
        feature_importance_items.append(
            {
                "feature": feature_name,
                "importance": float(perm.importances_mean[idx]),
                "std": float(perm.importances_std[idx]),
            }
        )

    feature_importance_items.sort(key=lambda item: item["importance"], reverse=True)

    feature_importance_path = data_dir / "feature_importance.json"
    feature_importance_path.write_text(
        json.dumps(feature_importance_items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 11) Сохраняем модель и скейлеры в формате joblib
    model_path = data_dir / "isolation_forest.pkl"
    scaler_path = data_dir / "scaler.pkl"
    minmax_scaler_path = data_dir / "minmax_scaler.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(minmax_scaler, minmax_scaler_path)

    # 12.a) test_scores.json (для гистограммы score)
    test_scores = [
        {"score": float(score), "label": int(label)}
        for score, label in zip(risk_scores_normalized, y_test.to_numpy())
    ]
    test_scores_path = data_dir / "test_scores.json"
    test_scores_path.write_text(
        json.dumps(test_scores, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 12.b) roc_curve_data.json
    fpr, tpr, roc_thresholds = roc_curve(y_test, risk_scores_raw)
    roc_curve_payload = {
        "fpr": _to_float_list(fpr),
        "tpr": _to_float_list(tpr),
        "auc": float(roc_auc),
        "thresholds": _to_threshold_list(roc_thresholds),
    }
    roc_curve_path = data_dir / "roc_curve_data.json"
    roc_curve_path.write_text(
        json.dumps(roc_curve_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 12.c) pr_curve_data.json (bonus)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, risk_scores_raw)
    # Площадь PR-кривой считаем численным интегрированием по recall.
    # Для корректного интеграла сортируем по recall по возрастанию.
    pr_order = np.argsort(pr_recall)
    pr_auc = np.trapezoid(pr_precision[pr_order], pr_recall[pr_order])

    pr_curve_payload = {
        "precision": _to_float_list(pr_precision),
        "recall": _to_float_list(pr_recall),
        "auc": float(pr_auc),
    }
    pr_curve_path = data_dir / "pr_curve_data.json"
    pr_curve_path.write_text(
        json.dumps(pr_curve_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 12.d) pca_projection.json (2D-проекция test set)
    pca = PCA(n_components=2, random_state=42)
    X_test_pca = pca.fit_transform(X_test_scaled)
    pca_projection = [
        {
            "x": float(point[0]),
            "y": float(point[1]),
            "label": int(label),
            "score": float(score),
        }
        for point, label, score in zip(
            X_test_pca, y_test.to_numpy(), risk_scores_normalized
        )
    ]
    pca_projection_path = data_dir / "pca_projection.json"
    pca_projection_path.write_text(
        json.dumps(pca_projection, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 12.e) threshold_analysis.json
    threshold_analysis: list[dict[str, float]] = []
    thresholds = np.arange(0.1, 0.901, 0.05)

    for threshold in thresholds:
        # Если риск >= threshold, считаем сессию аномальной.
        y_pred_threshold = (risk_scores_normalized >= threshold).astype(int)
        threshold_precision = precision_score(y_test, y_pred_threshold, zero_division=0)
        threshold_recall = recall_score(y_test, y_pred_threshold, zero_division=0)
        threshold_f1 = f1_score(y_test, y_pred_threshold, zero_division=0)

        threshold_analysis.append(
            {
                "threshold": float(threshold),
                "precision": float(threshold_precision),
                "recall": float(threshold_recall),
                "f1": float(threshold_f1),
            }
        )

    threshold_analysis_path = data_dir / "threshold_analysis.json"
    threshold_analysis_path.write_text(
        json.dumps(threshold_analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 13) Красивый финальный вывод в консоль:
    # сравнение трех подходов + отдельный default при contamination=0.01.
    print("\n" + "=" * 95)
    print("СРАВНЕНИЕ ПОДХОДОВ ОЦЕНКИ (TEST)")
    print("=" * 95)
    print(
        f"{'Подход':<28}"
        f"{'Precision':>11}"
        f"{'Recall':>11}"
        f"{'F1':>11}"
        f"{'Accuracy':>11}"
        f"{'Specificity':>13}"
    )
    print("-" * 95)
    print(
        f"{'Default (cont=0.06)':<28}"
        f"{metrics_default['precision']:>11.4f}"
        f"{metrics_default['recall']:>11.4f}"
        f"{metrics_default['f1']:>11.4f}"
        f"{metrics_default['accuracy']:>11.4f}"
        f"{metrics_default['specificity']:>13.4f}"
    )
    print(
        f"{'Tuned threshold=0.50':<28}"
        f"{metrics_tuned_05['precision']:>11.4f}"
        f"{metrics_tuned_05['recall']:>11.4f}"
        f"{metrics_tuned_05['f1']:>11.4f}"
        f"{metrics_tuned_05['accuracy']:>11.4f}"
        f"{metrics_tuned_05['specificity']:>13.4f}"
    )
    print(
        f"{'Production (0.30/0.70)':<28}"
        f"{metrics_production['precision']:>11.4f}"
        f"{metrics_production['recall']:>11.4f}"
        f"{metrics_production['f1']:>11.4f}"
        f"{metrics_production_binary['accuracy']:>11.4f}"
        f"{metrics_production_binary['specificity']:>13.4f}"
    )
    print("-" * 95)
    print(f"ROC-AUC (raw risk score): {roc_auc:.4f}")

    print("\nDefault-метрики при contamination=0.01 (сравнение порога модели):")
    print(
        f"Precision={metrics_default_c001['precision']:.4f}, "
        f"Recall={metrics_default_c001['recall']:.4f}, "
        f"F1={metrics_default_c001['f1']:.4f}, "
        f"Accuracy={metrics_default_c001['accuracy']:.4f}, "
        f"Specificity={metrics_default_c001['specificity']:.4f}"
    )

    print("\nРаспределение реальных классов по production-зонам:")
    print(
        f"- Нормальные: allowed={zones['normal_in_allowed']}, "
        f"mfa={zones['normal_in_mfa']}, blocked={zones['normal_in_blocked']}"
    )
    print(
        f"- Аномалии:   allowed={zones['anomaly_in_allowed']}, "
        f"mfa={zones['anomaly_in_mfa']}, blocked={zones['anomaly_in_blocked']}"
    )

    print("\nСохраненные файлы:")
    saved_paths = [
        metrics_path,
        feature_importance_path,
        model_path,
        scaler_path,
        minmax_scaler_path,
        test_scores_path,
        roc_curve_path,
        pr_curve_path,
        pca_projection_path,
        threshold_analysis_path,
    ]
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    train_and_evaluate()
