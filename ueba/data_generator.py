"""
Генерация синтетических сессий пользователей для UEBA.

Идея файла:
1) Сгенерировать "нормальные" пользовательские сессии (label=0).
2) Сгенерировать "аномальные" сессии, похожие на действия нарушителя (label=1).
3) Объединить, перемешать и сохранить датасет в CSV для обучения модели.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Фиксируем seed для полной воспроизводимости результатов.
# Это важно в обучении: при одинаковых параметрах мы получим одинаковый датасет.
np.random.seed(42)


def generate_dataset(n_normal: int = 8000, n_anomaly: int = 500) -> pd.DataFrame:
    """
    Генерирует синтетический датасет поведения пользователей в сессиях.

    Параметры:
        n_normal: количество нормальных сессий (label=0)
        n_anomaly: количество аномальных сессий (label=1)

    Возвращает:
        pandas.DataFrame с 6 признаками + колонкой label.
    """

    # ----------------------------
    # 1) Генерация нормальных сессий (label=0)
    # ----------------------------
    # Чтобы модель лучше распознавала "реальное" нормальное поведение в ранние моменты,
    # делим нормальные сессии на 3 типа:
    # - 60%: полные рабочие сессии;
    # - 25%: короткие сессии;
    # - 15%: очень короткие (cold start).
    full_count = int(round(n_normal * 0.60))
    short_count = int(round(n_normal * 0.25))
    very_short_count = n_normal - full_count - short_count

    # 1.a) Полные сессии: 20-40 мин, 8-12 cpm, 4-7 страниц, 2-5 MB.
    full_clicks_per_minute = np.random.uniform(8.0, 12.0, size=full_count)
    full_avg_time_between_clicks = np.random.normal(loc=7.0, scale=1.5, size=full_count)
    full_unique_pages = np.random.randint(4, 8, size=full_count)
    full_sensitive_access = np.random.poisson(lam=0.4, size=full_count)
    full_session_duration = np.random.uniform(20.0, 40.0, size=full_count)
    full_data_downloaded = np.random.uniform(2.0, 5.0, size=full_count)

    # 1.b) Короткие сессии: 1-5 мин, 3-8 cpm, 2-4 страницы, 0-2 MB.
    short_clicks_per_minute = np.random.uniform(3.0, 8.0, size=short_count)
    short_avg_time_between_clicks = np.random.normal(loc=8.0, scale=2.0, size=short_count)
    short_unique_pages = np.random.randint(2, 5, size=short_count)
    short_sensitive_access = np.random.poisson(lam=0.2, size=short_count)
    short_session_duration = np.random.uniform(1.0, 5.0, size=short_count)
    short_data_downloaded = np.random.uniform(0.0, 2.0, size=short_count)

    # 1.c) Очень короткие (cold start): 0.1-1 мин, 2-6 cpm, 1-2 страницы, 0-0.5 MB.
    # sensitive_pages_accessed фиксируем в 0 по требованию.
    very_short_clicks_per_minute = np.random.uniform(2.0, 6.0, size=very_short_count)
    very_short_avg_time_between_clicks = np.random.normal(
        loc=9.0, scale=2.0, size=very_short_count
    )
    very_short_unique_pages = np.random.randint(1, 3, size=very_short_count)
    very_short_sensitive_access = np.zeros(very_short_count, dtype=int)
    very_short_session_duration = np.random.uniform(0.1, 1.0, size=very_short_count)
    very_short_data_downloaded = np.random.uniform(0.0, 0.5, size=very_short_count)

    # Объединяем все подтипы нормальных сессий в единые массивы признаков.
    normal_clicks_per_minute = np.concatenate(
        [full_clicks_per_minute, short_clicks_per_minute, very_short_clicks_per_minute]
    )
    normal_avg_time_between_clicks = np.concatenate(
        [
            full_avg_time_between_clicks,
            short_avg_time_between_clicks,
            very_short_avg_time_between_clicks,
        ]
    )
    normal_unique_pages = np.concatenate(
        [full_unique_pages, short_unique_pages, very_short_unique_pages]
    )
    normal_sensitive_access = np.concatenate(
        [full_sensitive_access, short_sensitive_access, very_short_sensitive_access]
    )
    normal_session_duration = np.concatenate(
        [full_session_duration, short_session_duration, very_short_session_duration]
    )
    normal_data_downloaded = np.concatenate(
        [full_data_downloaded, short_data_downloaded, very_short_data_downloaded]
    )

    # ----------------------------
    # 2) Генерация аномальных сессий (label=1)
    # ----------------------------
    # clicks_per_minute: вокруг 25, std=8 (быстрые клики)
    anomaly_clicks_per_minute = np.random.normal(loc=25, scale=8, size=n_anomaly)

    # avg_time_between_clicks_sec: вокруг 1.5, std=0.5 (почти без пауз)
    anomaly_avg_time_between_clicks = np.random.normal(loc=1.5, scale=0.5, size=n_anomaly)

    # unique_pages_visited: вокруг 12, std=4, затем округляем до целых
    anomaly_unique_pages = np.random.normal(loc=12, scale=4, size=n_anomaly)
    anomaly_unique_pages = np.round(anomaly_unique_pages).astype(int)

    # sensitive_pages_accessed: вокруг 4, std=2, затем округляем до целых
    anomaly_sensitive_access = np.random.normal(loc=4, scale=2, size=n_anomaly)
    anomaly_sensitive_access = np.round(anomaly_sensitive_access).astype(int)

    # session_duration_min: смесь двух "режимов":
    # - очень короткие сессии (~3 мин)
    # - очень длинные сессии (~90 мин)
    # Делим аномалии на две группы и генерируем каждую отдельно.
    short_count = n_anomaly // 2
    long_count = n_anomaly - short_count
    anomaly_duration_short = np.random.normal(loc=3, scale=1.0, size=short_count)
    anomaly_duration_long = np.random.normal(loc=90, scale=15, size=long_count)
    anomaly_session_duration = np.concatenate([anomaly_duration_short, anomaly_duration_long])

    # data_downloaded_mb: вокруг 50, std=20 (эксфильтрация данных)
    anomaly_data_downloaded = np.random.normal(loc=50, scale=20, size=n_anomaly)

    # ----------------------------
    # 3) Гарантируем неотрицательность всех признаков
    # ----------------------------
    # Используем np.clip(..., 0, None), чтобы заменить любые отрицательные значения на 0.
    normal_clicks_per_minute = np.clip(normal_clicks_per_minute, 0, None)
    normal_avg_time_between_clicks = np.clip(normal_avg_time_between_clicks, 0, None)
    normal_unique_pages = np.clip(normal_unique_pages, 0, None)
    normal_sensitive_access = np.clip(normal_sensitive_access, 0, None)
    normal_session_duration = np.clip(normal_session_duration, 0, None)
    normal_data_downloaded = np.clip(normal_data_downloaded, 0, None)

    anomaly_clicks_per_minute = np.clip(anomaly_clicks_per_minute, 0, None)
    anomaly_avg_time_between_clicks = np.clip(anomaly_avg_time_between_clicks, 0, None)
    anomaly_unique_pages = np.clip(anomaly_unique_pages, 0, None)
    anomaly_sensitive_access = np.clip(anomaly_sensitive_access, 0, None)
    anomaly_session_duration = np.clip(anomaly_session_duration, 0, None)
    anomaly_data_downloaded = np.clip(anomaly_data_downloaded, 0, None)

    # ----------------------------
    # 4) Собираем два DataFrame: нормальный и аномальный
    # ----------------------------
    normal_df = pd.DataFrame(
        {
            "clicks_per_minute": normal_clicks_per_minute,
            "avg_time_between_clicks_sec": normal_avg_time_between_clicks,
            "unique_pages_visited": normal_unique_pages.astype(int),
            "sensitive_pages_accessed": normal_sensitive_access.astype(int),
            "session_duration_min": normal_session_duration,
            "data_downloaded_mb": normal_data_downloaded,
            "label": 0,
        }
    )

    anomaly_df = pd.DataFrame(
        {
            "clicks_per_minute": anomaly_clicks_per_minute,
            "avg_time_between_clicks_sec": anomaly_avg_time_between_clicks,
            "unique_pages_visited": anomaly_unique_pages.astype(int),
            "sensitive_pages_accessed": anomaly_sensitive_access.astype(int),
            "session_duration_min": anomaly_session_duration,
            "data_downloaded_mb": anomaly_data_downloaded,
            "label": 1,
        }
    )

    # ----------------------------
    # 5) Объединяем и перемешиваем датасет
    # ----------------------------
    # ignore_index=True пересобирает индекс от 0 до N-1.
    df = pd.concat([normal_df, anomaly_df], ignore_index=True)

    # sample(frac=1) означает "взять 100% строк в случайном порядке".
    # random_state=42 нужен для воспроизводимого перемешивания.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ----------------------------
    # 6) Сохраняем CSV в data/synthetic_sessions.csv
    # ----------------------------
    # Строим путь относительно корня проекта:
    # <project_root>/data/synthetic_sessions.csv
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_path = data_dir / "synthetic_sessions.csv"

    # На случай, если папка data отсутствует, создаем ее.
    data_dir.mkdir(parents=True, exist_ok=True)

    # index=False чтобы не добавлять в CSV лишнюю колонку индекса DataFrame.
    df.to_csv(output_path, index=False, encoding="utf-8")

    return df


if __name__ == "__main__":
    # Генерируем датасет с параметрами по умолчанию.
    df = generate_dataset()

    # Выводим первые строки, чтобы быстро посмотреть структуру и пример значений.
    print("\n=== Первые 5 строк датасета ===")
    print(df.head())

    # Выводим описательную статистику по всем числовым колонкам.
    print("\n=== Описательная статистика ===")
    print(df.describe())

    # Показываем распределение классов (сколько нормальных и аномальных сессий).
    print("\n=== Распределение классов ===")
    print(df["label"].value_counts())
