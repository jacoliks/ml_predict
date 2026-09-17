"""Вспомогательные функции для ноутбука с моделью цены квартир.

Здесь собраны функции, которые вызываются в ноутбуке больше одного раза
или содержат ветвления: генерация признаков, метрики, утилиты для SHAP.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Координаты Кремля — условный центр Москвы
CENTER_LAT = 55.7520
CENTER_LON = 37.6175
KM_PER_DEG_LAT = 111.13

# Шаблон признака «флаг по ключевым словам в тексте объявления»
TEXT_FLAGS = {
    "is_apartments": r"апартамент",
    "is_premium": r"пентхаус|террас",
    "has_designer_repair": r"дизайнерск",
    "has_euro_repair": r"евроремонт",
    "is_no_finish": r"без отделки|черновая отделка|черновой отделк",
    "is_white_box": r"предчистов|white box|вайт бокс",
    "has_parking": r"паркинг|машиномест",
    "has_view": r"вид на|видовая|видовые",
    "is_owner": r"собственник",
    "is_new_building": r"застройщик|новостро|дду|переуступ|сдача дома",
}

METRO_MINUTES_PATTERN = (
    r"(\d{1,2})\s*(?:-|–)?\s*(?:ти|х)?\s*мин\w*\s+(?:пешком|пешей|ходьбы)"
)
HOUSE_FLOORS_PATTERN = r"(\d{1,2})\s*-?\s*этажн"

CATEGORICAL_FEATURES = ["district", "locality", "microdistrict", "source"]


def haversine_km(lat, lon, lat0=CENTER_LAT, lon0=CENTER_LON):
    """Расстояние по сфере (км) от точек до опорной точки."""
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = (
        np.sin((lat - lat0) / 2) ** 2
        + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    )
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def bearing_deg(lat, lon, lat0=CENTER_LAT, lon0=CENTER_LON):
    """Направление от центра на объект в градусах (0 — восток, 90 — север)."""
    dy = (lat - lat0) * KM_PER_DEG_LAT
    dx = (lon - lon0) * KM_PER_DEG_LAT * np.cos(np.radians(lat0))
    return np.degrees(np.arctan2(dy, dx))


def extract_number(text, pattern, max_value):
    """Достаёт первое число по регулярке; нереалистичные значения -> NaN."""
    values = text.str.extract(pattern)[0].astype(float)
    return values.where((values > 0) & (values <= max_value))


def build_features(df):
    """Строит матрицу признаков из сырого датафрейма объявлений.

    Функция не использует целевую переменную и статистики по выборке,
    поэтому её безопасно применять до разбиения на train/test.
    """
    text = df["description"].fillna("").str.lower()
    layout = df["product_name"].str.split(",").str[0]

    features = pd.DataFrame(index=df.index)
    features["total_square"] = df["total_square"]
    features["is_studio"] = (layout == "Студия").astype(int)
    features["is_free_layout"] = (layout == "Квартира").astype(int)
    # У студий комнаты указаны то как NaN, то как 1 — приводим к 0
    features["rooms"] = df["rooms"].mask(features["is_studio"] == 1, 0)
    features["square_per_room"] = features["total_square"] / features["rooms"].clip(
        lower=1
    )
    features["floor"] = df["floor"]

    house_floors = extract_number(text, HOUSE_FLOORS_PATTERN, max_value=99)
    features["house_floors"] = house_floors.where(house_floors >= df["floor"])
    features["floor_ratio"] = features["floor"] / features["house_floors"]

    features["lat"] = df["lat"]
    features["lon"] = df["lon"]
    features["dist_to_center_km"] = haversine_km(df["lat"], df["lon"])
    features["bearing_deg"] = bearing_deg(df["lat"], df["lon"])
    features["metro_walk_min"] = extract_number(
        text, METRO_MINUTES_PATTERN, max_value=40
    )

    features["is_moscow"] = (df["city"] == "Москва").astype(int)
    features["address_has_zhk"] = (
        df["address_name"].str.contains("ЖК", regex=False).astype(int)
    )
    for name, pattern in TEXT_FLAGS.items():
        features[name] = text.str.contains(pattern, regex=True).astype(int)
    features["description_len_log"] = np.log1p(text.str.len())

    features["district"] = df["district"].fillna("нет")
    features["locality"] = df["city"].fillna(df["settlement"]).fillna("нет")
    features["microdistrict"] = df["area"].fillna("нет")
    features["source"] = df["source"]
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("category")
    return features


def make_listing_groups(df):
    """Идентификатор «физической» квартиры: адрес + площадь + этаж.

    Одна и та же квартира часто размещена на нескольких площадках;
    такие объявления должны попадать целиком либо в train, либо в test.
    """
    key = (
        df["address_name"].astype(str)
        + "|"
        + df["total_square"].astype(str)
        + "|"
        + df["floor"].astype(str)
    )
    return pd.Series(pd.factorize(key)[0], index=df.index, name="group")


def mape_optimal_constant(y):
    """Константа, минимизирующая MAPE: взвешенная медиана с весами 1/y."""
    order = np.argsort(y)
    y_sorted = np.asarray(y)[order]
    weights = 1 / y_sorted
    cumulative = np.cumsum(weights) / weights.sum()
    return y_sorted[np.searchsorted(cumulative, 0.5)]


def regression_report(y_true, y_pred):
    """Набор метрик: основная (MAPE) и вспомогательные."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "MAPE, %": 100 * ape.mean(),
        "MdAPE, %": 100 * np.median(ape),
        "Доля ошибок <10%, %": 100 * (ape < 0.10).mean(),
        "MAE, млн руб.": mean_absolute_error(y_true, y_pred) / 1e6,
        "R2": r2_score(y_true, y_pred),
    }


def categories_to_codes(features):
    """Категории -> коды, чтобы SHAP мог раскрасить точки на графиках."""
    numeric = features.copy()
    for column in numeric.select_dtypes("category").columns:
        numeric[column] = numeric[column].cat.codes
    return numeric


def shap_contribution_table(explanation, top_n=8):
    """Топ вкладов признаков для одного объекта в понятном виде.

    Модель обучена на log(price), поэтому SHAP-вклад s означает,
    что признак умножает цену на exp(s).
    """
    table = pd.DataFrame(
        {
            "признак": explanation.feature_names,
            "значение": explanation.display_data,
            "shap (log)": explanation.values,
        }
    )
    table["влияние на цену, %"] = 100 * (np.exp(table["shap (log)"]) - 1)
    order = table["shap (log)"].abs().sort_values(ascending=False).index
    return table.loc[order].head(top_n).round(3).reset_index(drop=True)
