# Модель стоимости квартир в Москве и МО

Учебный проект: предсказание цены квартиры в объявлении о продаже (`price`).

## Структура

|Файл|Назначение|
|-|-|
|`realty\\\\\\\_price\\\\\\\_model.ipynb`|основной ноутбук: EDA, бейзлайны, LightGBM с подбором гиперпараметров, интерпретация|
|`helpers.py`|генерация признаков, метрики, утилиты для SHAP|
|`requirements.txt`|зафиксированные версии библиотек|
|`data/realty\\\\\\\_data.csv`|исходные данные (необходимо загрузить)|

## Воспроизведение

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data \\\\\\\&\\\\\\\& cp /path/to/realty\\\\\\\_data.csv data/
jupyter nbconvert --to notebook --execute realty\\\\\\\_price\\\\\\\_model.ipynb --inplace
```

Все источники случайности зафиксированы через `RANDOM\\\\\\\_STATE = 42`.
Для LightGBM включены `deterministic=True` и `force\\\\\\\_row\\\\\\\_wise=True`.
Полный прогон на одном ядре занимает около 15–20 минут, на многоядерной машине заметно быстрее.

