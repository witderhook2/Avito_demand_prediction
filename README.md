# Avito Demand Prediction

https://www.kaggle.com/competitions/avito-demand-prediction

Соревнование по прогнозированию вероятности сделки (`deal_probability`) объявлений Avito. Итоговая модель — **CatBoostRegressor** с категориальными признаками и обработкой текстовых полей `title` и `description`. По результатам загрузки было получено RMSE=0.22659 на Public Score

## Структура проекта

```text
avito_demand/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_avito_eda.ipynb
│   ├── 02_avito_baseline_models.ipynb
│   ├── 03_avito_text_features.ipynb
│   └── 04_avito_hyperparameter_tuning.ipynb
├── .dockerignore
├── app_api.py
├── catboost_model.pkl
├── Dockerfile
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

* **`data/`** — директория с исходными датасетами (`train.csv`, `test.csv`). Папка не включается в Git-репозиторий из-за большого объема файлов
* **`notebooks/`** — исследовательская часть проекта: разведочный анализ данных (EDA), построение бейзлайнов, генерация текстовых признаков и подбор гиперпараметров
* **`train.py` и `predict.py`** — отдельный воспроизводимый пайплайн для обучения модели и пакетного инференса (формирования предсказаний)
* **`catboost_model.pkl`** — сериализованные веса финальной обученной модели CatBoost (тоже нет в репозитория из-за большого веса)
* **`app_api.py`** — веб-сервис на базе FastAPI, реализующий REST API для онлайн-предсказаний;
* **`Dockerfile`, `.dockerignore` и `requirements.txt`** — конфигурация окружения, список зависимостей и инструкции для сборки Docker-образа

## Постановка задачи

Для каждого объявления требуется оценить `deal_probability` — вероятность совершения сделки. Это задача регрессии с целевой переменной в диапазоне `[0, 1]`. Основная метрика качества — RMSE, где наименьшее значение лучше

## Подход к решению

### 1. Exploratory Data Analysis

В первом ноутбуке исследованы структура данных, пропуски, распределения признаков и целевой переменной, а также временные и категориальные срезы. EDA использовался для понимания качества данных и выбора дальнейших экспериментов

### 2. Сравнение baseline-моделей

Во втором ноутбуке сравнивались линейные модели (Linear Regression и Ridge), SVM, Decision Tree, Random Forest, XGBoost, LightGBM и CatBoost. Сравнение выполнялось по RMSE и лучший результат показал CatBoost

### 3. Эксперименты с текстом

В третьем ноутбуке проверялись четыре варианта раюоты с текстовыми данными:

- CatBoost + TF-IDF по `title` (до 2 000 признаков);
- CatBoost + TF-IDF (до 2 000 признаков) и предварительная лемматизация заголовка;
- CatBoost с нативным `text_features=['title']`;
- CatBoost с нативными `text_features=['title', 'description']` - *этот вариант показал лучший результат*

### 4. Подбор гиперпараметров

В четвёртом ноутбуке использовалась Optuna для подбора гиперпараметров CatBoost. Пространство поиска включало `learning_rate`, `depth`, `l2_leaf_reg`, `random_strength` и `bagging_temperature`

## Используемые признаки в финальном скрипте

| Тип | Признаки |
|---|---|
| Категориальные | `region`, `city`, `parent_category_name`, `category_name`, `param_1`, `activation_day_of_week`, `user_type`, `image_top_1` |
| Числовые | `price`, `item_seq_number` |
| Текстовые | `title`, `description` |

`activation_day_of_week` извлекается из `activation_date`

## API взаимодействие с моделью

### Как запустить сервиса через Docker

1. **Собираем Docker-образ:**
```bash
docker build -t avito-demand-service .
```

2. **Запускаем контейнер:**
```bash
docker run -d --name avito-demand-service -p 8000:5000 avito-demand-service
```

После запуска сервис будет доступен по адресу `http://localhost:8000`. 

Интерактивная документация Swagger UI доступна по ссылке `http://localhost:8000/docs`.

---

### Описание эндпоинтов и примеры запросов

#### 1. Проверка работоспособности (`GET /health`)
Проверяем, что сервис запущен и готов принимать запросы.

**Запрос:**
```bash
curl -X GET http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "OK"
}
```

---

#### 2. Статистика запросов (`GET /stats`)
Смотрим количество обращений к эндпоинту предсказания с момента запуска контейнера.

**Запрос:**
```bash
curl -X GET http://localhost:8000/stats
```

**Ответ:**
```json
{
  "request_count": 1
}
```

---

#### 3. Предсказание вероятности сделки (`POST /predict_model`)
Принимает параметры объявления в формате JSON и предсказывает вероятность сделки (`deal_probability` в диапазоне от 0 до 1).

**Пример запроса (`curl`):**
```bash
curl -X POST http://localhost:8000/predict_model \
  -H "Content-Type: application/json" \
  -d '{
    "region": "Свердловская область",
    "city": "Екатеринбург",
    "parent_category_name": "Личные вещи",
    "category_name": "Одежда, обувь, аксессуары",
    "param_1": "Женская одежда",
    "activation_date": "2017-03-28",
    "user_type": "Private",
    "image_top_1": "1008",
    "price": 1500.0,
    "item_seq_number": 42,
    "title": "Платье вечернее",
    "description": "В отличном состоянии, надевала один раз"
  }'
```

**Пример ответа:**
```json
{
  "deal_probability": 0.24518
}
```
