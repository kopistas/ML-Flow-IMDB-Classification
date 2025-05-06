# 🎬 Демонстрация классификации настроений отзывов  IMDB с MLflow 🚀

Это демострация интеграции пайплайна ML-классификатора в ML Flow ✨ ML Flow здесь просто хранит логи обучения и сами модели. 

## 🧩 Структура проекта

- `data/` - Инструменты для работы с данными IMDB 📊
- `models/` - ML модели ✨
- `mlflow_server/` - Отслеживание экспериментов 📈
- `app/` - API для обслуживания модели 🔌
- `tests/` - Тесты API и модели 🧪
- `Dockerfile` и `docker-compose.yml` - Конфигурация контейнеров 🐳

## 🚀 Быстрый старт с Docker

Самый простой способ запустить всё - с помощью Docker Compose:

```bash
# Запуск 
docker-compose up -d
```

Это выполнит:
1. 📦 Сборку Docker образа
2. 📊 Загрузку и обработку набора данных IMDB
3. 🧠 Обучение модели анализа настроений
4. 📈 Запуск сервера отслеживания MLflow
5. 🚀 Запуск API для предсказаний

## 🔮 Использование API

После запуска вы можете делать запросы к API по адресу http://localhost:5000:

### Одиночное предсказание

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Этот фильм был абсолютно фантастическим!"}'
```

### Пакетные предсказания

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": ["Этот фильм был абсолютно фантастическим!", "Худший фильм, который я когда-либо видел."]}'
```

### Использование предоставленного клиентского скрипта

Есть удобный клиентский скрипт:

```bash
# Одиночный текст
python app/query_api.py --text "Этот фильм был абсолютно фантастическим!"

# Из файла (один отзыв на строку)
python app/query_api.py --file my_reviews.txt --batch
```

## 📈 Отслеживание MLflow

Доступ к интерфейсу MLflow для просмотра экспериментов:

```
http://localhost:5001
```

## 🔧 Ручная настройка (без Docker)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Загрузка и обработка данных
python data/prepare_data.py

# Запуск сервера MLflow (в отдельном терминале)
python mlflow_server/run_server.py

# Обучение модели
python models/train.py

# Запуск API
python app/api.py
```

## 🧪 Тестирование

```bash
# С запущенным Docker
pytest tests/

# Или вручную
python -m pytest tests/
```