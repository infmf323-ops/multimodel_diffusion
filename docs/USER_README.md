# Multimodal Diffusion Dataset Service

Краткое руководство для пользователя по запуску и демонстрации проекта.

## Что делает проект

Сервис генерирует синтетические пары `text + image` для расширения мультимодального датасета.

В проекте есть два основных сценария:

1. Одиночная генерация изображения по prompt через `POST /generate`
2. Batch-генерация датасета по теме через `POST /generation-jobs`

Дополнительно в стенд интегрированы:

- `PostgreSQL` для истории запросов и generation jobs
- `MinIO` для хранения PNG и `manifest.csv`
- `Prometheus` для метрик сервиса
- `MLflow` для хранения экспериментов ЛР3

## Минимальные требования

- `Docker Desktop`
- `docker compose`
- Windows/macOS/Linux с доступным Docker daemon
- Для генерации изображений желательно наличие GPU, но сервис можно запускать и без него в упрощённом режиме

## Быстрый запуск

Из корня проекта:

```bash
docker compose up --build
```

Если хочешь запустить в фоне:

```bash
docker compose up -d --build
```

Проверка контейнеров:

```bash
docker compose ps
```

Остановка стенда:

```bash
docker compose down
```

## Что откроется после запуска

- Frontend: [http://localhost:3000](http://localhost:3000)
- FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health endpoint: [http://localhost:8000/health](http://localhost:8000/health)
- Prometheus metrics endpoint: [http://localhost:8000/metrics](http://localhost:8000/metrics)
- Prometheus UI: [http://localhost:9090](http://localhost:9090)
- MLflow UI: [http://localhost:5000](http://localhost:5000)
- MinIO Console: [http://localhost:9001](http://localhost:9001)

## Как использовать сервис

### 1. Одиночная генерация

Во frontend:

- открыть `http://localhost:3000`
- в блоке `Генерация одного изображения` ввести prompt
- нажать `Сгенерировать изображение`

Что вернётся:

- PNG preview
- `generation_id`
- `seed`
- `latency`
- путь к локальному файлу
- `output_object_uri` в MinIO

### 2. Генерация датасета по теме

Во frontend:

- открыть блок `Догенерация по теме`
- указать тему, например `мебель`
- указать количество изображений, например `100`
- нажать `Создать batch job`

Что делает сервис:

- создаёт `job_id`
- worker в фоне генерирует изображения
- сохраняет PNG и `manifest.csv`
- показывает первые preview
- считает dataset-level метрики по batch-партии

Что можно скачать:

- `manifest.csv` для всей партии

## Где смотреть метрики сервиса

### Raw metrics

Открыть:

[http://localhost:8000/metrics](http://localhost:8000/metrics)

Там видны метрики Prometheus, например:

- `http_requests_total`
- `http_request_duration_seconds`
- `ml_generations_total`
- `ml_generation_latency_seconds`
- `db_health_status`
- `diffusion_model_loaded_status`

### Prometheus UI

Открыть:

[http://localhost:9090](http://localhost:9090)

Удобные запросы для демонстрации:

```text
http_requests_total
ml_generations_total
ml_generation_latency_seconds_sum
ml_generation_latency_seconds_count
db_health_status
diffusion_model_loaded_status
```

## Где смотреть эксперименты MLflow

Открыть:

[http://localhost:5000](http://localhost:5000)

Что показать:

1. Experiment `synthetic-dataset-parameter-search-v2`
2. Runs:
   - `guided_320`
   - `fast_256`
   - `balanced_320`
   - `detailed_256`
   - `balanced_256`
3. Внутри run:
   - параметры генерации
   - dataset-level метрики
   - artifacts

Как объяснить:

```text
Эксперименты ЛР3 хранятся в MLflow: там логируются параметры генерации,
метрики датасета и артефакты сравнительных запусков.
```

## Где смотреть объекты в MinIO

Открыть:

[http://localhost:9001](http://localhost:9001)

Логин по умолчанию:

- login: `minioadmin`
- password: `minioadmin`

Что показать:

1. Bucket `synthetic-datasets`
2. Префиксы:
   - `online-generations/...`
   - `online-generations/jobs/...`
3. Внутри:
   - PNG-файлы
   - `manifest.csv`

Как объяснить:

```text
Сгенерированные изображения и manifest-файлы сохраняются не только локально,
но и в S3-compatible object storage MinIO. В PostgreSQL лежит metadata и URI,
а сами бинарные объекты лежат в MinIO.
```

## Что показывать на демонстрации

1. `docker compose up --build`
2. `docker compose ps`
3. Frontend на `localhost:3000`
4. Одиночную генерацию через `/generate`
5. Batch job через `/generation-jobs`
6. `manifest.csv`
7. `http://localhost:8000/metrics`
8. `http://localhost:9090`
9. `http://localhost:5000`
10. `http://localhost:9001`

## Полезные файлы проекта

- Основная документация: [README.md](D:/pet's%20proektsii/multimodel_diffusion_push/README.md)
- Отчёт по ЛР4: [LR4.md](D:/pet's%20proektsii/multimodel_diffusion_push/docs/LR4.md)
- Docker orchestration: [docker-compose.yml](D:/pet's%20proektsii/multimodel_diffusion_push/docker-compose.yml)
