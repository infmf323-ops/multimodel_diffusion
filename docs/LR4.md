# Лабораторная работа №4 — отчёт (краткая форма)

**ФИО:** _указать_  
**Группа:** _указать_

Подробное описание шагов, архитектуры, примеров curl и ответов на вопросы для самопроверки находится в корневом [`README.md`](../README.md) (секции «Лабораторная работа №4», «Быстрый запуск», «Примеры использования»).

## Результаты по шагам

| Шаг | Содержание | Где в проекте |
| --- | --- | --- |
| 1 | Инференс-модуль: `POST /generate`, `POST /predict`, асинхронные `POST /generation-jobs` | `inference_service/app/main.py`, `diffusion_runtime.py`, `model_runtime.py` |
| 2 | Интеграция: UI :3000, API :8000, worker, PostgreSQL, MinIO, MLflow | `docker-compose.yml`, `frontend/`, `generation_worker.py` |
| 3 | Мониторинг: логи запросов, `/metrics`, Prometheus | `main.py` (middleware, Prometheus), `monitoring/` |
| 4 | Контейнеризация | `Dockerfile` в сервисах, `docker compose up --build` |
| 5 | Демонстрация | `demo/sample_requests.json`, UI (`frontend/index.html`): превью + лимит строк `response_items_limit` / `items_limit`, полный батч в manifest |

## Вопросы для самопроверки

Ответы развёрнуто — в [`README.md`](../README.md) (конец секции ЛР4). Кратко:

- **Старт сервиса и загрузка модели:** caption-router грузится при старте; diffusion — лениво при первом `POST /generate`. Ускорение: HF cache volume, warmup, GPU.
- **Ошибка предсказания:** HTTP 500, запись в логах; для `/predict` — неверный label в истории.
- **Безопасность логов:** в демо пишутся prompt/caption; в production — маскирование чувствительных полей.
- **Метрики деградации:** рост latency, `ml_generations_total{status="error"}`, аномалии по классам `/predict`.
- **Тестирование интеграции:** smoke по `/health`, `/generate`, `/metrics`; полноценный test-suite — опционально.
- **10 000 пользователей:** узкое место — GPU-очередь diffusion, затем API/БД/storage без rate limiting и масштабирования воркеров.
