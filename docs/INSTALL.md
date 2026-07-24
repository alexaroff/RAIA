# Установка RAIA Tool (Fully Local)

**Ветка:** `tool-local-v1`  
**Текущая фаза:** 1 — Локальная транскрипция

## Требования

- macOS с Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- FFmpeg
- Минимум 8 ГБ ОЗУ (16–32 ГБ рекомендуется)

## Установка

```bash
# 1. Клонировать и переключиться на ветку
git clone https://github.com/alexaroff/RAIA.git
cd RAIA
git checkout tool-local-v1

# 2. Системные зависимости
brew install ffmpeg

# 3. Python-окружение (рекомендуется uv)
# Вариант A: uv (быстрее)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd tool
uv venv
source .venv/bin/activate
uv pip install -e ".[vad]"   # или без [vad] если не нужен Silero

# Вариант B: обычный venv
cd tool
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# опционально:
pip install silero-vad torch torchaudio
```

## Быстрый тест транскрипции

```bash
# из корня репозитория или из tool/
python -m tool.transcription.cli path/to/interview.webm \
  --model turbo \
  --lang ru \
  -o ./output \
  -v
```

Модели скачиваются автоматически при первом запуске в `~/.cache/huggingface` / mlx cache.

## Модели по RAM

| Флаг `--model` | Примерный runtime RAM | Когда использовать |
|----------------|-----------------------|--------------------|
| `turbo` (default) | 1.6–2.5 ГБ | 8–16 ГБ, лучший баланс |
| `medium` | ~2–3 ГБ | Если turbo давит |
| `small` | <1.5 ГБ | Только для быстрых черновиков |
| `large` | 4–6+ ГБ | 16+ ГБ, максимальное качество |

## Что дальше

После проверки качества на реальных интервью:
- Диаризация (фаза 6.2)
- Локальный LLM (GigaChat 3 Lightning)
- Gradio UI
