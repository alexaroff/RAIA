# RAIA Tool — Fully Local Edition

**Статус:** Фаза 1 — Локальная транскрипция (ветка `tool-local-v1`)

Полностью локальный AI-инструмент для рекрутера.  
Никаких облачных API. Соответствует 152-ФЗ.

---

## Фаза 1: Локальная транскрипция

### Стек

| Компонент | Решение |
|-----------|---------|
| STT | **mlx-whisper** (default) / GigaAM-v3 (skeleton) |
| VAD + Segment-wise | Silero → нарезка длинных файлов → меньше peak RAM |
| Auto model | `psutil` → выбирает turbo/medium/small/large по доступной ОЗУ |
| Config | `tool/config.toml` + `RAIA_*` env + CLI |

### Быстрый старт

```bash
brew install ffmpeg
cd tool
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[vad]"          # + [gigaam] если хочешь попробовать

# Авто-выбор модели по RAM
python -m tool.transcription.cli path/to/interview.webm -o ./output -v

# Явно
python -m tool.transcription.cli interview.webm --model turbo --backend mlx-whisper
python -m tool.transcription.cli interview.webm --force-segment-wise
```

Env-переменные:
```bash
export RAIA_MODEL=medium
export RAIA_BACKEND=mlx-whisper
export RAIA_LANGUAGE=ru
```

### Segment-wise

Если длительность ≥ 90 с и VAD нашёл несколько сегментов — каждый кусок транскрибируется отдельно.  
Пиковое потребление памяти падает, особенно на 8 ГБ.

Порог настраивается в `config.toml` (`segment_wise_threshold_s`).

### GigaAM

Backend `gigaam` скачивает community MLX-порт (`al-bo/gigaam-v3-rnnt-mlx`).  
Полный inference loop пока не встроен (ждём стабильный pure-MLX decoder) — при вызове будет понятная ошибка + путь к скачанным весам.  
Архитектура готова: как только появится рабочий loader — достаточно дописать `_load_gigaam_mlx`.

### Структура

```
tool/
├── config.toml
├── pyproject.toml
├── transcription/
│   ├── backends.py      ← MLXWhisper + GigaAM skeleton
│   ├── config.py        ← TOML + env + auto RAM
│   ├── pipeline.py      ← segment-wise + full
│   ├── vad.py
│   ├── preprocess.py
│   └── cli.py
```
