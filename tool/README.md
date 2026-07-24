# RAIA Tool — Fully Local Edition

**Статус:** Фаза 1 — Локальная транскрипция (ветка `tool-local-v1`)

Полностью локальный AI-инструмент для рекрутера.  
Никаких облачных API, никаких токенов, никаких отправок данных наружу.  
Соответствует 152-ФЗ.

---

## Цели

1. Транскрипция интервью (русский язык) — локально
2. Диаризация (кто говорит) — локально
3. HR-анализ и генерация структурированного отчёта — локально (GigaChat 3 Lightning / открытые модели)
4. Удобный локальный интерфейс

---

## Фаза 1: Локальная транскрипция

### Выбранный стек (2026, Apple Silicon, 8 ГБ)

| Компонент | Решение | Почему |
|-----------|---------|--------|
| STT | **mlx-whisper** + `large-v3-turbo` | Лучший баланс скорость/качество/память на M-series. Native MLX. |
| VAD | Silero (опционально) | Лёгкий, уже проверен |
| Preprocess | FFmpeg → 16 kHz mono | Стабильно, почти нулевой overhead |
| Абстракция | `TranscriptionBackend` | Готов к GigaAM-v3 / whisper.cpp без ломки API |

**Почему не WhisperX:** конфликты библиотек + высокий peak memory + перегрев на M2 8 ГБ.

### Модели по RAM

- `turbo` (default) — ~1.6–2.5 ГБ → комфортно на 8 ГБ
- `medium` — безопаснее
- `large` — лучше качество, комфортнее с 16+ ГБ

### Быстрый старт

```bash
brew install ffmpeg
cd tool
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[vad]"          # или без [vad]

python -m tool.transcription.cli path/to/interview.webm \
  --model turbo --lang ru -o ./output -v
```

Или из кода:

```python
from tool.transcription import transcribe_file

result = transcribe_file("interview.webm", model="turbo", language="ru")
print(result.text)
result.save_json("out.json")
```

### Структура

```
tool/
├── README.md
├── pyproject.toml          ← современная упаковка (uv-ready)
├── requirements.txt        ← временный, для справки
├── .gitignore
├── transcription/
│   ├── __init__.py
│   ├── backends.py         ← абстракция (MLXWhisper + заготовка GigaAM)
│   ├── pipeline.py         ← LocalTranscriber
│   ├── preprocess.py
│   ├── vad.py
│   └── cli.py
├── diarization/            ← фаза 6.2
├── analysis/               ← фаза 6.3–6.4
└── ui/                     ← фаза 6.5
```

### Что уже сделано без тестов

- Backend abstraction (легко добавить GigaAM)
- VAD пишет speech ratio + segments в результат
- Жёсткое ограничение потоков (против перегрева)
- pyproject.toml + .gitignore
- Документация обновлена

### Следующие улучшения (после первых реальных прогонов)

1. Segment-wise transcription по VAD (если turbo будет давить на длинных файлах)
2. Реальный GigaAMBackend
3. Автовыбор модели по доступной RAM
4. Диаризация

---

**Важно:** код в активной разработке. Не используй с реальными кандидатами до проверки качества на своих файлах.
