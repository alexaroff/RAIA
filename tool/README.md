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

## Фаза 1: Локальная транскрипция (текущая)

### Выбранный стек (2026, Apple Silicon, 8 ГБ)

| Компонент       | Решение                          | Почему |
|-----------------|----------------------------------|--------|
| STT             | **mlx-whisper** + `large-v3-turbo` | Лучший баланс скорость/качество/память на M-series. Native MLX, unified memory. |
| VAD (опционально) | Silero VAD                      | Лёгкий, надёжный, уже проверен в предыдущих экспериментах |
| Preprocess      | FFmpeg → 16 kHz mono WAV        | Стабильно, минимальный overhead |
| Альтернатива RU | GigaAM-v3 (MLX community port)  | SOTA по русскому, меньше модель (~420 МБ). Планируется как backend №2 |

**Почему не WhisperX / faster-whisper / pyannote:**
- WhisperX + PyTorch давал конфликты библиотек, высокий peak memory и перегрев на M2 8 ГБ.
- faster-whisper (CTranslate2) слабее использует Metal/ANE.
- mlx-whisper специально сделан под Apple Silicon и unified memory.

**Модели по RAM:**

- `turbo` (default) — ~1.6–2.3 ГБ runtime → комфортно на 8 ГБ
- `medium` — ещё безопаснее
- `large` — лучше качество, но комфортнее с 16+ ГБ

### Быстрый старт (macOS Apple Silicon)

```bash
# 1. Системные зависимости
brew install ffmpeg

# 2. Python-окружение (рекомендуем uv или venv)
cd tool
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install mlx mlx-whisper soundfile numpy rich

# (опционально, для VAD)
pip install silero-vad torch torchaudio

# 3. Запуск
python -m tool.transcription.cli path/to/interview.webm \
  --model turbo \
  --lang ru \
  -o ./output \
  -v
```

Или из кода:

```python
from tool.transcription import transcribe_file

result = transcribe_file(
    "interview.webm",
    model="turbo",
    language="ru",
    output_dir="output/"
)
print(result.text)
```

### Структура

```
tool/
├── README.md
├── requirements.txt
├── transcription/
│   ├── __init__.py
│   ├── pipeline.py      ← основной класс LocalTranscriber
│   ├── preprocess.py
│   ├── vad.py
│   └── cli.py
├── diarization/         ← фаза 2
├── analysis/            ← фаза 3–4
└── ui/                  ← фаза 5
```

### Долгосрочные заметки

- Абстракция backend уже заложена: можно добавить `GigaAMBackend` без ломки API.
- На 16–32 ГБ можно спокойно брать `large-v3` + полноценную диаризацию.
- Перегрев контролируется через `OMP_NUM_THREADS=2` (уже в коде).
- Следующий шаг после стабилизации: настоящая диаризация (pyannote/MLX или WeSpeaker + clustering) и локальный LLM.

---

**Важно:** код пока в активной разработке. Не используй с реальными кандидатами до тестов на своих файлах и проверки качества русского.
