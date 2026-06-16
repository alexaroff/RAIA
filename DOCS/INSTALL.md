# INSTALL — Установка RAIA на macOS

> Инструкция проверена на MacBook Air M2, 8 ГБ ОЗУ, macOS Sonoma.
> На других конфигурациях возможны отличия — см. раздел "Известные проблемы".

---

## Требования

- macOS (проверено на M2)
- Python 3.9
- Homebrew
- FFmpeg
- ~3–4 ГБ свободного места (модель Whisper Turbo)

---

## Шаг 1 — Установка Homebrew и FFmpeg

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg
```

---

## Шаг 2 — Создание проекта и виртуального окружения

```bash
mkdir ~/RAIA && cd ~/RAIA
python3.9 -m venv ~/raia-env
source ~/raia-env/bin/activate
mkdir -p audio output
```

---

## Шаг 3 — Установка зависимостей

```bash
pip install --upgrade pip
pip install whisperx
pip install huggingface-hub==0.34.0
pip install gradio==4.43.0
```

---

## Шаг 4 — Скачивание модели Whisper Turbo

```bash
mkdir -p ~/whisper_models/turbo
curl -L "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/model.safetensors" \
  -o ~/whisper_models/turbo/model.safetensors
```

---

## Шаг 5 — Патч gradio-client (Python 3.9)

На Python 3.9 есть баг в gradio-client. Фиксим:

```bash
python3 - << 'EOF'
path = "/Users/YOUR_USERNAME/raia-env/lib/python3.9/site-packages/gradio_client/utils.py"
with open(path, "r") as f:
    lines = f.readlines()
guard = "    if not isinstance(schema, dict):\n        return \"any\"\n"
lines.insert(901, guard)
with open(path, "w") as f:
    f.writelines(lines)
print("Done")
EOF
```

> Замени `YOUR_USERNAME` на своё имя пользователя macOS.

---

## Шаг 6 — Создание app.py

Скопируй содержимое `app.py` из репозитория в папку `~/RAIA/`.

---

## Шаг 7 — Ярлык на рабочем столе

```bash
cat > ~/Desktop/RAIA.command << 'EOF'
#!/bin/bash
cd ~/RAIA
source ~/raia-env/bin/activate
python app.py
EOF

chmod +x ~/Desktop/RAIA.command
```

---

## Шаг 8 — Запуск

Дважды кликни по **RAIA.command** на рабочем столе.
Браузер откроется автоматически на `http://localhost:7860`.

---

## Известные проблемы

### Pyannote несовместим с PyTorch 2.8
**Симптом:** `_pickle.UnpicklingError`
**Решение:** использовать Silero VAD вместо Pyannote (`--vad_method silero`)

### Конфликт transformers и huggingface-hub
**Симптом:** `ImportError`
**Решение:**
```bash
pip install huggingface-hub==0.34.0
```

### Перегрев M2 при длинных файлах
**Симптом:** зависания, высокая температура
**Решение:**
```bash
export OMP_NUM_THREADS=2
```

### Модель не загружается через Hugging Face
**Симптом:** `ReadTimeoutError`
**Решение:** скачать модель вручную через `curl` (см. Шаг 4)

### localhost недоступен на macOS
**Симптом:** `ValueError: When localhost is not accessible`
**Решение:** в `app.launch()` добавить `server_name="0.0.0.0"`

### Баг gradio-client на Python 3.9
**Симптом:** `TypeError: argument of type 'bool' is not iterable`
**Решение:** патч из Шага 5
