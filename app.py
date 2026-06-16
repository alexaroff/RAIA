import gradio as gr
import subprocess
import os

RAIA_DIR = os.path.expanduser("~/RAIA")
AUDIO_DIR = os.path.join(RAIA_DIR, "audio")
OUTPUT_DIR = os.path.join(RAIA_DIR, "output")
WHISPER_MODEL = os.path.expanduser("~/whisper_models/turbo")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_transcription(audio_file):
    if audio_file is None:
        return "⚠️ Загрузите аудиофайл"

    flac_path = os.path.join(AUDIO_DIR, "interview.flac")
    txt_path = os.path.join(OUTPUT_DIR, "interview.txt")

    # FFmpeg: конвертация в flac
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", audio_file,
        "-ac", "1", "-ar", "16000", "-c:a", "flac",
        flac_path
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"❌ Ошибка FFmpeg:\n{result.stderr}"

    # WhisperX транскрипция
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    whisper_cmd = [
        "whisperx", flac_path,
        "--model", WHISPER_MODEL,
        "--language", "ru",
        "--compute_type", "int8",
        "--output_dir", OUTPUT_DIR,
        "--batch_size", "4",
        "--vad_method", "silero",
        "--output_format", "txt"
    ]
    result = subprocess.run(whisper_cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return f"❌ Ошибка WhisperX:\n{result.stderr}"

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    return "⚠️ Файл транскрипта не найден"

with gr.Blocks(theme=gr.themes.Base(), title="RAIA") as app:
    gr.Markdown("# RAIA — Recruitment AI Assistant")

    with gr.Tab("🎙 Транскрипция"):
        audio_input = gr.Audio(type="filepath", label="Аудио интервью")
        run_btn = gr.Button("▶ Запустить транскрипцию", variant="primary")
        transcript_output = gr.Textbox(
            label="Результат",
            lines=20,
            interactive=True,
            placeholder="Транскрипт появится здесь..."
        )
        run_btn.click(fn=run_transcription, inputs=audio_input, outputs=transcript_output)

    with gr.Tab("🏷 Разметка ролей"):
        gr.Markdown("*Будет доступно в v0.4*")

    with gr.Tab("📋 HR-анализ"):
        gr.Markdown("*Будет доступно в v0.5*")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
