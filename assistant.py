#!/usr/bin/env python3
"""
Local low-latency voice assistant
  Ears  : RealtimeSTT (faster-whisper on GPU + Silero VAD turn detection)
  Brain : Ollama (streaming tokens)
  Voice : RealtimeTTS + Kokoro (speaks each sentence as soon as it exists)
  Barge-in: start talking while it speaks -> it stops and listens.

Everything runs offline on your GPU. Tune via environment variables (see CONFIG).
"""
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime

import numpy as np
import requests
import torch
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import KokoroEngine, TextToAudioStream

from memory import Memory, worth_checking

# ---------------- CONFIG ----------------
# Override any of these without editing code, e.g.:
#   VOICE=am_michael LLM_MODEL=qwen2.5:7b SILENCE=0.6 ./run.sh
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")  # or another PC's IP
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")        # fast; try qwen2.5:7b for smarter
STT_MODEL = os.getenv("STT_MODEL", "distil-large-v3")    # accurate + fast on GPU; small.en = faster, less accurate
STT_COMPUTE = os.getenv("STT_COMPUTE", "float16")        # Whisper precision; int8 for GPUs without tensor cores
                                                         # (T1000: 1.7 s vs 7 s with float16), Pascal (Titan Xp)
WHISPER_GPU = int(os.getenv("WHISPER_GPU", "0"))         # GPU numbers as in `nvidia-smi -L` (run.sh sets PCI order)
KOKORO_GPU = int(os.getenv("KOKORO_GPU", "0"))           # e.g. 1 = put the voice on a second, smaller GPU
LANGUAGE = os.getenv("LANGUAGE", "en")                   # "" = auto-detect
VOICE = os.getenv("VOICE", "af_heart")                   # Kokoro voice: af_bella, am_michael, bf_emma...
                                                         # or a blend: "0.6*af_heart + 0.4*af_bella"
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))         # speaking speed; 0.9 = calmer, 1.1 = faster
SENTENCE_PAUSE = float(os.getenv("SENTENCE_PAUSE", "0.2"))  # seconds of breath between sentences; 0 = none
SILENCE = float(os.getenv("SILENCE", "0.45"))            # seconds of silence = end of turn
BARGE_IN = os.getenv("BARGE_IN", "1") == "1"             # talk over it to interrupt; set 0 when using speakers
MEMORY = os.getenv("MEMORY", "1") == "1"                 # long-term memory about you; 0 = off
MEMORY_FILE = os.getenv("MEMORY_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.md"))
LOG = os.getenv("LOG", "1") == "1"                       # save conversations to LOG_DIR (one file per day); 0 = off
LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations"))
WAKE_WORD = os.getenv("WAKE_WORD", "hey_jarvis")        # say it before talking; "" = off (always listening)
FOLLOW_UP = float(os.getenv("FOLLOW_UP", "8"))           # seconds after a reply to talk without the wake word; 0 = always need it
WAKE_SENSITIVITY = float(os.getenv("WAKE_SENSITIVITY", "0.5"))  # 0..1, higher = stricter (fewer false wake-ups)
# SYSTEM_PROMPT = the AI's instructions; edit the text below to change its personality.
# You can add: a name ("You are Nova..."), its character (funny, calm, formal),
# facts about you (your name, job, interests), how it should talk, and rules.
# Keep the "short sentences, no markdown" part: every reply is read aloud.
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a warm, helpful voice assistant. Your replies are spoken aloud, "
    "so answer conversationally in short sentences. No markdown, lists, emojis "
    "or code blocks. Keep most answers to 1-3 sentences unless asked for more.",
)
# temperature: creativity, 0.3 = focused/predictable, 1.0 = lively/random
# num_ctx: how many tokens of conversation the model remembers
LLM_OPTIONS = {"temperature": 0.7, "num_ctx": 4096}
# ----------------------------------------

memory = Memory(MEMORY_FILE) if MEMORY else None


def system_prompt():
    return SYSTEM_PROMPT + (memory.prompt() if memory else "")


def print_memory_status():
    if memory:
        print(f"🧠  Memory on: {len(memory.facts)} facts in {MEMORY_FILE}", flush=True)
    else:
        print("🧠  Memory off", flush=True)


def refresh_system_prompt():
    """Let the current conversation see newly remembered facts."""
    history[0]["content"] = system_prompt()


history = [{"role": "system", "content": system_prompt()}]
stop_llm = threading.Event()


_log_lock = threading.Lock()
_log_sessions = set()  # day files that already have this run's session header


def log_turn(source, user_text, reply, interrupted):
    """Append one exchange to LOG_DIR/YYYY-MM-DD.md (plain Markdown)."""
    if not LOG:
        return
    now = datetime.now()
    path = os.path.join(LOG_DIR, f"{now:%Y-%m-%d}.md")
    with _log_lock:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write(f"# Conversations {now:%Y-%m-%d}\n")
            if path not in _log_sessions:
                _log_sessions.add(path)
                f.write(f"\n## {now:%H:%M} new session ({LLM_MODEL})\n")
            typed = " (typed)" if source == "typed" else ""
            note = " *(interrupted)*" if interrupted else ""
            f.write(f"\n**{now:%H:%M} You{typed}:** {user_text}\n\n"
                    f"**{now:%H:%M} AI:** {reply.strip() or '…'}{note}\n")


def llm_stream(user_text, on_token=None, source="voice"):
    """Yield tokens from Ollama as they are generated."""
    history.append({"role": "user", "content": user_text})
    payload = {
        "model": LLM_MODEL,
        "messages": history,
        "stream": True,
        "keep_alive": -1,  # keep model in VRAM -> no reload delay
        "think": False,    # no silent "thinking" before answering (qwen3 etc.); ignored by other models
        "options": LLM_OPTIONS,
    }
    reply = ""
    interrupted = False
    stop_llm.clear()
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if stop_llm.is_set():
                interrupted = True
                break
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                reply += token
                print(token, end="", flush=True)
                if on_token:
                    on_token(token)
                yield token
            if chunk.get("done"):
                break
    print()
    history.append({"role": "assistant", "content": reply or "(interrupted)"})
    log_turn(source, user_text, reply, interrupted)
    if memory and worth_checking(user_text):  # skip questions and small talk
        memory.update_async(user_text, OLLAMA_URL, LLM_MODEL, LLM_OPTIONS,
                            on_change=refresh_system_prompt)


def warm_up_llm():
    requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "messages": [{"role": "user", "content": "hi"}],
              "stream": False, "keep_alive": -1, "think": False,
              "options": {**LLM_OPTIONS, "num_predict": 1}},
        timeout=300,
    )


def chime():
    """Short soft tone meaning "I'm listening" (played with aplay, silent if missing)."""
    rate = 24000
    t = np.arange(int(rate * 0.12)) / rate
    tone = (0.25 * np.sin(2 * np.pi * 880 * t) * np.hanning(len(t)) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(tone.tobytes())
    try:
        subprocess.Popen(["aplay", "-q", "-"], stdin=subprocess.PIPE,
                         stderr=subprocess.DEVNULL).communicate(buf.getvalue())
    except OSError:
        pass


def wake_word_options(wake_word, on_wake=None):
    """Extra recorder settings for wake word mode (openWakeWord, offline).
    on_wake runs once per wake-up; default plays the chime on this PC."""
    if not wake_word:
        return {}
    last = [0.0]

    def detected():
        # fires for every audio chunk while the wake word is heard; react once
        if time.time() - last[0] > 4:
            print("\n👂  Listening...", flush=True)
            threading.Thread(target=on_wake or chime, daemon=True).start()
        last[0] = time.time()

    return dict(
        wakeword_backend="oww",
        openwakeword_model_paths=wake_word,   # only this model, not all built-in ones
        wake_words=wake_word,
        wake_words_sensitivity=WAKE_SENSITIVITY,
        wake_word_activation_delay=0,          # wake word needed right away; main() opens the
                                               # FOLLOW_UP window only after a reply
        wake_word_timeout=5,                   # after the wake word, wait this long for speech
        on_wakeword_detected=detected,
    )


def make_recorder(interrupt, wake_word=WAKE_WORD, on_wake=None, **extra):
    """Ears. `extra` overrides defaults (web.py passes use_microphone=False)."""
    return AudioToTextRecorder(
        model=STT_MODEL,
        language=LANGUAGE,
        device="cuda",
        compute_type=STT_COMPUTE,
        gpu_device_index=WHISPER_GPU,
        spinner=False,
        silero_sensitivity=0.4,               # higher = hears quiet speech, but also noise
        silero_use_onnx=True,
        webrtc_sensitivity=3,                 # 0 (least strict) .. 3 (most strict)
        post_speech_silence_duration=SILENCE,
        min_length_of_recording=0.3,          # ignore sounds shorter than this (coughs, clicks)
        min_gap_between_recordings=0,
        beam_size=1,                          # 5 = slightly better transcription, slower
        early_transcription_on_silence=0.2,  # start transcribing before silence timer ends
        on_vad_start=interrupt,               # barge-in (fires on actual speech)
        on_recording_start=interrupt,
        **wake_word_options(wake_word, on_wake),
        **extra,
    )


# How replies are cut into pieces for Kokoro (shared by local and web mode)
PLAY_OPTIONS = dict(
    fast_sentence_fragment=True,       # speak the first clause ASAP
    # Text pieces sent to Kokoro (characters). Higher (30-40) = more
    # natural intonation, but it starts speaking a bit later.
    minimum_sentence_length=8,
    minimum_first_fragment_length=8,
    sentence_silence_duration=SENTENCE_PAUSE,
    comma_silence_duration=0.05,       # only when a piece ends at a comma (first clause)
    buffer_threshold_seconds=0.0,
    log_synthesized_text=False,
)


def make_voice():
    """Kokoro on KOKORO_GPU (it loads onto the current GPU; its work follows the model)."""
    with torch.cuda.device(KOKORO_GPU):
        return KokoroEngine(voice=VOICE, default_speed=TTS_SPEED)


def listen(recorder, inbox):
    """Put each spoken sentence into the inbox."""
    while True:
        text = recorder.text().strip()
        if text:
            inbox.put(("voice", text))


def read_keyboard(inbox):
    """Put each typed line into the inbox (ends quietly when there is no keyboard)."""
    for line in sys.stdin:
        if line.strip():
            inbox.put(("typed", line.strip()))


def set_wake_mode(recorder, on):
    """Switch between regular talking and wake word mode at runtime."""
    recorder.use_wake_words = bool(on and WAKE_WORD)
    recorder.wake_word_activation_delay = 0
    recorder.wakeword_detected = False
    return recorder.use_wake_words


COMMANDS_HELP = """Commands (type and press Enter):
  /wake        switch the wake word on/off
  /wake on     say the wake word before talking
  /wake off    regular talking: answers everything it hears
  /help        show this list"""


def main():
    print("Loading voice...", flush=True)
    engine = make_voice()
    tts = TextToAudioStream(engine, frames_per_buffer=256)

    def interrupt(force=False):
        # typed messages always interrupt (no risk of the mic hearing the speakers)
        if (BARGE_IN or force) and tts.is_playing():
            stop_llm.set()
            tts.stop()

    print("Loading ears...", flush=True)
    recorder = make_recorder(interrupt)

    wake_name = WAKE_WORD.replace("_", " ")

    def open_follow_up():
        # After a reply has finished playing, allow FOLLOW_UP seconds of talk
        # without the wake word (timed from now, not from when listening resumed)
        if recorder.use_wake_words and FOLLOW_UP > 0:
            recorder.wake_word_activation_delay = FOLLOW_UP
            recorder.listen_start = time.time()

    def close_follow_up():
        recorder.wake_word_activation_delay = 0
        if recorder.use_wake_words:
            print(f"\n💤  Say \"{wake_name}\" to talk", flush=True)

    def mode_line():
        if recorder.use_wake_words:
            return f"💤  Wake word on: say \"{wake_name}\", then talk"
        return "🎤  Wake word off: just talk"

    def command(text):
        """Typed lines starting with "/" are commands, not messages for the AI."""
        words = text.lower().split()
        if words[0] == "/wake":
            arg = words[1] if len(words) > 1 else ("off" if recorder.use_wake_words else "on")
            if arg not in ("on", "off"):
                print("Usage: /wake, /wake on or /wake off", flush=True)
            elif arg == "on" and not WAKE_WORD:
                print("The wake word is disabled (started with WAKE_WORD=). "
                      "Restart without it to use /wake.", flush=True)
            else:
                set_wake_mode(recorder, arg == "on")
                print(mode_line(), flush=True)
        elif words[0] in ("/help", "/?"):
            print(COMMANDS_HELP, flush=True)
        else:
            print(f"Unknown command {words[0]}. Type /help", flush=True)

    print(f"Loading brain ({LLM_MODEL})...", flush=True)
    warm_up_llm()
    print_memory_status()
    tts.feed("Ready.").play()
    if WAKE_WORD:  # hooked up after "Ready." so startup doesn't open a window
        tts.on_audio_stream_stop = open_follow_up
        recorder.on_wakeword_timeout = close_follow_up

    inbox = queue.Queue()
    threading.Thread(target=listen, args=(recorder, inbox), daemon=True).start()
    threading.Thread(target=read_keyboard, args=(inbox,), daemon=True).start()

    if WAKE_WORD:
        print(f"\n{mode_line()}, or type a message and press Enter.")
        print("    Type /wake to turn the wake word on/off, /help for all commands. "
              "Ctrl+C to quit.\n", flush=True)
    else:
        print("\n🎤  Speak any time, or type a message and press Enter.")
        print("    Type /help for commands. Ctrl+C to quit.\n", flush=True)
    try:
        while True:
            try:
                source, text = inbox.get(timeout=0.5)  # timeout keeps Ctrl+C responsive
            except queue.Empty:
                continue
            if source == "typed" and text.startswith("/"):
                command(text)
                continue
            interrupt(force=source == "typed")
            if source == "voice":
                print(f"\nYou: {text}")
            print("AI : ", end="", flush=True)
            tts.feed(llm_stream(text, source=source))
            tts.play_async(**PLAY_OPTIONS)
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        stop_llm.set()
        tts.stop()
        recorder.shutdown()


if __name__ == "__main__":
    sys.exit(main())
