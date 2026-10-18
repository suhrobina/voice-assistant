#!/usr/bin/env python3
"""
Local low-latency voice assistant
  Ears  : RealtimeSTT (faster-whisper on GPU + Silero VAD turn detection)
  Brain : Ollama (streaming tokens)
  Voice : RealtimeTTS + Kokoro (speaks each sentence as soon as it exists)
  Barge-in: start talking while it speaks -> it stops and listens.

Everything runs offline on your GPU. Tune via environment variables (see CONFIG).
"""
import json
import os
import sys
import threading

import requests
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import KokoroEngine, TextToAudioStream

from memory import Memory, worth_checking

# ---------------- CONFIG ----------------
# Override any of these without editing code, e.g.:
#   VOICE=am_michael LLM_MODEL=qwen2.5:7b SILENCE=0.6 ./run.sh
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")  # or another PC's IP
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")        # fast; try qwen2.5:7b for smarter
STT_MODEL = os.getenv("STT_MODEL", "distil-large-v3")    # accurate + fast on GPU; small.en = faster, less accurate
LANGUAGE = os.getenv("LANGUAGE", "en")                   # "" = auto-detect
VOICE = os.getenv("VOICE", "af_heart")                   # Kokoro voice: af_bella, am_michael, bf_emma...
                                                         # or a blend: "0.6*af_heart + 0.4*af_bella"
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))         # speaking speed; 0.9 = calmer, 1.1 = faster
SENTENCE_PAUSE = float(os.getenv("SENTENCE_PAUSE", "0.2"))  # seconds of breath between sentences; 0 = none
SILENCE = float(os.getenv("SILENCE", "0.45"))            # seconds of silence = end of turn
BARGE_IN = os.getenv("BARGE_IN", "1") == "1"             # talk over it to interrupt; set 0 when using speakers
MEMORY = os.getenv("MEMORY", "1") == "1"                 # long-term memory about you; 0 = off
MEMORY_FILE = os.getenv("MEMORY_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.md"))
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


def llm_stream(user_text, on_token=None):
    """Yield tokens from Ollama as they are generated."""
    history.append({"role": "user", "content": user_text})
    payload = {
        "model": LLM_MODEL,
        "messages": history,
        "stream": True,
        "keep_alive": -1,  # keep model in VRAM -> no reload delay
        "options": LLM_OPTIONS,
    }
    reply = ""
    stop_llm.clear()
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if stop_llm.is_set():
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
    if memory and worth_checking(user_text):  # skip questions and small talk
        memory.update_async(user_text, OLLAMA_URL, LLM_MODEL, LLM_OPTIONS,
                            on_change=refresh_system_prompt)


def warm_up_llm():
    requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "messages": [{"role": "user", "content": "hi"}],
              "stream": False, "keep_alive": -1, "options": {**LLM_OPTIONS, "num_predict": 1}},
        timeout=300,
    )


def make_recorder(interrupt, **extra):
    """Ears. `extra` overrides defaults (web.py passes use_microphone=False)."""
    return AudioToTextRecorder(
        model=STT_MODEL,
        language=LANGUAGE,
        device="cuda",
        compute_type="float16",               # int8 for old GPUs (e.g. Titan Xp / Pascal)
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


def main():
    print("Loading voice...", flush=True)
    engine = KokoroEngine(voice=VOICE, default_speed=TTS_SPEED)
    tts = TextToAudioStream(engine, frames_per_buffer=256)

    def interrupt():
        if BARGE_IN and tts.is_playing():
            stop_llm.set()
            tts.stop()

    print("Loading ears...", flush=True)
    recorder = make_recorder(interrupt)

    print(f"Loading brain ({LLM_MODEL})...", flush=True)
    warm_up_llm()
    print_memory_status()
    tts.feed("Ready.").play()

    print("\n🎤  Speak any time. Ctrl+C to quit.\n")
    try:
        while True:
            text = recorder.text().strip()
            if not text:
                continue
            interrupt()
            print(f"\nYou: {text}\nAI : ", end="", flush=True)
            tts.feed(llm_stream(text))
            tts.play_async(**PLAY_OPTIONS)
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        stop_llm.set()
        tts.stop()
        recorder.shutdown()


if __name__ == "__main__":
    sys.exit(main())
