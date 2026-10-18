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
import re
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
MEMORY_FILE = os.getenv("MEMORY_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory", "memory.md"))
LOG = os.getenv("LOG", "1") == "1"                       # save conversations to LOG_DIR (one file per day); 0 = off
LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations"))
WAKE_WORD = os.getenv("WAKE_WORD", "hey_jarvis")        # say it before talking; "" = off (always listening)
FOLLOW_UP = float(os.getenv("FOLLOW_UP", "8"))           # seconds after a reply to talk without the wake word; 0 = always need it
WAKE_SENSITIVITY = float(os.getenv("WAKE_SENSITIVITY", "0.5"))  # 0..1, higher = stricter (fewer false wake-ups)
PROFILE = os.getenv("PROFILE", "")                       # personality from profiles/<name>.md; "" = last one chosen
PROFILES_DIR = os.getenv("PROFILES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles"))
PROFILE_FILE = os.getenv("PROFILE_FILE", os.path.join(os.path.dirname(MEMORY_FILE), "profile.txt"))  # remembers the last profile
# SYSTEM_PROMPT = fallback instructions when there is no profile file (profiles/default.md has the same text).
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

# ---------------- PROFILES ----------------
# profiles/<name>.md: optional "key: value" lines, a blank line, then the prompt.
# Keys a profile leaves out fall back to the settings above. profiles/_template.md explains
# every key; files starting with "_" are not profiles.
PROFILE_KEYS = {"label": str, "voice": str, "speed": float, "pause": float,
                "temperature": float, "model": str}


def list_profiles():
    if not os.path.isdir(PROFILES_DIR):
        return []
    return sorted(f[:-3] for f in os.listdir(PROFILES_DIR) if f.endswith(".md") and not f.startswith("_"))


def load_profile(name):
    """Settings + prompt of profiles/<name>.md (None if there is no such file)."""
    p = {"name": name, "label": name.replace("_", " ").replace("-", " ").capitalize(),
         "prompt": SYSTEM_PROMPT, "voice": VOICE, "speed": TTS_SPEED, "pause": SENTENCE_PAUSE,
         "temperature": LLM_OPTIONS["temperature"], "model": LLM_MODEL}
    path = os.path.join(PROFILES_DIR, f"{name}.md")
    if not re.fullmatch(r"[\w-]+", name) or name.startswith("_") or not os.path.isfile(path):
        return p if name == "default" else None
    header, body = True, []
    for line in open(path, encoding="utf-8").read().splitlines():
        m = re.match(r"^(\w+):\s*(.+)$", line)
        if header and m and m.group(1) in PROFILE_KEYS:
            try:
                p[m.group(1)] = PROFILE_KEYS[m.group(1)](m.group(2).strip())
            except ValueError:  # e.g. "speed: fast"
                print(f"⚠️  profiles/{name}.md: '{line}' is not a number, ignored", flush=True)
        elif header and (not line.strip() or line.startswith("#")):
            continue
        else:
            header = False
            body.append(line)
    p["prompt"] = "\n".join(body).strip() or p["prompt"]
    return p


def check_profile(p):
    """Fall back to the normal voice/model if the profile names one that isn't there
    (Kokoro would only fail mid-reply, Ollama on the first question)."""
    from huggingface_hub import hf_hub_download
    try:  # offline: finds it in the cache; online: downloads it now
        for v in re.findall(r"[a-z]{2}_[a-z]+", p["voice"]) or [p["voice"]]:
            hf_hub_download("hexgrad/Kokoro-82M", f"voices/{v}.pt")
    except Exception:
        print(f"⚠️  Profile {p['name']}: voice '{p['voice']}' not found (not downloaded? "
              f"run once with HF_HUB_OFFLINE=0), using {VOICE}", flush=True)
        p["voice"] = VOICE
    try:
        tags = requests.get(OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=5).json()
        installed = {m["name"] for m in tags.get("models", [])}
        if installed and p["model"] not in installed:
            print(f"⚠️  Profile {p['name']}: model '{p['model']}' is not installed "
                  f"(ollama pull {p['model']}), using {LLM_MODEL}", flush=True)
            p["model"] = LLM_MODEL
    except (requests.RequestException, ValueError):
        pass  # Ollama not reachable; warm_up_llm will say so
    return p


def saved_profile():
    try:
        return open(PROFILE_FILE, encoding="utf-8").read().strip()
    except OSError:
        return ""


def save_profile(name):
    try:
        os.makedirs(os.path.dirname(PROFILE_FILE), exist_ok=True)
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            f.write(name + "\n")
    except OSError as e:
        print(f"Could not save the profile choice: {e}", flush=True)


# PROFILE from the environment wins; otherwise the last one chosen; a deleted one falls back to default
PROFILE = PROFILE or saved_profile() or "default"
profile = check_profile(load_profile(PROFILE) or load_profile("default"))
LLM_MODEL = profile["model"]
LLM_OPTIONS["temperature"] = profile["temperature"]


def system_prompt():
    return profile["prompt"] + (memory.prompt() if memory else "")


def print_memory_status():
    if memory:
        print(f"🧠  Memory on: {len(memory.facts)} facts in {MEMORY_FILE}", flush=True)
    else:
        print("🧠  Memory off", flush=True)


on_memory_change = None  # web.py: callback(added, removed, learned) to update the phone


def memory_changed(added=(), removed=(), learned=True):
    """Let the current conversation see the change at once (learned = found by the AI, not by hand)."""
    history[0]["content"] = system_prompt()
    if on_memory_change:
        on_memory_change(added, removed, learned)


def memory_add(fact):
    fact = memory.remember(fact)
    if fact:
        print(f"🧠  remembered: {fact}", flush=True)
        memory_changed([fact], [], learned=False)
    return fact


def memory_remove(fact):
    if memory.forget(fact):
        print(f"🧠  forgot: {fact}", flush=True)
        memory_changed([], [fact], learned=False)
        return True
    return False


def memory_edit(old, new):
    fact = memory.edit(old, new)
    if fact:
        merged = fact not in memory.facts  # it matched another fact, so the old one was just dropped
        print(f"🧠  forgot: {old} (same as another fact)" if merged else
              f"🧠  changed: {old} → {fact}", flush=True)
        memory_changed([] if merged else [fact], [old], learned=False)
    return fact


def memory_clear():
    removed = memory.forget_all()
    print(f"🧠  forgot everything ({len(removed)} facts)", flush=True)
    memory_changed([], removed, learned=False)
    return removed


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
                f.write(f"\n## {now:%H:%M} new session ({profile['label']}, {LLM_MODEL})\n")
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
                            on_change=memory_changed)


MARKDOWN_SYMBOLS = str.maketrans("", "", "*_`#~")


def speakable(tokens):
    """Drop markdown symbols before Kokoro (it reads "*" as "asterisk"); the chat keeps them."""
    for token in tokens:
        token = token.translate(MARKDOWN_SYMBOLS)
        if token:
            yield token


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
    sentence_silence_duration=profile["pause"],
    comma_silence_duration=0.05,       # only when a piece ends at a comma (first clause)
    buffer_threshold_seconds=0.0,
    log_synthesized_text=False,
)


def make_voice():
    """Kokoro on KOKORO_GPU (it loads onto the current GPU; its work follows the model)."""
    with torch.cuda.device(KOKORO_GPU):
        return KokoroEngine(voice=profile["voice"], default_speed=profile["speed"])


def set_profile_voice(engine):
    """Apply the profile's voice and speed. A voice in another accent needs a new Kokoro
    pipeline; build it now on KOKORO_GPU (lazily it would land on the default GPU)."""
    with torch.cuda.device(KOKORO_GPU):
        engine.set_voice(profile["voice"])
        engine._get_pipeline(engine.current_lang)
    engine.set_speed(profile["speed"])


def apply_profile(name, engine=None):
    """Switch personality: settings, voice, and a fresh conversation (memory stays shared)."""
    global profile, LLM_MODEL
    new = load_profile(name)
    if not new:
        return None
    check_profile(new)
    old_model = LLM_MODEL
    profile, LLM_MODEL = new, new["model"]
    save_profile(name)  # start with it next time
    LLM_OPTIONS["temperature"] = new["temperature"]
    PLAY_OPTIONS["sentence_silence_duration"] = new["pause"]
    if engine:
        set_profile_voice(engine)
    history[:] = [{"role": "system", "content": system_prompt()}]
    _log_sessions.clear()  # next log entry starts a new session header
    if LLM_MODEL != old_model:  # free the GPU for the new model, then load it
        try:
            requests.post(OLLAMA_URL.replace("/api/chat", "/api/generate"),
                          json={"model": old_model, "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass
        print(f"Loading {LLM_MODEL}...", flush=True)
        warm_up_llm()
    return new


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
  /profiles    list the profiles (personalities)
  /profile X   switch to profile X (fresh conversation, memory stays)
  /memory      list what it remembers about you
  /remember X  add a fact, e.g. /remember Has a cat named Luna
  /forget N    forget fact number N (see /memory); /forget all = everything
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
        elif words[0] == "/profiles":
            for name in list_profiles():
                mark = "▶" if name == profile["name"] else " "
                print(f" {mark} {name:12s} {load_profile(name)['label']}", flush=True)
        elif words[0] == "/profile":
            if len(words) < 2:
                print(f"Profile: {profile['label']}. Usage: /profile NAME (see /profiles)", flush=True)
            elif apply_profile(words[1], engine):
                print(f"🎭  {profile['label']}: fresh conversation, memory kept", flush=True)
                tts.feed(f"{profile['label']}.")
                tts.play_async(**PLAY_OPTIONS)
            else:
                print(f"No profile '{words[1]}'. Type /profiles", flush=True)
        elif words[0] in ("/memory", "/remember", "/forget") and not memory:
            print("Memory is off (started with MEMORY=0).", flush=True)
        elif words[0] == "/memory":
            if not memory.facts:
                print("🧠  Nothing remembered yet.", flush=True)
            for i, fact in enumerate(memory.facts, 1):
                print(f" {i:2d}. {fact}", flush=True)
        elif words[0] == "/remember":
            fact = text.split(maxsplit=1)[1].strip() if len(words) > 1 else ""
            if not fact:
                print("Usage: /remember FACT, e.g. /remember Has a cat named Luna", flush=True)
            elif not memory_add(fact):
                print("🧠  Already known.", flush=True)
        elif words[0] == "/forget":
            arg = words[1] if len(words) > 1 else ""
            if arg == "all":
                memory_clear()
            elif arg.isdigit() and 1 <= int(arg) <= len(memory.facts):
                memory_remove(memory.facts[int(arg) - 1])
            else:
                print("Usage: /forget N (number from /memory) or /forget all", flush=True)
        elif words[0] in ("/help", "/?"):
            print(COMMANDS_HELP, flush=True)
        else:
            print(f"Unknown command {words[0]}. Type /help", flush=True)

    print(f"Loading brain ({LLM_MODEL})...", flush=True)
    warm_up_llm()
    print_memory_status()
    print(f"🎭  Profile: {profile['label']}" + ("" if profile["name"] == PROFILE else
          f" (profiles/{PROFILE}.md is gone)"), flush=True)
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
            tts.feed(speakable(llm_stream(text, source=source)))
            tts.play_async(**PLAY_OPTIONS)
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        stop_llm.set()
        tts.stop()
        recorder.shutdown()


if __name__ == "__main__":
    sys.exit(main())
