#!/usr/bin/env python3
"""
Local low-latency voice assistant
  Ears  : RealtimeSTT (faster-whisper on GPU + Silero VAD turn detection)
  Brain : Ollama (streaming tokens)
  Voice : RealtimeTTS + Kokoro (speaks each sentence as soon as it exists)
  Barge-in: start talking while it speaks -> it stops and listens.

Everything runs offline on your GPU. Tune via environment variables (see CONFIG).
"""
import glob
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
LANGUAGE = os.getenv("STT_LANGUAGE", "en")               # what you speak; "" = auto-detect
                                                         # (not LANGUAGE: that's a Linux locale variable)
VOICE = os.getenv("VOICE", "af_heart")                   # Kokoro voice: af_bella, am_michael, bf_emma...
                                                         # or a blend: "0.6*af_heart + 0.4*af_bella"
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))         # speaking speed; 0.9 = calmer, 1.1 = faster
SENTENCE_PAUSE = float(os.getenv("SENTENCE_PAUSE", "0.2"))  # seconds of breath between sentences; 0 = none
SILENCE = float(os.getenv("SILENCE", "0.45"))            # seconds of silence = end of turn
BARGE_IN = os.getenv("BARGE_IN", "1") == "1"             # talk over it to interrupt; set 0 when using speakers
MEMORY = os.getenv("MEMORY", "1") == "1"                 # long-term memory about you; 0 = off
MEMORY_FILE = os.getenv("MEMORY_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory", "memory.md"))
LOG = os.getenv("LOG", "1") == "1"                       # save conversations to LOG_DIR (one file per day); 0 = off
# Print what you say, the replies and memory facts in the console. Off by default for the auto-start
# service (VA_SERVICE): its output goes to the system journal, so the conversation stays in LOG_DIR only.
PRINT_CHAT = os.getenv("PRINT_CHAT", "0" if os.getenv("VA_SERVICE") else "1") == "1"
LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations"))
WAKE_WORD = os.getenv("WAKE_WORD", "hey_jarvis")        # say it before talking; "" = off (always listening)
FOLLOW_UP = float(os.getenv("FOLLOW_UP", "8"))           # seconds after a reply to talk without the wake word; 0 = always need it
SLEEP_SOUND = os.getenv("SLEEP_SOUND", "1") == "1"       # soft falling tone when it stops listening (wake word needed again); 0 = silent
WAKE_SENSITIVITY = float(os.getenv("WAKE_SENSITIVITY", "0.5"))  # 0..1, higher = stricter (fewer false wake-ups)
PROFILE = os.getenv("PROFILE", "")                       # personality from profiles/<name>.md; "" = last one chosen
PROFILES_DIR = os.getenv("PROFILES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles"))
# Choices made while using it (active profile, save log, wake word), saved automatically; not in git
SETTINGS_FILE = os.getenv("SETTINGS_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json"))
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
BASE_MODEL, BASE_TEMPERATURE = LLM_MODEL, LLM_OPTIONS["temperature"]  # what profiles fall back to

memory = Memory(MEMORY_FILE, verbose=PRINT_CHAT) if MEMORY else None

# ---------------- PROFILES ----------------
# profiles/<name>.md: optional "key: value" lines, a blank line, then the prompt.
# Keys a profile leaves out fall back to the settings above. profiles/_template.md explains
# every key; files starting with "_" are not profiles.
def on_off(value):
    """"on"/"off" (also yes/no, true/false, 1/0) -> bool."""
    v = value.strip().lower()
    if v in ("on", "yes", "true", "1"):
        return True
    if v in ("off", "no", "false", "0"):
        return False
    raise ValueError(value)


PROFILE_KEYS = {"label": str, "voice": str, "speed": float, "pause": float,
                "temperature": float, "model": str, "think": on_off, "silence": float}


def list_profiles():
    if not os.path.isdir(PROFILES_DIR):
        return []
    return sorted(f[:-3] for f in os.listdir(PROFILES_DIR) if f.endswith(".md") and not f.startswith("_"))


def load_profile(name):
    """Settings + prompt of profiles/<name>.md (None if there is no such file)."""
    path = os.path.join(PROFILES_DIR, f"{name}.md")
    if not re.fullmatch(r"[\w-]+", name) or name.startswith("_") or not os.path.isfile(path):
        return _profile_defaults(name) if name == "default" else None
    return _read_profile(path, name)


def _profile_defaults(name):
    return {"name": name, "label": name.replace("_", " ").replace("-", " ").capitalize(),
            "prompt": SYSTEM_PROMPT, "voice": VOICE, "speed": TTS_SPEED, "pause": SENTENCE_PAUSE,
            "temperature": BASE_TEMPERATURE, "model": BASE_MODEL, "think": False, "silence": SILENCE}


def _read_profile(path, name):
    p = _profile_defaults(name)
    header, body = True, []
    for line in open(path, encoding="utf-8").read().splitlines():
        m = re.match(r"^(\w+):\s*(.+)$", line)
        if header and m and m.group(1) in PROFILE_KEYS:
            try:
                p[m.group(1)] = PROFILE_KEYS[m.group(1)](m.group(2).strip())
            except ValueError:  # e.g. "speed: fast", "think: maybe"
                print(f"⚠️  profiles/{name}.md: '{line}' is not a valid value, ignored", flush=True)
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
                  f"(ollama pull {p['model']}), using {BASE_MODEL}", flush=True)
            p["model"] = BASE_MODEL
    except (requests.RequestException, ValueError):
        pass  # Ollama not reachable; warm_up_llm will say so
    if p["think"] and not model_can_think(p["model"]):
        print(f"⚠️  Profile {p['name']}: {p['model']} can't think first, think is off", flush=True)
        p["think"] = False
    return p


_can_think = {}


def model_can_think(model):
    """Ollama reports "thinking" for models that can reason before answering (qwen3: yes)."""
    if model not in _can_think:
        try:
            r = requests.post(OLLAMA_URL.replace("/api/chat", "/api/show"), json={"model": model}, timeout=5)
            _can_think[model] = "thinking" in r.json().get("capabilities", [])
        except (requests.RequestException, ValueError):
            return False  # don't cache: Ollama may just not be up yet
    return _can_think[model]


# ---- profile editor (web page): voices/models to choose from, save, delete ----
VOICE_ACCENTS = {"a": "American", "b": "British", "e": "Spanish", "f": "French", "h": "Hindi",
                 "i": "Italian", "j": "Japanese", "p": "Portuguese", "z": "Chinese"}


def available_voices():
    """Kokoro voices on this PC as [{"name", "desc"}] (offline mode can only use downloaded ones)."""
    from huggingface_hub import constants
    files = glob.glob(os.path.join(constants.HF_HUB_CACHE, "models--hexgrad--Kokoro-82M",
                                   "snapshots", "*", "voices", "*.pt"))
    names = sorted({os.path.basename(f)[:-3] for f in files})
    return [{"name": n, "desc": f"{VOICE_ACCENTS.get(n[0], n[0])}, "
                                f"{ {'f': 'female', 'm': 'male'}.get(n[1:2], '?') }"} for n in names]


def installed_models():
    try:
        tags = requests.get(OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=5).json()
        return sorted(m["name"] for m in tags.get("models", []))
    except (requests.RequestException, ValueError):
        return [BASE_MODEL]


def profile_for_editor(name):
    """A profile as the editor shows it; model "" = the normal model (no "model:" line)."""
    if name:
        p = load_profile(name)
    else:  # "+ New profile" starts from the template; without one, from the built-in defaults
        template = os.path.join(PROFILES_DIR, "_template.md")
        p = _read_profile(template, "") if os.path.isfile(template) else _profile_defaults("")
    return {**p, "model": "" if p["model"] == BASE_MODEL else p["model"]}


NO_MODEL_LINE = "# model: (not set = LLM_MODEL from local.env)"
PROFILE_RANGES = {"speed": (0.5, 2.0), "pause": (0.0, 1.0), "temperature": (0.0, 1.5), "silence": (0.3, 3.0)}


def profile_path(name):
    """profiles/<name>.md, or None for a name that could point anywhere else ("../x") or isn't a profile."""
    if not re.fullmatch(r"[\w-]+", name or "") or name.startswith("_"):
        return None
    path = os.path.realpath(os.path.join(PROFILES_DIR, f"{name}.md"))
    return path if os.path.dirname(path) == os.path.realpath(PROFILES_DIR) else None


def write_profile(data, old_name=None):
    """Save profiles/<name>.md from the editor. Returns (name, error message or None)."""
    name = str(data.get("name", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name) or name.startswith("_"):
        return name, "Name: letters, digits, - and _ only (up to 40, not starting with _)"
    if old_name is not None:  # editing: it comes from the page, so check it as strictly as a new name
        old_path = profile_path(old_name)
        if not old_path or not os.path.isfile(old_path):
            return name, f"No profile {old_name}"
    if old_name == "default" and name != "default":
        return name, "The default profile can't be renamed"
    path = os.path.join(PROFILES_DIR, f"{name}.md")
    if name != old_name and os.path.exists(path):
        return name, f"A profile named {name} already exists"
    label = " ".join(str(data.get("label", "")).split())[:60] or name
    voice = str(data.get("voice", "")).strip()
    if not re.fullmatch(r"[a-z]{2}_[a-z]+", voice):
        return name, "Choose a voice"
    values = {}
    for key, (low, high) in PROFILE_RANGES.items():
        try:
            values[key] = float(data.get(key))
        except (TypeError, ValueError):
            return name, f"{key} must be a number"
        if not low <= values[key] <= high:
            return name, f"{key} must be between {low:g} and {high:g}"
    model = str(data.get("model", "")).strip()
    if model and model not in installed_models():
        return name, f"Model {model} is not installed"
    think = data.get("think") in (True, "on", "true", "1", 1)
    if think and not model_can_think(model or BASE_MODEL):
        return name, f"{model or BASE_MODEL} can't think first; choose a model that can (e.g. qwen3)"
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return name, "The prompt can't be empty"
    lines = [f"label: {label}", f"voice: {voice}", f"speed: {values['speed']:g}",
             f"pause: {values['pause']:g}", f"temperature: {values['temperature']:g}",
             f"silence: {values['silence']:g}", f"think: {'on' if think else 'off'}",
             # no model line = the normal model, so changing LLM_MODEL in local.env still works
             f"model: {model}" if model else NO_MODEL_LINE]
    os.makedirs(PROFILES_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n" + prompt[:8000] + "\n")
    if old_name and old_name != name:  # renamed (old_path was checked above)
        os.remove(old_path)
    return name, None


def delete_profile(name):
    """Returns an error message or None. The default profile stays (it's the fallback)."""
    if name == "default":
        return "The default profile can't be deleted"
    path = profile_path(name)
    if not path or not os.path.isfile(path):
        return f"No profile {name}"
    os.remove(path)
    return None


# ---- settings.json: {"profile", "save_log", "wake_word"}, written on every change, read at start ----
_settings_lock = threading.Lock()


def load_settings():
    s = {}
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            s = json.load(f)
    except FileNotFoundError:  # first start: take over profile.txt from older versions
        old = os.path.join(os.path.dirname(SETTINGS_FILE), "profile.txt")
        if os.path.isfile(old):
            s["profile"] = open(old, encoding="utf-8").read().strip()
            write_settings(s)
            os.remove(old)
    except (OSError, ValueError) as e:
        print(f"⚠️  {SETTINGS_FILE} unreadable ({e}), using defaults", flush=True)
    s = s if isinstance(s, dict) else {}
    defaults = {"profile": "default", "memory": True, "save_log": True, "wake_word": False}  # file order
    full = {k: s.get(k, v) for k, v in defaults.items()} | s  # every setting, in this order (+ unknown keys)
    if list(full) != list(s):  # missing or out of order: rewrite the file
        write_settings(full)
    return full


def write_settings(s):
    try:
        tmp = SETTINGS_FILE + ".tmp"  # write + rename: never a half-written file
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
            f.write("\n")
        os.replace(tmp, SETTINGS_FILE)
    except OSError as e:
        print(f"Could not save {SETTINGS_FILE}: {e}", flush=True)


def save_setting(key, value):
    with _settings_lock:
        settings[key] = value
        write_settings(settings)


settings = load_settings()


def saved_profile():
    return str(settings.get("profile", "")).strip()


def save_profile(name):
    save_setting("profile", name)


# PROFILE from the environment wins; otherwise the last one chosen; a deleted one falls back to default
PROFILE = PROFILE or saved_profile() or "default"
profile = check_profile(load_profile(PROFILE) or load_profile("default"))
LLM_MODEL = profile["model"]
LLM_OPTIONS["temperature"] = profile["temperature"]


# The menu's "Use memory" switch (console /memory on|off; saved): off = the AI neither sees nor learns
# facts, memory.md stays untouched. MEMORY=0 = no memory at all.
memory_on = settings.get("memory", True) is not False


MEMORY_NOTE = "Memory was just switched"  # start of the note set_memory_use adds to the conversation


def set_memory_use(on):
    global memory_on
    memory_on = bool(on)
    save_setting("memory", memory_on)
    history[0] = {"role": "system", "content": system_prompt()}  # takes effect mid-conversation
    # drop the note of an earlier switch (after "off" the facts must not stay in the conversation)
    history[1:] = [m for m in history[1:] if not m["content"].startswith(MEMORY_NOTE)]
    if len(history) > 1:  # otherwise the model sticks to what it said before ("I don't know your name")
        facts = "\n".join(f"- {f}" for f in memory.facts) or "(none yet)"
        history.append({"role": "system", "content": (
            MEMORY_NOTE + " on. What you know about the user now:\n" + facts +
            "\nUse this from now on, even if you said earlier that you didn't know something."
            if memory_on else
            MEMORY_NOTE + " off: from now on don't use or mention remembered facts about the "
            "user, unless they tell you again in this conversation.")})


def system_prompt():
    return profile["prompt"] + (memory.prompt() if memory and memory_on else "")


def print_memory_status():
    if memory and not memory_on:
        print(f"🧠  Memory switched off ({len(memory.facts)} facts kept in {MEMORY_FILE})", flush=True)
    elif memory:
        print(f"🧠  Memory on: {len(memory.facts)} facts in {MEMORY_FILE}", flush=True)
    else:
        print("🧠  Memory off", flush=True)


on_memory_change = None  # web.py: callback(added, removed, learned) to update the browser


def memory_changed(added=(), removed=(), learned=True):
    """Let the current conversation see the change at once (learned = found by the AI, not by hand)."""
    history[0]["content"] = system_prompt()
    if on_memory_change:
        on_memory_change(added, removed, learned)


def memory_add(fact):
    fact = memory.remember(fact)
    if fact:
        if PRINT_CHAT:
            print(f"🧠  remembered: {fact}", flush=True)
        memory_changed([fact], [], learned=False)
    return fact


def memory_remove(fact):
    if memory.forget(fact):
        if PRINT_CHAT:
            print(f"🧠  forgot: {fact}", flush=True)
        memory_changed([], [fact], learned=False)
        return True
    return False


def memory_edit(old, new):
    fact = memory.edit(old, new)
    if fact:
        merged = fact not in memory.facts  # it matched another fact, so the old one was just dropped
        if PRINT_CHAT:
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


# switched at runtime (console /log, the web menu's switch; saved): off = private, nothing saved. LOG=0 = always off
log_on = LOG and settings.get("save_log", True) is not False


def set_logging(on):
    """Private mode when off (remembered in settings.json). LOG=0 keeps it off whatever the switch says."""
    global log_on
    log_on = LOG and bool(on)
    save_setting("save_log", bool(on))
    _log_sessions.clear()  # when it's on again, the next entry starts a new session header


def log_turn(source, user_text, reply, interrupted):
    """Append one exchange to LOG_DIR/YYYY-MM-DD.md (plain Markdown)."""
    if not log_on:
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


def log_event(text):
    """A line like "*12:30 — switched to Chef*" in today's log, inside the current session."""
    if not log_on:
        return
    now = datetime.now()
    path = os.path.join(LOG_DIR, f"{now:%Y-%m-%d}.md")
    with _log_lock:
        if path in _log_sessions:  # no session yet: its header will name the profile anyway
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n*{now:%H:%M} — {text}*\n")


def llm_stream(user_text, on_token=None, source="voice"):
    """Yield tokens from Ollama as they are generated."""
    history.append({"role": "user", "content": user_text})
    payload = {
        "model": LLM_MODEL,
        "messages": history,
        "stream": True,
        "keep_alive": -1,  # keep model in VRAM -> no reload delay
        "think": thinking(),  # profile "think: on" = reason silently first (qwen3); otherwise answer at once
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
                if PRINT_CHAT:
                    print(token, end="", flush=True)
                if on_token:
                    on_token(token)
                yield token
            if chunk.get("done"):
                break
    if PRINT_CHAT:
        print()
    history.append({"role": "assistant", "content": reply or "(interrupted)"})
    log_turn(source, user_text, reply, interrupted)
    if memory and memory_on and worth_checking(user_text):  # skip questions and small talk
        memory.update_async(user_text, OLLAMA_URL, LLM_MODEL, LLM_OPTIONS,
                            on_change=memory_changed)


MARKDOWN_SYMBOLS = str.maketrans("", "", "*_`#~")


def speakable(tokens):
    """Drop markdown symbols before Kokoro (it reads "*" as "asterisk"); the chat keeps them."""
    for token in tokens:
        token = token.translate(MARKDOWN_SYMBOLS)
        if token:
            yield token


def thinking():
    """The current profile wants the model to think first, and the model can."""
    return profile["think"] and model_can_think(LLM_MODEL)


def warm_up_llm():
    requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "messages": [{"role": "user", "content": "hi"}],
              "stream": False, "keep_alive": -1, "think": False,
              "options": {**LLM_OPTIONS, "num_predict": 1}},
        timeout=300,
    )


SLEEP_NOTES = ((660, 0.1), (440, 0.14))  # falling = "stopped listening"; the wake chime is one high note


def chime(notes=((880, 0.12),), volume=0.25):
    """Short soft tone meaning "I'm listening" (played with aplay, silent if missing)."""
    rate = 24000
    parts = []
    for freq, secs in notes:
        t = np.arange(int(rate * secs)) / rate
        parts.append(volume * np.sin(2 * np.pi * freq * t) * np.hanning(len(t)))
    tone = (np.concatenate(parts) * 32767).astype(np.int16)
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


def wake_word_options(wake_word, on_wake=None, on_detect=None):
    """Extra recorder settings for wake word mode (openWakeWord, offline).
    on_wake runs once per wake-up; default plays the chime on this PC.
    on_detect (the interrupt) also runs once per wake-up: "Hey Jarvis" stops a reply at once."""
    if not wake_word:
        return {}
    last = [0.0]

    def detected():
        # fires for every audio chunk while the wake word is heard; react once
        if time.time() - last[0] > 4:
            print("\n👂  Listening...", flush=True)
            if on_detect:
                on_detect()
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
        post_speech_silence_duration=profile["silence"],  # profiles can change it (apply_profile)
        min_length_of_recording=0.3,          # ignore sounds shorter than this (coughs, clicks)
        min_gap_between_recordings=0,
        beam_size=1,                          # 5 = slightly better transcription, slower
        early_transcription_on_silence=0.2,  # start transcribing before silence timer ends
        on_vad_start=interrupt,               # barge-in (fires on actual speech)
        on_recording_start=interrupt,
        **wake_word_options(wake_word, on_wake, on_detect=interrupt),
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


def set_voice(engine, voice, speed):
    """A voice in another accent needs a new Kokoro pipeline; build it now on KOKORO_GPU
    (lazily it would land on the default GPU)."""
    with torch.cuda.device(KOKORO_GPU):
        engine.set_voice(voice)
        engine._get_pipeline(engine.current_lang)
    engine.set_speed(speed)


def set_profile_voice(engine):
    set_voice(engine, profile["voice"], profile["speed"])


def new_conversation():
    """Start over (memory kept), like a profile switch without switching."""
    history[:] = [{"role": "system", "content": system_prompt()}]
    _log_sessions.clear()  # next log entry starts a new session header


def apply_profile(name, engine=None, recorder=None):
    """Switch personality (or apply an edited one): settings and voice change, the conversation and
    memory stay. Only the instructions at the top of the conversation are replaced; /new starts over."""
    global profile, LLM_MODEL
    new = load_profile(name)
    if not new:
        return None
    check_profile(new)
    old_model, old_name = LLM_MODEL, profile["name"]
    profile, LLM_MODEL = new, new["model"]
    save_profile(name)  # start with it next time
    LLM_OPTIONS["temperature"] = new["temperature"]
    PLAY_OPTIONS["sentence_silence_duration"] = new["pause"]
    if engine:
        set_profile_voice(engine)
    if recorder:  # read live by RealtimeSTT
        recorder.post_speech_silence_duration = new["silence"]
    history[0] = {"role": "system", "content": system_prompt()}
    if new["name"] != old_name:
        log_event(f"switched to {new['label']} ({LLM_MODEL})")
    if LLM_MODEL != old_model:  # free the GPU for the new model, then load it
        try:
            requests.post(OLLAMA_URL.replace("/api/chat", "/api/generate"),
                          json={"model": old_model, "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass
        print(f"Loading {LLM_MODEL}...", flush=True)
        warm_up_llm()
    return new


# "Hey Jarvis, what time is it?" -> "what time is it?" (said during a reply, the wake word is transcribed too)
WAKE_PHRASE = re.compile(r"^\W*" + r"\W+".join(WAKE_WORD.split("_")) + r"\b\W*", re.I) if WAKE_WORD else None


def listen(recorder, inbox):
    """Put each spoken sentence into the inbox."""
    while True:
        text = recorder.text().strip()
        if WAKE_PHRASE and recorder.use_wake_words:
            text = WAKE_PHRASE.sub("", text).strip()  # only "Hey Jarvis" = just stop the reply
            text = text[:1].upper() + text[1:]
        if text:
            inbox.put(("voice", text))


def read_keyboard(inbox):
    """Put each typed line into the inbox (ends quietly when there is no keyboard)."""
    for line in sys.stdin:
        if line.strip():
            inbox.put(("typed", line.strip()))


def hold_window(recorder):
    """Wake word mode: while a reply plays, plain speech interrupts it (no wake word needed;
    the browser's echo cancellation squashes your voice, so the wake word is often missed then).
    The FOLLOW_UP window takes over when the reply ends or is interrupted. Not with BARGE_IN=0
    (speakers: it would hear itself)."""
    if recorder.use_wake_words and BARGE_IN:
        recorder.wake_word_activation_delay = 24 * 3600
        recorder.listen_start = time.time()


def set_wake_mode(recorder, on, save=True):
    """Switch between regular talking and wake word mode at runtime (remembered in settings.json)."""
    recorder.use_wake_words = bool(on and WAKE_WORD)
    recorder.wake_word_activation_delay = 0
    recorder.wakeword_detected = False
    if save and WAKE_WORD:
        save_setting("wake_word", recorder.use_wake_words)
    return recorder.use_wake_words


COMMANDS_HELP = """Commands (type and press Enter):
  /wake        switch the wake word on/off
  /wake on     say the wake word before talking
  /wake off    regular talking: answers everything it hears
  /profiles    list the profiles (personalities)
  /profile X   switch to profile X (the conversation continues; /new starts over)
  /new         start a new conversation (memory stays)
  /log off     private: stop saving the conversation log (/log on = save again)
  /memory      list what it remembers about you
  /memory off  the AI ignores memory and learns nothing (/memory on = use it again)
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
            open_follow_up()  # wake word mode: you have FOLLOW_UP s to talk, then it sleeps

    print("Loading ears...", flush=True)
    recorder = make_recorder(interrupt)
    set_wake_mode(recorder, settings.get("wake_word", False), save=False)  # as you left it

    wake_name = WAKE_WORD.replace("_", " ")

    def open_follow_up():
        # After a reply has finished playing, allow FOLLOW_UP seconds of talk
        # without the wake word (timed from now, not from when listening resumed)
        if recorder.use_wake_words and FOLLOW_UP > 0:
            recorder.wake_word_activation_delay = FOLLOW_UP
            recorder.listen_start = time.time()

    def close_follow_up():
        # Back to waiting for the wake word: after the follow-up window, or "Hey Jarvis" and then silence
        recorder.wake_word_activation_delay = 0
        if recorder.use_wake_words:
            print(f"\n💤  Say \"{wake_name}\" to talk", flush=True)
            if SLEEP_SOUND and not tts.is_playing():  # a long reply can outlast the window
                threading.Thread(target=chime, args=(SLEEP_NOTES, 0.15), daemon=True).start()

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
            elif apply_profile(words[1], engine, recorder=recorder):
                print(f"🎭  {profile['label']} (conversation continues; /new starts over)", flush=True)
                tts.feed(f"{profile['label']}.")
                tts.play_async(**PLAY_OPTIONS)
            else:
                print(f"No profile '{words[1]}'. Type /profiles", flush=True)
        elif words[0] == "/log":
            if len(words) > 1 and words[1] in ("on", "off"):
                set_logging(words[1] == "on")
            if not LOG:
                print("The conversation log is turned off for good (LOG=0).", flush=True)
            print("📝  Conversation log on" if log_on else "🔒  Private: the conversation log is off",
                  flush=True)
        elif words[0] == "/new":
            interrupt(force=True)
            new_conversation()
            print("✨  New conversation (memory kept)", flush=True)
        elif words[0] in ("/memory", "/remember", "/forget") and not memory:
            print("Memory is off (started with MEMORY=0).", flush=True)
        elif words[0] == "/memory" and len(words) > 1 and words[1] in ("on", "off"):
            set_memory_use(words[1] == "on")
            print("🧠  Memory on" if memory_on else "🧠  Memory off: not used, nothing learned", flush=True)
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
            if PRINT_CHAT:
                if source == "voice":
                    print(f"\nYou: {text}")
                print("AI : ", end="", flush=True)
            tts.feed(speakable(llm_stream(text, source=source)))
            hold_window(recorder)  # talk over the reply to interrupt it, even in wake word mode
            tts.play_async(**PLAY_OPTIONS)
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        stop_llm.set()
        tts.stop()
        recorder.shutdown()


if __name__ == "__main__":
    sys.exit(main())
