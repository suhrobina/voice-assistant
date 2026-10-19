#!/usr/bin/env python3
"""
Web mode for the voice assistant: use it from a browser on any device (phone, tablet, PC).
  Same ears, brain and voice as assistant.py, but the audio comes from and
  goes to a web page (web/index.html) instead of the local mic and speakers.

  Browser --mic, 16 kHz PCM-->  server: Whisper -> Ollama -> Kokoro
  Browser <--voice, 24 kHz PCM--

Browsers allow the microphone only over HTTPS, so a self-signed certificate
is created on first start (accept the warning once in each browser).
Start with:  ./run.sh --web
"""
import asyncio
import hashlib
import hmac
import json
import os
import queue
import secrets
import socket
import ssl
import subprocess
import threading
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import unquote

from RealtimeTTS import TextToAudioStream
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from assistant import (BARGE_IN, FOLLOW_UP, PRINT_CHAT, SLEEP_SOUND, LLM_MODEL, PLAY_OPTIONS, make_voice,
                       WAKE_WORD, listen, llm_stream, make_recorder, print_memory_status,
                       stop_llm, warm_up_llm)
from assistant import hold_window, set_wake_mode as set_recorder_wake_mode
from assistant import (BASE_MODEL, PROFILE_RANGES, apply_profile, available_voices, delete_profile,
                       installed_models, list_profiles, model_can_think, profile_for_editor,
                       set_profile_voice, thinking,
                       set_voice, write_profile)
from assistant import memory, memory_add, memory_clear, memory_edit, memory_remove, speakable
import assistant  # for the current profile (assistant.profile changes on switch)

# ---------------- CONFIG ----------------
HOST = os.getenv("HOST", "0.0.0.0")            # 0.0.0.0 = reachable from your network
PORT = int(os.getenv("PORT", "8443"))
HTTPS = os.getenv("HTTPS", "1") == "1"         # 0 = plain HTTP (e.g. behind `tailscale serve`)
SSL_CERT = os.getenv("SSL_CERT", "certs/cert.pem")  # created automatically if missing
SSL_KEY = os.getenv("SSL_KEY", "certs/key.pem")
PASSWORD = os.getenv("WEB_PASSWORD", "")       # page password (set it in local.env); "" = anyone on the Wi-Fi
# ----------------------------------------

HERE = Path(__file__).parent
INDEX_HTML = (HERE / "web" / "index.html").read_bytes()

tts = None
voice = None  # the Kokoro engine inside tts (voice/speed change with the profile)
recorder = None


class Client:
    """The one connected browser (a new connection replaces the old one)."""
    ws = None
    queue = None
    loop = None
    muted = False  # set by the page: True = text-only replies (talking is off)
    wake_mode = False  # set by the page's switch: require the wake word
    replying = False   # a spoken reply is playing in the browser (until "played" or an interrupt)


inbox = queue.Queue()  # ("voice" | "typed", text) from the mic and the text box


def send(msg):
    """Thread-safe send to the browser: bytes = audio, dict = JSON event."""
    ws, queue, loop = Client.ws, Client.queue, Client.loop
    if ws is None:
        return
    data = msg if isinstance(msg, bytes) else json.dumps(msg)
    loop.call_soon_threadsafe(queue.put_nowait, data)


def interrupt(force=False):
    """Stop the current reply on the server and in the browser."""
    if not (BARGE_IN or force):
        return
    stop_llm.set()
    if tts.is_playing():
        tts.stop()
    # The browser may still be playing buffered audio even if the server is done
    send({"type": "stop"})
    if Client.replying:  # wake word mode: you have FOLLOW_UP s to talk, then it sleeps
        Client.replying = False
        open_follow_up()


def wake_state(state):
    """Tell the page: "listening" (wake word heard), "window" (follow-up), "sleep"."""
    send({"type": "wake", "state": state})


def set_wake_mode(on):
    """The page's switch: regular talking (off) or "Hey Jarvis" first (on)."""
    Client.wake_mode = set_recorder_wake_mode(recorder, on)
    print(f"\n🌐  Wake word {'on' if Client.wake_mode else 'off'}", flush=True)


def open_follow_up():
    """The browser finished playing a reply: talk without the wake word for FOLLOW_UP s."""
    if Client.wake_mode and FOLLOW_UP > 0:
        recorder.wake_word_activation_delay = FOLLOW_UP
        recorder.listen_start = time.time()
        wake_state("window")


def close_follow_up():
    recorder.wake_word_activation_delay = 0
    wake_state("sleep")


def memory_event(added=(), removed=(), learned=False):
    """Send the fact list to the page's memory panel; learned = the AI found it (chat note)."""
    send({"type": "memory", "facts": list(memory.facts), "added": list(added),
          "removed": list(removed), "learned": learned})


def profiles_event():
    """Everything the page's profile panel and editor need."""
    return {"type": "profiles", "current": assistant.profile["name"],
            "list": sorted((profile_for_editor(n) for n in list_profiles()), key=lambda p: p["label"].lower()),
            "template": profile_for_editor(""), "voices": available_voices(),
            "models": (models := installed_models()), "default_model": BASE_MODEL, "ranges": PROFILE_RANGES,
            "thinking_models": [m for m in models if model_can_think(m)]}


def preview(data):
    """▶ Test voice in the editor: two sample sentences with the chosen voice, speed and pause (not saved)."""
    name, speed, pause = str(data.get("voice", "")), data.get("speed", 1.0), data.get("pause", 0.2)
    if name not in {v["name"] for v in available_voices()}:
        return send({"type": "profile_error", "error": "Choose a voice first"})
    try:
        speed, pause = min(max(float(speed), 0.5), 2.0), min(max(float(pause), 0.0), 1.0)
    except (TypeError, ValueError):
        speed, pause = 1.0, 0.2
    label = " ".join(str(data.get("label", "")).split())[:40] or "your assistant"
    send({"type": "preview"})  # the page plays this even while talking is off
    set_voice(voice, name, speed)
    tts.feed(f"Hi! I'm {label}. This is how I sound.")
    tts.play(**{**PLAY_OPTIONS, "sentence_silence_duration": pause},  # faster than real time, so short
             muted=True, on_audio_chunk=send)
    set_profile_voice(voice)  # back to the active profile's voice
    send({"type": "preview_end"})


def drain(tokens):
    for _ in tokens:
        pass


def conversation():
    """Same loop as assistant.py, but replies are streamed to the browser."""
    text_reply = None  # thread generating a text-only reply (when muted)
    while True:
        source, text = inbox.get()
        interrupt(force=True)
        if text_reply:
            text_reply.join()
        if source == "profile":  # the page's profile menu
            if apply_profile(text, voice, recorder=recorder):
                p = assistant.profile
                print(f"\n🎭  {p['label']}: fresh conversation, memory kept", flush=True)
                send({"type": "profile", "name": p["name"], "label": p["label"]})
                send(profiles_event())
            continue
        if source == "profile_reload":  # the active profile was edited: new settings, same conversation
            if apply_profile(text, voice, fresh=False, recorder=recorder):
                print(f"\n🎭  {assistant.profile['label']}: updated", flush=True)
                send(profiles_event())
            continue
        if source == "preview":
            preview(json.loads(text))
            continue
        if PRINT_CHAT:
            print(f"\nYou{' (typed)' if source == 'typed' else ''}: {text}\nAI : ", end="", flush=True)
        else:  # service: only that a turn happened, not what was said
            print(f"💬  {source} message", flush=True)
        send({"type": "user", "text": text})
        send({"type": "reply"})
        if thinking():  # "think: on": a few silent seconds before the first word
            send({"type": "thinking"})
        tokens = llm_stream(text, on_token=lambda t: send({"type": "token", "text": t}),
                            source=source)
        if Client.muted:
            text_reply = threading.Thread(target=drain, args=(tokens,), daemon=True)
            text_reply.start()
        else:
            tts.feed(speakable(tokens))
            Client.replying = True
            hold_window(recorder)  # talk over the reply to interrupt it, even in wake word mode
            tts.play_async(**PLAY_OPTIONS, muted=True, on_audio_chunk=send)


async def pump(ws, queue):
    """Send queued messages in order."""
    while True:
        await ws.send(await queue.get())


async def handler(ws):
    if Client.ws is not None:
        await Client.ws.close()
    queue = asyncio.Queue()
    Client.ws, Client.queue, Client.loop = ws, queue, asyncio.get_running_loop()
    print("\n🌐  Device connected", flush=True)
    send({"type": "hello", "wake": WAKE_WORD.replace("_", " "), "sleep_sound": SLEEP_SOUND,
          "memory": memory is not None, "password": bool(PASSWORD)})
    send(profiles_event())
    sender = asyncio.create_task(pump(ws, queue))
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                recorder.feed_audio(msg, original_sample_rate=16000)
                continue
            try:
                m = json.loads(msg)
            except ValueError:
                continue
            if m.get("type") == "text" and str(m.get("text", "")).strip():
                inbox.put(("typed", str(m["text"]).strip()))
            elif m.get("type") == "mute":
                Client.muted = bool(m.get("value"))
            elif m.get("type") == "profile":
                inbox.put(("profile", str(m.get("name", ""))))
            elif m.get("type") == "profile_save":  # the editor: new, edited or renamed profile
                old = str(m.get("old") or "") or None
                name, error = write_profile(m.get("data") or {}, old)
                if error:
                    send({"type": "profile_error", "error": error})
                    continue
                print(f"\n🎭  Profile {name} saved", flush=True)
                if assistant.profile["name"] in (old, name):  # the active one: apply, keep the conversation
                    inbox.put(("profile_reload", name))
                send({"type": "profile_saved", "name": name})
                send(profiles_event())
            elif m.get("type") == "profile_delete":
                name = str(m.get("name", ""))
                error = delete_profile(name)
                if error:
                    send({"type": "profile_error", "error": error})
                    continue
                print(f"\n🎭  Profile {name} deleted", flush=True)
                if name == assistant.profile["name"]:
                    inbox.put(("profile", "default"))
                send(profiles_event())
            elif m.get("type") == "profile_preview":
                inbox.put(("preview", json.dumps(m.get("data") or {})))
            elif m.get("type") == "wake_mode":
                set_wake_mode(m.get("value"))
            elif m.get("type") == "played":  # the browser finished speaking the reply
                Client.replying = False
                open_follow_up()
            elif str(m.get("type", "")).startswith("memory") and memory:  # the page's memory panel
                fact = str(m.get("fact", "")).strip()
                if m["type"] == "memory_add" and fact:
                    memory_add(fact) or memory_event()  # already known: just resend the list
                elif m["type"] == "memory_remove":
                    memory_remove(fact) or memory_event()
                elif m["type"] == "memory_edit":  # emptied = delete
                    old = str(m.get("old", ""))
                    (memory_edit(old, fact) if fact else memory_remove(old)) or memory_event()
                elif m["type"] == "memory_clear":
                    memory_clear()
                else:
                    memory_event()
    finally:
        sender.cancel()
        if Client.ws is ws:
            Client.ws = None
            print("\n🌐  Device disconnected", flush=True)


# ---- password: the page asks once, then a signed cookie remembers the browser for a year ----
COOKIE = "va_session"
COOKIE_AGE = 365 * 24 * 3600
MAX_TRIES, LOCK_SECONDS = 5, 30   # wrong passwords per device before it has to wait
_failed = {}                      # ip -> (wrong tries, locked until)


def session_token():
    """The cookie value: HMAC of the password with a key kept in certs/session.key.
    Survives restarts; changing WEB_PASSWORD logs out every device."""
    key_file = HERE / "certs" / "session.key"
    if not key_file.exists():
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(secrets.token_bytes(32))
        key_file.chmod(0o600)
    return hmac.new(key_file.read_bytes(), PASSWORD.encode(), hashlib.sha256).hexdigest()


TOKEN = session_token() if PASSWORD else ""


def logged_in(request):
    if not PASSWORD:
        return True
    cookie = SimpleCookie(request.headers.get("Cookie", "")).get(COOKIE)
    return bool(cookie) and hmac.compare_digest(cookie.value, TOKEN)


def cookie_response(connection, value, max_age):
    response = connection.respond(HTTPStatus.OK, "OK\n")
    response.headers["Set-Cookie"] = (f"{COOKIE}={value}; Max-Age={max_age}; Path=/; "
                                      "Secure; HttpOnly; SameSite=Strict")
    return response


async def login(connection, request):
    """GET /login with the password in an X-Password header (websockets only speaks GET)."""
    ip = connection.remote_address[0]
    tries, locked_until = _failed.get(ip, (0, 0))
    if time.time() < locked_until:
        return connection.respond(HTTPStatus.TOO_MANY_REQUESTS, f"{int(locked_until - time.time()) + 1}\n")
    given = unquote(request.headers.get("X-Password", ""))
    if PASSWORD and hmac.compare_digest(given.encode(), PASSWORD.encode()):
        _failed.pop(ip, None)
        print(f"\n🔓  Device logged in ({ip})", flush=True)
        return cookie_response(connection, TOKEN, COOKIE_AGE)
    await asyncio.sleep(1)  # slows down guessing
    tries += 1
    _failed[ip] = (0, time.time() + LOCK_SECONDS) if tries >= MAX_TRIES else (tries, 0)
    print(f"\n🔒  Wrong password from {ip}", flush=True)
    return connection.respond(HTTPStatus.UNAUTHORIZED, "Wrong password\n")


async def http(connection, request):
    """Serve the web page and the login endpoints; let /ws through (only when logged in)."""
    path = request.path.split("?")[0]
    if path == "/ws":
        return None if logged_in(request) else connection.respond(HTTPStatus.UNAUTHORIZED, "Log in first\n")
    if path == "/auth":  # the page asks before connecting: 204 = go ahead, 401 = show the lock screen
        if logged_in(request):
            return connection.respond(HTTPStatus.NO_CONTENT, "")
        return connection.respond(HTTPStatus.UNAUTHORIZED, "Log in first\n")
    if path == "/login":
        return await login(connection, request)
    if path == "/logout":
        return cookie_response(connection, "", 0)
    if path == "/":
        headers = Headers({"Content-Type": "text/html; charset=utf-8",
                           "Content-Length": str(len(INDEX_HTML)),
                           "Cache-Control": "no-cache"})
        return Response(HTTPStatus.OK, "OK", headers, INDEX_HTML)
    return connection.respond(HTTPStatus.NOT_FOUND, "Not found\n")


def ssl_context():
    cert, key = HERE / SSL_CERT, HERE / SSL_KEY
    if not cert.exists():
        print("Creating self-signed HTTPS certificate...", flush=True)
        cert.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-days", "3650", "-subj", "/CN=voice-assistant",
             "-keyout", str(key), "-out", str(cert)],
            check=True, capture_output=True,
        )
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    return ctx


def local_ip():
    """This machine's address on the network (no packets are sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "localhost"


async def main():
    global tts, voice, recorder
    print("Loading voice...", flush=True)
    voice = make_voice()
    # "end" = all audio of the reply was sent; the browser answers "played" when it's done playing
    tts = TextToAudioStream(voice, muted=True,
                            on_audio_stream_stop=lambda: send({"type": "end"}))

    print("Loading ears...", flush=True)
    recorder = make_recorder(interrupt, on_wake=lambda: wake_state("listening"),
                             use_microphone=False)
    if WAKE_WORD:
        recorder.on_wakeword_timeout = close_follow_up
    set_wake_mode(False)  # until the page says otherwise (its switch defaults to off)

    print(f"Loading brain ({LLM_MODEL})...", flush=True)
    warm_up_llm()
    print_memory_status()
    print("🔒  Password on (each device logs in once)" if PASSWORD else
          "⚠️   No password: anyone on your Wi-Fi can use the page (set WEB_PASSWORD in local.env)",
          flush=True)
    assistant.on_memory_change = memory_event
    threading.Thread(target=listen, args=(recorder, inbox), daemon=True).start()
    threading.Thread(target=conversation, daemon=True).start()

    async with serve(handler, HOST, PORT, process_request=http,
                     ssl=ssl_context() if HTTPS else None, max_size=None):
        scheme = "https" if HTTPS else "http"
        print(f"\n🌐  Open on any device:  {scheme}://{local_ip()}:{PORT}")
        print("    (same Wi-Fi; accept the certificate warning once). Ctrl+C to quit.\n")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        stop_llm.set()
        if recorder:
            recorder.shutdown()
