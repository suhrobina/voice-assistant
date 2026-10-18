#!/usr/bin/env python3
"""
Phone / browser mode for the voice assistant.
  Same ears, brain and voice as assistant.py, but the audio comes from and
  goes to a web page (web/index.html) instead of the local mic and speakers.

  Phone browser --mic, 16 kHz PCM-->  server: Whisper -> Ollama -> Kokoro
  Phone browser <--voice, 24 kHz PCM--

Browsers allow the microphone only over HTTPS, so a self-signed certificate
is created on first start (accept the warning once in the phone's browser).
Start with:  ./run.sh --web
"""
import asyncio
import json
import os
import queue
import socket
import ssl
import subprocess
import threading
import time
from http import HTTPStatus
from pathlib import Path

from RealtimeTTS import TextToAudioStream
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from assistant import (BARGE_IN, FOLLOW_UP, LLM_MODEL, PLAY_OPTIONS, make_voice,
                       WAKE_WORD, listen, llm_stream, make_recorder, print_memory_status,
                       stop_llm, warm_up_llm)
from assistant import set_wake_mode as set_recorder_wake_mode
from assistant import apply_profile, list_profiles, load_profile
from assistant import memory, memory_add, memory_clear, memory_edit, memory_remove, speakable
import assistant  # for the current profile (assistant.profile changes on switch)

# ---------------- CONFIG ----------------
HOST = os.getenv("HOST", "0.0.0.0")            # 0.0.0.0 = reachable from your network
PORT = int(os.getenv("PORT", "8443"))
HTTPS = os.getenv("HTTPS", "1") == "1"         # 0 = plain HTTP (e.g. behind `tailscale serve`)
SSL_CERT = os.getenv("SSL_CERT", "certs/cert.pem")  # created automatically if missing
SSL_KEY = os.getenv("SSL_KEY", "certs/key.pem")
# ----------------------------------------

HERE = Path(__file__).parent
INDEX_HTML = (HERE / "web" / "index.html").read_bytes()

tts = None
voice = None  # the Kokoro engine inside tts (voice/speed change with the profile)
recorder = None


class Client:
    """The one connected phone (a new connection replaces the old one)."""
    ws = None
    queue = None
    loop = None
    muted = False  # set by the page: True = text-only replies (talking is off)
    wake_mode = False  # set by the page's switch: require the wake word


inbox = queue.Queue()  # ("voice" | "typed", text) from the mic and the text box


def send(msg):
    """Thread-safe send to the phone: bytes = audio, dict = JSON event."""
    ws, queue, loop = Client.ws, Client.queue, Client.loop
    if ws is None:
        return
    data = msg if isinstance(msg, bytes) else json.dumps(msg)
    loop.call_soon_threadsafe(queue.put_nowait, data)


def interrupt(force=False):
    """Stop the current reply on the server and the phone."""
    if not (BARGE_IN or force):
        return
    stop_llm.set()
    if tts.is_playing():
        tts.stop()
    # The phone may still be playing buffered audio even if the server is done
    send({"type": "stop"})


def wake_state(state):
    """Tell the page: "listening" (wake word heard), "window" (follow-up), "sleep"."""
    send({"type": "wake", "state": state})


def set_wake_mode(on):
    """The page's switch: regular talking (off) or "Hey Jarvis" first (on)."""
    Client.wake_mode = set_recorder_wake_mode(recorder, on)
    print(f"\n📱  Wake word {'on' if Client.wake_mode else 'off'}", flush=True)


def open_follow_up():
    """The phone finished playing a reply: talk without the wake word for FOLLOW_UP s."""
    if Client.wake_mode and FOLLOW_UP > 0:
        recorder.wake_word_activation_delay = FOLLOW_UP
        recorder.listen_start = time.time()
        wake_state("window")


def close_follow_up():
    recorder.wake_word_activation_delay = 0
    wake_state("sleep")


def memory_event(added=(), removed=(), learned=False):
    """Send the fact list to the phone's memory panel; learned = the AI found it (chat note)."""
    send({"type": "memory", "facts": list(memory.facts), "added": list(added),
          "removed": list(removed), "learned": learned})


def drain(tokens):
    for _ in tokens:
        pass


def conversation():
    """Same loop as assistant.py, but replies are streamed to the phone."""
    text_reply = None  # thread generating a text-only reply (when muted)
    while True:
        source, text = inbox.get()
        interrupt(force=True)
        if text_reply:
            text_reply.join()
        if source == "profile":  # the page's profile menu
            if apply_profile(text, voice):
                p = assistant.profile
                print(f"\n🎭  {p['label']}: fresh conversation, memory kept", flush=True)
                send({"type": "profile", "name": p["name"], "label": p["label"]})
            continue
        print(f"\nYou{' (typed)' if source == 'typed' else ''}: {text}\nAI : ", end="", flush=True)
        send({"type": "user", "text": text})
        send({"type": "reply"})
        tokens = llm_stream(text, on_token=lambda t: send({"type": "token", "text": t}),
                            source=source)
        if Client.muted:
            text_reply = threading.Thread(target=drain, args=(tokens,), daemon=True)
            text_reply.start()
        else:
            tts.feed(speakable(tokens))
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
    print("\n📱  Phone connected", flush=True)
    send({"type": "hello", "wake": WAKE_WORD.replace("_", " "),
          "memory": memory is not None, "profile": assistant.profile["name"],
          "profiles": [{"name": n, "label": load_profile(n)["label"]} for n in list_profiles()]})
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
            elif m.get("type") == "wake_mode":
                set_wake_mode(m.get("value"))
            elif m.get("type") == "played":  # the phone finished speaking the reply
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
            print("\n📱  Phone disconnected", flush=True)


def http(connection, request):
    """Serve the web page; let /ws through to the WebSocket handler."""
    path = request.path.split("?")[0]
    if path == "/ws":
        return None
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
    # "end" = all audio of the reply was sent; the phone answers "played" when it's done playing
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
    assistant.on_memory_change = memory_event
    threading.Thread(target=listen, args=(recorder, inbox), daemon=True).start()
    threading.Thread(target=conversation, daemon=True).start()

    async with serve(handler, HOST, PORT, process_request=http,
                     ssl=ssl_context() if HTTPS else None, max_size=None):
        scheme = "https" if HTTPS else "http"
        print(f"\n🌐  Open on your phone:  {scheme}://{local_ip()}:{PORT}")
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
