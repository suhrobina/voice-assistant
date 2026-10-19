# How It Works

The assistant is a chain of three parts — **ears → brain → voice** — plus a few helpers. Everything runs on your own PC.

```
 🎤 Mic ──► EARS ──────────► BRAIN ──────────► VOICE ──► 🔊 Speakers
           RealtimeSTT        Ollama            RealtimeTTS
           (Silero VAD +      (llama3.2:3b)     (Kokoro)
            Whisper)
               ▲                                    │
               └──────── barge-in: you talk ────────┘
                         → it stops speaking
```

## The components

### 1. Ears — RealtimeSTT (speech-to-text)
Turns your voice into text. Inside it, three pieces work together:

- **PyAudio** reads the microphone continuously, in small chunks (~1/16 of a second).
- **VAD (voice activity detection)** decides "is someone talking or is it silence?". Two detectors run together:
  - **WebRTC VAD** — fast and rough.
  - **Silero VAD** — accurate (requires `onnxruntime`).

  When you have been quiet for `SILENCE` seconds (default 0.45 s), it decides your turn is over.
- **faster-whisper** — the actual speech recognizer, running on the GPU. The model is `distil-large-v3` (~1.5 GB, downloaded on first launch). It turns the recorded audio into text, e.g. `"Do you hear me?"`.

### 2. Brain — Ollama + LLM
Decides what to answer.

- **Ollama** is a separate program running in the background as a system service. It keeps the AI model (`llama3.2:3b` by default) loaded in GPU memory.
- The script sends your text to it over a local web address (`http://localhost:11434`), together with the **conversation history**, so it remembers earlier turns.
- The **system prompt** tells the model: be warm, use short sentences, no markdown — because the answer will be spoken, not read.
- **Streaming:** the model returns the answer word by word as it is generated, instead of waiting for the whole answer.

### 3. Voice — RealtimeTTS + Kokoro (text-to-speech)
Turns the answer into speech.

- **RealtimeTTS** collects the streaming words and cuts them into sentences or clauses. As soon as it has a first piece (~8 characters, e.g. `"Sure, I can"`), it sends it on — without waiting for the full answer.
- **Kokoro** is the speech model. It turns each piece of text into audio using the selected voice (`af_heart` by default).
- **PyAudio** plays the audio through your speakers.

## Typing instead of talking

Spoken and typed messages go into the same queue (the "inbox"), so the brain doesn't care
where the text came from — memory, personality and history work the same.

```
 🎤 voice → Whisper ─┐
                     ├─► inbox ─► Ollama ─► Kokoro (web page: only while talking is on)
 ⌨️ typed text ──────┘
```

A typed message always interrupts the current reply. On the web page, replies are spoken
only while talking is on (🎤 tapped); otherwise the server skips Kokoro entirely, so a silent
text chat doesn't use the GPU for speech.

## One conversation turn, step by step

1. You say: *"What's the weather like on Mars?"*
2. Silero VAD hears speech → recording starts.
3. You stop. After 0.45 s of silence → turn is over.
4. Whisper transcribes the audio → `"What's the weather like on Mars?"` (~0.1 s).
5. The script sends this to Ollama → it starts streaming: `"Mars" "is" "very" "cold" ","…`
6. RealtimeTTS receives `"Mars is very cold,"` → Kokoro speaks it immediately.
7. While that plays, the brain keeps generating and Kokoro prepares the next sentence.

Because each step starts before the previous one finishes, you hear an answer about **0.7 s** after you stop talking, even though the full answer takes several seconds to generate.

| Step | Approx. time |
|---|---|
| Silence detection | 0.45 s |
| Transcription (Whisper) | ~0.1 s |
| First words from the LLM | ~0.1 s |
| First audio (Kokoro) | ~0.1 s |
| **Total until first sound** | **~0.7 s** |

## Wake word ("Hey Jarvis")

In normal mode, and in phone mode when the page's **👂 wake word icon** is on, the ears first wait
for the wake word, detected offline by
**openWakeWord** (a tiny model, runs on the CPU):

```
 💤 waiting ──"Hey Jarvis"──► 👂 chime, listening ──► you talk ──► reply
     ▲                                                               │
     └── 🔉 falling tone ◄── after FOLLOW_UP s of silence ◄── free-talk window ◄──┘
                                            (starts when the reply finishes playing)
```

- While waiting, voice detection is off — the TV or other people can't start a turn or interrupt a reply.
- Saying "Hey Jarvis" during a reply interrupts it (barge-in).
- Phone mode: the phone plays audio after the server has finished generating it, so the page reports
  `played` when it's done speaking and the server opens the window then. The server also sends the
  wake state (`listening` / `window` / `sleep`) so the page can chime and show it.
- Switch at runtime: console `/wake` command, phone **👂 button** next to the talk button (both toggle `recorder.use_wake_words`).
- Going back to sleep (window over, or the wake word and then 5 s of silence) plays two soft falling notes —
  the opposite of the wake chime — except while a reply is still playing.
- Settings: `WAKE_WORD` (`""` = off), `FOLLOW_UP`, `WAKE_SENSITIVITY`, `SLEEP_SOUND` (`0` = no falling tone).

## Barge-in (interrupting)

If you start talking while the assistant is speaking, Silero VAD notices it and calls `interrupt()`, which:

1. stops the audio playback,
2. tells the brain to stop generating,
3. listens to you.

It is hooked to the `on_vad_start` event, which fires only when actual speech is detected.

With **speakers** (no headphones), the microphone hears the assistant's own voice and it interrupts itself. In that case run it with interruption turned off:

```bash
BARGE_IN=0 ./run.sh
```

## Memory

Short-term: during a session the whole conversation is sent to the AI on every turn
(up to `num_ctx` = 4096 tokens, roughly 15–20 minutes; older parts drop off).

Long-term (`memory.py`, on unless `MEMORY=0`):

```
 You: "I have a dog named Rex"
        │
        ├─► normal reply (spoken right away)
        │
        └─► in the background, after the reply is generated:
              is it about you? ("I", "my", "we"… or "remember"/"forget"; questions skipped)
                └─► the AI extracts facts as JSON  →  {"add": ["Has a dog named Rex"], "remove": []}
                      └─► saved to memory/memory.md, and added to the current conversation

 Next start: facts from memory/memory.md are added to the system prompt → it knows you from the first word.
```

- It uses the same AI model and settings, so nothing extra is loaded into GPU memory.
- Safety rule: a fact is removed only when you ask to forget it, or when a new fact replaces it.
- By hand: the phone's ☰ panel and the console commands (`/memory`, `/remember`, `/forget`) change the same
  list. Every change — by hand or learned — updates the current conversation at once; learned facts also
  show up in the phone chat as *🧠 Remembered: …*.

## Phone mode

`./run.sh --web` swaps the local mic and speakers for a web page; the ears, brain and voice stay on the server:

```
 🎧 Bluetooth ◄──► 📱 Phone browser ◄──── Wi-Fi ────► 🖥️ Server
    headphones       mic → 16 kHz PCM   ── audio ──►   Whisper → Ollama → Kokoro
                     plays 24 kHz PCM   ◄── audio ──   (sent as it's generated)
```

- The page and the audio share one encrypted WebSocket connection (`/ws`). With `WEB_PASSWORD` set, the server
  refuses that connection until the phone has logged in once (a signed cookie remembers it).
- The server also sends small text messages: what you said, the AI's words as they arrive, and "stop" when you interrupt — so the phone drops any audio it has buffered.
- Kokoro generates audio faster than real time, so the phone receives the reply ahead of playback and plays it back to back without gaps.

## The files

| File | Role |
|---|---|
| `assistant.py` | Connects ears, brain and voice; runs the conversation loop |
| `profiles/` | Personalities: settings + prompt per file; switch with `/profile` or the phone menu |
| `memory.py` | Long-term memory: extracts facts about you and keeps them in `memory/memory.md` |
| `conversations/` | Daily Markdown logs of every exchange (`LOG=0` = off; not in git) |
| `memory/memory.md` | What the assistant remembers about you (plain text, created automatically, not in git) |
| `profile.txt` | The last profile you chose, used on the next start (not in git) |
| `web.py` | Phone mode (`./run.sh --web`): same loop, but audio comes from / goes to a web page over a WebSocket |
| `web/index.html` | The phone page: records the mic (16 kHz), plays the replies (24 kHz), shows the chat |
| `certs/` | Self-signed HTTPS certificate, created on first `--web` start (browsers need HTTPS for the mic) |
| `setup.sh` | One-time install: system packages, Ollama, the AI model, Python environment |
| `requirements.txt` | Exact versions of all Python packages, so every install is identical to the tested one |
| `run.sh` | Starts the assistant and tells it where the CUDA libraries are |
| `install-service.sh`, `voice-assistant.service` | Optional: start phone mode automatically at boot (systemd service) |
| `.venv/` | Python environment with all installed packages |
| `realtimesst.log` | RealtimeSTT's debug log — useful when something breaks |

## Where things run

Whisper, Kokoro and the AI model all run on the **GPU**, so it is fast and nothing leaves your computer. Internet is needed only for setup and the first launch (to download the models). After that it works fully offline: `run.sh` sets `HF_HUB_OFFLINE=1`, so Kokoro and Whisper don't even check Hugging Face for updates.
