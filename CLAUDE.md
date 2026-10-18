# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Offline voice assistant on one NVIDIA GPU (RTX 3070 Ti, 8 GB): RealtimeSTT (faster-whisper + Silero VAD) → Ollama (streaming) → RealtimeTTS + Kokoro. Two front ends share one brain: local mic/speakers (`assistant.py`) and a phone browser over HTTPS/WebSocket (`web.py` + `web/index.html`). User-facing docs: `README.md`, architecture explainer: `HOW_IT_WORKS.md`, work list: `ROADMAP.md`.

## Commands

```bash
bash setup.sh            # one-time: apt deps, Ollama + model pull, .venv from requirements.txt
./run.sh                 # local mode (mic + speakers; typing + Enter also works)
./run.sh --web           # phone mode, serves https://<ip>:8443 (self-signed cert auto-created in certs/)
VOICE=am_michael LLM_MODEL=qwen2.5:7b MEMORY=0 ./run.sh   # all config is env vars (CONFIG block in assistant.py)
.venv/bin/python -m py_compile assistant.py web.py memory.py   # quick syntax check
```

Always launch via `run.sh`: it sets `LD_LIBRARY_PATH` to the pip-installed cuBLAS/cuDNN, without which faster-whisper fails.

There is no test suite or linter. Changes have been verified with throwaway scripts (keep them in the scratchpad, not the repo):
- **Brain/memory without audio:** `import assistant` and consume `assistant.llm_stream("...")`; set `MEMORY_FILE` to a scratch path first so the user's real `memory.md` isn't touched.
- **Phone mode without a phone:** start `./run.sh --web`, connect a `websockets` client to `wss://localhost:8443/ws` (cert verification off), send 16 kHz s16le PCM (e.g. `espeak-ng … | ffmpeg -ar 16000 -ac 1 -f s16le -`) or `{"type":"text","text":…}`, and collect JSON events + 24 kHz PCM audio.
- **Console typing:** `(sleep 45; echo "question") | timeout 75 ./run.sh`.
- **Voice changes:** render WAVs with `TextToAudioStream(..., muted=True).play(output_wavfile=...)` and let the user compare by ear.

## Architecture

- **`assistant.py` is also the shared library.** `web.py` imports its config, `llm_stream`, `make_recorder`, `PLAY_OPTIONS`, `listen`, `stop_llm`. Tune STT/TTS/LLM settings there once; both modes pick them up.
- **Inbox pattern:** spoken text (`listen()` thread around `recorder.text()`) and typed text go into one `queue.Queue` as `("voice"|"typed", text)`; a single conversation loop consumes it. Typed messages always interrupt (`interrupt(force=True)`), voice only when `BARGE_IN=1`.
- **`llm_stream` is a generator fed to `tts.feed()`**; RealtimeTTS pulls tokens and synthesizes per sentence. Conversation history is a module-level list (`history[0]` is the system prompt, rebuilt with memory facts). Single user only.
- **Web mode:** recorder runs with `use_microphone=False` and gets `feed_audio()` from the WebSocket; TTS runs `muted=True` with `on_audio_chunk=send`, so audio is streamed (faster than real time) instead of played. Worker threads talk to asyncio via `loop.call_soon_threadsafe` into a per-connection queue (`pump`) to keep message order. Because the phone buffers audio ahead of the server, interrupts also send `{"type":"stop"}` and the page ignores audio until the next `{"type":"reply"}`. Replies are spoken only while the page's mic is on (page sends `{"type":"mute"}`); when muted the server skips Kokoro and just drains the generator in a thread.
- **Memory (`memory.py`):** after a reply, if `worth_checking()` passes (first-person statement or remember/forget; questions skipped), a background thread asks the same Ollama model for `{"add":[…],"remove":[…]}` JSON and updates `memory.md`. Removals are ignored unless the user said a forget-word or new facts were added — small models otherwise delete facts on unrelated turns. These guards were tuned against real `llama3.2:3b` failures; keep them.

## Gotchas (learned the hard way)

- Barge-in must use `on_vad_start` (actual speech). `on_vad_detect_start` fires whenever the recorder *starts listening*, which killed every reply.
- Every Ollama call must send the same `LLM_OPTIONS` (`num_ctx`), or Ollama reloads the model. Requests use `keep_alive: -1`, so models stay in VRAM until `ollama stop <model>`.
- VRAM is tight: Ollama `llama3.2:3b` ≈2.6 GB + Whisper/Kokoro ≈3 GB + desktop ≈0.7 GB. Only one mode can run at a time. `qwen2.5:7b` (installed — don't remove it) does not fit alongside `distil-large-v3`; the user chose to stay on `llama3.2:3b`.
- RealtimeTTS 0.8.10 already hands Kokoro whole sentences; bigger `minimum_*_length` values change nothing. `comma_silence_duration` only applies when a piece ends at a comma; the audible pause is `SENTENCE_PAUSE`.
- Browsers require HTTPS for the mic, hence the self-signed cert. The page has no auth and binds `0.0.0.0`.
- `requirements.txt` is a full `pip freeze` (with the cu124 extra index header). Upgrade deliberately: change, test, re-freeze, keep the header.
- Not in git (keep it that way): `memory.md` (personal data), `certs/`, `voice-samples/`, `.venv/`, logs.

## Working with this user

- Exploratory questions ("is it possible…", "what do you think…") get discussion only — no commands, not even read-only checks — until they say to build.
- `ROADMAP.md` is the running work list: pick from it, move finished items to Done, add agreed ideas.
- Commit locally only when asked; there is no remote — never push. After a feature, update README / HOW_IT_WORKS / ROADMAP as relevant.
