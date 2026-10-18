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

- Wake word (console: `/wake` command, typed lines starting with `/` are commands; phone: the page's switch → `{"type":"wake_mode"}`; both call `assistant.set_wake_mode()`, which toggles `recorder.use_wake_words` live — web.py imports it as `set_recorder_wake_mode` because it has its own `set_wake_mode`; `wake_word_options()` in assistant.py; `on_wake` hook lets web.py chime on the phone instead of the PC): openwakeword 0.6 is installed with `--no-deps` (it declares tflite-runtime, which has no py3.12 build; we use the ONNX engine) — so `setup.sh` installs requirements with `--no-deps`. Always pass `openwakeword_model_paths`, or every built-in model (alexa, mycroft…) triggers. `on_wakeword_detected` fires per audio chunk — keep the debounce. The follow-up window is RealtimeSTT's `wake_word_activation_delay`, switched at runtime: 0 at startup (else the first FOLLOW_UP seconds accept any speech), set to FOLLOW_UP with `recorder.listen_start` reset when a reply finishes playing (TTS `on_audio_stream_stop`, hooked after "Ready."), back to 0 in `on_wakeword_timeout`. In web mode the server's TTS finishes before the phone does, so the server sends `end`, the page answers `played` when its playback queue is empty, and only then `open_follow_up()` runs.
- Barge-in must use `on_vad_start` (actual speech). `on_vad_detect_start` fires whenever the recorder *starts listening*, which killed every reply.
- Every Ollama call must send the same `LLM_OPTIONS` (`num_ctx`), or Ollama reloads the model. Requests use `keep_alive: -1`, so models stay in VRAM until `ollama stop <model>`.
- Two GPUs (since 2026-10): RTX 3070 Ti (index 0) runs Ollama `qwen3:8b` + Whisper; Quadro T1000 (index 1, 4 GB, no tensor cores) runs Kokoro and the desktop (monitor on the T1000 + BIOS primary display = T1000; without the BIOS change the desktop still renders on the 3070 Ti). Set in `local.env` (git-ignored, machine-specific; read by `run.sh`, command-line env wins): `WHISPER_GPU=0 STT_COMPUTE=int8_float16 KOKORO_GPU=1 LLM_MODEL=qwen3:8b`. Code defaults stay one GPU + `llama3.2:3b` + `float16`. `run.sh` exports `CUDA_DEVICE_ORDER=PCI_BUS_ID` so indices match nvidia-smi. Whisper uses RealtimeSTT's `gpu_device_index=WHISPER_GPU`; Kokoro has no device option, so `make_voice()` loads it inside `torch.cuda.device(KOKORO_GPU)` (inference follows the model's device). Measured for a 3.3 s sentence: Whisper on the 3070 Ti next to qwen `int8_float16` 141 ms (5.9 GB used), `float16` 159 ms (6.9 GB); on the T1000 `int8` 1.7 s, `float16` 7.1 s — too slow, the user rejected it. Kokoro on the T1000: first audio 0.10 s. Don't change model/STT settings unasked. Ollama is pinned to the 3070 Ti (applied 2026-10-18) by a systemd drop-in `/etc/systemd/system/ollama.service.d/gpu.conf` with `CUDA_VISIBLE_DEVICES=<3070 Ti UUID>` — Ollama can't see the T1000. Check: `systemctl show ollama -p Environment`.
- VRAM: 3070 Ti ≈6.7 GB used (qwen3 5.5 + Whisper 1.2), ~1.1 GB free (qwen2.5: ~2.3 GB free); T1000 ≈1.1 GB (Kokoro 0.7 + desktop). Only one mode (console or web) can run at a time. `qwen2.5:7b`, `llama3.2:3b` and `qwen3:8b` are installed — don't remove any. qwen3:8b is the model in use (since 2026-10-18): before the pinning it split across both GPUs and was slow; pinned, first word 54–267 ms, same speed as qwen2.5. All Ollama requests send `"think": false` (qwen3 would otherwise reason silently before answering; other models ignore it).
- RealtimeTTS 0.8.10 already hands Kokoro whole sentences; bigger `minimum_*_length` values change nothing. `comma_silence_duration` only applies when a piece ends at a comma; the audible pause is `SENTENCE_PAUSE`.
- Offline: `run.sh` exports `HF_HUB_OFFLINE=1` once `~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M` exists (otherwise Kokoro/Whisper ping Hugging Face on every start and print the HF_TOKEN warning). New HF models need `HF_HUB_OFFLINE=0` once. Verified offline by running with `HTTP(S)_PROXY=http://127.0.0.1:9` and `NO_PROXY=localhost,127.0.0.1`: Silero (torch.hub cache) and openWakeWord (downloaded models) also stay local.
- Browsers require HTTPS for the mic, hence the self-signed cert. The page has no auth and binds `0.0.0.0`.
- `requirements.txt` is a full `pip freeze` (with the cu124 extra index header). Upgrade deliberately: change, test, re-freeze, keep the header.
- Conversation log: `log_turn()` in assistant.py, called at the end of `llm_stream` (the one path for both modes); callers pass `source="voice"|"typed"`. Files `LOG_DIR/YYYY-MM-DD.md`, session header per run. When testing, set `LOG_DIR` to a scratch path (and `MEMORY=0`) so the user's real logs and memory stay clean.
- Not in git (keep it that way): `memory.md` and `conversations/` (personal data), `local.env`, `certs/`, `voice-samples/`, `.venv/`, logs.

## Working with this user

- Exploratory questions ("is it possible…", "what do you think…") get discussion only — no commands, not even read-only checks — until they say to build.
- `ROADMAP.md` is the running work list: pick from it, move finished items to Done, add agreed ideas.
- Commit locally only when asked; there is no remote — never push. After a feature, update README / HOW_IT_WORKS / ROADMAP as relevant.
