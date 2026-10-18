# Roadmap

Ideas to make this the perfect local AI assistant. Pick one, build it, tick it off.

## ✅ Done
- [x] Working local assistant: Whisper → Ollama → Kokoro, fixed missing packages and barge-in
- [x] Phone mode: `./run.sh --web`, works with Bluetooth headphones
- [x] Long-term memory: `memory.md`, can be turned off with `MEMORY=0`
- [x] Docs: README, HOW_IT_WORKS, comments on every setting in `assistant.py`
- [x] More natural Kokoro: 0.2 s breath between sentences (chosen by ear from samples); `TTS_SPEED`, `SENTENCE_PAUSE`, voice blends

## ⭐ Top priorities
- [ ] **Tools: time, date, timers, reminders.** "Remind me in 20 minutes to check the oven." Small models don't even know today's date without this.
- [ ] **Pin package versions in `setup.sh`.** Protects fresh installs from breaking updates (that's what broke the project at the start). Known-good versions: RealtimeSTT 1.1.2, RealtimeTTS 0.8.10, kokoro 0.9.4, faster-whisper 1.2.1, ctranslate2 4.8.3, onnxruntime 1.31.0, torch/torchaudio 2.6.0 (cu124).
- [ ] **Wake word** ("Hey Nova…") with openWakeWord, so it ignores the TV and other people. Most useful if it runs all day.

## 🛠️ Let it do things (more tools)
- [ ] Weather and news ("Do I need an umbrella tomorrow?")
- [ ] Web search for current events
- [ ] Control the PC: open apps, play music, volume
- [ ] Smart home via Home Assistant (lights, plugs)
- Note: `qwen2.5:7b` is much better at tools than `llama3.2:3b`.

## 🔧 Convenience
- [ ] Auto-start as a system service (phone mode always available)
- [ ] Password for the phone page (now anyone on the Wi-Fi can open it)
- [ ] Conversation log: save chats to files
- [ ] Response time display: print how long each step took

## 🎙️ Voice quality
- Tested: longer text pieces change nothing — RealtimeTTS 0.8.10 already sends Kokoro whole sentences
- Samples to compare voices/speeds: `voice-samples/` (not in git)
- [ ] Natural writing style in the system prompt: contractions, casual phrases, occasional "well," / "hmm,"
- [ ] More human-like speech model: Chatterbox (emotion, voice cloning, ~3–4 GB) or Orpheus (very natural, heavy)
- [ ] Other languages: Whisper supports many (`LANGUAGE=ru` or auto), Kokoro voices are mainly English

## 🖥️ Hardware: use more GPUs / PCs
Available: RTX 3070 Ti 8 GB (current), Quadro T1000 4 GB, Titan Xp 12 GB.
- [ ] **3070 Ti + T1000 in one PC** (best combo): Ollama on the 3070 Ti, Whisper + Kokoro on the T1000
  - Ollama: `systemctl edit ollama` → `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=<3070 Ti UUID>`
  - `run.sh`: optional `ASSISTANT_GPU` setting → `CUDA_VISIBLE_DEVICES`
  - Use `qwen2.5:7b` + `STT_MODEL=small.en` (~0.9–1.1 s response vs ~0.7 s now)
  - README "Two GPUs" section
- [ ] **AI model on another PC** over the network (`OLLAMA_URL` already supports it; `OLLAMA_HOST=0.0.0.0` on the AI PC); README section
- [ ] `STT_COMPUTE` setting (`int8`) so Whisper runs on the Titan Xp (Pascal has no float16 support)
- [ ] Run `qwen2.5:7b` on the single 3070 Ti: doesn't fit with `distil-large-v3` (tested). Options: `STT_MODEL=small.en`, or Whisper at `int8_float16` precision

## 🚀 Bigger features
- [ ] Vision: send a photo from the phone, ask "What's on this picture?" (`qwen2.5-vl`, needs more GPU memory)
- [ ] Remote access with Tailscale: use the assistant from anywhere, not just home Wi-Fi

## 💤 Parked (decided not now)
- Noise suppression (RNNoise): phone mode already has it via the browser; Whisper handles light noise well and filters can even hurt accuracy. Revisit only if noise becomes a real problem — measure before/after.
