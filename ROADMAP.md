# Roadmap

Ideas to make this the perfect local AI assistant. Pick one, build it, tick it off.

## ✅ Done
- [x] Working local assistant: Whisper → Ollama → Kokoro, fixed missing packages and barge-in
- [x] Phone mode: `./run.sh --web`, works with Bluetooth headphones
- [x] Long-term memory: `memory/memory.md`, can be turned off with `MEMORY=0`
- [x] Docs: README, HOW_IT_WORKS, comments on every setting in `assistant.py`
- [x] Typed messages: console (type + Enter) and web (text box, works without the mic; replies are spoken only while talking is on)
- [x] Pinned package versions: `requirements.txt` (all 136 packages, verified in a fresh environment), used by `setup.sh`
- [x] Wake word "Hey Jarvis" (openWakeWord, offline) with chime, 8 s follow-up window and a falling "sleep" tone when it stops listening (`SLEEP_SOUND=0` = off); `WAKE_WORD=` turns it off; works in console and phone mode (switch modes: `/wake` in the console, a switch on the phone)
- [x] Two GPUs: `qwen3:8b` + Whisper (`int8_float16`) on the RTX 3070 Ti (Ollama pinned to it), Kokoro + desktop on the Quadro T1000 (BIOS primary display). Smarter model, no extra delay. Settings in `local.env`
- [x] Fully offline after the first start: `HF_HUB_OFFLINE=1` in `run.sh` (verified with the internet blocked)
- [x] Conversation log: `conversations/YYYY-MM-DD.md`, both modes, spoken/typed marked; `LOG=0` turns it off
- [x] Phone page: one button in the text field (sound wave = talk / ➤ send / ■ stop) like the ChatGPT/Claude apps; composer card with 🧠 memory and profile buttons under the text (Claude app style); 👂 wake word toggle next to the talk button
- [x] Profiles: `profiles/*.md` (prompt + voice + settings), switch with `/profile` or the phone menu; shared memory, fresh conversation on switch; last choice remembered across restarts
- [x] Memory on the phone and console: 🧠 panel (list, edit, delete, add, forget everything), "🧠 Remembered: …" notes in the chat; `/memory`, `/remember`, `/forget N|all`
- [x] More natural Kokoro: 0.2 s breath between sentences (chosen by ear from samples); `TTS_SPEED`, `SENTENCE_PAUSE`, voice blends

## ⭐ Top priorities
- [ ] **Tools: time, date, timers, reminders.** "Remind me in 20 minutes to check the oven." Small models don't even know today's date without this.

## 🛠️ Let it do things (more tools)
- [ ] Weather and news ("Do I need an umbrella tomorrow?")
- [ ] Web search for current events
- [ ] Control the PC: open apps, play music, volume
- [ ] Smart home via Home Assistant (lights, plugs)
- Note: `qwen2.5:7b` is much better at tools than `llama3.2:3b`.

## 🔧 Convenience
- [ ] Auto-start as a system service (phone mode always available)
- [ ] Password for the phone page (now anyone on the Wi-Fi can open it)
- [ ] Response time display: print how long each step took

## 🎙️ Voice quality
- Tested: longer text pieces change nothing — RealtimeTTS 0.8.10 already sends Kokoro whole sentences
- Samples to compare voices/speeds: `voice-samples/` (not in git)
- [ ] Natural writing style in the system prompt: contractions, casual phrases, occasional "well," / "hmm,"
- [ ] More human-like speech model: Chatterbox (emotion, voice cloning, ~3–4 GB) or Orpheus (very natural, heavy)

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
- [ ] Custom wake word: train an openWakeWord model for any phrase (~1 h, free online notebook), or use Whisper to react to a name ("Nova, …") with no training
- [ ] Remote access with Tailscale: use the assistant from anywhere, not just home Wi-Fi

## 💤 Parked (decided not now)
- Other languages (e.g. Russian): English is enough. If ever: Whisper `large-v3-turbo` (distil-large-v3 is English-only), qwen3 already speaks Russian, Kokoro has no Russian → Silero TTS, routed per sentence by Cyrillic.
- Noise suppression (RNNoise): phone mode already has it via the browser; Whisper handles light noise well and filters can even hurt accuracy. Revisit only if noise becomes a real problem — measure before/after.
