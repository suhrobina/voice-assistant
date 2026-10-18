# Local Voice Assistant (RTX 3070 Ti, fully offline)

How it works under the hood: see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Install & run
```bash
unzip voice-assistant.zip && cd voice-assistant
bash setup.sh      # once, ~10 min (downloads models)
./run.sh
```
Wait for "Ready.", then just talk. Talk over it to interrupt.

**Using speakers instead of headphones?** Start it with interruption turned off,
otherwise the assistant hears its own voice and keeps cutting itself off:
```bash
BARGE_IN=0 ./run.sh
```
With `BARGE_IN=0` you can't interrupt it mid-answer; wait until it finishes speaking.

## Use it from your phone (Bluetooth headphones work too)
```bash
./run.sh --web
```
1. It prints an address like `https://192.168.1.20:8443`. Open it in your phone's browser (Chrome) — phone and server on the same Wi-Fi.
2. The first time, the browser warns about the certificate (it is self-made). Tap **Advanced → Proceed**.
3. Tap **Start**, allow the microphone, and just talk. Keep the page open.

Notes:
- Audio goes through whatever the phone uses, so Bluetooth headphones just work.
- The browser's echo cancellation lets you interrupt even on the phone speaker.
- One phone at a time; a new connection replaces the old one.
- **From outside your home** (mobile data): install [Tailscale](https://tailscale.com) on the server and phone, then open `https://<server's Tailscale IP>:8443`. Don't open port 8443 on your router — the page has no password.
- Settings: `PORT=9000 ./run.sh --web`; all the usual ones (`VOICE`, `LLM_MODEL`…) work too.

## Memory (remembers you between sessions)
On by default. When you share something about yourself ("My name is…", "I have a dog named Rex",
"Remember that my sister's birthday is May 5th"), it's saved to `memory.md` and the assistant
knows it next time — in both normal and phone mode.

- **Forget something:** just say it — "Forget about my dog".
- **See or edit what it knows:** open `memory.md` — one fact per line, edit or delete freely
  (restart the assistant afterwards).
- **Turn off:** `MEMORY=0 ./run.sh` (or `MEMORY=0 ./run.sh --web`). Then `memory.md` is
  neither read nor written.
- **Start over:** delete `memory.md`.
- Questions and small talk are never saved. Small models occasionally save something odd —
  just delete that line. `qwen2.5:7b` picks facts more accurately.
- Stays on your server; `memory.md` is excluded from git.

## Tips for the smoothest feel
- **Use headphones.** Otherwise the assistant hears itself and interrupts. Without headphones run: `BARGE_IN=0 ./run.sh`
- **Faster turn-taking:** `SILENCE=0.3 ./run.sh` (it may cut you off mid-pause).
- **Smarter answers:** `ollama pull qwen2.5:7b` then `LLM_MODEL=qwen2.5:7b ./run.sh` (fits in 8 GB, slightly slower).
- **Other language:** `LANGUAGE=ru ./run.sh` (or `LANGUAGE=` for auto-detect). Kokoro voices are mainly English; see Kokoro voice list for others.
- **Other voice:** `VOICE=am_michael ./run.sh` (also af_bella, bf_emma, am_adam...)

## Expected latency on a 3070 Ti
About 0.5–0.9 s from when you stop talking to first sound. Roughly: silence detection ~0.45 s, transcription ~0.1 s, first LLM words ~0.1 s, first audio ~0.1 s.

## Troubleshooting
- **First launch is slow** (several minutes): it downloads the Whisper model (~1.5 GB) and loads the LLM into the GPU for the first time. Later launches are much faster.
- It hears you but never answers / cuts itself off → if you use speakers, start with `BARGE_IN=0 ./run.sh`.
- `libcudnn` / `libcublas` errors → always start with `./run.sh` (it sets the library path).
- Out of GPU memory → use `STT_MODEL=small.en` or a smaller LLM.
- No mic input → check `pactl list sources short`, set default mic in system sound settings.
