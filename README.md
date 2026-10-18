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
