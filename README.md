# Local Voice Assistant (RTX 3070 Ti, fully offline)

How it works under the hood: see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).
Ideas and plans: see [ROADMAP.md](ROADMAP.md).

## Install & run
```bash
unzip voice-assistant.zip && cd voice-assistant
bash setup.sh      # once, ~10 min (downloads models)
./run.sh
```
Wait for "Ready.", then say **"Hey Jarvis"** — a short chime means it's listening — and talk.
For 8 seconds after each answer you can keep talking without the wake word. Say "Hey Jarvis" to interrupt.
You can also **type** a message in the console and press Enter — voice keeps working at the same time.

**Wake word settings:**
- Switch while it's running: type `/wake` (or `/wake on`, `/wake off`) and press Enter. `/help` lists all commands.
- Start without the wake word feature at all: `WAKE_WORD= ./run.sh`
- Follow-up window: `FOLLOW_UP=15 ./run.sh` (seconds), or `FOLLOW_UP=0` to need "Hey Jarvis" every time.
- When it stops listening (window over, or "Hey Jarvis" and then nothing), two soft falling notes play:
  say "Hey Jarvis" again. `SLEEP_SOUND=0` turns that sound off.
- Too many false wake-ups → `WAKE_SENSITIVITY=0.7`; not reacting → `0.3` (default 0.5).
- Phone mode: the **👂 wake word button** next to the talk button picks the mode — off = regular talking
  (default), on = say "Hey Jarvis" first (the phone chimes and shows the state above the text box). The page remembers it.

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
3. One button at the right of the text field changes with what you do (like the ChatGPT/Claude apps):
   - **sound-wave icon** (field empty): start a voice conversation — you talk, it talks back.
   - **➤** (as soon as you type): send. Enter makes a new line (on a PC keyboard, Ctrl+Enter sends).
     Without voice on, replies are silent text.
   - **■** (while talking): stop talking — back to silent text chat.
   Keep the page open.

Notes:
- Audio goes through whatever the phone uses, so Bluetooth headphones just work.
- The browser's echo cancellation lets you interrupt even on the phone speaker.
- One phone at a time; a new connection replaces the old one.
- **From outside your home** (mobile data): install [Tailscale](https://tailscale.com) on the server and phone, then open `https://<server's Tailscale IP>:8443`. Don't open port 8443 on your router.
- Settings: `PORT=9000 ./run.sh --web`; all the usual ones (`VOICE`, `LLM_MODEL`…) work too.

### Password
Add it to `local.env` (plain text, no quotes), then restart (`./run.sh --web`, or
`sudo systemctl restart voice-assistant` with auto-start):
```
WEB_PASSWORD=your secret phrase
```
- The page shows a lock screen; after the right password the phone is remembered for a year
  (also across restarts). Without a password set, the page stays open and the server prints a warning.
- **Log out:** ☰ panel → *Log out of this phone* (with `MEMORY=0` the ☰ button stays, offering only that). **Changing the password** logs out every phone.
- 5 wrong tries → that device waits 30 s. The password travels over HTTPS (encrypted).

### Auto-start at boot (phone mode always on)
```bash
sudo ./install-service.sh              # install, start now and at every boot
sudo ./install-service.sh --uninstall  # remove
```
It runs `./run.sh --web` as a system service under your user (same models, memory, profiles and
`local.env`), waits for Ollama, and restarts itself if it crashes. Nobody needs to be logged in.

| What | Command |
|---|---|
| Is it running? | `systemctl status voice-assistant` |
| Live log (conversation + errors) | `journalctl -u voice-assistant -f` |
| Stop / start / restart (e.g. after code changes) | `sudo systemctl stop\|start\|restart voice-assistant` |

- Only one copy fits on the GPU: while the service runs, `./run.sh` refuses to start and tells you to
  stop the service first. It starts again at the next boot (or `sudo systemctl start voice-assistant`).
- Models stay loaded in GPU memory all the time — stop the service before games or other GPU work.
- Set a password (below), or anyone on your Wi-Fi who knows the address can use it.
- A fixed IP for the PC (set in the router) keeps the phone's bookmark working.

## Two GPUs
This PC: **RTX 3070 Ti** runs the AI model (`qwen3:8b`) + Whisper; **Quadro T1000** runs the voice
(Kokoro) and the desktop. Answers start as fast as on one GPU, with the smarter model.
- **Settings:** in `local.env` (machine-specific, not in git). GPU numbers as in `nvidia-smi -L`:
  ```
  WHISPER_GPU=0                # ears on the 3070 Ti
  STT_COMPUTE=int8_float16     # 141 ms per sentence, ~1 GB less memory than float16 (159 ms)
  KOKORO_GPU=1                 # voice on the T1000
  LLM_MODEL=qwen3:8b           # ~5.6 GB on the 3070 Ti (qwen2.5:7b: 4.8 GB, also installed)
  ```
  One-off override, e.g. `LLM_MODEL=llama3.2:3b ./run.sh`. Back to one GPU + llama: delete `local.env`.
- **Desktop on the T1000:** monitor plugged into the T1000 *and* the BIOS primary display set to it.
  Otherwise the desktop still renders on the 3070 Ti (~0.3 GB, more with a browser video).
- **Memory on the 3070 Ti:** ~6.7 GB of 8 GB used, ~1.1 GB free (with qwen2.5:7b: ~2.3 GB free).
- **Why not Whisper on the T1000?** It has no tensor cores: 1.7 s per sentence even at `int8` (7 s at `float16`).
- **Ollama's GPU** (one time, needs sudo; applied on this PC). Needed for qwen3 — without it, Ollama
  splits the model across both cards and it gets slow:
  ```bash
  sudo mkdir -p /etc/systemd/system/ollama.service.d
  printf '[Service]\nEnvironment="CUDA_VISIBLE_DEVICES=<3070 Ti UUID>"\n' | sudo tee /etc/systemd/system/ollama.service.d/gpu.conf
  sudo systemctl daemon-reload && sudo systemctl restart ollama
  ```
- Check with `nvidia-smi` while it runs: `llama-server` + `python` on the 3070 Ti, `python` + Xorg on the T1000.

## Memory (remembers you between sessions)
On by default. When you share something about yourself ("My name is…", "I have a dog named Rex",
"Remember that my sister's birthday is May 5th"), it's saved to `memory/memory.md` and the assistant
knows it next time — in both normal and phone mode.

- **Forget something:** just say it — "Forget about my dog".
- **See or edit what it knows:**
  - Phone: the **☰ button** (three lines) under the text field opens the list — tap a fact to edit it, ✕ deletes it, the field adds one,
    "Forget everything" clears it. When it learns something, a grey *🧠 Remembered: …* line appears in the chat.
  - Console: `/memory` lists the facts, `/remember Has a cat named Luna` adds one, `/forget 3` deletes
    fact 3, `/forget all` clears everything.
  - Both take effect immediately. Or edit `memory/memory.md` by hand (one fact per line, newest on top; restart afterwards).
- **Turn off:** `MEMORY=0 ./run.sh` (or `MEMORY=0 ./run.sh --web`). Then `memory/memory.md` is
  neither read nor written.
- **Start over:** delete `memory/memory.md`.
- Questions and small talk are never saved. Small models occasionally save something odd —
  just delete that line. `qwen2.5:7b` picks facts more accurately.
- Stays on your server; `memory/memory.md` is excluded from git.

## Tips for the smoothest feel
- **Use headphones.** Otherwise the assistant hears itself and interrupts. Without headphones run: `BARGE_IN=0 ./run.sh`
- **Faster turn-taking:** `SILENCE=0.3 ./run.sh` (it may cut you off mid-pause).
- **Smarter answers:** `ollama pull qwen2.5:7b` then `LLM_MODEL=qwen2.5:7b ./run.sh` (fits in 8 GB, slightly slower).
- **Other language:** `LANGUAGE=ru ./run.sh` (or `LANGUAGE=` for auto-detect). Kokoro voices are mainly English; see Kokoro voice list for others.
- **Other voice:** `VOICE=am_michael ./run.sh` (also af_bella, bf_emma, am_adam...), or a blend: `VOICE="0.6*af_heart + 0.4*af_bella" ./run.sh`
- **Speaking speed:** `TTS_SPEED=0.92 ./run.sh` (calmer) or `1.1` (faster).
- **Pause between sentences:** `SENTENCE_PAUSE=0.3 ./run.sh` (default 0.2 s; `0` = none).

## Expected latency on a 3070 Ti
About 0.5–0.9 s from when you stop talking to first sound. Roughly: silence detection ~0.45 s, transcription ~0.1 s, first LLM words ~0.1 s, first audio ~0.1 s.

## Profiles (personalities)
Each file in `profiles/` is a personality with its own prompt, voice and settings. Memory is shared
(every profile knows you); switching starts a fresh conversation.

| Profile | What it does | Voice |
|---|---|---|
| `default` | the everyday assistant | af_heart |
| `teacher` | English teacher: gently corrects your grammar, then answers | bf_emma (British) |
| `chef` | cooking helper, one step at a time | am_michael |
| `storyteller` | tells stories, slower and calmer | af_bella |

- **Switch:** console `/profile teacher` (`/profiles` lists them); phone: the profile button under the text field.
- **Remembered:** it starts with the profile you chose last (saved in `memory/profile.txt`). If that profile file was deleted, it starts with `default`.
- **Force one at startup:** `PROFILE=chef ./run.sh`, or `PROFILE=chef` in `local.env` (this wins over the remembered one).
- **Make your own:** copy `profiles/_template.md` (it explains every setting) to e.g. `profiles/coach.md` and edit it.
  The file name is the profile name (letters, digits, `_`, `-`); files starting with `_` are not profiles. Example:
  ```
  label: Fitness coach
  voice: am_michael
  speed: 1.05
  temperature: 0.6

  You are an energetic fitness coach. ... (the prompt; keep "spoken aloud: short sentences, no markdown")
  ```
  Settings at the top, one blank line, then the prompt. All settings are optional; left out = the normal value:

  | Setting | What it does | Values | Default |
  |---|---|---|---|
  | `label` | name in the phone menu, spoken on switch | any text | file name |
  | `voice` | Kokoro voice: 1st letter `a` American / `b` British, 2nd `f` female / `m` male | downloaded: `af_heart`, `af_bella`, `am_michael`, `bf_emma`; others (`af_sarah`, `am_adam`, `bm_george`…) need one run with `HF_HUB_OFFLINE=0` | `af_heart` |
  | `speed` | speaking speed (0.9 calmer, 1.1 faster) | 0.8 – 1.2 | 1.0 |
  | `pause` | seconds of silence between sentences | 0 – 0.5 | 0.2 |
  | `temperature` | creativity (0.3 focused, 0.9 lively) | 0.0 – 1.0 | 0.7 |
  | `model` | AI model; a different one takes ~20–60 s to load on every switch | installed: `qwen3:8b`, `qwen2.5:7b`, `llama3.2:3b` (`ollama list`) | `LLM_MODEL` |

  A wrong value (not a number, a voice or model that isn't there) is skipped with a ⚠️ warning and the normal value is used.

## Conversation log
Every conversation is saved to `conversations/YYYY-MM-DD.md` (one Markdown file per day), in both
normal and phone mode: time, what you said (spoken or typed) and the AI's reply; interrupted replies
are marked. Open the files in any editor.
- **Turn off:** `LOG=0 ./run.sh` (or put `LOG=0` in `local.env`).
- **Other folder:** `LOG_DIR=/path/to/folder ./run.sh`.
- Private: stays on your PC and is excluded from git. Delete the files any time.

## Fully offline
After the first start (models downloaded), the assistant never contacts the internet: `run.sh` sets
`HF_HUB_OFFLINE=1` once the models are on disk (tested with all internet traffic blocked).
- **New voice or Whisper model** (not downloaded yet): run once with `HF_HUB_OFFLINE=0 ./run.sh`,
  after that it works offline too.
- Ollama models are separate: `ollama pull <model>` needs internet once.

## Troubleshooting
- **First launch is slow** (several minutes): it downloads the Whisper model (~1.5 GB) and loads the LLM into the GPU for the first time. Later launches are much faster.
- It hears you but never answers / cuts itself off → if you use speakers, start with `BARGE_IN=0 ./run.sh`.
- `libcudnn` / `libcublas` errors → always start with `./run.sh` (it sets the library path).
- Out of GPU memory → use `STT_MODEL=small.en` or a smaller LLM.
- No mic input → check `pactl list sources short`, set default mic in system sound settings.
