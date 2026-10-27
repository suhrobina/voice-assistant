# TEMPLATE: copy this file to profiles/<name>.md (letters, digits, _ and -), e.g. coach.md, then switch
# with "/profile coach" or the web page's Profiles panel (which can also create and edit profiles from this
# template). Files starting with "_" are not profiles.
#
# Layout: settings lines at the top, ONE blank line, then the prompt (the rest of the file).
# Every setting is optional: delete a line to use the normal value. Lines starting with # are ignored.
#
# label        name in the Profiles panel, spoken on switch     any text           (file name)
# voice        Kokoro voice: 1st letter a = American, b = British; 2nd f = female, m = male
#              downloaded: af_heart, af_bella, am_michael, bf_emma   others: af_sarah, af_nicole,
#              am_adam, bf_isabella, bm_george, bm_lewis... (new ones: run once with HF_HUB_OFFLINE=0)
#                                                                                 (af_heart)
# speed        speaking speed, 0.9 = calmer, 1.1 = faster       0.8 - 1.2          (1.0)
# pause        seconds of silence between sentences            0 - 0.5            (0.2)
# temperature  0.3 = focused, 0.9 = lively                     0.0 - 1.0          (0.7)
# silence      seconds you can pause before it answers         0.3 - 2.0          (SILENCE, 0.45)
#              longer = time to think mid-sentence (e.g. a teacher profile: 0.8)
# model        AI model (must be installed, see "ollama list")  qwen3:8b, qwen2.5:7b, llama3.2:3b
#              a different model takes ~20-60 s to load on every switch     (LLM_MODEL in local.env)
# think        reason silently before answering: smarter on hard questions,  on / off  (off)
#              a few seconds slower. Only models that can (qwen3 yes; qwen2.5, llama3.2 no)
# context      how much of the conversation it sees, in tokens       2048 - 32768   (NUM_CTX, 4096)
#              (~0.75 words each; older messages drop out). Bigger = longer chats, but more
#              GPU memory: on 8 GB, qwen3:8b fits 8192. A change reloads the model (~5 s)
#
# A wrong value (not a number, voice or model not found, think with a model that can't) is skipped
# with a warning.
label: My profile
voice: af_heart
speed: 1.0
pause: 0.2
temperature: 0.7
silence: 0.45
think: off
# model: (not set = LLM_MODEL from local.env)
# context: (not set = NUM_CTX from local.env)

You are a warm, helpful voice assistant. Your replies are spoken aloud, so answer conversationally in short sentences. No markdown, lists, emojis or code blocks. Keep most answers to 1-3 sentences unless asked for more.
