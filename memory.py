"""
Long-term memory: short facts about the user, kept in a plain text file.

  - At start, the facts are added to the system prompt.
  - After each reply, the LLM checks (in the background) whether the user
    shared something worth remembering, or asked to forget something.
  - The file is plain Markdown (memory.md): read, edit or delete it any time.

Turn off with MEMORY=0 (then the file is neither read nor written).
"""
import json
import threading
from pathlib import Path

import requests

EXTRACT_PROMPT = (
    "You maintain a list of long-term facts about the user of a voice assistant. "
    "Look only at the user's latest message.\n\n"
    "add: lasting personal facts the user states about themselves, or anything they "
    "ask you to remember. One short fact per item, in third person. Examples:\n"
    '  "My name is Alex and I am a nurse" -> ["Name is Alex", "Works as a nurse"]\n'
    '  "We got a cat, a Siamese named Luna" -> ["Has a Siamese cat named Luna"]\n'
    '  "Remember my mom\'s birthday is June 2" -> ["Mom\'s birthday is June 2"]\n'
    '  "Hi, how are you?" -> []\n'
    "Each fact must make sense on its own (keep details together with what they describe).\n"
    "Never add questions, small talk, general knowledge, or what the assistant said.\n\n"
    "remove: copy an existing fact exactly, ONLY if the user asks to forget it or says "
    "it is no longer true (then add the new fact). Questions and other topics never "
    "remove anything; in that case remove is [].\n\n"
    'Reply with JSON only: {"add": [...], "remove": [...]}'
)

# Without an explicit request, a fact may only be removed when a new one replaces it
FORGET_WORDS = ("forget", "delete", "remove", "erase", "no longer", "not anymore",
                "don't remember", "do not remember")

FIRST_PERSON = {"i", "i'm", "im", "i've", "i'd", "i'll", "my", "mine", "myself",
                "we", "we're", "we've", "our", "ours"}

_lock = threading.Lock()


def _norm(text):
    return " ".join(text.lower().strip(" .").split())


def worth_checking(user_text):
    """Only statements about the user (or remember/forget requests) can hold facts.
    Small models invent facts from questions like "What is the capital of Japan?"."""
    text = user_text.lower()
    if "remember" in text or any(w in text for w in FORGET_WORDS):
        return True
    if text.rstrip().endswith("?"):
        return False
    words = {w.strip(".,!?;:\"") for w in text.split()}
    return bool(words & FIRST_PERSON)


class Memory:
    def __init__(self, path):
        self.path = Path(path)
        self.facts = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.startswith("- "):
                    self.facts.append(line[2:].strip())

    def prompt(self):
        """Text appended to the system prompt."""
        text = ("\n\nYou have long-term memory: facts the user shares are saved "
                "automatically. Don't mention it, unless the user asks you to "
                "remember or forget something; then just confirm briefly.")
        if self.facts:
            text += ("\nWhat you remember about the user (use it naturally, "
                     "don't list it back):\n" + "\n".join(f"- {f}" for f in self.facts))
        return text

    def _save(self):
        self.path.write_text(
            "# What the assistant remembers about you\n"
            "# Edit or delete lines freely; one fact per line starting with \"- \".\n\n"
            + "".join(f"- {f}\n" for f in self.facts),
            encoding="utf-8",
        )

    def apply(self, add, remove):
        """Apply changes; returns (added, removed) lists."""
        removed = []
        for r in remove:
            n = _norm(r)
            for f in list(self.facts):
                if n and (_norm(f) == n or n in _norm(f)):
                    self.facts.remove(f)
                    removed.append(f)
        known = {_norm(f) for f in self.facts}
        added = [a.strip() for a in add if a.strip() and _norm(a) not in known]
        self.facts += added
        if added or removed:
            self._save()
        return added, removed

    def update(self, user_text, url, model, options, on_change=None):
        """Ask the LLM what to remember from the user's message (blocking)."""
        current = "\n".join(f"- {f}" for f in self.facts) or "(none)"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": f"Current facts:\n{current}\n\n"
                                            f"User's latest message:\n{user_text}"},
            ],
            "stream": False,
            "format": "json",
            "keep_alive": -1,
            # same num_ctx as the chat, otherwise Ollama reloads the model
            "options": {**options, "temperature": 0, "num_predict": 200},
        }
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = json.loads(r.json()["message"]["content"])
            add = [str(x) for x in data.get("add", []) if x]
            remove = [str(x) for x in data.get("remove", []) if x]
        except (requests.RequestException, ValueError, KeyError, AttributeError) as e:
            print(f"\n🧠  memory update failed: {e}", flush=True)
            return
        # Small models sometimes "remove" facts on unrelated turns; guard against it
        if not add and not any(w in user_text.lower() for w in FORGET_WORDS):
            remove = []
        with _lock:
            added, removed = self.apply(add, remove)
        for f in added:
            print(f"🧠  remembered: {f}", flush=True)
        for f in removed:
            print(f"🧠  forgot: {f}", flush=True)
        if (added or removed) and on_change:
            on_change()

    def update_async(self, *args, **kwargs):
        threading.Thread(target=self.update, args=args, kwargs=kwargs, daemon=True).start()
