"""Small client for a local Ollama server (https://ollama.com). Uses only the standard library."""
import json
import os
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


class LLMUnavailable(Exception):
    pass


def _post(path, payload, timeout):
    req = urllib.request.Request(OLLAMA_URL + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise LLMUnavailable(f"Ollama returned {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}")
    except (urllib.error.URLError, OSError) as e:
        raise LLMUnavailable(f"Ollama is not reachable at {OLLAMA_URL} ({e})")


def installed_models():
    """Names of models installed in Ollama, or None if Ollama isn't running."""
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return None


def has_model(name, models):
    if models is None:
        return False
    return any(m == name or m == name + ":latest" or m.split(":")[0] == name for m in models)


def generate_raw(model, prompt, on_token=None, stop=None, max_tokens=1000, timeout=900):
    """Plain text completion with NO chat template applied (raw=True). Streams tokens to on_token."""
    payload = {"model": model, "prompt": prompt, "raw": True, "stream": True, "keep_alive": "30m",
               "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096,
                           "stop": stop or []}}
    out = []
    with _post("/api/generate", payload, timeout) as r:
        for line in r:
            if not line.strip():
                continue
            msg = json.loads(line)
            if msg.get("error"):
                raise LLMUnavailable(msg["error"])
            piece = msg.get("response", "")
            if piece:
                out.append(piece)
                if on_token:
                    on_token(piece)
            if msg.get("done"):
                break
    return "".join(out)


def chat_json(model, messages, schema, max_tokens=500, temperature=0.4, timeout=300):
    """Chat completion constrained to a JSON schema. Returns the parsed object."""
    payload = {"model": model, "messages": messages, "stream": False, "format": schema, "keep_alive": "30m",
               "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096}}
    with _post("/api/chat", payload, timeout) as r:
        data = json.load(r)
    if data.get("error"):
        raise LLMUnavailable(data["error"])
    content = data.get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise LLMUnavailable("assistant model returned invalid JSON")
