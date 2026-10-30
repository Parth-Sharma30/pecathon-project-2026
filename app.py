import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3-flash-preview,gemini-2.5-flash"
).split(",") if m.strip()]

ALLOWED_EXT = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".cpp", ".c", ".h",
    ".java", ".cs", ".html", ".css", ".json", ".go", ".rs",
}
LANGS = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "javascript", ".tsx": "javascript", ".cpp": "cpp",
    ".c": "cpp", ".h": "cpp", ".cs": "csharp",
}
MAX_FILE_BYTES = 500 * 1024          # per file
MAX_CODE_CHARS = 120_000             # per transform request

client = genai.Client(api_key=API_KEY) if API_KEY else None

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # whole upload


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[\w+#-]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/upload")
def upload():
    """Receive files, validate them, return their text. Nothing is executed."""
    files, skipped = [], []
    for f in request.files.getlist("files"):
        name = os.path.basename(f.filename or "")
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXT:
            skipped.append(f"{name} (unsupported type)")
            continue
        raw = f.read()
        if len(raw) > MAX_FILE_BYTES:
            skipped.append(f"{name} (over 500 KB)")
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        files.append({
            "name": name,
            "path": name,
            "language": LANGS.get(ext, "python"),
            "content": text,
            "loc": len(text.splitlines()) or 1,
            "size": f"{len(raw) / 1024:.1f}",
        })
    return jsonify(files=files, skipped=skipped)


@app.post("/api/transform")
def transform():
    if client is None:
        return jsonify(error="GEMINI_API_KEY is missing in the .env file"), 500

    data = request.get_json(silent=True) or {}
    code = data.get("code", "")
    instruction = (data.get("instruction") or "").strip()
    filename = data.get("filename") or "file.txt"

    if not code.strip() or not instruction:
        return jsonify(error="Both code and instruction are required"), 400
    if len(code) > MAX_CODE_CHARS:
        return jsonify(error="File is too large for one request"), 413

    prompt = (
        "You are a world-class code refactoring engine.\n"
        "Modify the source code below according to the user's instruction.\n\n"
        "RULES:\n"
        "1. Output ONLY valid source code for the whole file.\n"
        "2. Do NOT use markdown fences.\n"
        "3. No explanations before or after the code.\n\n"
        f"USER INSTRUCTION:\n{instruction}\n\n"
        f"FILE NAME: {filename}\n"
        f"ORIGINAL CODE:\n{code}"
    )

    last_error = "Empty response from model"
    for model in MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = (resp.text or "").strip()
            if text:
                return jsonify(code=strip_fences(text), model=model)
        except Exception as exc:  # try next model
            last_error = str(exc)
    return jsonify(error=last_error), 502


# ---------------- GitHub repository fetch ----------------
GH_TOKEN = os.getenv("GITHUB_TOKEN")  # optional: private repos + higher rate limit
MAX_REPO_FILES = 20
MAX_REPO_FILE_BYTES = 200 * 1024
SKIP_PARTS = {"node_modules", "vendor", "dist", "build", ".git", "__pycache__",
              "venv", ".venv", "migrations", "static_collected"}
GH_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:github\.com/)?([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)"
    r"(?:\.git)?(?:/tree/([^/]+)(?:/(.+?))?)?/?$"
)


def gh_get(url: str, raw: bool = False):
    headers = {"User-Agent": "cyberscan-studio"}
    if not raw:
        headers["Accept"] = "application/vnd.github+json"
    if GH_TOKEN:
        headers["Authorization"] = f"{'token' if raw else 'Bearer'} {GH_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = resp.read()
    return body if raw else json.loads(body)


@app.post("/api/github")
def github_repo():
    """Fetch code files from a public GitHub repo (host is fixed to github.com)."""
    data = request.get_json(silent=True) or {}
    m = GH_URL_RE.match((data.get("url") or "").strip())
    if not m:
        return jsonify(error="Use a link like https://github.com/owner/repo"), 400
    owner, repo, branch, subpath = m.groups()

    try:
        if not branch:
            branch = gh_get(f"https://api.github.com/repos/{owner}/{repo}")["default_branch"]
        qb = urllib.parse.quote(branch, safe="")
        tree = gh_get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{qb}?recursive=1")

        cands = []
        for item in tree.get("tree", []):
            if item.get("type") != "blob":
                continue
            path = item["path"]
            size = item.get("size", 0)
            if subpath and not path.startswith(subpath.rstrip("/") + "/"):
                continue
            if os.path.splitext(path)[1].lower() not in ALLOWED_EXT or path.endswith(".min.js"):
                continue
            if set(path.split("/")[:-1]) & SKIP_PARTS:
                continue
            if size == 0 or size > MAX_REPO_FILE_BYTES:
                continue
            cands.append(item)
        cands.sort(key=lambda i: (i["path"].count("/"), i["size"]))
        picked = cands[:MAX_REPO_FILES]

        def load(item):
            try:
                raw = gh_get(
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{qb}/"
                    + urllib.parse.quote(item["path"], safe="/"), raw=True)
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                return None
            ext = os.path.splitext(item["path"])[1].lower()
            return {
                "name": os.path.basename(item["path"]),
                "path": item["path"],
                "language": LANGS.get(ext, "python"),
                "content": text,
                "loc": len(text.splitlines()) or 1,
                "size": f"{item['size'] / 1024:.1f}",
            }

        with ThreadPoolExecutor(max_workers=8) as pool:
            files = [f for f in pool.map(load, picked) if f]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            msg = "Repository or branch not found (private repos need GITHUB_TOKEN)"
        elif exc.code in (403, 429):
            msg = "GitHub rate limit reached. Wait a while or set GITHUB_TOKEN"
        else:
            msg = f"GitHub returned error {exc.code}"
        return jsonify(error=msg), 502
    except Exception:
        return jsonify(error="Could not reach GitHub"), 502

    if not files:
        return jsonify(error="No supported code files found in this repo"), 404
    return jsonify(repo=f"{owner}/{repo}", branch=branch, total=len(cands), files=files)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
