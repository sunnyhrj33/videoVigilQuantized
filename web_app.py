"""
web_app.py
────────────────────────────────────────────────────────────────────────────────
VideoVigil web dashboard – upload a video, get a people gallery.

    python web_app.py                 → opens http://127.0.0.1:8000
    python web_app.py --no-browser    (host / port / folders: config.yaml → web)

A simplified front end for non-technical users: they upload a video and get the
annotated video, a gallery of every person in it (recognised / unrecognised /
watch list) and an activity timeline, and can teach the system new people.
Settings are fixed to the recommended ones – there are no model, tracker or
threshold controls. The full operator console is app.py.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import webbrowser
from pathlib import Path

import yaml
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from web_pipeline import JobManager, web_settings

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
logging.getLogger("werkzeug").setLevel(logging.WARNING)       # the page polls; keep the log readable
logger = logging.getLogger("web_app")


def create_app(cfg: dict) -> Flask:
    settings = web_settings(cfg)
    jobs = JobManager(cfg)
    app = Flask(__name__, static_folder=str(WEB / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = int(settings["max_upload_mb"]) * 2**20
    app.config["JOBS"] = jobs

    def error(message: str, code: int):
        return jsonify({"error": message}), code

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        if exc.code == 413:
            return error(f"That file is too big – the limit is {settings['max_upload_mb']} MB.", 413)
        if request.path.startswith("/api/"):
            return error(exc.description or exc.name, exc.code or 500)
        return exc

    # ── page ─────────────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        return send_from_directory(WEB, "index.html")

    # ── API ──────────────────────────────────────────────────────────────────

    @app.get("/api/status")
    def status():
        return jsonify({**jobs.status(), "detector": settings["detector"],
                        "reid": cfg.get("reid", {}).get("model"), "max_upload_mb": settings["max_upload_mb"]})

    @app.get("/api/jobs")
    def list_jobs():
        return jsonify(jobs.list())

    @app.post("/api/jobs")
    def upload():
        file = request.files.get("video")
        if file is None or not file.filename:
            return error("Choose a video file to upload.", 400)
        name = secure_filename(file.filename) or "video.mp4"
        try:
            job = jobs.create(name, file.save)
        except ValueError as exc:
            return error(str(exc), 400)
        logger.info("Upload: %s (%.1f MB) → job %s", name, job["size_mb"], job["id"])
        return jsonify(job), 201

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        job = jobs.get(job_id)
        return jsonify(job) if job else error("That analysis doesn't exist (any more).", 404)

    @app.delete("/api/jobs/<job_id>")
    def delete_job(job_id):
        try:
            jobs.delete(job_id)
        except KeyError:
            return error("That analysis doesn't exist (any more).", 404)
        except RuntimeError as exc:
            return error(str(exc), 409)
        return jsonify({"ok": True})

    @app.post("/api/jobs/<job_id>/people/<pid>/remember")
    def remember(job_id, pid):
        name = (request.get_json(silent=True) or {}).get("name", "")
        try:
            person = jobs.remember_person(job_id, pid, name)
        except KeyError:
            return error("That person doesn't exist (any more).", 404)
        except ValueError as exc:
            return error(str(exc), 400)
        return jsonify(person)

    @app.get("/api/known")
    def known():
        listing = jobs.known.listing()
        gallery = jobs.suspects()
        listing["watchlist"] = [{"id": s.suspect_id, "name": s.label.title(), "reason": s.reason,
                                 "source": s.source, "created": s.created, "sightings": len(s.sightings),
                                 "photos": [p.name for p in gallery.crops(s.suspect_id)]}
                                for s in gallery.suspects.values()]
        return jsonify(listing)

    @app.delete("/api/known/<path:name>")
    def forget(name):
        try:
            jobs.known.remove(name)
        except KeyError:
            return error(f"{name} isn't on your list.", 404)
        return jsonify({"ok": True})

    @app.delete("/api/watchlist/<sid>")
    def unwatch(sid):
        try:
            jobs.delete_suspect(sid)
        except KeyError:
            return error("That watch-list entry doesn't exist (any more).", 404)
        except RuntimeError as exc:
            return error(str(exc), 409)
        return jsonify({"ok": True})

    # ── media ────────────────────────────────────────────────────────────────

    @app.get("/media/jobs/<job_id>/<path:file>")
    def job_file(job_id, file):
        d = jobs.job_dir(job_id)
        if d is None:
            return error("Not found", 404)
        return send_from_directory(d, file, max_age=3600)

    @app.get("/media/known/<path:name>/<file>")
    def known_photo(name, file):
        d = jobs.known.person_dir(name)
        return send_from_directory(d, file) if d else error("Not found", 404)

    @app.get("/media/staff/<label>/<file>")
    def staff_photo(label, file):
        d = jobs.known.staff_dir(label)
        return send_from_directory(d, file, max_age=86400) if d else error("Not found", 404)

    @app.get("/media/watchlist/<sid>/<file>")
    def watch_photo(sid, file):
        return send_from_directory(jobs.suspects().dir / sid, file)

    return app


def main():
    ap = argparse.ArgumentParser(description="VideoVigil web dashboard")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    os.chdir(ROOT)                     # config paths (models/, enrolled_gallery/, output/) are repo-relative
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    settings = web_settings(cfg)
    host, port = args.host or settings["host"], args.port or int(settings["port"])

    app = create_app(cfg)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    logger.info("VideoVigil dashboard → %s  (AI models are loading in the background)", url)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
