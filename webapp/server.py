"""
webapp/server.py
────────────────────────────────────────────────────────────────────────────────
VideoVigil web dashboard – upload a video, get a people gallery.

    python run_web.py                 → opens http://127.0.0.1:8000
    python run_web.py --no-browser    (host / port / folders: config.yaml → web)
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

from core.enrollment import EnrollmentError
from core.open_set import is_iso_date
from webapp.pipeline import JobManager, date_in_name, web_settings

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
logging.getLogger("werkzeug").setLevel(logging.WARNING)
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

    @app.get("/")
    def index():
        return send_from_directory(WEB, "index.html")

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
            # The date in the original name is read before secure_filename strips CHIRLA's ':'s;
            # it is only a hint – the staff gallery follows the date the user chose.
            job = jobs.create(name, file.save, video_date=request.form.get("video_date"),
                              filename_date=date_in_name(file.filename))
        except ValueError as exc:
            return error(str(exc), 400)
        logger.info("Upload: %s (%.1f MB, recorded %s) → job %s", name, job["size_mb"],
                    job["video_date"], job["id"])
        return jsonify(job), 201

    @app.get("/api/galleries")
    def galleries():
        return jsonify({"days": jobs.galleries(), "default": jobs.known.default_staff_day()})

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

    @app.get("/api/jobs/<job_id>/enroll")
    def enroll_preview(job_id):
        try:
            return jsonify(jobs.enroll_preview(job_id))
        except KeyError:
            return error("That analysis doesn't exist (any more).", 404)
        except RuntimeError as exc:
            return error(str(exc), 409)
        except EnrollmentError as exc:
            return error(str(exc), 400)

    @app.post("/api/jobs/<job_id>/enroll")
    def enroll(job_id):
        body = request.get_json(silent=True) or {}
        try:
            summary = jobs.enroll(job_id, body.get("assignments"), str(body.get("mode") or "merge"))
        except KeyError:
            return error("That analysis doesn't exist (any more).", 404)
        except RuntimeError as exc:
            return error(str(exc), 409)
        except EnrollmentError as exc:
            return error(str(exc), 400)
        except OSError as exc:
            logger.exception("Writing the staff gallery failed")
            return error(f"The staff gallery could not be written ({exc.strerror or exc}). "
                         f"The previous gallery was left in place – close anything using it and try again.", 500)
        return jsonify(summary)

    @app.get("/api/known")
    def known():
        day = request.args.get("day") or None
        if day is not None and not is_iso_date(day):
            return error("Use a date like 2023-06-02.", 400)
        listing = jobs.known.listing(day)
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
        d = jobs.known.staff_dir(label, request.args.get("day") or None)
        # absolute: Flask resolves a relative folder against webapp/, not the working directory
        return send_from_directory(d.resolve(), file, max_age=86400) if d else error("Not found", 404)

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

    os.chdir(ROOT)
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
