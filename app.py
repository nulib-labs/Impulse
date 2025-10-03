import os
import uuid
import shlex
from pathlib import Path
from typing import List, Dict, Any, Optional

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from fireworks.core.launchpad import LaunchPad
from fireworks.utilities.filepad import FilePad
from fireworks.core.firework import Firework, Workflow
from fireworks.user_objects.firetasks.script_task import PyTask

try:
    from fabric import Connection
except Exception:
    Connection = None  # Fabric not installed; remote qlaunch will be skipped

ALLOWED_EXTENSIONS = {"jp2", "jpg", "jpeg", "png", "tif", "tiff"}
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_ROOT = BASE_DIR / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

CONN_STR = os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")
if not CONN_STR:
    raise RuntimeError("MONGODB_OCR_DEVELOPMENT_CONN_STRING is not set.")

lp = LaunchPad(
    host=CONN_STR,
    port=27017,
    uri_mode=True,
    name="fireworks",
    logdir=str(BASE_DIR / "logs"),
)

fp = FilePad(
    host=(CONN_STR + "/fireworks?"),
    port=27017,
    uri_mode=True,
    database="fireworks",
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def trigger_remote_qlaunch(
    batch_id: str, fw_ids: List[int], ensure_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    SSH into a remote host and run qlaunch to submit jobs.
    - Sources shell profiles so uv (and PATH) are available.
    - Exports Mongo URI so auxiliary can connect on remote.
    - Optionally ensures a writable output dir exists (ensure_dir).
    - Runs from project workdir.
    Env:
      QREMOTE_HOST, QREMOTE_USER, QREMOTE_PORT, QREMOTE_KEY, QREMOTE_PASSWORD,
      QREMOTE_WORKDIR (default /projects/p32234/projects/aerith/Impulse),
      QREMOTE_OUTPUT_BASE (default /projects/p32234/projects/aerith/Impulse/uploads),
      QREMOTE_QLAUNCH_CMD (default 'uv run qlaunch singleshot')
    """
    if not os.getenv("QREMOTE_HOST"):
        return {"enabled": False, "reason": "QREMOTE_HOST not set"}

    if Connection is None:
        return {"enabled": False, "reason": "fabric not installed"}

    host = os.getenv("QREMOTE_HOST")
    user = os.getenv("QREMOTE_USER")
    port = int(os.getenv("QREMOTE_PORT", "22"))
    key = os.getenv("QREMOTE_KEY")
    password = os.getenv("QREMOTE_PASSWORD")
    workdir = os.getenv("QREMOTE_WORKDIR", "/projects/p32234/projects/aerith/Impulse")
    qlaunch_cmd = os.getenv("QREMOTE_QLAUNCH_CMD", "uv run qlaunch singleshot")

    connect_kwargs: Dict[str, Any] = {}
    if key:
        connect_kwargs["key_filename"] = key
    if password:
        connect_kwargs["password"] = password

    # cd target (keep ~ unquoted for expansion)
    if not workdir or not workdir.strip():
        cd_target = "$HOME"
    elif "~" in workdir:
        cd_target = workdir
    else:
        cd_target = f'"{workdir}"'

    # Export Mongo URI into the remote environment for this command
    mongo_uri = os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING", "")
    export_env = ""
    if mongo_uri:
        export_env = (
            f"export MONGODB_OCR_DEVELOPMENT_CONN_STRING={shlex.quote(mongo_uri)}; "
        )

    mkcmd = f"mkdir -p {shlex.quote(ensure_dir)}; " if ensure_dir else ""

    # Source profiles, export env, ensure output dir, cd to project, run
    inner_cmd = (
        "source ~/.bash_profile >/dev/null 2>&1 || true; "
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        "source ~/.profile >/dev/null 2>&1 || true; "
        f"{export_env}"
        f"{mkcmd}"
        f"cd {cd_target} && {qlaunch_cmd}"
    )
    bash_cmd = f"bash -lc {shlex.quote(inner_cmd)}"

    try:
        conn = Connection(
            host=host, user=user, port=port, connect_kwargs=connect_kwargs
        )
        result = conn.run(bash_cmd, pty=True, hide=True, warn=True)
        return {
            "enabled": True,
            "ok": result.ok,
            "command": bash_cmd,
            "stdout": result.stdout.strip() if result.stdout else "",
            "stderr": result.stderr.strip() if result.stderr else "",
        }
    except Exception as e:
        return {"enabled": True, "ok": False, "error": str(e)}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        flash("Please select at least one image file.")
        return redirect(url_for("index"))

    # Optional barcode provided by user
    barcode_raw = request.form.get("barcode", "").strip()
    barcode = secure_filename(barcode_raw) if barcode_raw else ""

    # Group this upload as one batch -> one Firework
    batch_id = str(uuid.uuid4())
    # local path grouped by barcode if provided
    batch_dir = UPLOAD_ROOT / (barcode if barcode else "") / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Remote output directory to avoid PermissionError on cluster nodes
    remote_out_base = os.getenv(
        "QREMOTE_OUTPUT_BASE", "/projects/p32234/projects/aerith/Impulse/uploads"
    )
    remote_batch_dir = (
        f"{remote_out_base}/{barcode}/{batch_id}"
        if barcode
        else f"{remote_out_base}/{batch_id}"
    )
    use_remote_dir = bool(os.getenv("QREMOTE_HOST"))

    saved_paths: List[Path] = []
    identifiers: List[str] = []

    # Save and register files locally (inputs come from FilePad remotely)
    for f in files:
        if f and allowed_file(f.filename):
            fn = secure_filename(f.filename)
            dest = batch_dir / fn
            f.save(dest)
            saved_paths.append(dest)

            # Add to FilePad and collect the identifier
            _, identifier = fp.add_file(str(dest), identifier=str(uuid.uuid4()))
            identifiers.append(identifier)

    if not identifiers:
        flash("No valid images were uploaded. Allowed: jp2, jpg, jpeg, png, tif, tiff.")
        return redirect(url_for("index"))

    # Create Firework with same task chain; barcode_dir points to remote when using remote qlaunch
    fw = Firework(
        [
            PyTask(
                func="auxiliary.image_conversion_task",
                inputs=["identifiers", "barcode_dir"],
                outputs="converted_images",
            ),
            PyTask(
                func="auxiliary.image_to_pdf",
                inputs=["converted_images", "barcode_dir"],
                outputs="PDF_id",
            ),
            PyTask(func="auxiliary.marker_on_pdf", inputs=["PDF_id"]),
        ],
        spec={
            "identifiers": identifiers,
            "barcode_dir": remote_batch_dir if use_remote_dir else str(batch_dir),
            "barcode": barcode,
            "batch_id": batch_id,
        },
        name=f"OCR Firework - {batch_id}",
    )
    wf = Workflow([fw])
    fw_ids = lp.add_wf(wf)

    # Trigger remote qlaunch (ensure remote output dir exists)
    qlaunch_info = trigger_remote_qlaunch(
        batch_id=batch_id,
        fw_ids=fw_ids,
        ensure_dir=remote_batch_dir if use_remote_dir else None,
    )

    return render_template(
        "success.html",
        batch_id=batch_id,
        barcode=barcode,
        file_count=len(saved_paths),
        fw_ids=fw_ids,
        upload_dir=str(batch_dir),
        output_dir=(remote_batch_dir if use_remote_dir else str(batch_dir)),
        qlaunch_info=qlaunch_info,
    )


@app.route("/monitor", methods=["GET"])
def monitor():
    # States considered "in-progress"/queued
    states = request.args.get("states")
    if states:
        states = [s.strip().upper() for s in states.split(",") if s.strip()]
    else:
        states = ["RUNNING", "RESERVED", "READY", "WAITING"]

    query = {"state": {"$in": states}}
    try:
        fw_ids = lp.get_fw_ids(query=query, sort=[("updated_on", -1)], limit=200)
    except TypeError:
        fw_ids = lp.get_fw_ids(query=query, limit=200)

    items = []
    for fid in fw_ids:
        d = lp.get_fw_dict_by_id(fid)
        items.append(
            {
                "fw_id": d.get("fw_id", fid),
                "name": d.get("name", ""),
                "state": d.get("state", ""),
                "created_on": d.get("created_on"),
                "updated_on": d.get("updated_on"),
                "launches": d.get("launches", []),
            }
        )

    return render_template("monitor.html", items=items, states=states)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5600, debug=True)
