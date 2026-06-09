"""Copyright © 2025-2026, Empa.

File upload element for large files with a progress bar.
"""

import tempfile
from pathlib import Path

from dash import Dash, Input, Output, clientside_callback, dcc, html
from flask import Response, jsonify, request
from werkzeug.utils import secure_filename

UPLOAD_DIR = Path(tempfile.gettempdir()) / "aurora_upload_tmp"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)


def upload_component() -> html.Div:
    """Upload element and filepath store."""
    return html.Div(
        [
            html.Iframe(
                src="/uploader-ui",
                style={"width": "100%", "height": "125px", "border": "none"},
            ),
            dcc.Store(id="upload-filepath"),
        ]
    )


def register_upload_callbacks(app: Dash) -> None:
    """Register callbacks for uploading files."""
    # Max file size 2 GB
    app.server.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

    @app.server.route("/uploader-ui")
    def uploader_ui() -> str:
        return """
<!DOCTYPE html>
<html>
<body style="margin:0;font-family:sans-serif;padding:8px;box-sizing:border-box">
    <div id="drop" style="
        border: 2px dashed #aaa;
        border-radius: 6px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
        transition: background 0.2s;
    ">
        Drop file here or <u>click to browse</u>
        <input type="file" id="f" style="display:none">
    </div>
    <progress id="p" value="0" max="100" style="width:100%;display:none;margin-top:6px"></progress>
    <div id="status" style="font-size:0.85em;margin-top:2px"></div>
    <script>
        let xhr;
        const drop = document.getElementById('drop');
        const input = document.getElementById('f');
        const statusEl = document.getElementById('status');

        drop.addEventListener('click', () => input.click());
        input.addEventListener('change', () => upload(input.files[0]));

        drop.addEventListener('dragover', e => {
            e.preventDefault();
            drop.style.background = '#f0f0f0';
        });
        drop.addEventListener('dragleave', () => drop.style.background = '');
        drop.addEventListener('drop', e => {
            e.preventDefault();
            drop.style.background = '';
            upload(e.dataTransfer.files[0]);
        });

        function upload(file) {
            if (!file) return;
            const MAX_BYTES = 2 * 1024 * 1024 * 1024; // 2 GB
            if (file.size > MAX_BYTES) {
                statusEl.textContent = 'Error: file too large (max 2 GB)';
                return;
            }
            statusEl.textContent = '⬆️ uploading ' + file.name + '...';
            p.style.display = 'block';
            p.value = 0;

            const fd = new FormData();
            fd.append('file', file);

            if (xhr) xhr.abort();
            xhr = new XMLHttpRequest();
            xhr.upload.onprogress = e => p.value = (e.loaded / e.total) * 100;
            xhr.onload = () => {
                const resp = JSON.parse(xhr.responseText);
                if (resp.error) {
                    statusEl.textContent = 'Error: ' + resp.error;
                    p.style.display = 'none';
                    return;
                }
                statusEl.textContent = '✅ uploaded: ' + file.name;
                window.parent.postMessage({uploadFilepath: resp.path}, '*');
            };
            xhr.onabort = () => { statusEl.textContent = ''; p.style.display = 'none'; p.value = 0; };
            xhr.onerror = () => { statusEl.textContent = 'Upload failed'; };
            xhr.open('POST', '/upload');
            xhr.send(fd);
        }
    </script>
</body>
</html>
"""

    @app.server.route("/upload", methods=["POST"])
    def upload() -> tuple[Response, int]:
        f = request.files["file"]
        filename = secure_filename(f.filename)
        if not filename:
            return jsonify({"error": "Invalid filename"}), 400

        path = UPLOAD_DIR / filename
        if not path.resolve().is_relative_to(UPLOAD_DIR.resolve()):
            return jsonify({"error": "Invalid path"}), 400

        f.save(path)
        return jsonify({"path": str(path)}), 200

    clientside_callback(
        """
        function(_) {
            window.addEventListener('message', e => {
                if (e.data && e.data.uploadFilepath) {
                    window.dash_clientside.set_props('upload-filepath', {data: e.data.uploadFilepath});
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output("upload-filepath", "data"),
        Input("upload-filepath", "id"),
    )
