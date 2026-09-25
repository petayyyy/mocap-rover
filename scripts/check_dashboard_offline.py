#!/usr/bin/env python3
"""Browser smoke check using saved files only. Does not connect to Gazebo."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.dashboard import DashboardHandler, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default="artifacts/post-reboot/live_static")
    parser.add_argument("--output", default="artifacts/post-reboot/dashboard_offline")
    args = parser.parse_args()
    source = Path(args.recording)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads((source / "status.json").read_text())
    saved["phase"] = "OFFLINE_UI_FIXTURE"
    DashboardHandler.status_provider = staticmethod(lambda: saved)
    DashboardHandler.cameras_provider = staticmethod(
        lambda: [
            {"camera_id": cid, "channels": data}
            for cid, data in saved["cameras"].items()
        ]
    )
    DashboardHandler.preview_files_provider = staticmethod(
        lambda: {cid: str(source / (cid + ".jpg")) for cid in saved["cameras"]}
    )
    DashboardHandler.settings_provider = staticmethod(
        lambda: {"role": "offline_fixture", "cameras": []}
    )
    server = serve("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        raise RuntimeError("Chrome/Chromium unavailable")
    try:
        with tempfile.TemporaryDirectory(prefix="mocap-browser-") as profile:
            proc = subprocess.Popen(
                [
                    chrome,
                    "--headless",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    f"--user-data-dir={profile}",
                    f'--screenshot={output / "dashboard.png"}',
                    "--window-size=1440,1500",
                    "--virtual-time-budget=3000",
                    "--dump-dom",
                    f"http://127.0.0.1:{server.server_port}/",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                dom, errors = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                raise
            (output / "page.html").write_text(dom)
            (output / "browser.log").write_text(errors)
            if proc.returncode:
                raise RuntimeError("browser failed; see browser.log")
            if (
                "<polyline" not in dom
                or "opponent</text>" not in dom
                or "tag_rover</text>" not in dom
            ):
                raise AssertionError("both rover positions were not rendered")
            if (
                len(re.findall(r'<img[^>]+src="[^"]*/preview/camera_[1-6][^"]*"', dom))
                != 6
            ):
                raise AssertionError("missing camera slots")
            report = {
                "scope": "offline_browser_fixture_only",
                "browser_exit_code": proc.returncode,
                "two_tracks_rendered": True,
                "six_preview_slots": True,
                "screenshot": str(output / "dashboard.png"),
                "live_gazebo_tested": False,
            }
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
