"""Windows lifecycle manager for the local Jev/Decider installation."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
STATE = ROOT.parent
CONFIG = json.loads((ROOT / "installation.json").read_text(encoding="utf-8"))
STOP = ROOT / "paused"
LOG = STATE / "local-service.log"
BACKEND = "http://127.0.0.1:8000"
GATEWAY = "http://127.0.0.1:8790"


def request(url, post=False):
    try:
        req = urllib.request.Request(url, data=b"" if post else None)
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def log(message):
    with LOG.open("a", encoding="utf-8") as file:
        file.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")


def occupied(port):
    with socket.socket() as connection:
        connection.settimeout(.3)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def gateway_start():
    current = request(GATEWAY + "/health")
    if current:
        if current.get("jev") != "typesafe" or current.get("upstream") != CONFIG["upstream"]:
            raise RuntimeError("Port 8790 belongs to an unexpected gateway; no process was changed")
        return
    if occupied(8790):
        raise RuntimeError("Port 8790 is occupied; no process was changed")
    with LOG.open("ab") as file:
        result = subprocess.run([CONFIG["node"], CONFIG["gateway_launcher"], "--start"],
            stdout=file, stderr=file, creationflags=subprocess.CREATE_NO_WINDOW, timeout=20)
    if result.returncode or not request(GATEWAY + "/health"):
        raise RuntimeError("Jev Gateway failed to start; see local-service.log")


def stop_backend():
    health = request(BACKEND + "/health")
    if health and health.get("service") == "jev-local-decider":
        try:
            os.kill(int(health["pid"]), signal.SIGTERM)
        except ProcessLookupError:
            pass


def supervise():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    mutex = kernel.CreateMutexW(None, False, "Local\\JevLocalDecider-" + os.environ.get("USERNAME", "user"))
    if not mutex:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        kernel.CloseHandle(mutex)
        return
    child = None
    ready_before = False
    recycle_requested = False
    attempts = []
    try:
        if STOP.exists():
            return
        gateway_start()
        request(GATEWAY + "/dashboard/routing?enabled=false", post=True)
        log("Supervisor started")
        while not STOP.exists():
            gateway_start()
            health = request(BACKEND + "/health")
            if health and health.get("service") != "jev-local-decider":
                raise RuntimeError("Port 8000 belongs to an unexpected service; no process was changed")
            planned_recycle = bool(health and health.get("phase") == "recycle_pending")
            recycle_requested = recycle_requested or planned_recycle
            if health and health.get("phase") in ("failed", "recycle_pending"):
                log("Deep idle: recycling the GPU process into a fresh CPU-only backend" if planned_recycle else "Backend reported failure; restarting it")
                request(GATEWAY + "/dashboard/routing?enabled=false", post=True)
                stop_backend()
                health = None
                time.sleep(1)
            if not health and not (child and child.poll() is None):
                if occupied(8000):
                    raise RuntimeError("Port 8000 is occupied; no process was changed")
                attempts = [] if recycle_requested else [at for at in attempts if time.monotonic() - at < 600]
                if len(attempts) >= 3:
                    raise RuntimeError("Backend failed three times in ten minutes; automatic restart stopped")
                if not recycle_requested:
                    attempts.append(time.monotonic())
                ready_before = False
                request(GATEWAY + "/dashboard/routing?enabled=false", post=True)
                env = os.environ.copy()
                env.update(FLA_CACHE_RESULTS="0", FLA_CACHE_MODE="disabled", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                           TRITON_CACHE_DIR=CONFIG["cache"], JEV_LOCAL_PRECISION=CONFIG.get("precision", "int8"),
                           JEV_LOCAL_IDLE_SECONDS=str(CONFIG.get("idle_seconds", 10)),
                           JEV_LOCAL_DEEP_IDLE_SECONDS=str(CONFIG.get("deep_idle_seconds", 30)))
                with (STATE / "decider.log").open("ab") as file:
                    child = subprocess.Popen([CONFIG["python"], "-u", "-m", "uvicorn", "local_rocm_server:app",
                        "--app-dir", str(ROOT), "--host", "127.0.0.1", "--port", "8000", "--no-access-log"], cwd=CONFIG["decider_dir"],
                        env=env, stdin=subprocess.DEVNULL, stdout=file, stderr=file,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                log("Backend started; loading Decider into RAM")
                recycle_requested = False
            if health and health.get("ready") and not ready_before:
                request(GATEWAY + "/dashboard/routing?enabled=true", post=True)
                log("Backend ready; routing enabled with GPU on demand")
                ready_before = True
            for _ in range(10):
                if STOP.exists():
                    break
                time.sleep(1)
        request(GATEWAY + "/dashboard/routing?enabled=false", post=True)
        stop_backend()
        log("Local GPU stopped; gateway retained for normal Codex use")
    except Exception as exc:
        log(type(exc).__name__ + ": " + str(exc))
        request(GATEWAY + "/dashboard/routing?enabled=false", post=True)
        raise
    finally:
        kernel.CloseHandle(mutex)


def start():
    STOP.unlink(missing_ok=True)
    gateway_start()
    with LOG.open("ab") as file:
        subprocess.Popen([CONFIG["pythonw"], str(Path(__file__).resolve()), "run"], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=file, stderr=file, creationflags=subprocess.CREATE_NO_WINDOW)
    print("Jev started. Decider v10 loads into RAM; GPU wakes on demand, offloads after 10 seconds and fully releases after 30-40 seconds idle.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="status", choices=["start", "stop", "restart", "status", "run", "logs", "dashboard"])
    action = parser.parse_args().action
    if action == "run":
        supervise()
    elif action == "start":
        start()
    elif action == "logs":
        for name in ("local-service.log", "decider.log", "codex.log"):
            path = STATE / name
            print("\n" + str(path))
            if path.exists():
                print("\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
    elif action == "dashboard":
        webbrowser.open(GATEWAY + "/dashboard")
        print(GATEWAY + "/dashboard")
    elif action in ("stop", "restart"):
        STOP.touch()
        request(GATEWAY + "/dashboard/routing?enabled=false", post=True)
        # Give the supervisor time to finish and release its singleton mutex.
        for _ in range(15):
            if not request(BACKEND + "/health"):
                break
            time.sleep(1)
        if request(BACKEND + "/health"):
            stop_backend()
        if action == "restart":
            time.sleep(2)
            start()
        else:
            print("Local Jev stopped. Codex still works through the passthrough gateway.")
    else:
        print(json.dumps({"paused": STOP.exists(), "gateway": request(GATEWAY + "/health"),
                          "decider": request(BACKEND + "/health")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
