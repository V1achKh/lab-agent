"""Песочница для кода, который пишет модель (режим «универсальный агент»).

Код модели никогда не запускается «как есть». Он выполняется в изоляции:
  * нет сети                      — нельзя ничего скачать или отправить наружу;
  * система только для чтения     — нельзя ничего сломать или удалить;
  * домашняя папка не видна       — нельзя прочитать ~/.ssh, документы, пароли;
  * писать можно только в /work   — отдельная папка этого запуска;
  * лимиты памяти, CPU, процессов и времени — нельзя «повесить» компьютер.

Бэкенды (LAB_SANDBOX):
  podman / docker — контейнер. Образ LAB_SANDBOX_IMAGE:
      lab-sandbox — свой образ (собрать: python -m labagent sandbox build);
      host        — без своего образа: в пустой контейнер подключаются системные папки
                    хоста только для чтения (удобно, если образ не собрать).
  bwrap           — bubblewrap, лёгкая песочница (на ней работает Flatpak).
  auto            — podman → docker → bwrap, что найдётся.
  none            — БЕЗ изоляции, только для отладки; включается лишь явно.

Внутри песочницы:
  /work  — рабочая папка (чтение и запись), сюда сохраняют графики и таблицы;
  /data  — результаты simulate_circuit (circuit.json, sim.json) только для чтения;
  /app   — пакет labagent только для чтения (можно импортировать решатель).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import config

SANDBOX = os.environ.get("LAB_SANDBOX", "auto")
IMAGE = os.environ.get("LAB_SANDBOX_IMAGE", "lab-sandbox")
TIMEOUT = int(os.environ.get("LAB_SANDBOX_TIMEOUT", "60"))
MEMORY = os.environ.get("LAB_SANDBOX_MEMORY", "1g")
HOST_IMAGE = "lab-sandbox-empty"  # пустой образ для режима host
CONTAINERFILE_DIR = config.ROOT / "sandbox"


class SandboxError(RuntimeError):
    pass


# ---------------------------------------------------------------- выбор бэкенда

def _image_exists(engine: str, image: str) -> bool:
    try:
        r = subprocess.run([engine, "image", "inspect", image], capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def _engine_works(engine: str) -> bool:
    if not shutil.which(engine):
        return False
    try:
        return subprocess.run([engine, "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def backend() -> str | None:
    if SANDBOX != "auto":
        return SANDBOX
    for engine in ("podman", "docker"):
        if _engine_works(engine) and (IMAGE == "host" or _image_exists(engine, IMAGE)):
            return engine
    if shutil.which("bwrap"):
        return "bwrap"
    return None


def describe() -> str:
    b = backend()
    if b in ("podman", "docker"):
        return f"{b}, образ {IMAGE}"
    return b or "нет"


# ---------------------------------------------------------------- подготовка образа

def build_image(engine: str | None = None) -> str:
    engine = engine or next((e for e in ("podman", "docker") if _engine_works(e)), None)
    if not engine:
        raise SandboxError("не найден podman или docker (sudo pacman -S podman)")
    if IMAGE == "host":
        _ensure_empty_image(engine)
        return f"{engine}: пустой образ {HOST_IMAGE} готов (режим host)"
    r = subprocess.run([engine, "build", "-t", IMAGE, "-f", str(CONTAINERFILE_DIR / "Containerfile"),
                        str(CONTAINERFILE_DIR)], text=True)
    if r.returncode != 0:
        raise SandboxError("сборка образа не удалась")
    return f"{engine}: образ {IMAGE} собран"


def _ensure_empty_image(engine: str):
    if _image_exists(engine, HOST_IMAGE):
        return
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for d in ("work", "data", "app", "tmp", "usr", "etc", "dev", "proc", "sys"):
            info = tarfile.TarInfo(d)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tar.addfile(info)
        # в Arch и Ubuntu /bin, /lib, /lib64 — ссылки на /usr/...; без них не найдётся загрузчик программ
        for link in ("bin", "sbin", "lib", "lib64"):
            if os.path.islink("/" + link):
                info = tarfile.TarInfo(link)
                info.type = tarfile.SYMTYPE
                info.linkname = os.readlink("/" + link)
                tar.addfile(info)
    r = subprocess.run([engine, "import", "-", HOST_IMAGE], input=buf.getvalue(), capture_output=True)
    if r.returncode != 0:
        raise SandboxError(f"не удалось создать пустой образ: {r.stderr.decode()[-300:]}")


# ---------------------------------------------------------------- команды запуска

def _host_ro_mounts() -> list[tuple[str, str]]:
    """Системные папки хоста (только чтение) для режима host и для bwrap."""
    mounts = []
    for p in ("/usr", "/etc", "/bin", "/sbin", "/lib", "/lib64", "/opt"):
        if os.path.exists(p) and not os.path.islink(p):
            mounts.append((p, p))
    prefix = sys.prefix  # виртуальное окружение с numpy, matplotlib (часто лежит в /home)
    if not prefix.startswith(("/usr", "/opt")):
        mounts.append((prefix, prefix))
        base = getattr(sys, "base_prefix", prefix)
        if base != prefix and not base.startswith(("/usr", "/opt")):
            mounts.append((base, base))
    return mounts


def _python_in_box(engine_image: bool) -> str:
    return "python" if engine_image else sys.executable


def container_cmd(engine: str, name: str, work: Path, data: Path | None) -> list[str]:
    cmd = [engine, "run", "--rm", "--name", name,
           "--network=none", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
           f"--memory={MEMORY}", "--cpus=1", "--pids-limit=128",
           "--cap-drop=ALL", "--security-opt=no-new-privileges",
           "-v", f"{work}:/work:rw",
           "-v", f"{config.ROOT / 'labagent'}:/app/labagent:ro",
           "-w", "/work",
           "-e", "PYTHONPATH=/app", "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp",
           "-e", "PYTHONDONTWRITEBYTECODE=1"]
    if data:
        cmd += ["-v", f"{data}:/data:ro"]
    if engine == "podman":
        cmd += ["--userns=keep-id"]          # файлы в /work будут принадлежать вам
    else:
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    if IMAGE == "host":
        for src, dst in _host_ro_mounts():
            cmd += ["-v", f"{src}:{dst}:ro"]
        cmd += [HOST_IMAGE, sys.executable, "/work/_main.py"]
    else:
        cmd += [IMAGE, "python", "/work/_main.py"]
    return cmd


def bwrap_cmd(work: Path, data: Path | None) -> list[str]:
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
           "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    for src, dst in _host_ro_mounts():
        cmd += ["--ro-bind", src, dst]
    for link in ("/bin", "/sbin", "/lib", "/lib64"):
        if os.path.islink(link):
            cmd += ["--symlink", os.readlink(link), link]
    cmd += ["--bind", str(work), "/work",
            "--ro-bind", str(config.ROOT / "labagent"), "/app/labagent"]
    if data:
        cmd += ["--ro-bind", str(data), "/data"]
    cmd += ["--setenv", "PATH", "/usr/bin:/bin", "--setenv", "PYTHONPATH", "/app",
            "--setenv", "HOME", "/tmp", "--setenv", "MPLCONFIGDIR", "/tmp",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--chdir", "/work", sys.executable, "/work/_main.py"]
    return cmd


def _limits():
    """Лимиты для процесса bwrap/none (у контейнеров свои)."""
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT + 5, TIMEOUT + 5))
    resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (100 << 20, 100 << 20))


# ---------------------------------------------------------------- запуск

def run_code(code: str, work: Path, data: Path | None = None, timeout: int | None = None) -> dict:
    """Выполнить код модели. Возвращает stdout, stderr, код выхода и список новых файлов в /work."""
    timeout = timeout or TIMEOUT
    b = backend()
    if b is None:
        raise SandboxError("песочница не найдена: установите podman (sudo pacman -S podman) или "
                           "bubblewrap (sudo pacman -S bubblewrap). Без изоляции код модели не запускается.")
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "_main.py").write_text(code, encoding="utf-8")
    before = {p: p.stat().st_mtime for p in work.rglob("*") if p.is_file()}
    name = f"labsbx-{uuid.uuid4().hex[:8]}"
    if b in ("podman", "docker"):
        if IMAGE == "host":
            _ensure_empty_image(b)
        cmd, preexec = container_cmd(b, name, work, data), None
    elif b == "bwrap":
        cmd, preexec = bwrap_cmd(work, data), _limits
    elif b == "none":
        cmd, preexec = [sys.executable, str(work / "_main.py")], _limits
    else:
        raise SandboxError(f"неизвестная песочница: {b}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              preexec_fn=preexec, cwd=work,
                              env=None if b != "none" else {**os.environ, "MPLCONFIGDIR": str(work)})
        out, err, code_ = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        if b in ("podman", "docker"):
            subprocess.run([b, "kill", name], capture_output=True)
        out = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err, code_ = f"превышено время выполнения ({timeout} с) — процесс остановлен", -9
    files = sorted(str(p.relative_to(work)) for p in work.rglob("*")
                   if p.is_file() and p.name != "_main.py" and before.get(p) != p.stat().st_mtime)
    return {"backend": b, "returncode": code_, "stdout": out, "stderr": err,
            "files": files, "seconds": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- проверка изоляции

PROBES = [
    ("код выполняется", "print(2 + 2)", lambda r: r["returncode"] == 0 and "4" in r["stdout"]),
    ("запись в /work разрешена", "open('/work/ok.txt','w').write('1'); print('ok')",
     lambda r: r["returncode"] == 0 and "ok" in r["stdout"]),
    ("нет сети", "import socket\ns=socket.socket(); s.settimeout(3)\ntry:\n    s.connect(('1.1.1.1',80)); print('CONNECTED')\n"
                 "except OSError as e:\n    print('blocked', e)",
     lambda r: "CONNECTED" not in r["stdout"]),
    ("личные файлы не видны", "import os\nh={home!r}\nprint(sorted(os.listdir(h)) if os.path.isdir(h) else [])",
     lambda r: r["returncode"] == 0 and set(eval(r["stdout"].strip() or "[]")) <= _ALLOWED_IN_HOME),
    ("система только для чтения", "try:\n    open('/usr/hacked','w'); print('WROTE')\nexcept OSError as e:\n    print('denied')",
     lambda r: "WROTE" not in r["stdout"]),
    ("бесконечный цикл остановлен", "while True: pass", lambda r: r["returncode"] != 0),
]


# в домашней папке может «просвечивать» только путь к виртуальному окружению проекта
_home = Path.home()
try:
    _ALLOWED_IN_HOME = {Path(sys.prefix).relative_to(_home).parts[0]}
except ValueError:
    _ALLOWED_IN_HOME = set()


def self_test(tmp: Path) -> list[tuple[str, bool, str]]:
    results = []
    for title, code, check in PROBES:
        code = code.replace("{home!r}", repr(str(_home)))
        timeout = 5 if "while True" in code else 60
        try:
            r = run_code(code, tmp, timeout=timeout)
            ok = check(r)
            detail = (r["stdout"] or r["stderr"]).strip().splitlines()[-1:] or [""]
            results.append((title, ok, detail[0][:80]))
        except SandboxError as exc:
            results.append((title, False, str(exc)))
            break
    return results
