#!/usr/bin/env python3
"""Run a command in a resource-capped Docker container and kill it if the machine runs hot.

Runs on the host (macOS or Linux), standard library only. Local model servers such as Ollama run
outside Docker, so the container's own caps cannot limit them; this watchdog watches both.

It kills the container (and unloads the Ollama model) when any of these holds:
  - macOS reports a thermal warning or a CPU speed limit below 100% (`pmset -g therm`);
  - Ollama's processes use more than --ollama-cpu percent CPU for --sustain seconds;
  - the container uses more than --container-cpu percent CPU for --sustain seconds;
  - the run exceeds --max-minutes.
Every sample is logged to --log; the reason for any kill is logged and printed.

Example:
  python3 scripts/watchdog.py --name ceilings-smoke --model gemma3:1b --cpus 1.5 --memory 2g \
      --env OLLAMA_HOST=http://host.docker.internal:11434 -- bash scripts/local_smoke.sh
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Limits:
    ollama_cpu: float = 300.0  # percent of one core, summed over Ollama processes
    container_cpu: float = 175.0  # percent of one core
    sustain: float = 60.0  # seconds a CPU limit may be exceeded before the kill
    max_seconds: float = 3 * 3600


@dataclass
class Sample:
    t: float
    ollama_cpu: float
    container_cpu: float | None
    thermal: str  # "ok" or a description of the warning


@dataclass
class Monitor:
    """Decides when to kill. Pure logic, so it can be tested with made-up samples."""

    limits: Limits
    start: float
    over_since: dict = field(default_factory=dict)

    def check(self, s: Sample) -> str | None:
        if s.thermal != "ok":
            return f"thermal: {s.thermal}"
        if s.t - self.start > self.limits.max_seconds:
            return f"wall clock above {self.limits.max_seconds / 60:.0f} minutes"
        for name, value, limit in (("ollama cpu", s.ollama_cpu, self.limits.ollama_cpu),
                                   ("container cpu", s.container_cpu, self.limits.container_cpu)):
            if value is not None and value > limit:
                first = self.over_since.setdefault(name, s.t)
                if s.t - first >= self.limits.sustain:
                    return (f"{name} {value:.0f}% above {limit:.0f}% for "
                            f"{s.t - first:.0f}s")
            else:
                self.over_since.pop(name, None)
        return None


def thermal_state() -> str:
    """'ok', or the warning text. macOS only; other systems report 'ok'."""
    if sys.platform != "darwin":
        return "ok"
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "ok"
    for m in re.finditer(r"(CPU_Speed_Limit|GPU_Speed_Limit|CPU_Available_CPUs)\s*=\s*(\d+)",
                         out):
        if m.group(1).endswith("Speed_Limit") and int(m.group(2)) < 100:
            return f"{m.group(1)} = {m.group(2)}"
    for line in out.splitlines():
        low = line.lower()
        if "warning level" in low and not low.strip().startswith("note: no"):
            return line.strip()
    return "ok"


def ollama_cpu() -> float:
    try:
        out = subprocess.run(["ps", "-A", "-o", "%cpu=,comm="], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    total = 0.0
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "ollama" in parts[1].lower():
            try:
                total += float(parts[0])
            except ValueError:
                pass
    return total


def container_cpu(name: str) -> float | None:
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", name],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return float(out.rstrip("%")) if out else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def running(name: str) -> bool:
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                         capture_output=True, text=True).stdout.strip()
    return out == "true"


def exit_code(name: str) -> int:
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.ExitCode}}", name],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.lstrip("-").isdigit() else 1


def unload(model: str | None) -> None:
    if model:
        subprocess.run(["ollama", "stop", model], capture_output=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", default="ceilings-smoke")
    p.add_argument("--image", default="ceilings:dev")
    p.add_argument("--workdir", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--cpus", default="1.5")
    p.add_argument("--memory", default="2g")
    p.add_argument("--pids", default="256")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for the container")
    p.add_argument("--model", default=None, help="Ollama model to unload on exit or kill")
    p.add_argument("--ollama-cpu", type=float, default=Limits.ollama_cpu)
    p.add_argument("--container-cpu", type=float, default=Limits.container_cpu)
    p.add_argument("--sustain", type=float, default=Limits.sustain)
    p.add_argument("--max-minutes", type=float, default=Limits.max_seconds / 60)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--log", default=None)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("give the command to run after --")

    log_path = Path(a.log or Path(a.workdir) / "data" / f"{a.name}-watchdog.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")

    def say(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    if thermal_state() != "ok":
        say(f"not starting: {thermal_state()}")
        return 3
    subprocess.run(["docker", "rm", "-f", a.name], capture_output=True)
    docker = ["docker", "run", "-d", "--name", a.name, "--cpus", a.cpus, "--memory", a.memory,
              "--memory-swap", a.memory, "--pids-limit", a.pids,
              "-v", f"{a.workdir}:/work", "-w", "/work"]
    for kv in a.env:
        docker += ["-e", kv]
    started = subprocess.run(docker + [a.image] + cmd, capture_output=True, text=True)
    if started.returncode != 0:
        say(f"docker run failed: {started.stderr.strip()}")
        return 1
    say(f"started {a.name}: cpus={a.cpus} memory={a.memory} pids={a.pids}; limits: ollama "
        f"{a.ollama_cpu:.0f}%, container {a.container_cpu:.0f}% for {a.sustain:.0f}s, "
        f"{a.max_minutes:.0f} min; cmd: {' '.join(cmd)}")
    mon = Monitor(Limits(a.ollama_cpu, a.container_cpu, a.sustain, a.max_minutes * 60),
                  time.monotonic())
    last_log = 0.0
    try:
        while running(a.name):
            s = Sample(time.monotonic(), ollama_cpu(), container_cpu(a.name), thermal_state())
            if s.t - last_log >= 30:
                say(f"sample ollama_cpu={s.ollama_cpu:.0f}% container_cpu="
                    f"{s.container_cpu if s.container_cpu is not None else 'n/a'}% "
                    f"thermal={s.thermal}")
                last_log = s.t
            reason = mon.check(s)
            if reason:
                subprocess.run(["docker", "kill", a.name], capture_output=True)
                unload(a.model)
                say(f"KILLED: {reason}")
                return 3
            time.sleep(a.interval)
    except KeyboardInterrupt:
        subprocess.run(["docker", "kill", a.name], capture_output=True)
        unload(a.model)
        say("KILLED: interrupted")
        return 130
    code = exit_code(a.name)
    unload(a.model)
    tail = subprocess.run(["docker", "logs", "--tail", "40", a.name], capture_output=True,
                          text=True)
    say(f"finished with exit code {code}; last output:\n{tail.stdout}{tail.stderr}")
    return code


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
