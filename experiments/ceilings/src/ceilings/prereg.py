"""Hash-locked pre-registration (research R11)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

PREREG_DIR = Path("prereg")


class PreregHashMismatch(RuntimeError):
    """The frozen pre-registration no longer matches its recorded hash (CLI exit code 3)."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(
    config_path: Path | str = PREREG_DIR / "config.json", prereg_dir: Path | str = PREREG_DIR
) -> str:
    """Write config_frozen.json (canonical JSON) and HASH; return the hash."""
    prereg_dir = Path(prereg_dir)
    config = json.loads(Path(config_path).read_text())
    frozen = prereg_dir / "config_frozen.json"
    frozen.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    digest = _sha(frozen)
    (prereg_dir / "HASH").write_text(digest + "\n")
    return digest


def check_hash(prereg_dir: Path | str = PREREG_DIR) -> str:
    prereg_dir = Path(prereg_dir)
    frozen, hash_file = prereg_dir / "config_frozen.json", prereg_dir / "HASH"
    if not frozen.exists() or not hash_file.exists():
        raise PreregHashMismatch("pre-registration is not frozen; run `ceilings prereg freeze`")
    digest = _sha(frozen)
    if digest != hash_file.read_text().strip():
        raise PreregHashMismatch("prereg/config_frozen.json does not match prereg/HASH")
    return digest


def current_hash(prereg_dir: Path | str = PREREG_DIR) -> str:
    """Frozen hash if frozen and valid; otherwise the hash of the draft config (dev runs only)."""
    prereg_dir = Path(prereg_dir)
    try:
        return check_hash(prereg_dir)
    except PreregHashMismatch:
        draft = prereg_dir / "config.json"
        return _sha(draft) if draft.exists() else "0" * 64


def load_frozen(prereg_dir: Path | str = PREREG_DIR) -> dict:
    check_hash(prereg_dir)
    return json.loads((Path(prereg_dir) / "config_frozen.json").read_text())


def log_deviation(
    section: str,
    planned: str,
    actual: str,
    reason: str,
    impact: str,
    prereg_dir: Path | str = PREREG_DIR,
) -> None:
    row = (
        f"| {date.today().isoformat()} | {section} | {planned} | {actual} | {reason} | {impact} |\n"
    )
    with open(Path(prereg_dir) / "DEVIATIONS.md", "a") as f:
        f.write(row)
