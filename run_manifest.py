"""Write run_manifest.json for experiment reproducibility."""

from __future__ import annotations

import json
import random
import subprocess
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit_hash(repo_root: Optional[Path] = None) -> Optional[str]:
    root = repo_root or Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def apply_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _read_text(path: Path) -> Optional[str]:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return None


def build_config_snapshot(
    config_path: Path,
    personas_path: Path,
    prompts_path: Optional[Path] = None,
) -> Dict[str, str]:
    snapshot = {}
    text = _read_text(config_path)
    if text is not None:
        snapshot["config.yaml"] = text
    text = _read_text(personas_path)
    if text is not None:
        snapshot["personas.yaml"] = text
    if prompts_path is not None:
        text = _read_text(prompts_path)
        if text is not None:
            snapshot["prompts.yaml"] = text
    return snapshot


def _redact_llm_config(llm: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(llm)
    if "api_key" in out and out["api_key"] is not None:
        out["api_key"] = "***"
    return out


def create_run_manifest(
    *,
    run_name: str,
    run_index: int,
    phase: str,
    args: Namespace,
    config_path: Path,
    personas_path: Path,
    prompts_path: Path,
    random_seed: int,
    llm_config: Dict[str, Any],
    git_commit: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "run_index": run_index,
        "phase": phase,
        "status": "running",
        "started_at": _utc_now_iso(),
        "finished_at": None,
        "git_commit": git_commit or git_commit_hash(config_path.parent),
        "random_seed": random_seed,
        "llm": _redact_llm_config(llm_config),
        "cli_args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "config_path": str(config_path.resolve()),
        "personas_path": str(personas_path.resolve()),
        "prompts_path": str(prompts_path.resolve()),
        "config_snapshot": build_config_snapshot(config_path, personas_path, prompts_path),
        "error": None,
    }


def write_run_manifest(path: Path | str, manifest: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def finalize_run_manifest(manifest: Dict[str, Any], *, status: str = "completed", error: Optional[str] = None) -> Dict[str, Any]:
    manifest["status"] = status
    manifest["finished_at"] = _utc_now_iso()
    if error is not None:
        manifest["error"] = error
    return manifest
