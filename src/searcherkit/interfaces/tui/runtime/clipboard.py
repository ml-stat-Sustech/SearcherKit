from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _copy_text_to_clipboard(text: str) -> str | None:
    if _copy_text_with_osc52(text):
        osc_destination = "OSC 52 clipboard"
    else:
        osc_destination = None
    command_destination = _copy_text_with_clipboard_command(text)
    if command_destination is not None:
        return command_destination
    pyperclip_destination = _copy_text_with_pyperclip(text)
    if pyperclip_destination is not None:
        return pyperclip_destination
    return osc_destination


def _copy_text_with_osc52(text: str) -> bool:
    if not sys.stdout.isatty():
        return False
    sequence = "\033]52;c;" + base64.b64encode(text.encode("utf-8")).decode("ascii") + "\a"
    try:
        if os.environ.get("TMUX") or os.environ.get("STY"):
            sys.stdout.write("\033Ptmux;\033" + sequence + "\033\\")
        else:
            sys.stdout.write(sequence)
        sys.stdout.flush()
    except OSError:
        return False
    return True


def _copy_text_with_pyperclip(text: str) -> str | None:
    try:
        import pyperclip
    except ImportError:
        return None
    try:
        pyperclip.copy(text)
    except pyperclip.PyperclipException:
        return None
    return "system clipboard"


def _copy_text_with_clipboard_command(text: str) -> str | None:
    commands: list[tuple[str, list[str]]] = []
    powershell_set_clipboard = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; Set-Clipboard -Value ([Console]::In.ReadToEnd())",
    ]
    if sys.platform == "darwin":
        commands.append(("system clipboard", ["pbcopy"]))
    elif sys.platform == "win32":
        commands.append(("system clipboard", powershell_set_clipboard))
        commands.append(("system clipboard", ["clip.exe"]))
    else:
        if os.environ.get("WAYLAND_DISPLAY"):
            commands.append(("system clipboard", ["wl-copy"]))
        commands.extend(
            [
                ("system clipboard", ["xclip", "-selection", "clipboard"]),
                ("system clipboard", ["xsel", "--clipboard", "--input"]),
            ]
        )
        if _looks_like_wsl():
            commands.append(("Windows clipboard", powershell_set_clipboard))
            commands.append(("Windows clipboard", ["clip.exe"]))
    for destination, command in commands:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(
                command,
                input=text,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        return destination
    return None


def _looks_like_wsl() -> bool:
    try:
        version = Path("/proc/version").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    lowered = version.lower()
    return "microsoft" in lowered or "wsl" in lowered
