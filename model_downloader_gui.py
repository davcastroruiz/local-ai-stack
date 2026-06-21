"""
Unified model downloader for Trellis2 GGUF and addon model packs.

Supports:
1. Built-in profiles for known model groups.
2. Custom URL entries for downloading any additional model files locally.

Custom URL entry format (one per line):
  https://example.com/model.bin
  https://example.com/model.bin | custom/path/model.bin
  https://example.com/model.bin | custom/path/model.bin | Friendly Label
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, messagebox, ttk
from urllib.parse import unquote, urlparse


def _ensure_requests() -> None:
    try:
        import requests  # noqa: F401
    except ImportError:
        print("[INFO] 'requests' not found, installing now...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "requests"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print("[INFO] 'requests' installed successfully.\n")


_ensure_requests()

import requests


CHUNK_SIZE = 8 * 1024 * 1024
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "downloader_config.json"


def hf_resolve_url(repo_id: str, relative_path: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{relative_path}"


TRELLIS_REPO = "Aero-Ex/Trellis2-GGUF"
TRELLIS_FILES = [
    ("pipeline.json", "Pipeline Config"),
    ("refiner/ss_flow_img_dit_1_3B_64_bf16.json", "Refiner Config"),
    ("refiner/ss_flow_img_dit_1_3B_64_bf16_Q4_K_M.gguf", "Refiner Model (Q4_K_M)"),
    ("shape/slat_flow_img2shape_dit_1_3B_512_bf16.json", "Shape 512 Config"),
    ("shape/slat_flow_img2shape_dit_1_3B_512_bf16_Q4_K_M.gguf", "Shape 512 Model (Q4_K_M)"),
    ("shape/slat_flow_img2shape_dit_1_3B_1024_bf16.json", "Shape 1024 Config"),
    ("shape/slat_flow_img2shape_dit_1_3B_1024_bf16_Q4_K_M.gguf", "Shape 1024 Model (Q4_K_M)"),
    ("texture/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json", "Texture 512 Config"),
    ("texture/slat_flow_imgshape2tex_dit_1_3B_512_bf16_Q4_K_M.gguf", "Texture 512 Model (Q4_K_M)"),
    ("texture/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json", "Texture 1024 Config"),
    ("texture/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_Q4_K_M.gguf", "Texture 1024 Model (Q4_K_M)"),
    ("decoders/Stage1/ss_dec_conv3d_16l8_fp16.json", "SS Decoder Config"),
    ("decoders/Stage1/ss_dec_conv3d_16l8_fp16.safetensors", "SS Decoder Model"),
    ("decoders/Stage2/shape_dec_next_dc_f16c32_fp16.json", "Shape Decoder Config"),
    ("decoders/Stage2/shape_dec_next_dc_f16c32_fp16.safetensors", "Shape Decoder Model"),
    ("decoders/Stage2/tex_dec_next_dc_f16c32_fp16.json", "Texture Decoder Config"),
    ("decoders/Stage2/tex_dec_next_dc_f16c32_fp16.safetensors", "Texture Decoder Model"),
    ("encoders/shape_enc_next_dc_f16c32_fp16.json", "Shape Encoder Config"),
    ("encoders/shape_enc_next_dc_f16c32_fp16.safetensors", "Shape Encoder Model"),
]


BUILTIN_PROFILES = [
    {
        "key": "trellis2_gguf",
        "title": "Trellis2 GGUF core pack",
        "target_subdir": "trellis_gguf",
        "default_enabled": True,
        "items": [
            {
                "relative_path": rel_path,
                "label": label,
                "url": hf_resolve_url(TRELLIS_REPO, rel_path),
            }
            for rel_path, label in TRELLIS_FILES
        ],
    },
    {
        "key": "hymotion",
        "title": "HyMotion model pack",
        "target_subdir": "",
        "default_enabled": False,
        "items": [
            {
                "relative_path": "hymotion/HY-Motion-1.0/latest.ckpt",
                "label": "HY-Motion-1.0 Full",
                "url": "https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/hymotion/HY-Motion-1.0/latest.ckpt",
            },
            {
                "relative_path": "hymotion/HY-Motion-1.0-Lite/latest.ckpt",
                "label": "HY-Motion-1.0-Lite",
                "url": "https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/hymotion/HY-Motion-1.0-Lite/latest.ckpt",
            },
            {
                "relative_path": "text_encoders/clip-vit-large-patch14.safetensors",
                "label": "CLIP ViT Large Patch14",
                "url": "https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/text_encoders/clip-vit-large-patch14.safetensors",
            },
            {
                "relative_path": "text_encoders/Qwen3-8B-Q8_0.gguf",
                "label": "Qwen3-8B GGUF Q8_0",
                "url": "https://huggingface.co/Aero-Ex/Hy-Motion1.0/resolve/main/text_encoders/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf",
            },
        ],
    },
    {
        "key": "ultrashape",
        "title": "UltraShape model",
        "target_subdir": "",
        "default_enabled": False,
        "items": [
            {
                "relative_path": "ultrashape/ultrashape_v1.pt",
                "label": "UltraShape v1",
                "url": "https://huggingface.co/infinith/UltraShape/resolve/main/ultrashape_v1.pt",
            }
        ],
    },
]

PROFILE_BY_KEY = {profile["key"]: profile for profile in BUILTIN_PROFILES}


def load_config() -> dict:
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_config(data: dict) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def normalize_default_folder(raw_folder: str) -> str:
    text = raw_folder.strip()
    if not text:
        return ""

    path = Path(text)
    if path.name.lower() == "trellis_gguf":
        return str(path.parent)
    return str(path)


def guess_relative_path(url: str) -> str:
    parsed = urlparse(url)
    decoded_path = unquote(parsed.path)
    marker = "/resolve/main/"
    if marker in decoded_path:
        return decoded_path.split(marker, 1)[1].lstrip("/")

    filename = Path(decoded_path).name or "download.bin"
    return f"custom/{filename}"


def parse_custom_entries(raw_text: str) -> tuple[list[dict], list[str]]:
    items: list[dict] = []
    errors: list[str] = []

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part.strip() for part in line.split("|")]
        if not parts:
            continue

        url = parts[0]
        if not (url.startswith("http://") or url.startswith("https://")):
            errors.append(f"Line {line_no}: URL must start with http:// or https://")
            continue

        relative_path = parts[1] if len(parts) > 1 and parts[1] else guess_relative_path(url)
        relative_path = relative_path.replace("\\", "/").lstrip("/")
        if not relative_path:
            errors.append(f"Line {line_no}: relative path is empty")
            continue

        label = parts[2] if len(parts) > 2 and parts[2] else Path(relative_path).name
        items.append(
            {
                "relative_path": relative_path,
                "label": label,
                "url": url,
            }
        )

    return items, errors


def build_download_plan(root_folder: str, selected_profile_keys: list[str], custom_text: str) -> tuple[list[dict], list[str], list[str]]:
    root_dir = Path(root_folder)
    plan: list[dict] = []
    warnings: list[str] = []
    seen_destinations: set[str] = set()

    for key in selected_profile_keys:
        profile = PROFILE_BY_KEY.get(key)
        if not profile:
            continue

        target_subdir = profile["target_subdir"].strip("/")
        for item in profile["items"]:
            relative_path = item["relative_path"].replace("\\", "/").lstrip("/")
            if target_subdir:
                relative_path = f"{target_subdir}/{relative_path}"

            destination = (root_dir / relative_path).resolve()
            destination_key = str(destination).lower()
            if destination_key in seen_destinations:
                warnings.append(f"Duplicate destination skipped: {relative_path}")
                continue

            seen_destinations.add(destination_key)
            plan.append(
                {
                    "profile": profile["title"],
                    "label": item["label"],
                    "url": item["url"],
                    "relative_path": relative_path,
                    "destination": destination,
                }
            )

    custom_items, custom_errors = parse_custom_entries(custom_text)
    for item in custom_items:
        destination = (root_dir / item["relative_path"]).resolve()
        destination_key = str(destination).lower()
        if destination_key in seen_destinations:
            warnings.append(f"Duplicate destination skipped: {item['relative_path']}")
            continue

        seen_destinations.add(destination_key)
        plan.append(
            {
                "profile": "Custom URLs",
                "label": item["label"],
                "url": item["url"],
                "relative_path": item["relative_path"],
                "destination": destination,
            }
        )

    return plan, warnings, custom_errors


def _download_worker(download_plan: list[dict], msg_queue: queue.Queue, cancel_event: threading.Event) -> None:
    def put(tag: str, *args):
        msg_queue.put((tag, *args))

    total = len(download_plan)
    downloaded_count = 0
    skipped_count = 0
    failed: list[str] = []

    for idx, item in enumerate(download_plan, start=1):
        if cancel_event.is_set():
            put("log", "[CANCELLED] Download stopped by user.")
            put("done", total, downloaded_count, skipped_count, failed, True)
            return

        profile = item["profile"]
        label = item["label"]
        url = item["url"]
        destination: Path = item["destination"]
        relative_path = item["relative_path"]
        tmp_destination = Path(str(destination) + ".tmp")

        put("log", f"[{idx:2d}/{total}] [{profile}] {label} -> {relative_path}")

        if destination.exists() and destination.stat().st_size > 0:
            size_mb = destination.stat().st_size / (1024 * 1024)
            if size_mb >= 1:
                put("log", f"  already exists ({size_mb:.1f} MB) - skipped")
            else:
                put("log", "  already exists - skipped")
            skipped_count += 1
            put("progress", idx, total)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        success = False

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_bytes = int(response.headers.get("content-length", 0))
            received = 0

            with tmp_destination.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if cancel_event.is_set():
                        break
                    if not chunk:
                        continue

                    file_handle.write(chunk)
                    received += len(chunk)
                    if total_bytes:
                        pct = received / total_bytes * 100
                        done_mb = received / (1024 * 1024)
                        total_mb = total_bytes / (1024 * 1024)
                        put("progress_file", pct, done_mb, total_mb)

            if cancel_event.is_set():
                tmp_destination.unlink(missing_ok=True)
                put("log", "  cancelled - partial file removed")
                put("done", total, downloaded_count, skipped_count, failed, True)
                return

            tmp_destination.replace(destination)
            size_mb = destination.stat().st_size / (1024 * 1024)
            if size_mb >= 1:
                put("log", f"  done ({size_mb:.1f} MB)")
            else:
                put("log", "  done")

            downloaded_count += 1
            success = True

        except requests.exceptions.HTTPError as exc:
            put("log", f"  HTTP error: {exc}")
        except requests.exceptions.ConnectionError as exc:
            put("log", f"  connection error: {exc}")
        except requests.exceptions.Timeout:
            put("log", "  timeout")
        except Exception as exc:
            put("log", f"  failed: {exc}")
        finally:
            tmp_destination.unlink(missing_ok=True)

        if not success:
            failed.append(f"{profile} :: {relative_path}")

        put("progress", idx, total)

    put("done", total, downloaded_count, skipped_count, failed, False)


class DownloaderApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Trellis2 + Addon Model Downloader")
        self.resizable(True, True)
        self.minsize(820, 680)

        self._queue: queue.Queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        cfg = load_config()
        self._folder_var = tk.StringVar(value=normalize_default_folder(cfg.get("last_folder", "")))

        selected_from_config = set(cfg.get("selected_profiles", []))
        self._profile_vars: dict[str, tk.BooleanVar] = {}
        for profile in BUILTIN_PROFILES:
            if selected_from_config:
                enabled = profile["key"] in selected_from_config
            else:
                enabled = bool(profile["default_enabled"])
            self._profile_vars[profile["key"]] = tk.BooleanVar(value=enabled)

        self._build_ui(cfg.get("custom_entries", ""))
        self._poll_queue()

    def _build_ui(self, custom_entries: str) -> None:
        pad = 10

        folder_frame = ttk.LabelFrame(self, text="Model Root Folder (Host)", padding=pad)
        folder_frame.pack(fill="x", padx=pad, pady=(pad, 4))

        self._folder_entry = ttk.Entry(folder_frame, textvariable=self._folder_var, font=("Consolas", 10))
        self._folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ttk.Button(folder_frame, text="Browse...", command=self._browse).pack(side="left")

        profile_frame = ttk.LabelFrame(self, text="Built-in Profiles", padding=pad)
        profile_frame.pack(fill="x", padx=pad, pady=4)

        for profile in BUILTIN_PROFILES:
            key = profile["key"]
            count = len(profile["items"])
            text = f"{profile['title']} ({count} files)"
            ttk.Checkbutton(profile_frame, text=text, variable=self._profile_vars[key]).pack(anchor="w")

        custom_frame = ttk.LabelFrame(self, text="Custom URL Entries (optional)", padding=pad)
        custom_frame.pack(fill="both", expand=False, padx=pad, pady=4)

        ttk.Label(
            custom_frame,
            text="Format: url | relative/path | label   (relative/path and label are optional)",
            foreground="#666666",
        ).pack(anchor="w", pady=(0, 4))

        self._custom_text = tk.Text(custom_frame, height=8, wrap="none", font=("Consolas", 9))
        self._custom_text.pack(fill="x", expand=True)
        if custom_entries:
            self._custom_text.insert("1.0", custom_entries)

        custom_btns = ttk.Frame(custom_frame)
        custom_btns.pack(fill="x", pady=(6, 0))
        ttk.Button(custom_btns, text="Load URL List File...", command=self._load_custom_file).pack(side="left")
        ttk.Button(custom_btns, text="Clear", command=self._clear_custom_entries).pack(side="left", padx=(6, 0))

        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill="x", padx=pad, pady=(4, 0))

        self._progress_label = ttk.Label(progress_frame, text="Ready")
        self._progress_label.pack(anchor="w")

        self._progress_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate",
            maximum=1,
        )
        self._progress_bar.pack(fill="x", pady=(2, 0))

        self._file_progress_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
        )
        self._file_progress_bar.pack(fill="x", pady=(2, 0))

        self._file_progress_label = ttk.Label(progress_frame, text="", font=("Consolas", 9))
        self._file_progress_label.pack(anchor="w")

        output_frame = ttk.LabelFrame(self, text="Output", padding=4)
        output_frame.pack(fill="both", expand=True, padx=pad, pady=4)

        mono = font.Font(family="Consolas", size=9)
        self._output = tk.Text(
            output_frame,
            wrap="none",
            state="disabled",
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            font=mono,
            relief="flat",
        )
        self._output.pack(side="left", fill="both", expand=True)

        scrollbar_y = ttk.Scrollbar(output_frame, orient="vertical", command=self._output.yview)
        scrollbar_y.pack(side="right", fill="y")
        self._output.configure(yscrollcommand=scrollbar_y.set)

        scrollbar_x = ttk.Scrollbar(self, orient="horizontal", command=self._output.xview)
        scrollbar_x.pack(fill="x", padx=pad)
        self._output.configure(xscrollcommand=scrollbar_x.set)

        self._output.tag_configure("ok", foreground="#4ec9b0")
        self._output.tag_configure("skip", foreground="#808080")
        self._output.tag_configure("error", foreground="#f48771")
        self._output.tag_configure("header", foreground="#dcdcaa")
        self._output.tag_configure("summary_ok", foreground="#6a9955")
        self._output.tag_configure("summary_fail", foreground="#f48771")

        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=pad, pady=(2, pad))

        self._start_btn = ttk.Button(button_frame, text="Start Download", command=self._start)
        self._start_btn.pack(side="left", padx=(0, 6))

        self._cancel_btn = ttk.Button(button_frame, text="Cancel", command=self._cancel, state="disabled")
        self._cancel_btn.pack(side="left")

        ttk.Label(
            button_frame,
            text=f"Profiles: {len(BUILTIN_PROFILES)} built-in + custom URL support",
            foreground="#808080",
        ).pack(side="right")

    def _browse(self) -> None:
        chosen = filedialog.askdirectory(
            title="Select host model root folder",
            initialdir=self._folder_var.get() or str(Path.home()),
        )
        if chosen:
            self._folder_var.set(chosen)

    def _load_custom_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select URL list file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not selected:
            return

        try:
            content = Path(selected).read_text(encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("File error", f"Could not read file:\n{exc}")
            return

        if self._custom_text.get("1.0", "end").strip():
            self._custom_text.insert("end", "\n")
        self._custom_text.insert("end", content)

    def _clear_custom_entries(self) -> None:
        self._custom_text.delete("1.0", "end")

    def _start(self) -> None:
        folder = self._folder_var.get().strip()
        if not folder:
            messagebox.showerror("No folder", "Please select a model root folder first.")
            return

        root_dir = Path(folder)
        try:
            root_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Folder error", f"Cannot create folder:\n{exc}")
            return

        selected_profiles = [key for key, var in self._profile_vars.items() if var.get()]
        custom_text = self._custom_text.get("1.0", "end").strip()
        plan, warnings, custom_errors = build_download_plan(str(root_dir), selected_profiles, custom_text)

        if custom_errors:
            messagebox.showerror("Custom URL errors", "\n".join(custom_errors[:10]))
            return

        if not plan:
            messagebox.showerror(
                "No downloads",
                "No download items selected. Enable a profile or add custom URL entries.",
            )
            return

        save_config(
            {
                "last_folder": str(root_dir),
                "selected_profiles": selected_profiles,
                "custom_entries": custom_text,
            }
        )

        self._cancel_event.clear()
        self._progress_bar["maximum"] = len(plan)
        self._progress_bar["value"] = 0
        self._file_progress_bar["value"] = 0
        self._file_progress_label.config(text="")
        self._progress_label.config(text="Starting...")

        self._clear_output()
        self._log(f"Unified model downloader - {len(plan)} files queued", tag="header")
        self._log(f"Target root: {root_dir}", tag="header")
        self._log("=" * 70, tag="header")

        for warning in warnings:
            self._log(f"WARNING: {warning}", tag="error")

        self._start_btn.config(state="disabled")
        self._cancel_btn.config(state="normal")

        self._worker_thread = threading.Thread(
            target=_download_worker,
            args=(plan, self._queue, self._cancel_event),
            daemon=True,
        )
        self._worker_thread.start()

    def _cancel(self) -> None:
        self._cancel_event.set()
        self._cancel_btn.config(state="disabled")
        self._progress_label.config(text="Cancelling...")

    def _poll_queue(self) -> None:
        try:
            while True:
                message = self._queue.get_nowait()
                tag = message[0]

                if tag == "log":
                    text = message[1]
                    log_tag = None
                    lower_text = text.lower()
                    if "done" in lower_text and "failed" not in lower_text:
                        log_tag = "ok"
                    if "skipped" in lower_text:
                        log_tag = "skip"
                    if "error" in lower_text or "failed" in lower_text or "cancelled" in lower_text:
                        log_tag = "error"
                    self._log(text, tag=log_tag)

                elif tag == "progress":
                    step = message[1]
                    total = message[2]
                    self._progress_bar["value"] = step
                    pct = int(step / total * 100) if total else 0
                    self._progress_label.config(text=f"File {step} / {total} ({pct}%)")
                    self._file_progress_bar["value"] = 0
                    self._file_progress_label.config(text="")

                elif tag == "progress_file":
                    pct = message[1]
                    done_mb = message[2]
                    total_mb = message[3]
                    self._file_progress_bar["value"] = pct
                    self._file_progress_label.config(
                        text=f"  {done_mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%)"
                    )

                elif tag == "done":
                    total, downloaded, skipped, failed, cancelled = message[1:]
                    self._on_done(total, downloaded, skipped, failed, cancelled)

        except queue.Empty:
            pass

        self.after(60, self._poll_queue)

    def _on_done(self, total: int, downloaded: int, skipped: int, failed: list[str], cancelled: bool) -> None:
        self._start_btn.config(state="normal")
        self._cancel_btn.config(state="disabled")
        self._file_progress_bar["value"] = 0
        self._file_progress_label.config(text="")

        self._log("")
        self._log("=" * 70, tag="header")

        if cancelled:
            self._progress_label.config(text="Cancelled")
            self._log("Download cancelled by user.", tag="error")
            return

        if not failed:
            self._progress_bar["value"] = total
            self._progress_label.config(text=f"Complete - {downloaded} downloaded, {skipped} skipped")
            self._log(
                f"All files ready. {downloaded} downloaded, {skipped} skipped.",
                tag="summary_ok",
            )
            return

        self._progress_label.config(text=f"Completed with {len(failed)} failures")
        self._log(
            f"Downloaded: {downloaded}, skipped: {skipped}, failed: {len(failed)}",
            tag="summary_fail",
        )
        for item in failed:
            self._log(f"  - {item}", tag="error")

    def _log(self, text: str, tag: str | None = None) -> None:
        self._output.configure(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}\n"
        if tag:
            self._output.insert("end", line, tag)
        else:
            self._output.insert("end", line)
        self._output.configure(state="disabled")
        self._output.see("end")

    def _clear_output(self) -> None:
        self._output.configure(state="normal")
        self._output.delete("1.0", "end")
        self._output.configure(state="disabled")


if __name__ == "__main__":
    app = DownloaderApp()
    app.mainloop()
