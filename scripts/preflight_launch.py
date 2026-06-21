#!/usr/bin/env python3
"""Container startup preflight for Trellis2 GGUF on ComfyUI."""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

COMFY_DIR = Path("/root/ComfyUI")
CUSTOM_NODES_DIR = COMFY_DIR / "custom_nodes"
STAMPS_DIR = COMFY_DIR / "storage" / "preflight_stamps"
_INITIAL_SPARSE_CONV_BACKEND = os.environ.get("SPARSE_CONV_BACKEND")
_ALLOWED_SPARSE_CONV_BACKENDS = {"none", "spconv", "torchsparse", "flex_gemm"}
_TRUTHY_VALUES = {"1", "true", "yes", "on"}

if str(COMFY_DIR) not in sys.path:
    sys.path.insert(0, str(COMFY_DIR))

PROTECTED_REQUIREMENT_PINS_DEFAULT = (
    "torch,torchvision,torchaudio,numpy,transformers,"
    "onnxruntime,onnxruntime-gpu,huggingface-hub,huggingface_hub,"
    "triton,flash-attn,flash_attn"
)

REQUIRED_NODE_REPOS = [
    {
        "key": "trellis2",
        "name": "ComfyUI-Trellis2",
        "url": "https://github.com/visualbruno/ComfyUI-Trellis2.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-Trellis2",
        "requirements": "requirements.txt",
        "pip_args": [],
        "required": True,
    },
    {
        "key": "trellis2_gguf",
        "name": "ComfyUI-Trellis2-GGUF",
        "url": "https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-Trellis2-GGUF",
        "requirements": "requirements.txt",
        "pip_args": ["--no-deps"],
        "required": True,
    },
]

OPTIONAL_ADDON_REPOS = [
    {
        "key": "sam3",
        "name": "ComfyUI-SAM3",
        "url": "https://github.com/PozzettiAndrea/ComfyUI-SAM3.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-SAM3",
        "requirements": "requirements.txt",
        "pip_args": ["--no-deps"],
        # Keep SAM3 tokenizer/runtime imports explicit because optional addon
        # installs run with --no-deps in this stack.
        "extra_packages": ["comfy-env==0.2.61", "ftfy", "regex", "iopath", "portalocker"],
        "extra_pip_args": ["--no-deps"],
        "enable_env": "COMFYUI_ADDON_SAM3",
        "enabled_by_default": True,
    },
    {
        "key": "segvigen",
        "name": "ComfyUI-SegviGen",
        "url": "https://github.com/Aero-Ex/ComfyUI-SegviGen.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-SegviGen",
        "requirements": None,
        "pip_args": [],
        "extra_packages": [
            "pillow==12.0.0",
            "imageio==2.37.2",
            "imageio-ffmpeg==0.6.0",
            "tqdm==4.67.1",
            "easydict==1.13",
            "opencv-python-headless==4.12.0.88",
            "trimesh==4.10.1",
            "transformers==4.57.3",
            "zstandard==0.25.0",
            "kornia==0.8.2",
            "timm==1.0.22",
            "rembg",
            "fast_simplification",
        ],
        "extra_pip_args": ["--no-deps"],
        "run_install_py_env": "COMFYUI_RUN_SEGVIGEN_INSTALLER",
        "enable_env": "COMFYUI_ADDON_SEGVIGEN",
        "enabled_by_default": True,
    },
    {
        "key": "faithcontouring",
        "name": "ComfyUI-FaithContouring",
        "url": "https://github.com/krishnancr/ComfyUI-FaithContouring.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-FaithContouring",
        "requirements": None,
        "pip_args": [],
        "extra_packages": ["trimesh", "scipy", "einops"],
        "extra_pip_args": ["--no-deps"],
        "enable_env": "COMFYUI_ADDON_FAITHC",
        "enabled_by_default": True,
    },
    {
        "key": "pulse_meshaudit",
        "name": "ComfyUI-Pulse-MeshAudit",
        "url": "https://github.com/krishnancr/ComfyUI-Pulse-MeshAudit.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-Pulse-MeshAudit",
        "requirements": None,
        "pip_args": [],
        "extra_packages": ["comfy-env==0.2.7"],
        "extra_pip_args": ["--no-deps"],
        "enable_env": "COMFYUI_ADDON_PULSE_MESHAUDIT",
        "enabled_by_default": True,
    },
    {
        "key": "hymotion",
        "name": "ComfyUI-HyMotion",
        "url": "https://github.com/Aero-Ex/ComfyUI-HyMotion.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-HyMotion",
        "requirements": "requirements.txt",
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_HYMOTION",
        "enabled_by_default": True,
    },
    {
        "key": "hy_motion1",
        "name": "ComfyUI-HY-Motion1",
        "url": "https://github.com/jtydhr88/ComfyUI-HY-Motion1.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-HY-Motion1",
        "requirements": "requirements.txt",
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_HYMOTION",
        "enabled_by_default": True,
    },
    {
        "key": "ultrashape",
        "name": "ComfyUI-UltraShape",
        "url": "https://github.com/Rizzlord/ComfyUI-UltraShape.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-UltraShape",
        "requirements": ["UltraShape-1.0/requirements.txt"],
        "pip_args": ["--no-build-isolation"],
        "enable_env": "COMFYUI_ADDON_ULTRASHAPE",
        "enabled_by_default": True,
    },
    {
        "key": "memory_cleanup",
        "name": "Comfyui-Memory_Cleanup",
        "url": "https://github.com/LAOGOU-666/Comfyui-Memory_Cleanup.git",
        "path": CUSTOM_NODES_DIR / "Comfyui-Memory_Cleanup",
        "requirements": None,
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_MEMORY_CLEANUP",
        "enabled_by_default": True,
    },
    {
        "key": "geometrypack",
        "name": "ComfyUI-GeometryPack",
        "url": "https://github.com/PozzettiAndrea/ComfyUI-GeometryPack.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-GeometryPack",
        "requirements": "requirements.txt",
        "pip_args": [],
        "extra_packages": [
            "comfy-env==0.2.7",
            "comfy-dynamic-widgets",
            "pymeshfix>=0.17.0",
        ],
        "enable_env": "COMFYUI_ADDON_GEOMETRYPACK",
        "enabled_by_default": True,
    },
    {
        "key": "rmbg",
        "name": "ComfyUI-RMBG",
        "url": "https://github.com/1038lab/ComfyUI-RMBG.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-RMBG",
        "requirements": "requirements.txt",
        "pip_args": [],
        "required_modules": ["rembg"],
        "enable_env": "COMFYUI_ADDON_RMBG",
        "enabled_by_default": True,
    },
    {
        "key": "agsoft",
        "name": "comfyui-AGSoft",
        "url": "https://github.com/Art-xmaster/comfyui-AGSoft.git",
        "path": CUSTOM_NODES_DIR / "comfyui-AGSoft",
        "requirements": None,
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_AGSOFT",
        "enabled_by_default": True,
    },
    {
        "key": "cg_use_everywhere",
        "name": "cg-use-everywhere",
        "url": "https://github.com/chrisgoringe/cg-use-everywhere.git",
        "path": CUSTOM_NODES_DIR / "cg-use-everywhere",
        "requirements": None,
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_CG_USE_EVERYWHERE",
        "enabled_by_default": True,
    },
    {
        "key": "comfy_cup",
        "name": "ComfyUI-CUP",
        "url": "https://github.com/AIGODLIKE/ComfyUI-CUP.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-CUP",
        "requirements": None,
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_COMFY_CUP",
        "enabled_by_default": True,
    },
    {
        "key": "comfy_mtb",
        "name": "comfy_mtb",
        "url": "https://github.com/melMass/comfy_mtb.git",
        "path": CUSTOM_NODES_DIR / "comfy_mtb",
        "requirements": "requirements.txt",
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_COMFY_MTB",
        "enabled_by_default": True,
    },
    {
        "key": "essentials",
        "name": "ComfyUI_essentials",
        "url": "https://github.com/cubiq/ComfyUI_essentials.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI_essentials",
        "requirements": None,
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_ESSENTIALS",
        "enabled_by_default": True,
    },
    {
        "key": "allor",
        "name": "ComfyUI-Allor",
        "url": "https://github.com/Nourepide/ComfyUI-Allor.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-Allor",
        "requirements": None,
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_ALLOR",
        "enabled_by_default": True,
    },
    {
        "key": "kjnodes",
        "name": "ComfyUI-KJNodes",
        "url": "https://github.com/kijai/ComfyUI-KJNodes.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-KJNodes",
        "requirements": "requirements.txt",
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_KJNODES",
        "enabled_by_default": True,
    },
    {
        "key": "unload_model",
        "name": "ComfyUI-Unload-Model",
        "url": "https://github.com/SeanScripts/ComfyUI-Unload-Model.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI-Unload-Model",
        "requirements": None,
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_UNLOAD_MODEL",
        "enabled_by_default": True,
    },
    {
        "key": "rgthree",
        "name": "rgthree-comfy",
        "url": "https://github.com/rgthree/rgthree-comfy.git",
        "path": CUSTOM_NODES_DIR / "rgthree-comfy",
        "requirements": None,
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_RGTHREE",
        "enabled_by_default": True,
    },
    {
        "key": "was_suite",
        "name": "was-node-suite-comfyui",
        "url": "https://github.com/ltdrdata/was-node-suite-comfyui.git",
        "path": CUSTOM_NODES_DIR / "was-node-suite-comfyui",
        "requirements": "requirements.txt",
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_WAS_SUITE",
        "enabled_by_default": True,
    },
    {
        "key": "make_it_animatable",
        "name": "ComfyUI_Make-It-Animatable",
        "url": "https://github.com/speige/ComfyUI_Make-It-Animatable.git",
        "path": CUSTOM_NODES_DIR / "ComfyUI_Make-It-Animatable",
        "requirements": "requirements.txt",
        "pip_args": [],
        "enable_env": "COMFYUI_ADDON_MAKE_IT_ANIMATABLE",
        "enabled_by_default": False,
    },
]

ADDON_MODEL_DIRS = [
    COMFY_DIR / "models" / "sam3",
    COMFY_DIR / "models" / "SegviGen",
    COMFY_DIR / "models" / "FaithC",
    COMFY_DIR / "models" / "hymotion" / "HY-Motion-1.0",
    COMFY_DIR / "models" / "hymotion" / "HY-Motion-1.0-Lite",
    COMFY_DIR / "models" / "ultrashape",
    COMFY_DIR / "models" / "make_it_animatable",
    COMFY_DIR / "models" / "text_encoders",
]

DINO_FILES = [
    (
        "model.safetensors",
        "https://huggingface.co/PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m/resolve/main/model.safetensors",
    ),
    (
        "config.json",
        "https://huggingface.co/PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m/resolve/main/config.json",
    ),
    (
        "preprocessor_config.json",
        "https://huggingface.co/PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m/resolve/main/preprocessor_config.json",
    ),
]

IMPORT_CHECKS = [
    ("flash_attn", "flash_attn"),
    ("cumesh", "cumesh"),
    ("nvdiffrast.torch", "nvdiffrast"),
    ("nvdiffrec_render", "nvdiffrec_render"),
    ("flex_gemm", "flex_gemm"),
    ("o_voxel", "o_voxel"),
    ("bpy", "bpy"),
    ("bmesh", "bmesh"),
]

COMFY_CORE_IMPORT_CHECKS = [
    ("alembic", "alembic"),
    ("sqlalchemy", "sqlalchemy"),
    ("comfy_aimdo.control", "comfy_aimdo"),
    ("blake3", "blake3"),
]

OPTIONAL_IMPORT_CHECKS = [
    ("torch_scatter", "torch_scatter"),
    ("atom3d", "atom3d"),
    ("pytorch_lightning", "pytorch_lightning"),
    ("lightning_utilities", "lightning_utilities"),
    ("comfy_env", "comfy_env"),
    ("comfy_dynamic_widgets", "comfy_dynamic_widgets"),
]


def env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY_VALUES


def touch_stamp(name: str) -> None:
    STAMPS_DIR.mkdir(parents=True, exist_ok=True)
    (STAMPS_DIR / f"{name}.done").touch()


def stamp_exists(name: str) -> bool:
    return (STAMPS_DIR / f"{name}.done").exists()


def remove_stamp(name: str) -> None:
    (STAMPS_DIR / f"{name}.done").unlink(missing_ok=True)


def disabled_repo_path(repo_path: Path) -> Path:
    return repo_path.with_name(f"{repo_path.name}.disabled")


def restore_disabled_repo(repo: dict) -> None:
    repo_path = repo["path"]
    disabled_path = disabled_repo_path(repo_path)
    if repo_path.exists() or not disabled_path.exists():
        return

    print(f"[preflight] restoring previously disabled addon repo: {repo['name']}")
    disabled_path.rename(repo_path)


def disable_repo(repo: dict, reason: str) -> bool:
    repo_path = repo["path"]
    disabled_path = disabled_repo_path(repo_path)

    if disabled_path.exists() and not repo_path.exists():
        print(f"[preflight] addon already disabled: {repo['name']} ({reason})")
        return True

    if not repo_path.exists():
        return False

    if disabled_path.exists():
        print(
            f"[preflight] WARNING: cannot auto-disable {repo['name']} because "
            f"{disabled_path.name} already exists"
        )
        return False

    print(f"[preflight] auto-disabling addon {repo['name']}: {reason}")
    shutil.move(str(repo_path), str(disabled_path))
    return True


def run_command(command: list[str], required: bool = False) -> bool:
    display = " ".join(shlex.quote(part) for part in command)
    print(f"[preflight] $ {display}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"[preflight] command failed with exit code {result.returncode}")
        if required:
            raise SystemExit(result.returncode)
        return False
    return True


def module_exists(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def canonical_package_name(spec: str) -> str:
    text = spec.strip()
    if not text or text.startswith("#"):
        return ""

    if "#egg=" in text:
        text = text.split("#egg=", 1)[1]

    if text.startswith(("-", "--")):
        return ""

    # Extract package token before version markers/extras/spaces.
    match = re.match(r"^([A-Za-z0-9_.-]+)", text)
    if not match:
        return ""
    return match.group(1).lower().replace("_", "-")


def get_protected_packages() -> set[str]:
    configured = os.environ.get(
        "COMFYUI_PROTECTED_PINS",
        PROTECTED_REQUIREMENT_PINS_DEFAULT,
    )
    values = [item.strip().lower().replace("_", "-") for item in configured.split(",")]
    return {item for item in values if item}


def filter_requirements_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    if env_enabled("COMFYUI_ALLOW_ADDON_PIN_OVERRIDES", False):
        return lines, []

    protected = get_protected_packages()
    kept: list[str] = []
    skipped: list[str] = []

    for line in lines:
        package = canonical_package_name(line)
        if package and package in protected:
            skipped.append(line.rstrip())
            continue
        kept.append(line)

    return kept, skipped


def install_python_packages(
    source: str,
    packages: list[str],
    pip_args: list[str] | None = None,
    protect_pins: bool = True,
    required: bool = False,
) -> bool:
    if not packages:
        return True

    selected_packages = packages
    skipped: list[str] = []
    if protect_pins and not env_enabled("COMFYUI_ALLOW_ADDON_PIN_OVERRIDES", False):
        protected = get_protected_packages()
        selected_packages = []
        for package in packages:
            canonical = canonical_package_name(package)
            if canonical and canonical in protected:
                skipped.append(package)
            else:
                selected_packages.append(package)

    if skipped:
        print(f"[preflight] {source}: skipped protected package pins: {', '.join(skipped)}")

    if not selected_packages:
        print(f"[preflight] {source}: package install skipped (all entries protected)")
        return True

    command = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *selected_packages]
    if pip_args:
        command.extend(pip_args)
    return run_command(command, required=required)


def install_requirements_file(
    source: str,
    requirements_path: Path,
    pip_args: list[str] | None = None,
    protect_pins: bool = True,
    required: bool = False,
) -> bool:
    if not requirements_path.exists():
        print(f"[preflight] {source}: requirements file missing: {requirements_path}")
        return False

    lines = requirements_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if protect_pins:
        filtered_lines, skipped = filter_requirements_lines(lines)
    else:
        filtered_lines, skipped = lines, []

    if skipped:
        print(f"[preflight] {source}: skipped protected requirement pins")
        for item in skipped:
            if item.strip():
                print(f"[preflight]   - {item}")

    if not any(line.strip() and not line.strip().startswith("#") for line in filtered_lines):
        print(f"[preflight] {source}: requirements install skipped (empty after filtering)")
        return True

    if filtered_lines == lines:
        command = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements_path)]
        if pip_args:
            command.extend(pip_args)
        return run_command(command, required=required)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp_file:
        tmp_file.writelines(filtered_lines)
        filtered_path = Path(tmp_file.name)

    try:
        command = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(filtered_path)]
        if pip_args:
            command.extend(pip_args)
        return run_command(command, required=required)
    finally:
        filtered_path.unlink(missing_ok=True)


def install_faithcontouring_runtime_dependencies(repo_path: Path) -> None:
    if not module_exists("torch_scatter"):
        wheel_index_url = ""
        if module_exists("torch"):
            import torch  # type: ignore

            torch_version = torch.__version__.split("+", 1)[0]
            cuda_version = (torch.version.cuda or "").replace(".", "")
            if cuda_version:
                wheel_index_url = f"https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html"

        if wheel_index_url:
            run_command(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "torch-scatter",
                    "-f",
                    wheel_index_url,
                ],
                required=False,
            )

        if not module_exists("torch_scatter"):
            run_command(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", "torch-scatter"],
                required=False,
            )

    if module_exists("atom3d"):
        return

    linux_wheels_dir = repo_path / "wheels" / "Linux"
    cp_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel_candidates: list[Path] = []

    if linux_wheels_dir.exists():
        wheel_candidates = sorted(linux_wheels_dir.glob(f"**/atom3d-*-{cp_tag}-{cp_tag}-linux_x86_64.whl"))

    if not wheel_candidates and linux_wheels_dir.exists():
        available_wheels = sorted(path.name for path in linux_wheels_dir.glob("**/atom3d-*-linux_x86_64.whl"))
        if available_wheels:
            print(
                "[preflight] FaithContouring: no local atom3d wheel matches "
                f"{cp_tag}. Available: {', '.join(available_wheels)}"
            )

    preferred_torch_folder = ""
    if module_exists("torch"):
        import torch  # type: ignore

        version_parts = torch.__version__.split("+", 1)[0].split(".")
        if len(version_parts) >= 3 and all(part.isdigit() for part in version_parts[:3]):
            preferred_torch_folder = f"Torch{version_parts[0]}{version_parts[1]}{version_parts[2]}"

    if preferred_torch_folder:
        wheel_candidates.sort(
            key=lambda path: (0 if preferred_torch_folder in str(path) else 1, str(path))
        )

    for wheel_path in wheel_candidates:
        if run_command(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", str(wheel_path)],
            required=False,
        ) and module_exists("atom3d"):
            print(f"[preflight] FaithContouring atom3d wheel installed: {wheel_path.name}")
            break

    if not module_exists("atom3d"):
        run_command(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "atom3d==0.1.0"],
            required=False,
        )


def load_module_from_file(module_name: str, file_path: Path) -> tuple[bool, object | str]:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return False, f"spec load failed for {file_path}"

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return False, str(exc)

    return True, module


def check_faithcontouring_readiness(repo: dict) -> tuple[bool, str]:
    if not module_exists("torch_scatter"):
        return False, "missing torch_scatter"

    if not module_exists("atom3d"):
        return False, "missing atom3d"

    try:
        atom3d_module = importlib.import_module("atom3d")
        if not hasattr(atom3d_module, "MeshBVH"):
            return False, "atom3d does not expose MeshBVH"
    except Exception as exc:
        return False, f"atom3d import failed: {exc}"

    nodes_path = repo["path"] / "nodes.py"
    faithcontour_dir = repo["path"] / "faithcontour"
    if not nodes_path.exists() or not faithcontour_dir.exists():
        return False, "FaithContouring source files missing"

    return True, "ready"


def check_ultrashape_readiness(repo: dict) -> tuple[bool, str]:
    for module_name in ("pytorch_lightning", "lightning_utilities", "omegaconf", "pymeshlab"):
        if not module_exists(module_name):
            return False, f"missing {module_name}"

    ultra_nodes_path = repo["path"] / "ultra_nodes.py"
    if not ultra_nodes_path.exists():
        return False, "ultra_nodes.py missing"

    ok, loaded = load_module_from_file("preflight_ultrashape_nodes", ultra_nodes_path)
    if not ok:
        return False, f"ultra_nodes import failed: {loaded}"

    import_error = getattr(loaded, "ULTRASHAPE_IMPORT_ERROR", None)
    if import_error is not None:
        return False, f"ultra_nodes import chain failed: {import_error}"

    return True, "ready"


def check_required_modules(module_names: tuple[str, ...]) -> tuple[bool, str]:
    for module_name in module_names:
        if not module_exists(module_name):
            return False, f"missing {module_name}"

    return True, "ready"


def check_geometrypack_readiness() -> tuple[bool, str]:
    ok, reason = check_required_modules(("comfy_env", "comfy_dynamic_widgets", "pymeshfix"))
    if not ok:
        return False, reason

    try:
        dynamic_widgets_module = importlib.import_module("comfy_dynamic_widgets")
    except Exception as exc:
        return False, f"comfy_dynamic_widgets import failed: {exc}"

    if not hasattr(dynamic_widgets_module, "write_mappings"):
        return False, "comfy_dynamic_widgets missing write_mappings"

    return True, "ready"


def check_optional_addon_readiness(repo: dict) -> tuple[bool, str]:
    key = repo.get("key")
    if key == "faithcontouring":
        return check_faithcontouring_readiness(repo)
    if key == "ultrashape":
        return check_ultrashape_readiness(repo)
    if key == "sam3":
        return check_required_modules(("comfy_env", "ftfy", "regex", "iopath", "portalocker"))
    if key == "geometrypack":
        return check_geometrypack_readiness()
    if key == "hymotion":
        return check_required_modules(("smplx",))

    required_modules = tuple(repo.get("required_modules", []))
    if required_modules:
        return check_required_modules(required_modules)

    return True, "ready"


def clone_or_update_repo(repo: dict, update_flag_name: str | None = None) -> bool:
    repo_path = repo["path"]
    cloned = False

    if not repo_path.exists():
        print(f"[preflight] missing node repo {repo['name']}, cloning now")
        cloned = run_command(["git", "clone", "--depth", "1", repo["url"], str(repo_path)])
        if not cloned:
            return False
    elif update_flag_name and env_enabled(update_flag_name, False):
        print(f"[preflight] updating repo {repo['name']}")
        run_command(["git", "-C", str(repo_path), "pull"], required=False)

    return cloned or repo_path.exists()


def install_repo_dependencies(repo: dict, required: bool = False) -> bool:
    repo_path = repo["path"]
    pip_args = repo.get("pip_args", [])
    ok = True

    requirements_value = repo.get("requirements")
    protect_pins = not bool(repo.get("required", False))
    requirements_list: list[str] = []
    if isinstance(requirements_value, str):
        requirements_list = [requirements_value]
    elif isinstance(requirements_value, list):
        requirements_list = requirements_value

    for requirement_file in requirements_list:
        ok = install_requirements_file(
            repo["name"],
            repo_path / requirement_file,
            pip_args=pip_args,
            protect_pins=protect_pins,
            required=required,
        ) and ok

    extra_packages = repo.get("extra_packages", [])
    extra_pip_args = repo.get("extra_pip_args", [])
    if extra_packages:
        ok = install_python_packages(
            repo["name"],
            extra_packages,
            pip_args=extra_pip_args,
            protect_pins=protect_pins,
            required=required,
        ) and ok

    run_install_py_env = repo.get("run_install_py_env")
    if run_install_py_env and env_enabled(run_install_py_env, True):
        install_script = repo_path / "install.py"
        if install_script.exists():
            ok = run_command([sys.executable, str(install_script)], required=False) and ok

    if repo.get("key") == "faithcontouring":
        install_faithcontouring_runtime_dependencies(repo_path)

    if repo.get("key") == "pulse_meshaudit":
        print("[preflight] Pulse MeshAudit requires Vulkan runtime (libvulkan1).")

    return ok


def ensure_required_nodes() -> None:
    CUSTOM_NODES_DIR.mkdir(parents=True, exist_ok=True)
    STAMPS_DIR.mkdir(parents=True, exist_ok=True)

    force_reinstall = env_enabled("COMFYUI_FORCE_ADDON_REINSTALL", False)

    for repo in REQUIRED_NODE_REPOS:
        repo_ready = clone_or_update_repo(repo)
        if not repo_ready:
            if repo.get("required", False):
                raise SystemExit(f"required repo unavailable: {repo['name']}")
            continue

        stamp_key = f"{repo['key']}.install"
        if stamp_exists(stamp_key) and not force_reinstall:
            continue

        install_repo_dependencies(repo, required=False)
        touch_stamp(stamp_key)


def ensure_addon_model_directories() -> None:
    for directory in ADDON_MODEL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_optional_addons() -> None:
    if not env_enabled("COMFYUI_ENABLE_OPTIONAL_ADDONS", True):
        print("[preflight] optional addons disabled by COMFYUI_ENABLE_OPTIONAL_ADDONS=0")
        return

    force_reinstall = env_enabled("COMFYUI_FORCE_ADDON_REINSTALL", False)
    auto_disable_failed = env_enabled("COMFYUI_AUTO_DISABLE_FAILED_ADDONS", True)

    if env_enabled("COMFYUI_ALLOW_ADDON_PIN_OVERRIDES", False):
        print(
            "[preflight] WARNING: COMFYUI_ALLOW_ADDON_PIN_OVERRIDES=1. "
            "Optional addon installs may override baseline package pins."
        )

    for repo in OPTIONAL_ADDON_REPOS:
        enabled = env_enabled(repo["enable_env"], repo["enabled_by_default"])
        status_text = "enabled" if enabled else "disabled"
        print(f"[preflight] addon {repo['name']}: {status_text}")

        if not enabled:
            continue

        restore_disabled_repo(repo)

        if not clone_or_update_repo(repo, update_flag_name="COMFYUI_AUTO_UPDATE_OPTIONAL_REPOS"):
            print(f"[preflight] WARNING: could not prepare optional addon {repo['name']}")
            continue

        stamp_key = f"{repo['key']}.install"
        if not stamp_exists(stamp_key) or force_reinstall:
            install_ok = install_repo_dependencies(repo, required=False)
            if install_ok:
                touch_stamp(stamp_key)
            else:
                remove_stamp(stamp_key)
                print(f"[preflight] WARNING: dependency install reported errors for {repo['name']}")

        ready, reason = check_optional_addon_readiness(repo)
        if not ready:
            print(f"[preflight] WARNING: addon readiness failed for {repo['name']}: {reason}")
            print(f"[preflight] attempting addon dependency repair for {repo['name']}")
            repair_ok = install_repo_dependencies(repo, required=False)
            if repair_ok:
                touch_stamp(stamp_key)

            ready, reason = check_optional_addon_readiness(repo)

        if ready:
            print(f"[preflight] addon readiness OK: {repo['name']}")
            continue

        print(f"[preflight] WARNING: addon remains not ready for {repo['name']}: {reason}")
        if auto_disable_failed:
            if disable_repo(repo, reason):
                remove_stamp(stamp_key)
        else:
            print(
                "[preflight] WARNING: addon left enabled because "
                "COMFYUI_AUTO_DISABLE_FAILED_ADDONS=0"
            )

    # Keep the base numeric stack deterministic for Trellis2 GGUF.
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            "--no-deps",
            "numpy==1.26.4",
        ],
        required=False,
    )


def maybe_run_gguf_installer_once() -> None:
    if os.environ.get("COMFYUI_RUN_GGUF_INSTALLER", "1").lower() in {"0", "false", "no"}:
        print("[preflight] skipping GGUF installer by configuration")
        return

    installer = CUSTOM_NODES_DIR / "ComfyUI-Trellis2-GGUF" / "install.py"
    stamp_file = Path("/tmp/trellis2_gguf_installer.done")

    if not installer.exists():
        print("[preflight] GGUF install.py not found; skipping installer step")
        return

    if stamp_file.exists():
        print("[preflight] GGUF installer already executed for this container")
        return

    if run_command([sys.executable, str(installer)]):
        stamp_file.touch()


def enforce_numpy_baseline() -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            "--no-deps",
            "numpy==1.26.4",
        ],
        required=False,
    )


def ensure_dino_assets() -> None:
    dino_dir = COMFY_DIR / "models" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
    dino_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in DINO_FILES:
        destination = dino_dir / filename
        if destination.exists():
            print(f"[preflight] DINO asset already present: {destination.name}")
            continue

        print(f"[preflight] downloading DINO asset: {destination.name}")
        try:
            urlretrieve(url, destination)
        except Exception as exc:
            print(f"[preflight] WARNING: could not download {destination.name}: {exc}")


def ensure_comfyui_core_requirements() -> None:
    if os.environ.get("COMFYUI_AUTO_INSTALL_CORE_REQS", "1").lower() in {"0", "false", "no"}:
        print("[preflight] skipping ComfyUI core requirements install by configuration")
        return

    requirements_path = COMFY_DIR / "requirements.txt"
    if not requirements_path.exists():
        print("[preflight] ComfyUI requirements.txt not found; skipping core dependency check")
        return

    missing = [label for module_name, label in COMFY_CORE_IMPORT_CHECKS if not module_exists(module_name)]
    if not missing:
        print("[preflight] ComfyUI core requirements already satisfied")
        return

    print("[preflight] missing ComfyUI core dependencies: " + ", ".join(missing))
    run_command(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements_path)],
        required=False,
    )

    still_missing = [label for module_name, label in COMFY_CORE_IMPORT_CHECKS if not module_exists(module_name)]
    if still_missing:
        print("[preflight] WARNING: still missing ComfyUI core dependencies: " + ", ".join(still_missing))
    else:
        print("[preflight] ComfyUI core dependencies ready")


def log_runtime_versions() -> None:
    print(f"[preflight] Python: {sys.version.split()[0]}")

    if module_exists("numpy"):
        import numpy  # type: ignore

        print(f"[preflight] NumPy: {numpy.__version__}")

    if module_exists("huggingface_hub"):
        import huggingface_hub  # type: ignore

        print(f"[preflight] huggingface_hub: {huggingface_hub.__version__}")

    if module_exists("torch"):
        import torch  # type: ignore

        print(f"[preflight] Torch: {torch.__version__}")
        print(f"[preflight] Torch CUDA runtime: {torch.version.cuda}")

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            print(f"[preflight] CUDA device: {name} (compute capability {major}.{minor})")
        else:
            print("[preflight] WARNING: CUDA not available inside container")


def maybe_apply_blackwell_patch() -> None:
    if os.environ.get("COMFYUI_ENABLE_BLACKWELL_PATCH", "1").lower() in {"0", "false", "no"}:
        print("[preflight] Blackwell patch disabled by configuration")
        return

    if not module_exists("torch"):
        print("[preflight] torch is unavailable, skipping Blackwell patch")
        return

    import torch  # type: ignore

    if not torch.cuda.is_available():
        print("[preflight] CUDA unavailable, skipping Blackwell patch")
        return

    major, minor = torch.cuda.get_device_capability(0)
    if major < 10:
        print(f"[preflight] GPU compute capability {major}.{minor}; Blackwell patch not required")
        return

    print(f"[preflight] Blackwell-class GPU detected ({major}.{minor}), attempting patch")

    for path in (
        CUSTOM_NODES_DIR / "ComfyUI-Trellis2",
        CUSTOM_NODES_DIR / "ComfyUI-Trellis2-GGUF",
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        blackwell_fix = importlib.import_module("blackwell_fix")
        patch_all = getattr(blackwell_fix, "patch_all", None)
        if callable(patch_all):
            patch_all(force=False, verbose=True)
            print("[preflight] applied Blackwell compatibility patch")
        else:
            print("[preflight] WARNING: blackwell_fix.patch_all not found")
    except Exception as exc:
        print(f"[preflight] WARNING: Blackwell patch could not be applied: {exc}")


def enforce_sparse_conv_backend() -> None:
    desired_backend = os.environ.get("COMFYUI_SPARSE_CONV_BACKEND", "").strip().lower()

    # Preserve any pre-configured backend from the container environment if no
    # explicit preflight override is provided.
    if not desired_backend and _INITIAL_SPARSE_CONV_BACKEND:
        desired_backend = _INITIAL_SPARSE_CONV_BACKEND.strip().lower()

    if desired_backend not in _ALLOWED_SPARSE_CONV_BACKENDS:
        if desired_backend:
            print(
                f"[preflight] WARNING: unsupported sparse backend '{desired_backend}', "
                "falling back to flex_gemm"
            )
        desired_backend = "flex_gemm"

    os.environ["SPARSE_CONV_BACKEND"] = desired_backend
    print(f"[preflight] effective sparse conv backend: {desired_backend}")

    if desired_backend == "spconv" and not module_exists("spconv.pytorch"):
        print(
            "[preflight] WARNING: SPARSE_CONV_BACKEND=spconv but module "
            "'spconv.pytorch' is unavailable. Set COMFYUI_SPARSE_CONV_BACKEND=flex_gemm "
            "or install spconv."
        )


def check_runtime_imports() -> None:
    missing = []

    for module_name, label in IMPORT_CHECKS:
        if module_exists(module_name):
            print(f"[preflight] import OK: {label}")
        else:
            print(f"[preflight] WARNING: import missing: {label}")
            missing.append(label)

    for module_name, label in OPTIONAL_IMPORT_CHECKS:
        if module_exists(module_name):
            print(f"[preflight] optional import OK: {label}")
        else:
            print(f"[preflight] optional import missing: {label}")

    if missing:
        print("[preflight] Some optional dependencies are missing. Workflows may be partially degraded:")
        print("[preflight] " + ", ".join(missing))

    blender_ready = module_exists("bpy") and module_exists("bmesh")
    if blender_ready:
        print("[preflight] Blender unwrap support: available (bpy + bmesh)")
    else:
        print(
            "[preflight] Blender unwrap support: unavailable; Trellis2 will fallback to Xatlas "
            "when Blender unwrap is selected."
        )


def launch_comfyui() -> None:
    os.chdir(COMFY_DIR)

    args = [
        str(COMFY_DIR / "main.py"),
        "--listen",
        "0.0.0.0",
        "--disable-dynamic-vram",
        "--disable-pinned-memory",
    ]

    extra_args = os.environ.get("COMFYUI_EXTRA_ARGS", "").strip()
    if extra_args:
        args.extend(shlex.split(extra_args))

    print("[preflight] starting ComfyUI")
    os.execv(sys.executable, [sys.executable, *args])


def main() -> None:
    ensure_required_nodes()
    ensure_addon_model_directories()
    ensure_optional_addons()
    maybe_run_gguf_installer_once()
    enforce_numpy_baseline()
    ensure_dino_assets()
    ensure_comfyui_core_requirements()
    log_runtime_versions()
    maybe_apply_blackwell_patch()
    enforce_sparse_conv_backend()
    check_runtime_imports()
    launch_comfyui()


if __name__ == "__main__":
    main()
