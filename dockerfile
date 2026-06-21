# Use a development image to allow for compilation of CUDA extensions.
# Requires CUDA 12.8 to match PyTorch 2.8.0+cu128.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Target RTX 5080 (Blackwell = sm_120) while keeping broad compatibility.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    MAX_JOBS=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Install system dependencies.
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    libgl1 \
    libopengl0 \
    libglib2.0-0 \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure python command is available.
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and build tools.
RUN python3 -m pip install --upgrade pip setuptools wheel ninja

# Install PyTorch 2.8.0 with CUDA 12.8 support (required by Trellis2 GGUF).
# NOTE: If this fails, check https://download.pytorch.org/whl/cu128 for the exact available version.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# FaithContouring dependency helper: prefer prebuilt torch-scatter wheel for
# the active Torch/CUDA pair; keep image build non-blocking.
RUN pip install --no-cache-dir --prefer-binary torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html || true

# Install ComfyUI core dependencies.
RUN pip install --no-cache-dir aiohttp tqdm requests psutil pandas scipy Pillow matplotlib opencv-python-headless packaging

# rembg requires onnxruntime at runtime; use GPU variant for this CUDA stack.
RUN pip install --no-cache-dir onnxruntime-gpu

# triton must be installed before flash-attn.
RUN pip install --no-cache-dir triton

# flash-attn requires --no-build-isolation so it can see the already-installed torch.
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# General Trellis2 dependencies and BAT-equivalent version behavior.
RUN pip install --no-cache-dir hf_transfer gguf
RUN pip install --no-cache-dir --upgrade huggingface_hub pooch
RUN pip install --no-cache-dir --force-reinstall numpy==1.26.4

# Install nvdiffrast from source.
# --no-build-isolation is required: nvdiffrast imports torch at build time,
# which fails in pip's isolated env.
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

# CuMesh cubvh extension needs Eigen headers at /usr/include/eigen3.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install CuMesh from source as Linux fallback.
RUN git clone --recursive https://github.com/visualbruno/CuMesh /tmp/cumesh \
    && cd /tmp/cumesh \
    && CPLUS_INCLUDE_PATH=/usr/include/eigen3 CPATH=/usr/include/eigen3 pip install --no-cache-dir --no-build-isolation . \
    && rm -rf /tmp/cumesh

# Setup working directory.
WORKDIR /root

# Clone ComfyUI.
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Install ComfyUI's own requirements to stay aligned with upstream runtime
# changes (database + comfy_aimdo dependencies in recent versions).
RUN pip install --no-cache-dir -r /root/ComfyUI/requirements.txt

# Include ComfyUI-Manager (UI Manager button and node management panel).
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git \
    /root/ComfyUI/custom_nodes/ComfyUI-Manager
RUN pip install --no-cache-dir -r /root/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt

# Clone visualbruno's Trellis2 node — provides all core Trellis2 nodes.
RUN git clone https://github.com/visualbruno/ComfyUI-Trellis2.git \
    /root/ComfyUI/custom_nodes/ComfyUI-Trellis2
RUN pip install --no-cache-dir -r /root/ComfyUI/custom_nodes/ComfyUI-Trellis2/requirements.txt

# Clone Aero-Ex's GGUF fork — provides Trellis2LoadModel_GGUF.
RUN git clone https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF.git \
    /root/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF
RUN pip install --no-cache-dir -r /root/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/requirements.txt --no-deps

# Try the GGUF installer to pull matching Linux CUDA wheels (cumesh, flex_gemm,
# nvdiffrast, nvdiffrec_render, o_voxel) and apply upstream patches.
RUN python3 /root/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/install.py || true

# Re-assert onnxruntime provider after custom-node requirements and installer
# side effects, so rembg import remains available at runtime.
# Use --no-deps here to avoid unintentionally changing the pinned numpy baseline.
RUN pip install --no-cache-dir --no-deps onnxruntime-gpu

# Fallback installers: continue even if an individual package is unavailable.
RUN python3 - <<'PY'
import importlib
import subprocess
import sys


def module_exists(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def run_pip(args):
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + args
    print("[docker:fallback]", " ".join(cmd))
    subprocess.run(cmd, check=False)


fallback_matrix = [
    ("nvdiffrec_render", [["nvdiffrec-render"], ["nvdiffrec_render"]]),
    ("flex_gemm", [["flex-gemm"], ["flex_gemm"]]),
    ("o_voxel", [["o-voxel"], ["o_voxel"]]),
]

for module_name, package_variants in fallback_matrix:
    if module_exists(module_name):
        print(f"[docker:fallback] {module_name} already available")
        continue

    for variant in package_variants:
        run_pip(variant)
        if module_exists(module_name):
            print(f"[docker:fallback] installed {module_name} via {' '.join(variant)}")
            break
    else:
        print(f"[docker:fallback] WARNING: could not install {module_name}")

if not module_exists("cumesh"):
    run_pip(["git+https://github.com/visualbruno/CuMesh.git"])

if not module_exists("nvdiffrast.torch"):
    run_pip(["--no-build-isolation", "git+https://github.com/NVlabs/nvdiffrast.git"])
PY

# Optional Blender Python API for Blender UV unwrap support in Trellis2 GGUF.
# Best-effort only: if no compatible wheel exists for this Python stack,
# the image still builds and Trellis2 falls back to Xatlas at runtime.
RUN python3 - <<'PY'
import importlib
import subprocess
import sys


def modules_ready():
    try:
        importlib.import_module("bpy")
        importlib.import_module("bmesh")
        return True
    except Exception:
        return False


if modules_ready():
    print("[docker:bpy] bpy + bmesh already available")
    raise SystemExit(0)

candidates = [
    "bpy==4.2.0",
    "bpy==4.1.0",
    "bpy==4.0.0",
    "bpy",
]

for spec in candidates:
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--prefer-binary", spec]
    print("[docker:bpy] trying", " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        continue

    if modules_ready():
        print(f"[docker:bpy] installed successfully via {spec}")
        break
else:
    print("[docker:bpy] WARNING: bpy unavailable in this environment. Trellis2 will use Xatlas fallback.")
PY

# BAT parity: patch CuMesh remeshing.py with the latest upstream file if available.
RUN python3 - <<'PY'
import importlib.util
import pathlib
import shutil
import urllib.request

url = "https://raw.githubusercontent.com/visualbruno/CuMesh/main/cumesh/remeshing.py"
spec = importlib.util.find_spec("cumesh")

if spec and spec.origin:
    package_dir = pathlib.Path(spec.origin).resolve().parent
    target = package_dir / "remeshing.py"
    backup = package_dir / "remeshing.py.bak"

    if target.exists():
        shutil.copy2(target, backup)

    try:
        urllib.request.urlretrieve(url, target)
        print(f"[docker:patch] patched {target}")
    except Exception as exc:
        print(f"[docker:patch] WARNING: remeshing patch skipped: {exc}")
else:
    print("[docker:patch] WARNING: cumesh not found; remeshing patch skipped")
PY

# Create model/runtime directories expected by the workflows and startup preflight.
RUN mkdir -p /root/ComfyUI/models/Trellis2 \
    /root/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m \
    /root/ComfyUI/models/sam3 \
    /root/ComfyUI/models/SegviGen \
    /root/ComfyUI/models/FaithC \
    /root/ComfyUI/models/hymotion/HY-Motion-1.0 \
    /root/ComfyUI/models/hymotion/HY-Motion-1.0-Lite \
    /root/ComfyUI/models/ultrashape \
    /root/ComfyUI/models/make_it_animatable \
    /root/ComfyUI/models/text_encoders \
    /root/ComfyUI/scripts

# Add startup preflight launcher.
COPY ./scripts/preflight_launch.py /root/ComfyUI/scripts/preflight_launch.py

EXPOSE 8188

WORKDIR /root/ComfyUI
CMD ["python3", "/root/ComfyUI/scripts/preflight_launch.py"]