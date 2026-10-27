# Driving gaze and distraction demo

This folder combines the EyeCue video-and-gaze distraction classifier with
GazeFollower webcam gaze tracking. The GazeFollower model assets are included.
The EyeCue training checkpoint is hosted publicly on Hugging Face; the training
dataset and driving videos are not included.

## Required EyeCue checkpoint

Place the trained `best_model.pth` file at:

```text
driver-gaze-demo/EyeCue/checkpoints/best_model.pth
```

The checkpoint is about 1.66 GB and is intentionally not stored in Git. It
contains the trained EyeCue encoders, fusion module, classifier, and optimizer
state. Download it from the public
[Hugging Face model repository](https://huggingface.co/Thorfinn11111/eyecue_best_model)
using a regular VS Code PowerShell terminal from this `driver-gaze-demo`
directory:

To copy a checkpoint you already have on Windows:

```powershell
New-Item -ItemType Directory -Force .\EyeCue\checkpoints | Out-Null
Copy-Item "D:\path\to\best_model.pth" .\EyeCue\checkpoints\best_model.pth
```

To download the public checkpoint instead:

```powershell
New-Item -ItemType Directory -Force .\EyeCue\checkpoints | Out-Null
curl.exe -L --fail --retry 5 --output .\EyeCue\checkpoints\best_model.pth "https://huggingface.co/Thorfinn11111/eyecue_best_model/resolve/main/best_model.pth"
Get-Item .\EyeCue\checkpoints\best_model.pth | Select-Object Name,Length
```

The model repository is public, and its `best_model.pth` file is available
without a token. The expected file size is 1,661,075,654 bytes.

## Install

Use Python 3.12 for the EyeCue environment and a Python version supported by
GazeFollower/MNN for the webcam environment. Run these commands from this
directory in a regular VS Code PowerShell terminal. Install PyTorch separately
to select the appropriate CPU or CUDA build.

```powershell
py -3.12 -m venv .\EyeCue\.venv
.\EyeCue\.venv\Scripts\python.exe -m pip install --upgrade pip
.\EyeCue\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
.\EyeCue\.venv\Scripts\python.exe -m pip install -r .\EyeCue\requirements.txt

py -3.11 -m venv .\GazeFollower\.venv
.\GazeFollower\.venv\Scripts\python.exe -m pip install --upgrade pip
.\GazeFollower\.venv\Scripts\python.exe -m pip install -r .\GazeFollower\requirements.txt
```

For a CPU-only EyeCue install, replace the PyTorch install line with:

```powershell
.\EyeCue\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

## Run live gaze overlay and distraction prediction

Connect a webcam, have the trained checkpoint in place, and provide a local
MP4 video (or a directory of videos). Run from this directory:

```powershell
.\GazeFollower\.venv\Scripts\python.exe .\GazeFollower\watch_and_assess.py `
  --video "D:\videos\driving.mp4" `
  --eyecue-python ".\EyeCue\.venv\Scripts\python.exe" `
  --checkpoint ".\EyeCue\checkpoints\best_model.pth" `
  --device cpu
```

The application prompts for webcam gaze calibration, plays the selected video,
draws the estimated gaze point and the latest EyeCue distraction probability on
the video, and saves an annotated MP4 plus CSV/JSON results under
`EyeCue/outputs/live_assessments`. Use `--device cuda` when the EyeCue
environment has a compatible CUDA PyTorch installation.

The first EyeCue run downloads the TimeSformer image processor and pretrained
backbone from Hugging Face, so internet access is required. The webcam must be
available to OpenCV. This is an exploratory estimate, not a validated driving
safety assessment.
