# QGIS Deepness & Whitebox Workflows, conda `geo` environment

## Environment setup

QGIS should be started with the conda `geo` environment active so its Python
plugins (Deepness, Whitebox Workflows) can import the same libs your scripts
use here (`osgeo`, `onnxruntime`, `opencv`, `numpy<2`, ...).

A ready-made environment file is at:

    ~/QGIS/envs/qgis-geo.env

Source it before launching QGIS:

    source ~/QGIS/envs/qgis-geo.env
    qgis

What that does:

- `QGIS_PREFIX_PATH=/usr` → uses the system QGIS 3.44.7 from apt
- `PATH` first contains `~/miniconda3/envs/geo/bin` → QGIS Python
  plugins inherit the `geo` conda environment (GDAL 3.8.5, ONNX Runtime GPU
  1.17.0, OpenCV 4.9.0, numpy 1.26.4, …)
- `PYTHONHOME` / `PYTHONPATH` cleared → avoids clashing with system Python
- `LD_LIBRARY_PATH` limited to `/usr/lib/x86_64-linux-gnu` + Qt/dotnet dirs
  → keeps QGIS libs isolated from conda libs
- `DOTNET_ROOT` / `dotnet` in `PATH` → needed by QGIS 3.44 on this machine

### Desktop / launcher

If you start QGIS from a desktop launcher, point it at a wrapper like:

    #!/usr/bin/env bash
    source ~/QGIS/envs/qgis-geo.env
    exec /usr/bin/qgis "$@"

Save that as `~/bin/qgis-geo` and make it executable, then set the launcher
command to `qgis-geo`.

### Verify QGIS sees the right Python

Inside QGIS Python console:

    import sys
    print(sys.executable)
    import onnxruntime, cv2
    print(onnxruntime.__version__, cv2.__version__)

Expected: executable inside `miniconda3/envs/geo`, and the onnxruntime / cv2
versions installed above.

## Notes / caveats

- `onnxruntime-gpu` requires a CUDA-capable GPU and the right driver/cuDNN
  stack. In this env `ort.get_available_providers()` currently returns
  `CUDAExecutionProvider` / `CPUExecutionProvider` / …, confirm the GPU
  provider is actually usable on your hardware before relying on GPU inference.
- `numpy` was downgraded to `<2` (`1.26.4`) because the pinned
  `opencv-python-headless<=4.9.0.80` and `onnxruntime-gpu<=1.17.0` wheels are
  built against NumPy 1.x and refuse to import under NumPy 2.x.
- This `source …; qgis` approach is a runtime env wiring, not a QGIS plugin
  install. If Deepness / Whitebox expect a specific plugin package or repo,
  install that separately through QGIS Plugin Manager.
