NanoSAM on this project expects two TensorRT engine files in this directory:

- `resnet18_image_encoder.engine`
- `mobile_sam_mask_decoder.engine`

The official NanoSAM project builds these engine files with `trtexec` on supported NVIDIA / TensorRT environments.

This macOS machine does not have `TensorRT` or `trtexec`, so the Python package can be installed, but the engine files cannot be built locally here.

Official setup references:

- https://github.com/NVIDIA-AI-IOT/nanosam
- https://github.com/NVIDIA-AI-IOT/nanosam#-setup

If you later move to a supported NVIDIA machine, place the built `.engine` files here and the builder will pick them up automatically through `spatial_rag/config.py`.
