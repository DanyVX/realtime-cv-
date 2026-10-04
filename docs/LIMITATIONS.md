# Limitations

- No detector weights or licensed dataset have been selected yet.
- The current development host has no NVIDIA GPU, CUDA, TensorRT, Docker, or ffmpeg.
- There are no serving, accuracy, or load-test claims until the relevant milestones are measured.
- The planned service is single-node; autoscaling and multi-GPU are not initially supported.
- Dynamic batching primitives are implemented, but a model session and verified weights are required before readiness can become true.
- Docker configuration is supplied but not locally tested because Docker is not installed on the development host.
