# Design

## Decision log

### 2026-10-04 — Detector license path

Options considered: Ultralytics YOLO under AGPL-3.0 (or paid enterprise terms), and YOLOX
source under Apache-2.0. Chosen: YOLOX source. Reason: the repository can remain MIT-licensed
without AGPL copyleft obligations. This does **not** clear any checkpoint; weights require their
own verified license before use.

### 2026-10-04 — CPU-first baseline

The development host has no CUDA-capable GPU, so CPU ONNX Runtime and INT8 will be the primary
measured path. GPU/TensorRT results are optional and must be recorded on the hardware that
produces them.

## Planned serving design

One batching worker owns admission from a bounded queue. It gathers until `max_batch_size` or a
deadline, runs ORT in an executor, then resolves each request's future. Preprocessing and model
execution stay off the event loop. Detailed queue semantics and the open-loop benchmark protocol
will be added with M4 and M6, before any numbers are published.
