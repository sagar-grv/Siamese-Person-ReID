
## Successful verification

After exporting the ONNX model as a self-contained file with `external_data=False`, the production preview reports `Model ready`. The default query loads, browser inference completes in approximately 48 ms in the sandbox, and the UI renders ten gallery matches with cosine scores. The sample query's top match is the expected same identity (`TRACK 0001 · CAM 1`) with a score of `0.990`.
