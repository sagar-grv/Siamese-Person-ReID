
## Successful verification

After exporting the ONNX model as a self-contained file with `external_data=False`, the production preview reports `Model ready`. The default query loads, browser inference completes in approximately 48 ms in the sandbox, and the UI renders ten gallery matches with cosine scores. The sample query's top match is the expected same identity (`TRACK 0001 · CAM 1`) with a score of `0.990`.

## Vercel verification

The anonymous production deployment at `https://temporary-speedy-chestnut-xrs34fd.vercel.app` loaded successfully with `Model ready`. Running the default query completed browser inference in approximately 52 ms and rendered ten ranked matches. The top result was `TRACK 0001 · CAM 1` with cosine score `0.990`, matching the local preview behavior.
