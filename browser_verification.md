
## Successful verification

After exporting the ONNX model as a self-contained file with `external_data=False`, the production preview reports `Model ready`. The default query loads, browser inference completes in approximately 48 ms in the sandbox, and the UI renders ten gallery matches with cosine scores. The sample query's top match is the expected same identity (`TRACK 0001 · CAM 1`) with a score of `0.990`.

## Vercel verification

The anonymous production deployment at `https://temporary-speedy-chestnut-xrs34fd.vercel.app` loaded successfully with `Model ready`. Running the default query completed browser inference in approximately 52 ms and rendered ten ranked matches. The top result was `TRACK 0001 · CAM 1` with cosine score `0.990`, matching the local preview behavior.

## Follow-up after user report

The reported URL was checked again and currently returns the Trace/Lab application rather than a 403. The model status is `Model ready`; running the default query completed in approximately 17 ms and returned the same ten ranked matches, with top score `0.990`. The mobile 403 may have been a transient anonymous-deployment access/expiry response or a cached failure; a permanent Vercel deployment still requires claiming the anonymous deployment or connecting a Vercel account.
