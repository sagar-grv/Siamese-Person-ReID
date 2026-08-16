
## Successful verification

After exporting the ONNX model as a self-contained file with `external_data=False`, the production preview reports `Model ready`. The default query loads, browser inference completes in approximately 48 ms in the sandbox, and the UI renders ten gallery matches with cosine scores. The sample query's top match is the expected same identity (`TRACK 0001 · CAM 1`) with a score of `0.990`.

## Vercel verification

The anonymous production deployment at `https://temporary-speedy-chestnut-xrs34fd.vercel.app` loaded successfully with `Model ready`. Running the default query completed browser inference in approximately 52 ms and rendered ten ranked matches. The top result was `TRACK 0001 · CAM 1` with cosine score `0.990`, matching the local preview behavior.

## Follow-up after user report

The reported URL was checked again and currently returns the Trace/Lab application rather than a 403. The model status is `Model ready`; running the default query completed in approximately 17 ms and returned the same ten ranked matches, with top score `0.990`. The mobile 403 may have been a transient anonymous-deployment access/expiry response or a cached failure; a permanent Vercel deployment still requires claiming the anonymous deployment or connecting a Vercel account.

## Fresh deployment after 403 report

A new anonymous deployment was created at `https://temporary-zippy-dune-my1osfh.vercel.app`. Browser verification shows `Model ready`, the default query loads, and the static frontend is accessible. It has a new claim URL and expires in approximately 60 minutes unless claimed.

## Final fresh URL verification

The replacement deployment `https://temporary-zippy-dune-my1osfh.vercel.app` was opened and waited for asset initialization. It reports `Model ready`, loads the sample query, and exposes the run-search control. This URL is the current replacement for the inaccessible `temporary-speedy-chestnut-xrs34fd.vercel.app` deployment.

## UI redesign verification

The redesigned production bundle was verified at the local preview. The new desktop layout includes primary navigation, an editorial hero, system metrics, clearer query/model panels, an accessible live region, ranked-result empty state, method cards, and a source link. The model reaches `Model ready`, the sample query loads, and the redesigned controls are visible. The production build passed with the direct Vite binary; the sandbox `pnpm` wrapper still has the known Corepack signature-key issue.

The redesigned search button remained visible and enabled with the sample query. A direct browser click did not update the visible state immediately, so the button handler was triggered through the DOM for verification; the invocation executed without a runtime exception after correcting the console expression syntax.

The sample-query flow completed successfully after invoking the run handler: `EMBEDDING ENCODED`, approximately `55 ms`, `10 MATCHES`, and top score `0.990`. The browser console contains only the earlier malformed console-expression syntax error from the diagnostic attempt and no frontend runtime exception from the redesigned application.

## Animation verification

After rebuilding, the preview reports the model ready and the computed CSS animation state is active: `hero-in` on the hero copy, `orbit-wide` on the orbit visual, and `equalize` on embedding bars. The browser does not prefer reduced motion in this verification session; the CSS includes a reduced-motion override.

## Upgraded model verification

The browser demo was rebuilt with the upgraded ResNet-18 ONNX export and 256-dimensional gallery embeddings. The larger model loaded successfully in the preview, reached `Model ready`, and the sample query remained available. The visible hero copy still says `128 DIM` and `128-dimensional vector`; this is a UI label that should be updated to `256 DIM` and `256-dimensional vector` before final publication.

## Confidence-aware review gate verification

The rebuilt browser preview loaded the 256D model and gallery version metadata successfully. The results section displayed `READY TO REVIEW`, a calibrated threshold of `0.590`, and gallery version `df0ef1d10ccb-92f18a0447f7`. The model state was `Model ready`, and the sample query was available.

## Review interaction check

The review strip loaded the calibrated `REVIEW ≥ 0.590` threshold and gallery index version. The preview remained in `READY TO REVIEW` after the first click attempt, so the search event requires a final asynchronous/browser-event check before publication.

## End-to-end confidence review verification

The default query completed browser inference in 143 ms and rendered 10 ranked gallery matches. The top score was `0.893`, above the calibrated `0.590` review threshold, so the review strip displayed `REVIEW CANDIDATE` with the explicit message that human confirmation is required. Cards above the threshold were labeled `REVIEW CANDIDATE`; lower-scoring cards were labeled `LOW CONFIDENCE`. Gallery version `df0ef1d10ccb-92f18a0447f7` remained visible.
