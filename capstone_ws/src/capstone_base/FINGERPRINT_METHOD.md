# capstone_base — fingerprint discovery method (write-up reference)

The base station is LABEL-FREE: it clusters only embeddings, never sees truth. A
reader reconstructing the method from a paper must find every step below — in
particular the **centroid-merge post-process is a distinct step**, not part of
"eigengap."

## Pipeline
1. **Per-burst track.** Pool the per-receiver embeddings of one `burst_seq`
   (same emission ⇒ same unit) → mean + L2. Accumulate tracks over time.
2. **Raw eigengap (published as-is).** On a density-controlled subsample
   (≤ `COMPUTE_CAP=300` tracks): build a symmetrized kNN connectivity graph,
   take the **normalized-Laplacian eigenvalue spectrum**, and set the raw estimate
   `k_raw = argmax_{k∈[2,kmax]} (λ_k − λ_{k−1})`. This matches the canonical RFF
   `gen_rff/demo_router/base_station.eigengap_k` verbatim. The spectrum
   (`eigenvalues`) and winning gap (`eigengap_value`) are published on
   `FingerprintSet` **unmodified** — they describe the raw spectral analysis.
3. **Centroid-merge refinement (SEPARATE step, threshold stated).** Raw eigengap
   has a Goldilocks per-cluster sample count (~15–30 tracks/unit); at higher track
   density its spectrum smooths and `argmax`-gap drifts HIGH, **over-splitting a
   single unit into fragments**. Because no fixed track budget fixes this across
   K=2..8 (low K wants few tracks, high K wants many), we partition into `k_raw`
   clusters and then **union-merge any two clusters whose L2 centroids have cosine
   > `MERGE_TAU = 0.85`**. `k_final` = merged cluster count = the published
   `k_estimated`. `MERGE_TAU` sits ABOVE the provisioned between-unit worst
   centroid cosine (0.785, by unit selection) so distinct units never merge, and
   BELOW same-unit fragment similarity so over-split fragments do. A genuine
   near-duplicate unit-pair (cosine > τ) correctly merges to one ID — an honest
   "indistinguishable" outcome, not an error.
4. **Assignment + confidence.** Partition the full track set into `k_final`
   clusters, assign global IDs `d1..dK` (first-seen order). Confidence per unit:
   - `0` (explicitly WITHHELD) if the cluster has `< min_tracks_confident=15`
     accumulated tracks (accumulation gate), **or**
   - the cluster's **mean silhouette (cosine), clamped [0,1]** otherwise — an
     identity-**separability** measure, NOT tightness. Cohesion alone is fooled by
     low track diversity (a unit whose tracks haven't averaged across conditions is
     internally tight but sits close to other units); silhouette is low exactly in
     that case. See "channel-dominated embedding" finding.

## Key empirical facts behind the design
- **Channel-dominated embedding (outdoor):** a unit's own Loc1-vs-Loc4 centroids
  are ~orthogonal (cosine ≈ −0.2), more different than *other* units. Device
  identity emerges only when a track averages across locations; single-location
  makes units ≈0.979-identical. So tracks must span condition variety — and in
  the SIMULATOR that variety is injected by pool sampling, which means the
  count-based gate looks sufficient in sim but is not sufficient in principle
  (real hardware could under-accumulate diversity while clearing the count).
- **Near-duplicate units:** 27/31 XIAO units share a >0.9-cosine partner; the
  demo/gate unit set is provisioned to be mutually <0.785 (disclosed selection).
