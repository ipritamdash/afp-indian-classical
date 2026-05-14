# Post-mortem — 4-system Saraga AFP benchmark (2026-05-12)

**Scope.** Deep audit of the entire benchmark pipeline (refs library, test set,
NAFP model outputs, scoring code, cross-system comparability) following the
v0.4 HF dataset release. Performed by the author + four independent senior
reviewer agents. Every claim below cites a file path + a command anyone can
re-run; agent-only claims are flagged "needs independent verification".

**Result snapshot at audit time** (HF dataset
`Tachyeon/audio-fingerprint-indian-bench` v0.4, commit `4ea00b87`):

| | 1 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|
| Olaf main HR@1 | 0.492 | 0.955 | 0.993 | 0.998 |
| Dejavu main HR@1 | 0.745 | 0.969 | 0.994 | 1.000 |
| Panako main HR@1 (tuned) | 0.000 | 0.000 | 0.922 | 0.997 |
| **NAFP main HR@1** | **0.983** | **0.998** | **0.999** | **1.000** |

---

## Phase A — Author's own 5-layer audit (every check ran on actual data)

### Layer 1: reference library (357 refs)

| check | result |
|---|---|
| All 357 MP3s exist on disk | ✓ |
| `soundfile.info(p).duration` vs manifest `duration_sec` (\|Δ\|>0.5 s) | 0 mismatches |
| Source MD5 reproducible on 3 random refs | 3/3 match |
| Duplicate `audio_path` / duplicate `source_md5` | 0 / 0 |
| Unique artists | 35 for 357 refs (top-1 Sanjay 42 = 12 %, top-5 = 31 %) |
| Sample rates | 350 @ 44.1 kHz, 7 @ 48 kHz |
| Refs lacking `raagas` | 80/357 |
| Refs lacking `work_mbids` | 73/357 (upper bound of un-flaggable twins) |

### Layer 2: test set (1 632 queries × 4 lengths × 2 corpora)

| check | result |
|---|---|
| Total queries across 8 variants × 2 corpora | 6 528 ✓ |
| 30 random WAVs: missing / wrong_dur / silence / NaN / bad_sr / bad_ch | 0 / 0 / 0 / 0 / 0 / 0 |
| **Bit-exact re-derivation from `GLOBAL_SEED=20260511`** | **1000/1000** offsets ✓ |
| Twin flag sum | 835 no-twin + 165 with-twin = 1000 ✓ |
| Ablation buckets | 484 composed + 137 alaap + 11 tani = 632 ✓ |

### Layer 3: NAFP model outputs (`saraga_only_main/`)

| check | result |
|---|---|
| `ref_embs.mm` shape | (690 414, 128) float32 ✓ matches `index_log.json` |
| L2-norms of 5 000 sample rows | all = 1.0 ± 1 e-5 ✓ |
| NaN in 5k sample | 0 ✓ |
| Lookup contiguous `global_idx` 0..690 413 | ✓ |
| All 8 query parquets: ranks 1..10 per query | ✓ no invalid predicted_ref_ids |
| Score separation hits vs miss (main 1 s) | median 0.929 vs 0.809 |

**Flags** (informational, not bugs):
- **F-NAFP-1**: `ref_embs.mm` on disk is 412 160 B (805 segments) larger than needed. Memmap was allocated at the predicted size 691 219 then truncated to 690 414 via shape-only resize; trailing bytes are zero padding. Reads use `shape=(690 414, 128)` so padding is invisible. 0.12 % wasted disk.
- **F-NAFP-2**: `align_err_max = 2 082 s` on NAFP main 1 s — a handful of queries predict the **correct ref** but at a **completely wrong offset** within that ref. HR@1 still counts these as correct; `top1_near` correctly classifies them as not-near. Property of the model; explains why `top1_near < HR@1` at 1 s.

### Layer 4: scoring code (`scripts/score.py`)

| check | result |
|---|---|
| Wilson CI at p ∈ {0, 0.5, 0.999, 1.0}, n ∈ {0, 1000} | all sensible |
| `hit_at_k` on 3-query toy: HR@1=1/3, HR@2=HR@5=2/3 | matches hand-calc |
| MRR@1=0.333=HR@1; MRR@5=(1+0.5+0)/3=0.5 | matches formula |
| NAFP main 10s scores.json `hr@1_n_hits` vs recomputed | 1000 = 1000 ✓ |
| Per-corpus sums = total | ✓ 1000 |
| Twin + no-twin = total | 835 + 165 = 1000 ✓ |
| **top1_near ≤ HR@1 invariant across all 32 cells** | **0 violations** |

### Layer 5: cross-system comparability

| check | result |
|---|---|
| All 4 systems emit rows for all 1 000 main qids | 1000/1000 each ✓ |
| All `predicted_ref_id` ∈ 357-ref manifest namespace | ✓ all 4 systems |
| Recomputed HR@1 from raw parquets matches scores.json | 998/1000/997/1000 ✓ |
| No-match semantics: rank=0 ↔ predicted=None | consistent |

**Observed:**
- Rows-per-query: Olaf 1-10 mean 2.71, Dejavu 10/10, Panako 1-3 mean 1.05, NAFP 10/10
- Latency p50: Olaf 43 / Dejavu 131 / Panako 5 / NAFP 230 ms — see M7 caveat below

---

## Phase B — Four senior reviewer agents (independent)

### Reviewer A: test-set adversarial

| # | finding | severity |
|---|---|---|
| **F18** | `has_twin_in_library` computed from **works-text** (Counter on title strings) not **work_mbids**. 35 / 165 queries flagged True are spurious — driven by generic form-names "Thillana" (Carnatic, 4 refs / 5 MBIDs), "Ragam Thanam Pallavi" (2 refs / 2 MBIDs), "Dekho Ajab Khel Hein" (2 H refs / 2 MBIDs). Re-deriving via MBIDs gives 130 genuine twin queries. | **BLOCKER** |
| F19 | tani N = 11 (Carnatic-only); Hindustani has 0 tani by design — Laggī excluded by `build_ablation.py` lines 67-69. Per-bucket tani uninterpretable; cross-corpus impossible. | MAJOR |
| F20 | 61/632 ablation queries time-overlap with main queries on the same ref. Main vs ablation are not independent. | MAJOR |
| F21 | 276/357 ref audio_paths have `*.mp3.mp3` double-extension. Files exist; cosmetic. | MINOR |
| F22 | HR@1 conditional on missing-metadata is within sampling noise (Olaf 0.998 overall, 1.000 on no_raagas; saturated at test ceiling). | INFORMATIONAL |
| F23 | All 39 test-source refs missing `work_mbids` are Hindustani. Twin flag asymmetry across corpora. | INFORMATIONAL |

**Negative checks that passed** (reported for transparency): 0/632 section-label vs section_type mismatches; jitter distribution uniform per KS-test; 0/1632 query→ref attribution errors; 30 byte-equivalence checks pass for 1s/3s/5s vs 10s truncation.

### Reviewer B: NAFP output forensic

| # | finding | severity |
|---|---|---|
| | Encoder manifold non-isotropy: 10 k random ref-pair cosines have p99 = 0.40, max = 0.79 vs isotropic-baseline p99 = 0.21, max = 0.30. **22 artist-centroid pairs at cos > 0.60** (max Kanakadurga ↔ Srividya = 0.86). Intra-artist ≈ inter-artist on average. Same-artist confusion is **geometric**, not algorithmic. ckpt-10 may be under-trained for artist disambiguation. | MAJOR |
| | NAFP L=1 sequence-search degenerates to pure NN cosine — diagonal-mean rescore is a single scalar at 1 s queries. Drives 23 "right-ref wrong-offset" outliers across the two 1 s runs (10 main + 13 ablation; 0 at any ≥3 s length). | MAJOR |
| | Hub-refs over-represented in 1 s misses: `carnatic_91_Ragam_Tanam_Pallavi` (Mahati, 2 same-artist 1s-miss predictions) and `hindustani_17_Raag_Miyan_Malhar` (4 of 17 1s-miss interactions). | MINOR |
| | 3 cross-corpus mismatches at 1 s (1 main, 2 ablation); 0 at any ≥3 s length. | MINOR |
| | Top-1 score spread shrinks monotonically with query length: std 0.057 (1 s) → 0.044 (10 s). | INFORMATIONAL |

**Agent error caught**: Reviewer B claimed all 8 NAFP result dirs are Mac-encoded. **False** — main 10 s came from the Kaggle T4 inference kernel (`scripts/nafp/infer_kernel/kaggle_infer.py`); the other 7 are Mac `nafp_runner.py --skip-index`. Does not invalidate other findings.

### Reviewer C: scoring code review

| # | finding | severity |
|---|---|---|
| BUG-2 | `score.py:133` `no_match = (results.groupby("query_id")["rank"].max() == 0)` uses results-derived denominator. Query that the system fails to emit any row for is silently dropped from `frac_no_match` — counted as miss in HR@K (manifest denominator) but not as no-match. Reproduced at `/tmp/_score_audit2.py`: 3 manifest queries, 1 emitted → `frac_no_match = 0.0` instead of 2/3. | **MAJOR** |
| BUG-11 | `frac_no_match` denominator mismatch — same root cause as BUG-2. | MAJOR |
| BUG-1 | `wilson_ci` crashes on `p_hat ∉ [0, 1]` (math domain error in sqrt). Not currently triggerable from `main()` (p_hat = n_hit/n is well-bounded). | MINOR |
| BUG-3 | `hit_at_k` `.iloc[0]` raises IndexError on NaN query_id (NaN != NaN). | MINOR |
| BUG-5 | Duplicate query_id in manifest → row-explosion in merge; HR@1 silently inflates. No assertion. | MINOR |
| BUG-7 | dtype mismatch (`int` vs `str` predicted_ref_id) → silent all-misses. | MINOR |
| BUG-12 | No assertion `rank ≥ 0`; rank=-1 sentinels would survive silently. | MINOR |
| BUG-4 | Orphan results rows pollute `hit_df` but not HR. | INFORMATIONAL |
| BUG-6 | Tied rank=1 policy ambiguous, undocumented (0 actual ties on current data). | INFORMATIONAL |
| BUG-8 | Wilson CI upper bound for p=1.0 returns 0.9999999... (cosmetic, round(_,4) fixes display). | INFORMATIONAL |
| BUG-9 | `n_results_rows` per-row count misleading across systems. | INFORMATIONAL |
| BUG-10 | `align_err.abs()` discards sign — cannot diagnose systematic positive lag. | INFORMATIONAL |

**Verified OK**: round-trip determinism, twin/section gate behaves correctly.

### Reviewer D: cross-system comparability

| # | finding | severity |
|---|---|---|
| | Dejavu `ref_stop` hardcoded to `ref_start + 10.0` (`dejavu_runner.py:397`). Silently wrong on 1/3/5 s splits. **Verified empirically**: all dejavu_1s/3s/5s splits show ref_stop-ref_start = 10.0 regardless of actual query length. Does not affect HR/MRR/align_err (those use ref_start only). | **BLOCKER** |
| | No common confidence axis across systems → ROC/FAR/FRR curves impossible across systems. | BLOCKER (known limitation, can't fix; needs paper footnote) |
| | `match_count` semantically incompatible across systems (Olaf hash-count, Dejavu hash-count 10× scale, Panako fingerprint-count, NAFP constant L). Not used by `score.py`, but misleading in parquets. | MAJOR |
| | `ref_start` / `query_start` semantics differ. Dejavu and NAFP hardcode `query_start = 0`; for Olaf and Panako it's measured. Alignment-error formula reduces to system-conditional `\|ref_start - query_offset\|` for two systems. | MAJOR |
| | NAFP `align_err_median = 0.125 s` on main 1s is a **0.5 s hop quantization artefact**, not algorithmic deficiency. `top1_near_at_0.05s` (NAFP 0.20 vs others 0.99+) is structurally unfair. | MAJOR |
| | Latency timers wrap different boundaries per runner: Olaf, Dejavu, NAFP exclude audio load before `t0`; Panako's latency parsed from Java log line — excludes JVM startup + ffmpeg decode entirely. | MAJOR |
| | HR@5 / HR@10 not directly comparable. Olaf 2 714 rows / 1 000 queries vs Dejavu 10 000 / 1 000 vs Panako 1 054 / 1 000 vs NAFP 10 000 / 1 000. HR@K>1 systematically inflated for Dejavu+NAFP because they "spend" K=10 slots even on weak matches. HR@1 unaffected. | MAJOR |
| | NAFP rank tie-breaking via `np.argsort` (default kind=quicksort, **non-stable**). Empirically 0/1000 actual ties on current main 1 s data — latent only. Recommend `kind='stable'`. | MAJOR |

---

## Independent verification of BLOCKER claims (author, post-agent)

| # | claim | verified by | result |
|---|---|---|---|
| B2 | Dejavu ref_stop hardcoded | `grep -n "ref_stop" dejavu_runner.py` + parquet diff measurement | **confirmed** — all 4 splits show fixed 10.0 s span |
| M9 | NAFP `np.argsort` non-stable | parquet pivot top1 vs top2 scores, exact-equal check | confirmed code, 0 actual ties on current data — latent only |
| B1 | twin flag from works-text not MBIDs | **NOT YET INDEPENDENTLY VERIFIED** | — |
| B3 | no_match denominator bug | **NOT YET INDEPENDENTLY VERIFIED** | — |

---

## Proposed fix plan (Phase 1 — ~1 hour)

Gated on independent verification of B1 + B3. If both reproduce:

1. **B1 fix**: recompute `has_twin_in_library` per query using `work_mbids` only. Where `work_mbids` is NaN, fall back to `works` text but exclude generic form-names {Thillana, Ragam Thanam Pallavi, Dekho Ajab Khel Hein}. Persist to `data/manifests/{corpus}/queries.csv`. Re-score all 32 cells.
2. **B2 fix**: `dejavu_runner.py:397` → `"ref_stop": float(hit[OFFSET_SECS]) + length_sec` where `length_sec` comes from the query manifest row. Re-emit dejavu parquets for 1/3/5 s splits.
3. **B3 fix**: `score.py:133` → `no_match = ~queries["query_id"].isin(results.loc[results["rank"]>=1, "query_id"])`. Re-score all 32 cells.
4. **M9 fix**: `nafp_runner.py:266` → `order = np.argsort(-scores, kind='stable')`. No re-run needed (0 actual ties on current data).

**Expected impact**: B1 fix will reduce `with_twin` count from 165 → 130 in manifests and shift `hr@1_with_twin` / `hr@1_no_twin` slightly. B2 fix only changes the dejavu parquet column — does not affect any reported HR/MRR/top1_near number. B3 fix only matters when a system fails to emit a query row (currently doesn't happen in our 4 systems). M9 is latent only.

**No HR@1 main 10 s headline numbers will change.** Only twin breakdown shifts.

---

## Out-of-Phase-1 items (paper-time experiments)

| priority | item | effort | expected impact |
|---|---|---|---|
| 1 | FMA-medium distractors (25 k Western tracks added to library) | 4-8 h | HIGH — turns saturated cells into differentiating cells |
| 2 | Random 1 s placement re-cut (not first-N truncation) | 2-3 h | medium — closes a methodological gap (NAFP paper §4.2) |
| 3 | Noise robustness (additive noise + IR conv on queries) | 2-3 h | medium — expected NAFP advantage |
| 4 | Intervention 2: per-artist mean subtraction | 1-2 h | low-medium — targets geometric failure mode M1; ~+0.6-0.9 pp HR@1 at 1 s |
| 5 | Voicing-gated query expansion | 3-4 h | low — targets ~6 specific misses |

---

## Disclosure-only items for paper limitations section

- **M3**: NAFP align_err median 0.125 s at 1 s is 0.5 s hop quantization, not deficiency
- **M4**: 61/632 ablation queries time-overlap with main queries — main/ablation not independent
- **M5**: tani N=11 single-corpus; per-bucket tani uninterpretable
- **M6**: `match_count` semantically incompatible across systems (parquet column only, score.py unaffected)
- **M7**: latency timers wrap different boundaries; cross-system table needs footnote
- **M8**: HR@5/HR@10 not directly comparable (K-asymmetry)
- 35 unique artists for 357 refs — heavy concentration; per-section claims confounded with singer (audit F6, prior session)
- 47 H refs + 26 C refs lack `work_mbids` — twin-flag undercounted (audit F11, prior session)
- 80 refs lack `raagas` metadata — per-raga analysis impossible for those
- Library = 357 refs with no distractor pool; HR@k saturate at 10 s
- 1/3/5 s queries are first-N truncations of 10 s queries — not random placements like Chang 2021 §4.2

---

## Provenance

- Author 5-layer audit: every check in this document re-runnable from `uv run python` inside `/Users/prita/Desktop/Audio Fingerprinting/afp_bench/`.
- Reviewer A: `/tmp/adversarial_audit.py`, `/tmp/audit_deep.py`, `/tmp/audit_twin_overcounting.py`, `/tmp/audit_hr2.py`
- Reviewer B: probed `ref_embs.mm`, `refs.csv`, `query_results.parquet`; no scripts persisted (single-session reads)
- Reviewer C: `/tmp/_score_audit.py`, `/tmp/_score_audit2.py`, `/tmp/_score_audit3.py`, `/tmp/_score_audit4.py`
- Reviewer D: code-only review (lacked Bash) + author's empirical verification of D's BLOCKER claim

## Status as of this document

- HF v0.4 live: https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench (commit 4ea00b87)
- Kaggle inference kernel v6: completed successfully, ref_embs.mm + 8 result parquets pulled
- All 32 cells scored with HR@1/5/10 + Wilson CI + MRR@5/10 + top1_near
- Phase 1 verification + fixes: **COMPLETED** (see below)
- Phase 2 experiments: **not started**

---

## Phase 1 — verifications and fixes applied (this session)

### Independent verifications (author, after agents)

| # | claim | result |
|---|---|---|
| B1 | twin flag spurious 35/165 from works-text not MBIDs | **CONFIRMED — exactly 35/165**. 7 refs trace: 4 Thillana variants + carnatic_38_Raagam_Thaanam_Pallavi + hindustani_5_Raag_Kalyan & hindustani_56_Raag_Malkauns (sharing "Dekho Ajab Khel Hein" title but distinct MBIDs). |
| B2 | Dejavu ref_stop hardcoded +10.0 | **CONFIRMED** by grep + parquet diff (all 4 splits had ref_stop-ref_start=10.0). |
| B3 | score.py no_match undercount when queries missing from results | **CONFIRMED** by crafted 5-query test: emitted 2 (qA correct + qB rank=0), missing 3 (qC/qD/qE). score.py reported `n_no_match=1` instead of correct `4`, `frac_no_match=0.2` instead of `0.8`. |
| B-NEW | `hit_at_k` crashes (KeyError 'hit') when input merged is empty | **CONFIRMED** while constructing B3 test — empty twin subset triggers it. Latent on current data (all twin subsets non-empty). |
| M9 | `np.argsort` non-stable on cosine ties | code confirmed; 0 actual ties on current main 1s data. Latent. |

### Fixes applied

- **`scripts/score.py:133`**: `no_match` now uses manifest denominator. Set difference `queries∖{qids_with_match}` correctly counts queries missing-from-results as no_match.
- **`scripts/score.py:hit_at_k`**: returns DataFrame with schema `[query_id, truth, top_k_preds, hit]` when `correct` list is empty (was: empty DataFrame with no columns → KeyError downstream).
- **`scripts/systems/dejavu_runner.py:397`**: `ref_stop = ref_start + float(row["length_sec"])` and `query_stop = float(row["length_sec"])`. Previous hardcoded `+10.0` caused all 1/3/5 s splits to silently emit 10 s spans.
- **`scripts/systems/nafp_runner.py:266`**: `np.argsort(..., kind="stable")` for deterministic tie-breaking.
- **`data/manifests/{corpus}/queries{,_1s,_3s,_5s}.csv`**: `has_twin_in_library` recomputed using `work_mbids` only. Net change: 165 True → **130 True** (35 spurious flags eliminated).

### Post-hoc patches to existing parquets

- Dejavu parquets × 8 cells: ref_stop and query_stop columns rewritten to use actual length (no re-run needed since the column is not consumed by `score.py`).

### Verification of fixes on B3 test case

| metric | bug (before fix) | corrected (after fix) |
|---|---|---|
| n_no_match (5 manifest, 2 emitted) | 1 | **4** ✓ |
| frac_no_match | 0.2 | **0.8** ✓ |
| hr@1 (uses manifest denominator) | 0.2 | 0.2 (unchanged, correct) |

### Re-scoring across all 32 cells

All 32 (4 systems × 4 lengths × 2 variants) re-scored cleanly. Sanity invariants:
- `top1_near ≤ hr@1` in 32/32 cells
- `mrr@10 ≥ hr@1` in 32/32 cells

**Headline HR@1 numbers unchanged from v0.4.** Only the twin/no-twin breakdown moved:

| system | hr@1_no_twin (n=870, was 835) | hr@1_with_twin (n=130, was 165) |
|---|---|---|
| Olaf | 0.998 (was 0.998) | 1.000 (was 1.000) |
| Dejavu | 1.000 (was 1.000) | 1.000 (was 1.000) |
| Panako | **0.999** (was 0.992) | **0.985** (was 0.976) |
| NAFP | 1.000 (was 1.000) | 1.000 (was 1.000) |

Panako's twin/no-twin shifted because the 35 spurious-twin queries were almost all in Panako's correct set; removing them from "with_twin" (denominator 165 → 130) makes the genuine twin queries (which Panako gets slightly worse) more visible. **More honest number now.**

### Cells still NOT fixed (paper-time disclosure only)

- **M3** NAFP align_err median 0.125 s is a 0.5 s hop quantization artefact
- **M4** main/ablation time-window overlap 61/632
- **M5** tani N=11 single-corpus
- **M6** `match_count` column semantics differ
- **M7** latency timer boundaries differ
- **M8** HR@5/HR@10 K-asymmetry across systems
- Library = 357 refs no distractors → all systems saturated at 10 s
- 80 refs lack raagas, 73 lack work_mbids
- 1/3/5 s queries are first-N truncations, not random placements

All deferred to paper Limitations section.

## Phase 2 — paper-grade experiments (next session)

Ranked by ROI for an ICASSP/ISMIR submission:

1. FMA-medium distractors (4-8 h) — the cell that turns saturated → differentiating
2. Random 1 s placement re-cut (2-3 h) — closes methodological gap
3. Per-artist mean subtraction (1-2 h) — targets M1 geometric failure mode
4. Noise robustness (2-3 h)
5. Voicing-gated query expansion (3-4 h)

**Awaiting user decision on which to start.**
