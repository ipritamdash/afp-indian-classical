# Deploy your own copy of this Space

This folder contains everything needed to host your own clone of
[Tachyeon/afp-indian-classical-demo](https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo)
on **your** Hugging Face account.

There are two paths:

- **[Path A](#path-a-one-shot-script-recommended) (recommended): one Python command**
- **[Path B](#path-b-fully-manual): UI-only, no scripting required**

Both produce an identical, working public demo at
`https://huggingface.co/spaces/<your_user>/<your_repo>`.

---

## Prerequisites (both paths)

1. **Hugging Face account** — sign up free at https://huggingface.co/join
2. **HF API token with `write` scope** — create at https://huggingface.co/settings/tokens
3. **Free disk space**: ~600 MB temporarily (for snapshot)
4. **Python 3.10+** with `pip` (Path A only)

You do **not** need a GPU. You do **not** need the Saraga MP3s. The
public Space's pre-computed reference embeddings are downloaded automatically
from Hugging Face Hub.

---

## Path A: one-shot script (recommended)

From this directory (`demo/`), run:

```bash
# 1) Install dependencies
pip install 'huggingface_hub>=0.30'

# 2) Set your token (https://huggingface.co/settings/tokens — needs write scope)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx

# 3) Deploy
python deploy.py --repo-name my-afp-demo
```

The script will:

1. Verify your token
2. Snapshot the upstream Space (~537 MB; pulled from HF's CDN, fast)
3. Create `<your_user>/my-afp-demo` on Hugging Face
4. Upload everything
5. Wait for the build + print the live URL when ready

Total wall time: **~5-10 minutes**, mostly waiting for Spaces to build the
TF + kapre + Gradio environment.

### Common flags

```bash
# Private Space (only you can see it)
python deploy.py --repo-name my-afp-demo --private

# Deploy under an organization you control
python deploy.py --user my-org --repo-name afp-demo

# Dry-run (verify token + show plan, don't actually deploy)
python deploy.py --repo-name my-afp-demo --dry-run
```

---

## Path B: fully manual (no scripting)

Use this if you want to inspect every file before pushing, or if Path A fails
for any reason.

### Step 1 — Create an empty Space

1. Go to https://huggingface.co/new-space
2. **Name**: anything you want, e.g., `my-afp-demo`
3. **Space SDK**: choose **Gradio**
4. **Hardware**: **CPU basic (free)** is sufficient
5. **Visibility**: **Public** (recommended) or **Private**
6. Click **Create Space**

You now have an empty Space at `https://huggingface.co/spaces/<your_user>/my-afp-demo`.

### Step 2 — Clone the upstream Space locally

```bash
# Install git-lfs first (https://git-lfs.com/)
git lfs install

# Clone the public source-of-truth
git clone https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo
cd afp-indian-classical-demo
```

This pulls **everything** including model weights and embeddings (~537 MB via LFS).

### Step 3 — Point your local clone at your new Space and push

```bash
# Replace the remote with your own Space URL
git remote set-url origin https://huggingface.co/spaces/<your_user>/my-afp-demo

# Authenticate (use your HF token as the password when prompted)
git push origin main
```

You'll be asked for credentials:
- **Username**: your HF username
- **Password**: paste your HF token (it won't echo; that's normal)

The push transfers ~537 MB. Coffee break.

### Step 4 — Wait for the build

1. Open `https://huggingface.co/spaces/<your_user>/my-afp-demo`
2. Watch the build log (top-right of the Space page)
3. Stages: `BUILDING` → `APP_STARTING` → `RUNNING`
4. **Total time**: ~5-15 minutes (downloading TF + kapre wheels, then warming the encoder)

When the stage hits `RUNNING`, your demo is live.

---

## Troubleshooting

### `huggingface_hub` says "not authorized"
- Your token doesn't have `write` scope. Regenerate at https://huggingface.co/settings/tokens
  and tick **Write** access.

### Build stalls at `BUILDING` for >20 min
- Hugging Face's infra is sometimes slow on first build (downloading TF 2.19 wheels).
  Reload the page and check the build log; first build can take 10-20 minutes.

### Build fails with `tensorflow==2.19.0` not found
- Make sure `python_version: "3.11"` is in the README YAML frontmatter. Python 3.13
  (HF Spaces' new default) has no TF 2.19 wheels — only 2.20+. We pin 3.11.

### Build fails with numpy / librosa conflict
- The pinned versions in `requirements.txt` (`numpy>=2.0,<2.2`, `librosa>=0.11,<1.0`) are
  the intersection of TF 2.19 + kapre 0.3.7 (PyPI rebuild). Do not relax these.

### App starts but inference fails with "checkpoint silently partial-restored"
- `TF_USE_LEGACY_KERAS=1` must be set **before** `import tensorflow`. It is, in `app.py` line 19.
  Don't move the env-var line; don't import any TF-using package before it.

### Space sleeps after 48 h of inactivity
- This is HF Spaces' free-tier policy. Anyone visiting wakes it up automatically (~90s cold start).
- To keep it always-on, upgrade to CPU Upgrade ($0.03/h) at Settings → Hardware.

---

## What's in this folder

```
demo/
├── app.py                          # Gradio app (single file)
├── requirements.txt                # Pinned Python deps (TF 2.19 + kapre + gradio)
├── packages.txt                    # apt-get packages (ffmpeg)
├── README.md                       # Space card (HF reads YAML frontmatter)
├── .gitattributes                  # LFS patterns for *.mm, ckpt-*, *.parquet
├── DEPLOY.md                       # this file
├── deploy.py                       # one-shot deployment script
├── artifacts/                      # NOT in git — fetched from upstream Space
│   ├── ckpt-30.data-00000-of-00001 # 194 MB Recipe v3 seed 42 weights
│   ├── ckpt-30.index               # 102 KB
│   ├── recipe_v3.yaml              # NAFP config
│   ├── ref_embs.mm                 # 354 MB precomputed Saraga embeddings
│   ├── ref_segment_lookup.parquet  # 5 MB segment → ref_id lookup
│   ├── refs_hindustani.csv         # 357-track ref metadata
│   ├── refs_carnatic.csv
│   └── threshold.json              # calibrated FPR≤5% threshold
└── upstream/                       # patched NAFP model code (MIT, Chang et al. 2021)
    ├── config/
    └── model/
```

The `artifacts/` and `upstream/` directories are downloaded as part of the
snapshot in Path A, or auto-cloned by `git lfs` in Path B.

---

## What about Saraga audio?

You do **not** need any Saraga MP3 files. The demo ships **pre-computed embeddings**
of the Saraga library (the `ref_embs.mm` file). These are:

- 690,414 segments × 128-D float32 vectors
- A one-way fingerprint hash — audio cannot be reconstructed from them
- Derivative of Saraga 1.5 audio under CC-BY-NC-SA 4.0

This is exactly how Shazam works: your phone has the fingerprint, not the
original songs. The embeddings are enough to do retrieval.

---

## License

- **App code**: MIT (this folder, except `upstream/`)
- **Model weights** (`artifacts/ckpt-30.*`): MIT (trained on FMA-medium)
- **Reference embeddings** (`artifacts/ref_embs.mm`): CC-BY-NC-SA 4.0 (derivative of Saraga 1.5)
- **Patched NAFP upstream code** (`upstream/`): MIT (Chang et al. 2021)

**Non-commercial use only** because the embeddings inherit Saraga's NC clause.
If you redeploy publicly, keep the about-tab disclaimer and don't market it as commercial.

---

## Test the deployed Space

After deployment, your Space accepts:

- **Upload** any WAV/MP3/M4A/OGG/WebM clip (1-30 seconds)
- **Microphone** record (works on desktop; iOS Safari is flaky)

Expected outcomes (use clips from the
[demo_clips/](https://github.com/ipritamdash/afp-indian-classical/tree/main/demo_clips)
pack):

- **Saraga in-library clip** → green "✓ Match found (high confidence)"
- **Bollywood / Western / random audio** → red "❌ No match in library"
- **Silent / noise** → red "❌ No match in library"

If the deployed Space doesn't behave like this, something went wrong with the
upload. Re-clone from upstream and retry Path B.

---

## Questions / problems

Open an issue on the [GitHub repo](https://github.com/ipritamdash/afp-indian-classical)
or on the [upstream Space's community tab](https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo/discussions).
