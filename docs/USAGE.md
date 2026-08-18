# Usage and development

Python 3.12 or later is required.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The installed commands are `ac-cfr-train`, `ac-cfr-benchmark`, `ac-cfr-evaluate`, `ac-cfr-plot-results`, `ac-cfr-download-models` and `ac-cfr-web`. Checkout-local Python wrappers are also provided for the main workflows.

## Training and resume

Start a tabular run with an explicit budget and snapshot milestones:

```bash
python train.py \
    --game leduc \
    --solver cfr_plus \
    --iterations 500000 \
    --run-id leduc-cfr-plus-final \
    --evaluation-interval 5000 \
    --checkpoint-interval 50000 \
    --snapshot-iterations 50000,200000,500000 \
    --averaging-delay 10 \
    --plot
```

Resume from the stored configuration and solver state:

```bash
python train.py --resume runs/leduc-cfr-plus-final/checkpoints/latest.npz
```

Start or resume the selected Leduc Deep CFR preset:

```bash
python train.py \
    --config configs/deep_cfr/leduc_final.toml \
    --run-id leduc-deep-cfr-final

python train.py --resume runs/leduc-deep-cfr-final/checkpoints/latest.pt
```

Tabular and neural checkpoints retain the state required for compatible resume. Playable strategy snapshots are smaller exports containing only the selected average policy and its reconstruction metadata.

## Evaluation and plots

Evaluate registered policies by strategy ID:

```bash
python evaluate.py leduc_cfr_plus_final
python evaluate.py leduc_mccfr_final
python evaluate.py leduc_deep_cfr_final
```

Plot one run or compare several runs:

```bash
python plot_results.py runs/leduc-cfr-final runs/leduc-cfr-plus-final
```

Run output belongs under ignored `runs/`, selected policy files under ignored `artifacts/`, and compact evidence under version-controlled `results/`.

## Policy files

Install every file-backed policy declared by the registry from its published release:

```bash
ac-cfr-download-models
```

Release assets can instead be installed from a local staging directory:

```bash
ac-cfr-download-models --source-directory /path/to/release-assets
```

Installation is atomic and requires the declared file size and SHA-256 checksum to match. Random and rule-based agents require no policy file.

## Web application

After installing the registered policies, start the local single-worker application:

```bash
ac-cfr-web --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Hands are intentionally stored in one process and disappear when the page is refreshed or the server restarts.

The production Docker image downloads and verifies all registered policy files during its build:

```bash
docker build -t ac-cfr-web:models-v1 .
docker run --rm -p 8000:8000 ac-cfr-web:models-v1
```

The image listens on port 8000, runs as a non-root user and uses CPU-only PyTorch.

## Cloud Run deployment

Set the target project, region and repository:

```bash
PROJECT_ID="your-project-id"
REGION="europe-west2"
REPOSITORY="ac-cfr"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/web:models-v1"
```

Enable Artifact Registry and Cloud Run, then create the Docker repository once:

```bash
gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    --project "${PROJECT_ID}"

gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format docker \
    --location "${REGION}" \
    --project "${PROJECT_ID}"
```

Authenticate Docker, tag the production image and push it:

```bash
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
docker tag ac-cfr-web:models-v1 "${IMAGE}"
docker push "${IMAGE}"
```

Deploy the public service:

```bash
gcloud run deploy ac-cfr-web \
    --image "${IMAGE}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --port 8000 \
    --cpu 1 \
    --memory 1Gi \
    --min-instances 0 \
    --max-instances 1 \
    --allow-unauthenticated
```

The single-instance limit matches the application's in-memory hand model; zero minimum instances allows it to scale down while idle.

## Checks

Run the normal development checks before committing:

```bash
ruff check .
ruff format --check .
pyright
pytest
```

The default suite excludes expensive evaluator tests. Run them separately when changing evaluator code or lookup data:

```bash
pytest -m slow
```

CI also builds the wheel and production container and verifies installed command-line entry points outside the checkout.

The Hold'em evaluator uses committed deterministic lookup tables. Regenerate them only when changing evaluator generation:

```bash
python tools/generate_holdem_evaluator.py
```
