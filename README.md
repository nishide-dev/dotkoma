# dotkoma

> **Status: active development.** The pipeline, file formats and configs may change without notice, and no dataset has been released yet.

dotkoma builds a dataset of short, looping **16×16 pixel-art animations**.

It starts from still, transparent 16×16 sprites in [Pixel Art Atlas 1.4M](https://huggingface.co/datasets/Angais/pixel-art-atlas-1.4m), asks an image-to-video model ([Wan2.2](https://huggingface.co/Wan-AI)) to animate each sprite with a single motion (blink, hop, wave, …), and keeps only the results that still read correctly once reduced back to 16×16.

The dataset is meant for training small pixel-art video models that animate a still sprite from an action label.

## How it works

```
prepare   (CPU)  pick sprites, build reference images        -> jobs.jsonl
generate  (GPU)  image-to-video for each job                 -> raw/<job_id>.mp4
finalize  (CPU)  select loop frames, key out the background, -> 16×16 RGBA frames,
                 reduce to 16×16, run quality checks            timing, preview GIF
```

- The stages communicate only through files, so GPU work can run on a remote machine such as Colab while everything else runs locally.
- Each job ID is a hash of the requested generation settings: source sprite, action, prompt, seed, model ID and revision, and inference and reference settings. Pin the model revision to a commit in the config so that the ID also identifies the weights. The revision actually loaded is recorded separately in `results.jsonl`.
- Re-running `generate` processes only unfinished jobs. Failed jobs are skipped unless `--retry-failed` is passed.
- Raw videos, job lists and finalized samples are written to a temporary file and then renamed, so an interrupted run never leaves partial output that looks complete. `results.jsonl` is an append-only log. A record counts only once its line ends with a newline, so a final line cut short by a crash is ignored and removed on the next append.
- `finalize` rebuilds samples from the raw videos. Post-processing can be changed and re-run without regenerating anything.
- By default `finalize` takes evenly spaced frames over the whole clip. A frame window or explicit frame indices can be set per clip in `selections.jsonl`.
- Every frame of a clip is reduced with the same grid and palette rules, so real motion such as a hop is preserved rather than re-centred away.
- Every sample can be traced back to its origin through its `job_id`:
  - `sample.json` holds the source sprite (`row_id`, dataset revision) and the requested action.
  - `jobs.jsonl` holds the prompt, seed and inference settings.
  - `results.jsonl` holds the resolved model revision and runtime.
- Automatic checks (frame count, empty or identical frames, loop seams, cropped motion, unstable background) are kept separate from human review decisions.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- A CUDA GPU with PyTorch already installed, for the `generate` stage only. Google Colab GPU runtimes ship with it. The local environment does not install PyTorch.

## Quick start

```sh
git clone https://github.com/nishide-dev/dotkoma.git
cd dotkoma
uv sync
uv run pytest
```

Run the whole pipeline on the CPU, using the bundled test sprites and a synthetic `fake` video backend:

```sh
uv run dotkoma prepare  configs/smoke.toml
uv run dotkoma generate configs/smoke.toml
uv run dotkoma finalize configs/smoke.toml
```

Results are written to `work/smoke/final/<job_id>/`.

## Generating with Wan2.2

Two configs pull sprites from Pixel Art Atlas:

- `configs/pilot.toml` targets `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (fits a 24 GB GPU).
- `configs/pilot-a14b.toml` targets `Wan-AI/Wan2.2-I2V-A14B-Diffusers` with 4-step distillation LoRAs and fp8 weight storage. It runs on a 40 GB GPU and needs about 85 GB of CPU RAM while loading.

```sh
uv run dotkoma prepare configs/pilot.toml
```

Copy `src/`, `scripts/` and `work/pilot/` to a GPU machine and run:

```sh
python scripts/colab_pilot.py work/pilot
```

This installs Diffusers and the other generation dependencies on top of the existing PyTorch, and records the versions it used in `colab_env.json`. Bring `work/pilot/raw/`, `work/pilot/results.jsonl` and `work/pilot/colab_env.json` back, then run:

```sh
uv run dotkoma finalize configs/pilot.toml
```

## Configuration

A run is described by a single TOML file:

- **Source:** the dataset, its pinned revision, and which shards to read.
- **Sprites:** which subjects to animate, and with which actions.
- **Actions:** one or more prompt wordings per action. All wordings of an action share a seed, so their acceptance rates can be compared directly.
- **Generation:** model, resolution, frame count and inference settings.
- **Post-processing:** frames per loop, keying tolerance, palette handling and check thresholds.

See [`configs/smoke.toml`](configs/smoke.toml) and [`configs/pilot.toml`](configs/pilot.toml).

## Output format

Each finalized sample is a directory:

```
final/<job_id>/
  frames/000.png …   ordered 16×16 RGBA frames (canonical data)
  sample.json        durations (ms), loop flag, source sprite, action, prompt variant, checks
  preview.gif        transparent looping preview, regenerated from the frames
```

## Further reading

- [`docs/pipeline.md`](docs/pipeline.md): file contracts, frame selection overrides and quality-check flags (developer notes, in Japanese).

## License

- **Code:** MIT. See [LICENSE](LICENSE).
- **Data:** generated data is licensed separately.
  - The source sprites are released by their author under CC0 1.0, to the extent of the rights they can waive. The upstream model terms still apply.
  - See [`docs/dataset-card.md`](docs/dataset-card.md) for provenance and terms. These will be finalized before any data is published.
