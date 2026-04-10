# LoRA A/B Experiment: seed-only vs seed+synthetic

## Goal

Check whether adding synthetic `text + image` pairs improves a diffusion model after LoRA fine-tuning.

## Data

- Proxy seed mode: no, used the provided seed image manifest
- Seed train size: 48
- Synthetic train size: 50
- Seed + synthetic train size: 98
- Validation size: 8
- Test prompts/images: 8

## Training

- Base model: `segmind/tiny-sd`
- LoRA rank: `4`
- Learning rate: `0.0001`
- Train steps per run: `60`
- Resolution: `256x256`

## Metrics

| Metric | Seed only | Seed + synthetic | Delta | Direction |
| --- | ---: | ---: | ---: | --- |
| `clip_text_image_mean` | 0.26917 | 0.271936 | 0.002766 | higher_is_better |
| `clip_kid_proxy` | 0.00051785 | 0.00051403 | -4e-06 | lower_is_better |
| `diversity_score` | 0.426146 | 0.437324 | 0.011178 | higher_is_better_until_too_high |
| `duplicate_rate` | 0 | 0 | 0 | lower_is_better |
| `technical_image_score_mean` | 0.7461 | 0.7358 | -0.0103 | higher_is_better |
| `latency_seconds_mean` | 0.3865 | 0.3885 | 0.002 | lower_is_better |

- CLIP metric model: `openai/clip-vit-base-patch32`
- CLIP load error: `None`

## Artifacts

- A/B preview grid: `eval/ab_preview_grid.png`
- Metric bar chart: `eval/metric_comparison.png`
- Full summary JSON: `summary.json`
- LoRA adapters: `runs/*/adapters/`

## Interpretation

Use CLIP text-image similarity as the primary metric. Use KID/FID-like distribution metrics, diversity, duplicate rate, technical quality and latency as secondary checks. If seed+synthetic improves CLIPScore without a large loss of diversity or latency, the synthetic augmentation is useful for the chosen domain.