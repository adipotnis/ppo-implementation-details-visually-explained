# PPO Implementation Details, Visually Explained

A small visual explainer of some of the implementation details from
[The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
(ICLR Blog Track, 2022).

> **Work in progress.** More sections are on the way.

## Sections

1. Vectorized environments: how `next_obs` and `next_done` link rollout phases
2. Orthogonal initialization of weights, constant initialization of biases
5. Generalized Advantage Estimation (GAE), with an interactive walk through PPO's GAE loop
6. Mini-batch updates
7. Normalization of advantages

## Run locally

It's a plain static site with no build step:

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000.

## Publish with GitHub Pages

Push to GitHub, then in the repo go to **Settings → Pages**, set the source to
**Deploy from a branch**, branch `main`, folder `/ (root)`.

## Layout

- `index.html`: the blog post
- `assets/style.css`: styles (light and dark mode)
- `assets/rollout-phases.js`: the interactive rollout-phase grid for section 1
- `assets/gae.js`: the interactive GAE table for section 5
