# mini-DiT

A small Diffusion Transformer written from scratch, to actually understand how the
architecture works rather than just reading about it. Based on
[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
(Peebles & Xie, 2022), with [explainingai-code/DiT-PyTorch](https://github.com/explainingai-code/DiT-PyTorch)
as a reference when I get stuck.

The goal is to write each function explicitly. For example not using `einops`, no `F.scaled_dot_product_attention` and instead using `reshape` and `permute` to accomplish those tasks.

## The idea

DiT replaces the U-Net in a diffusion model with a plain transformer. You cut the image
into patches, treat them as a sequence of tokens, and run standard transformer blocks
over them. The timestep is injected through the LayerNorm parameters instead of being
added to feature maps.

Training is the usual DDPM setup: noise an image to a random timestep, ask the model to
predict the noise that was added, take the MSE.

## What's implemented

- **DDPM forward process** — linear beta schedule, closed-form `x_t` from `x_0`
- **Patchify / unpatchify** — `(B, C, H, W) ↔ (B, N, p²C)`
- **2D sin-cos position embeddings** — frozen, not learned, built by applying the 1D
  formula to row and column indices separately with frequencies
- **Multi-head self-attention** — bidirectional
- **adaLN-Zero blocks** — the timestep embedding predicts six vectors per block
  shift/scale for each of the two LayerNorms, plus a gate on each residual branch.
  The output layer is zero-initialized, so at init every block is an exact identity
  and the whole network is a no-op. This was the paper's biggest improvement/finding.
- **DDPM reverse sampling** — full T-step loop

## What's left out

- **The VAE.** I'm diffuaing in pixel space on
  small images instead, which means the transformer is doing the same job on the same
  token-grid size without me having to train an autoencoder first.
- **Class conditioning and classifier-free guidance.** Unconditional only. Adding it
  is mostly an embedding table summed into the timestep embedding can be a
  reasonable extension later.
- **Learned variance.** The paper predicts both ε and a diagonal covariance training
  the latter with the full KL term. I predict ε only and use the fixed posterior
  variance.
- **EMA.**

## Config

| | |
|---|---|
| Data | MNIST, padded to 32×32, single channel |
| Patch size | 4 → 8×8 grid = 64 tokens |
| Hidden size | 256 |
| Layers / heads | 6 / 4 |
| Timesteps | 1000, β from 1e-4 to 0.02 |
| Parameters | ~7M |

## Training

AdamW, MSE against the sampled noise, batches off MNIST. Runs on MPS
on a laptop.
Sampling starts from pure noise at t=999 and walks back to t=0 one step at a time
decoding to an image at the end.

## Layout

```
scheduler.py   noise schedule, forward and reverse process
patching.py    patchify/unpatchify, 2D position embeddings
blocks.py      attention, adaLN-Zero transformer block
dit.py         the model
train.py       training loop
sample.py      generation
```

## Status

Scheduler, patching and blocks are done and tested. Model, training loop and sampling
are in progress.
