# TTI-Diffusion

A text-to-image diffusion model built from scratch in PyTorch — trained on the Stanford Cars dataset with BLIP captions. Includes a custom BPE text tokenizer/encoder, a conditional U-Net denoiser, a cosine noise schedule, and a separate super-resolution upscaler that takes the 64×64 diffusion output up to 256×256.

## How it works

1. **Text encoding** — Captions are tokenized with a custom-trained BPE tokenizer and passed through a small Transformer encoder to produce text embeddings.
2. **Diffusion (U-Net)** — A conditional U-Net predicts the noise added to an image at a given timestep, using sinusoidal time embeddings and cross-attention to the text embeddings.
3. **Noise schedule** — A cosine beta schedule (Nichol & Dhariwal) controls how noise is added/removed across 1000 timesteps.
4. **Upscaling** — A residual CNN with pixel-shuffle upsampling takes the 64×64 generated image to 256×256.

Pipeline: `prompt → BPE tokenizer → Text Encoder → U-Net denoising loop → 64×64 image → Upscaler → 256×256 image`

## Project structure

```
.
├── Text_encoder.py           # Transformer text encoder (self-attention blocks)
├── DefussionBlock.py         # Sinusoidal time embedding, time MLP, ResBlock
├── UNET.py                   # Conditional U-Net with cross-attention to text
├── CosineNoiseScheduler.py   # Cosine noise schedule + forward diffusion (add_noise)
├── Dataloader.py             # Dataset wrapper: image transform + tokenized caption + label
├── Train.ipynb               # Trains the text encoder + U-Net (diffusion model)
├── upscaler_train.ipynb      # Trains the upscaler
├── test.ipynb                 # Loads checkpoints and generates images from a prompt
│
├── checkpoints/
│   ├── TTI.pt                # U-Net + text encoder checkpoint
│   └── upscaler.pt           # Upscaler checkpoint
│
├── Token/
│   ├── TTI_Tokenizer.py      # Wraps a HuggingFace `tokenizers` BPE tokenizer (padding/truncation)
│   └── bpe_tokenizer.json    # Pretrained BPE tokenizer vocab/merges
│
├── Upscalar/
│   ├── Upscaler.py           # Residual CNN + PixelShuffle 64x64 -> 256x256 upscaler
│   └── Upscaler_dataset.py   # Paired low-res/high-res dataset for the upscaler
│
├── Data/
│   ├── DataDownloader.ipynb  # Downloads/inspects the Stanford Cars (Lance format) dataset
│   └── Datasets/             # Cached dataset files (gitignored)
│
└── readme.md
```

## Dataset

[`lance-format/stanford-cars-lance`](https://huggingface.co/datasets/lance-format/stanford-cars-lance) via 🤗 `datasets`, using the `blip_caption` field as the text condition.

## Training

- **Diffusion model**: `Train.ipynb` — AdamW, cosine LR schedule, MSE loss between predicted and true noise, gradient clipping, checkpointing/resume support.
- **Upscaler**: `upscaler_train.ipynb` — AdamW, L1 loss between predicted and ground-truth high-res image.

## Inference

`test.ipynb` loads the trained U-Net, text encoder, and upscaler checkpoints, runs the reverse diffusion process from a text prompt, and passes the result through the upscaler to produce the final image.

## Requirements

- Python, PyTorch, torchvision
- [`tokenizers`](https://github.com/huggingface/tokenizers) (HuggingFace)
- 🤗 `datasets`, `pylance` (for the Lance-format dataset)
- matplotlib (for visualizing generations)

## Notes

- The BPE tokenizer (`Token/bpe_tokenizer.json`) was trained separately as part of another project (RIRURU LLM) — the training code for it isn't included here.
- Image resolution: 64×64 for the diffusion model, upscaled to 256×256.
- `checkpoints/`, `Data/Datasets/`, and `__pycache__/` are gitignored — checkpoints and cached dataset files aren't tracked in the repo.