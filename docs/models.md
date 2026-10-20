# Models

## Getting the weights

The upstream checkpoint lives at
[BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2). It is
about 6.5 GB and the download must include the `audio_tokenizer/` subdirectory,
which holds the vocoder. The converter refuses to run without it.

```
huggingface-cli download BreezeBlue/Breeze-TTS-2 --local-dir ./Breeze-TTS-2
```

The layout the converter expects:

```
Breeze-TTS-2/
  config.json
  model-*.safetensors
  tokenizer.json
  audio_tokenizer/
    config.json
    model.safetensors
```

## Converting to GGUF

```
pip install -r scripts/requirements.txt
python scripts/convert_hf_to_gguf.py ./Breeze-TTS-2 -o breeze-tts-2-f16.gguf --dtype f16
```

Only `numpy` and `gguf` are needed. The converter reads safetensors directly
with a memory mapped reader and decodes BF16 by hand, so there is no torch
dependency and the peak memory stays low.

`--dtype f32` produces an unquantized model at roughly double the size. It is
useful as a reference for debugging but not for normal use; measurements put
the difference against F16 at around 78 dB SNR, which is inaudible.

### What the converter does

Beyond a straight tensor copy it applies a few transforms that the runtime
depends on:

- **Gemma RMSNorm offset.** `T5Gemma2RMSNorm` computes `x_norm * (1 + w)` while
  the weights are stored centred on zero. The converter bakes the `+1` into
  every `te.*_norm.weight` so the runtime can use a plain RMSNorm.
- **Codebook reconstruction.** Quantizer codebooks are stored as
  `embedding_sum` and `cluster_usage`. The converter precomputes
  `embedding_sum / max(cluster_usage, 1e-5)`.
- **Depthwise kernel repacking.** ggml convolutions have no group support, so
  the ConvNeXt depthwise kernels are stored transposed as `(K, C)` and applied
  in the runtime as a shift and multiply rather than expanded into a block
  diagonal matrix, which would have cost 59 MB for no benefit.
- **Head transpose.** `codebooks_head` is transposed from `(15, 1024, 2051)` to
  `(15, 2051, 1024)` so each slice is a ready to use matmul weight.
- **Precision pinning.** Convolution kernels, embeddings, codebooks, norms and
  biases stay F32 regardless of `--dtype`. The Vulkan convolution kernels reject
  F16 inputs, and the elementwise parameters are too small to be worth
  shrinking.

The in checkpoint Mimi codec is skipped entirely. The reference implementation
never uses it at inference time; it loads `audio_tokenizer/` instead.

## Quantizing

```
breeze-quantize breeze-tts-2-f16.gguf breeze-tts-2-q4_k.gguf q4_k
```

Accepts `f16`, `q8_0`, `q6_k`, `q4_k`, `q3_k` or `q2_k`.

| Variant | Size | Approximate VRAM |
| --- | --- | --- |
| F16 | 5.9 GB | ~7 GB |
| Q8_0 | 3.3 GB | ~4 GB |
| Q6_K | 2.9 GB | ~3.5 GB |
| Q4_K | 2.4 GB | ~3 GB |

The quantizer is deliberately conservative:

- Tensors already stored as F32 are left alone, so the vocoder is bit identical
  across every variant. Quantizing the network that writes the waveform is what
  makes a TTS model sound gritty, so it never happens here.
- Token and audio embeddings drop to Q8_0 at most.
- K-quants require a row length divisible by 256. Rows that do not divide fall
  back to Q8_0, then F16.
- 1D tensors, norms, biases and the 3D codebooks head are never quantized.

Q4_K is a reasonable default. The quality loss lands in the backbone and text
encoder, which affects phrasing more than timbre.

### Quantizing the depth decoder separately

`--depth` gives the depth decoder its own type:

```
breeze-quantize breeze-tts-2-f16.gguf breeze-tts-2-q4_k-dd2.gguf q4_k --depth q2_k
```

This buys throughput cheaply. The depth decoder runs 15 sequential single token
passes per frame and is the largest single cost in generation. The table below
was measured before the depth decoder moved to one graph per frame, when most of
its time was dispatch overhead. That overhead is gone now and what remains is
mostly weight reading, so every row is faster than shown and a smaller depth type
should save more than this suggests. With the current build Q8_0 with a Q4_K depth
decoder spends 23.86 ms per frame in depth. Measured on an RTX 3060 over five
runs, median, before the change:

| Model | Depth type | Size | Depth per frame | Total per frame | Real time factor |
| --- | --- | --- | --- | --- | --- |
| Q8_0 | Q8_0 | 3.3 GB | 42.96 ms | 64.20 ms | 1.25x |
| Q8_0 | Q4_K | 3.2 GB | 38.92 ms | 59.28 ms | 1.35x |
| Q8_0 | Q2_K | 3.1 GB | 35.21 ms | 55.53 ms | 1.44x |
| Q4_K | Q4_K | 2.4 GB | 39.20 ms | 58.64 ms | 1.30x |
| Q4_K | Q2_K | 2.3 GB | 36.89 ms | 55.71 ms | 1.44x |

Dropping the depth decoder to Q2_K is worth about 15% throughput for roughly
70 MB.

### It costs more than it first appears

Short clips hide the damage. A Q2_K depth decoder sounds fine for a sentence and
then falls apart over half a minute of continuous speech, going muffled and then
tinny. Measured mean energy above 6 kHz on the same passage and seed, where the
only difference between the middle two rows is the depth decoder:

| Model | Depth type | Energy above 6 kHz |
| --- | --- | --- |
| F16 | F16 | 2.2% |
| Q8_0 | Q8_0 | 13% |
| Q8_0 | Q2_K | 0.5% |
| Q4_K | Q2_K | 1.9% |

Two things are going on. The depth decoder predicts codebooks 1 to 15, the
residual detail on top of the coarse codebook the backbone chose, so quantizing
it strips the top of the spectrum directly. Worse, every frame it emits is fed
back into the backbone as the next input, so the error is inside the feedback
loop and accumulates the longer a passage runs. That is why it degrades with
time rather than uniformly, and why it recovers at the start of each piece of
split text, where the context restarts.

Use `--depth` when you want throughput on short utterances. For anything long or
anything where fidelity matters, leave the depth decoder at the base type. `Q8_0`
throughout is the safe choice.

`q3_k` and `q2_k` are accepted for the whole model too, but there is little
reason to use them. The backbone is bandwidth bound and degrades in ways you can
hear.

## Tensor naming

GGUF tensor names are grouped by stage.

| Prefix | Stage |
| --- | --- |
| `te.*` | T5Gemma2 text encoder |
| `bb.*` | Qwen3 backbone |
| `dd.*` | Depth decoder |
| `audio_embd.weight` | Audio embedding, shared between backbone and depth decoder |
| `codec.enc.*`, `codec.enct.*` | Vocoder encoder, used only for voice cloning |
| `codec.sq.*`, `codec.aq.*` | Encoder semantic and acoustic quantizers |
| `codec.dq.*` | Decoder split residual quantizer |
| `codec.dpre.*`, `codec.dtf.*` | Decoder pre convolution and transformer |
| `codec.dup.*` | ConvNeXt upsampling blocks |
| `codec.dhead.*`, `codec.dblk.*`, `codec.dfin.*` | Decoder blocks and output head |

Metadata keys live under `breeze.`, with per stage groups such as
`breeze.te.*`, `breeze.bb.*`, `breeze.dd.*`, `breeze.codec.*` and
`breeze.codec.dec.*`.

## Sampling defaults

Stored in the GGUF metadata and matching the reference implementation:

| Key | Value |
| --- | --- |
| `breeze.temperature` | 0.9 |
| `breeze.top_k` | 50 |
| `breeze.top_p` | 1.0 |
| `breeze.repetition_penalty` | 1.1 |
| `breeze.depth_temperature` | 0.9 |
| `breeze.depth_top_k` | 50 |
| `breeze.depth_top_p` | 1.0 |
| `breeze.max_new_tokens` | 750 |
