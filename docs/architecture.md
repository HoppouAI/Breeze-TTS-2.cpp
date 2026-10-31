# Architecture

This is a from scratch C++ port of Breeze-TTS-2 on top of ggml. Where the
reference uses PyTorch modules, this repo builds ggml graphs; the maths is meant
to match exactly.

## Pipeline

```mermaid
flowchart TD
    A[text + instruction] --> B[Gemma BPE tokenizer]
    B --> C[T5Gemma2 text encoder<br/>26 layers, 1152 hidden]
    C --> D[projection 1152 to 2048]
    D --> E[Qwen3 backbone<br/>28 layers, 2048 hidden]
    E -->|codebook 0| F[depth decoder<br/>12 layers, 1024 hidden]
    F -->|codebooks 1 to 15| G[vocoder]
    E -->|last hidden state| F
    G --> H[24 kHz waveform]
```

One backbone step produces one frame. A frame is 16 codebook entries and
becomes 1920 samples, so generation runs at 12.5 frames per second.

## Text encoder

T5Gemma2, 26 layers, hidden 1152, 4 query heads over 1 KV head, head dim 256,
FFN 6912, `gelu_pytorch_tanh`.

Two details are easy to get wrong and both produce speech that is intelligible
but paced incorrectly:

**Attention is bidirectional.** `T5Gemma2SelfAttention` sets `is_causal=False`.
The sliding layers use an asymmetric bidirectional window: with
`left = (w + 1) / 2` and `right = w / 2 + 1`, a key at distance `d = q - kv` is
visible when `0 <= d < left` or `0 < -d < right`. Applying a causal mask here
makes the model emit far too few frames for the text.

**RMSNorm is Gemma style.** `T5Gemma2RMSNorm` computes `x_norm * (1 + w)`, and
the weights are stored centred on zero. The converter bakes the `+1` in so the
runtime uses a plain RMSNorm.

Layers alternate between sliding window and full attention, with full attention
every sixth layer, and the two kinds use different RoPE settings: theta 1e6 with
a linear factor of 8 for full layers, theta 1e4 for sliding layers.

## Backbone

Qwen3, 28 layers, hidden 2048, 16 query heads over 8 KV heads, head dim 128,
FFN 6144, RoPE theta 1e6, with QK normalisation.

It reads the projected encoder output plus the running audio embedding and
predicts codebook 0 through a 2048 to 2052 head. Token 2051 ends the utterance;
tokens 2048 to 2050 are suppressed during sampling.

The audio embedding for a frame is the sum of the 16 codebook embeddings, each
looked up at `codebook_index * vocab_size + token` in one shared table.

## Depth decoder

12 layers, hidden 1024, 8 query heads over 2 KV heads, head dim 128, FFN 8192,
RoPE theta 5e5 with llama3 scaling. No QK normalisation.

For every frame it runs 15 short autoregressive steps to fill codebooks 1 to 15.
The KV cache is cleared at the start of each frame.

Position 0 takes the backbone's last hidden state as its input embedding.
Position `p` for `p >= 1` takes the token of codebook `p - 1`, offset by
`(p - 1) * vocab_size` into the shared embedding table, and the logits at that
position use head `p - 1`. The head weight is a `(15, 2051, 1024)` tensor sliced
per codebook.

All 15 steps and their sampling are built into one graph, so a frame is one
submit and one tiny readback instead of 15 round trips. Inside that graph the KV
cache just grows by concat. Sampling is top-k plus Gumbel noise that the CPU draws
from the seed up front, which gives the same distribution as softmax sampling over
the top k. Top-p below 1, forced codes and top-k above 256 fall back to one graph
per step.

## Vocoder

The vocoder is `Qwen3TTSTokenizerV2`, loaded from the `audio_tokenizer/`
directory. The Mimi codec inside the main checkpoint is dead weight; the
reference never touches it at inference time, and using it produces garbled
audio.

Its encoder is a Mimi model (SEANet plus transformer plus split RVQ) and is only
needed for voice cloning. Its decoder is a different network entirely:

```
codes [16, T]
  split residual quantizer      first codebook + sum of the other 15, both projected to 512
  causal conv 512 to 1024, k=3
  transformer                   8 layers, hidden 512, 16 heads, head dim 64,
                                SwiGLU, per branch layer scale, causal
                                sliding window 72, RoPE theta 1e4
  2x ConvNeXt upsample          transposed conv stride 2, then depthwise k=7,
                                LayerNorm, 1024 to 4096 to 1024, gamma scale
  causal conv 1024 to 1536, k=7
  4x decoder block              SnakeBeta, transposed conv at rates 8, 5, 4, 3,
                                then 3 residual units at dilations 1, 3, 9
  SnakeBeta, causal conv 96 to 1, k=7
  clamp to [-1, 1]
```

Total upsampling is `8 * 5 * 4 * 3 * 2 * 2 = 1920`, matching one frame.

SnakeBeta is `x + sin(x * alpha)^2 / beta`, where `alpha` and `beta` are stored
in log space and exponentiated at runtime.

Causal convolutions left pad by `(k - 1) * dilation - (stride - 1)` and never
pad on the right. Causal transposed convolutions run the transpose then trim
`k - stride` from the right, which is the same as keeping the first
`L * stride` samples.

## Classifier free guidance

When `cfg_scale != 1` the text encoder, backbone and depth decoder all run twice,
once on the conditioned prompt and once on a negative prompt that drops the
instruction. Logits combine as:

$$\text{logits} = \text{uncond} + s \cdot (\text{cond} - \text{uncond})$$

The depth decoder runs both branches in **one graph**, not two. Each weight is
read once for both branches, so the second one costs about 40% rather than a
doubling: 23.5 ms per frame at `cfg_scale` 1 against 32.3 ms at 3.

The two branches share one KV cache, interleaved as
`slot = pos * n_branch + branch`. That keeps the tokens written each step
contiguous, so the cache append is a single copy, and
`build_branch_causal_mask` makes each query causal within its own branch and
blind to the other.

The backbone runs its branches as rows of one batch. The conditioned and
unconditioned prompts have different lengths and do not interleave cleanly, so
each row keeps its own cache, positions and mask. That is the remaining cost of
guidance, worth about 4 ms per frame.

Measured end to end for voice design on an RTX 3060, Q8_0 with a Q4_K depth
decoder:

| `cfg_scale` | Backbone | Depth | Vocoder | Total per frame | Real time factor |
| --- | --- | --- | --- | --- | --- |
| 1 | 8.83 ms | 23.46 ms | 2.22 ms | 34.51 ms | 2.32x |
| 3 | 12.65 ms | 32.30 ms | 2.42 ms | 47.37 ms | 1.69x |

Cloning adds a one off reference encode of roughly 650 ms, which lands on time to
first audio and not on throughput.

## Streaming

Audio is flushed in growing chunks. The first is as small as the GPU can stay ahead
with, usually 1 frame (80 ms), and each flush grows by a third up to 25 frames (2 s)
by default. On an RTX 3060 at Q4_K a warmed up server reaches first audio in about
80 ms. Both ends of the ramp are configurable, see [performance.md](performance.md).

The vocoder keeps its state between flushes in `VocoderStream`, so a flush only
decodes the new frames. The state is held in persistent GPU tensors: the history of
every causal and depthwise convolution, the overlap tail of every transposed
convolution, and the transformer's keys and values for its attention window. Each
tensor is read through a concat and written back with a copy inside the same graph.
A one shot decode goes through a throwaway stream, and since a cleared state equals
the zero padding it used before, its output did not change.

A flush costs about 8.5 ms plus 1.1 ms per frame on Vulkan: 9.1 ms for 1 frame, 17.7
ms for 8 and 40 ms for 25. Chunked output matches a one shot decode apart from
rounding, because `ggml_conv_1d` rounds through F16 and GPU matmuls round too. It is
about 40 dB down on real speech. With F32 convolutions on the CPU every flush pattern
matched to 106 dB, which is what shows the state itself is right.

Chunking also shapes what the client has to do. The queue drains continuously but
refills once per chunk, so a client needs to hold more audio than a chunk takes
to produce, not merely delay its first chunk. Producing 25 frames takes about
1 s, so a client sitting on 0.5 s can run dry even when the average rate is well
ahead of playback. The server sends early to cover most of this, see
[server.md](server.md#queue-depth).

## Several streams

`GenScheduler` runs all GPU work on one thread. With `--parallel` above 1 it steps
every active `SpeechJob` by one frame at a time through `step_jobs`:

- **Depth decoder.** One fused graph for the whole batch, cached by the number of plain
  streams, the number of guided streams and top k. Plain streams take one row each and
  guided streams take two. Each stream keeps its own random numbers, guidance scale
  and temperature, so requests with different settings can share a batch.
- **Backbone.** One batch with a row per branch, each with its own cache, positions
  and mask, since the streams are at different lengths.
- **Vocoder.** Runs per stream, on its own `VocoderStream`, when that stream's chunk
  fills.

Sharing a batch reads each weight once for every row, which is why three streams cost
far less than three times one. Audio callbacks run on the request's own thread through
a queue, so a slow client only holds up itself. Other GPU work, like encoding a
reference voice, runs between frames.

Slots are counted, not streams, and a guided stream costs two. A new request's prompt
and prefill wait up to 400 ms while a paced stream has under 250 ms of audio left. The
first chunk is sized from how long frames have recently taken at that number of rows.

## Where the time goes

Measured on an RTX 3060 at Q8_0 with a Q4_K depth decoder, per frame of audio
(80 ms):

| Stage | Per frame | Share |
| --- | --- | --- |
| Backbone decode | 8.83 ms | 26% |
| Depth decode | 23.46 ms | 68% |
| Vocoder | 2.22 ms | 6% |

The depth decoder dominates because it runs 15 sequential steps per frame, one
per codebook, and each step is a full 12 layer forward pass over a single token.

It used to be bound by dispatch count, with a CPU round trip per step and the GPU
idle between them. With the whole frame in one graph that overhead is gone, and
the Vulkan profiler puts about 26 ms of kernel time on each frame, most of it in
the FFN matmuls. What is left is reading the depth decoder's weights 15 times per
frame, so the depth type now matters more for speed than it used to.

## ggml notes

Things worth knowing if you touch the graph code.

**Graph pattern.** Build the graph, call `ggml_gallocr_alloc_graph`, then set
input tensor data, then compute. Inputs cannot be written before allocation
because their backing memory does not exist yet. `Graph` in
[common.h](../include/breeze/common.h) stashes pending inputs and writes them at
the right moment.

**No grouped convolutions.** ggml convolution ops ignore groups. Depthwise
kernels are applied as a shift and multiply over `k` shifted views instead of
being expanded into a block diagonal matrix.

**Convolution kernels must be F32.** The Vulkan `conv_transpose_1d` path rejects
F16 inputs, so the converter pins every convolution kernel to F32 regardless of
the requested dtype.

**`ggml_conv_1d` uses an F16 im2col.** Activations round to F16 inside every
convolution. Measured against a full F32 path the difference is about 78 dB SNR,
which is inaudible, so it is left alone.

**Tensor layouts.** Convolution stages use `[length, channels]` with time in
`ne0`. Transformer stages use `[hidden, tokens]` with hidden in `ne0`. Transpose
between them, matching the reference's `transpose(1, 2)` and `permute(0, 2, 1)`.

## Source map

| File | Contents |
| --- | --- |
| `src/common.cpp` | Backend init, KV cache, graph helper, attention, masks |
| `src/gguf_loader.cpp` | GGUF loading and backend tensor allocation |
| `src/config.cpp` | Metadata to config structs |
| `src/tokenizer.cpp` | Gemma BPE with byte fallback |
| `src/text_encoder.cpp` | T5Gemma2 encoder |
| `src/backbone.cpp` | Qwen3 backbone and audio embedding |
| `src/depth_decoder.cpp` | Depth decoder and llama3 RoPE scaling |
| `src/codec.cpp` | Vocoder entry points |
| `src/codec_conv.cpp` | Causal convolution helpers and the SEANet encoder |
| `src/codec_transformer.cpp` | Mimi encoder transformer and vocoder transformer |
| `src/codec_decoder.cpp` | Quantizer decode, ConvNeXt, SnakeBeta, decoder blocks, the stateful `VocoderStream` |
| `src/sampling.cpp` | Repetition penalty, temperature, top-k, top-p |
| `src/generation.cpp` | Prompt assembly, `SpeechJob`, batched frame steps, chunked flush |
| `src/scheduler.cpp` | `GenScheduler`: the GPU thread, slots, warm up |
| `src/c_api.cpp` | C ABI wrapper |
