# Performance and tuning

How fast this runs, what changed recently, and which settings are worth touching.

Every number here was measured on one RTX 3060 12 GB with its tensor core limit
unlocked, on Vulkan unless a row says otherwise. The speed is the real time factor:
seconds of audio per second of wall clock, so anything above `1.00x` is faster than
playback. Repeat runs move by a few percent, and other programs using the GPU move them
further, so measure your own card before you size anything from these tables.
[Measuring](#measuring) shows how.

## What's new since v0.1.5

- **The vocoder only decodes new frames.** It used to decode about 88 frames of old
  audio again on every flush and throw that output away. It now keeps its state
  between flushes, so a flush only does the new work. At the default chunk sizes that
  took it from about 6.6 ms to about 2.4 ms per frame. With small chunks, 1 up to 8
  frames, it went from 14.95 ms to 2.74 ms.
- **First audio is about twice as fast.** On a warmed up server at Q4_K it went from
  181 ms to about 80 ms, because the first chunk can now be 1 frame without slowing
  the rest of the clip down.
- **The first chunk sizes itself.** `--chunk-first` now defaults to 1 and acts as a
  minimum. When the GPU is too busy for 1 frame to stay ahead of playback, the first
  chunk goes up to 2, 4 or 8 frames on its own.
- **Several streams at once.** `--parallel N` lets one server generate up to N streams
  together. They share one GPU batch per frame, so three streams cost far less than
  three times one stream. Streams with and without guidance can run together.
- **The WebSocket waits for a free slot.** A session only reports `queued` when every
  slot is taken, and it no longer holds the whole GPU while it speaks.

One stream on Vulkan, v0.1.5 and the current code run back to back, averaged over 4
seeds. All of the gain is in the vocoder, the backbone and depth decoder did not change:

| Model | `cfg_scale` | v0.1.5 | Now |
| --- | --- | --- | --- |
| Q4_K | 1 | 2.15x | 2.41x |
| Q4_K | 3 | 1.53x | 1.74x |
| Q8_0 | 1 | 1.78x | 1.99x |
| Q8_0 | 3 | 1.46x | 1.58x |

## The short version

1. **Leave `--chunk-first` and `--chunk-max` alone.** The defaults are now the best
   numbers measured here. Raising `--chunk-max` no longer helps.
2. **Use `--parallel` if you need more than one stream.** `--parallel 3` is the safe
   setting on a 12 GB RTX 3060 with Q8_0 or Q4_K.
3. **Keep `cfg_scale` at 1 unless you need guidance.** It costs about 25% of the speed
   and a stream using it takes two slots.
4. **Pick a smaller model if you are short on speed.** The depth decoder takes about
   two thirds of every frame, so its weight type matters more than anything else.
5. **Leave the warm up on.** It costs a few seconds at startup and saves that time on
   the requests your users are waiting for.
6. **Do not judge the first run.** See [The first run is slower](#the-first-run-is-slower).

## Picking a model

One stream, per frame of audio (80 ms):

| Model | File size | `cfg_scale` 1 | `cfg_scale` 3 |
| --- | --- | --- | --- |
| Q4_K | 2.5 GB | 2.41x | 1.74x |
| Q8_0 with a Q4_K depth decoder | 3.4 GB | 2.32x | 1.69x |
| Q8_0 | 3.6 GB | 1.99x | 1.58x |
| F16 | 6.3 GB | 1.42x | |

Quantising the depth decoder is the biggest single speed gain, and it holds up well
with several streams, but it damages the high end of the audio and the damage grows
over a long passage. Listen to it first, the numbers are in [models.md](models.md).
The per stage numbers, and how CUDA compares, are in [build.md](build.md#vulkan-or-cuda).

## Running several streams

```
breeze-server breeze-tts-2-q8_0.gguf --parallel 3
```

`--parallel N` sets how many slots the server has. It defaults to 1. A stream takes
one slot, or two if its `cfg_scale` is above or below 1, because guidance generates
two versions of every frame. At `--parallel 1` a stream using guidance still fits.

These are all with streams starting one second apart, and a client that starts
playing on the first packet and buffers nothing:

| Streams | Model | Backend | Speed of each stream | Gaps |
| --- | --- | --- | --- | --- |
| 1 | Q8_0 | Vulkan | 1.86x to 2.15x | none |
| 1 with guidance | Q8_0 | Vulkan | 1.59x | none |
| 3 | Q8_0 | Vulkan | 1.24x to 1.32x | none |
| 4 | Q8_0 | Vulkan | 1.08x to 1.20x | none, or up to 224 ms |
| 3 | Q8_0 with Q4_K depth | Vulkan | 1.42x to 1.51x | none |
| 4 | Q8_0 with Q4_K depth | Vulkan | 1.18x to 1.32x | none |
| 3 | Q4_K | CUDA | 1.38x to 1.52x | none |
| 2 | F16 | Vulkan | 1.26x to 1.29x | none |
| 1 with guidance, 1 without | Q8_0 | Vulkan | 1.23x to 1.36x | none |
| 1 with guidance, 2 without | Q8_0 | Vulkan | 1.15x to 1.28x | none |
| 1 with guidance, 2 without | Q4_K | Vulkan | 1.09x to 1.18x | none |
| 2 with guidance | Q8_0 | Vulkan | 1.16x to 1.19x | none |
| 3 with guidance | Q8_0 | Vulkan | 0.87x to 0.95x | **about 3 s, does not keep up** |

Three streams with guidance need 6 slots, so the settings recommended here refuse that
load instead of letting it stutter.

**How many slots to set on a 12 GB RTX 3060:**

| Model | `--parallel` | Notes |
| --- | --- | --- |
| Q8_0 or Q4_K | `3` | Plenty of margin without guidance. |
| Q8_0 or Q4_K | `4` | Works, and also fits 2 streams with guidance. The speed gets close enough to 1x that other GPU use can cause gaps. |
| Q8_0 with Q4_K depth | `4` | Still 1.18x or better on every stream. |
| F16 | `2` | |

On another card, start from how fast one stream runs. Batching helps more than you
would guess because the streams share each read of the weights: F16 runs one stream at
about 1.4x and two streams at about 1.3x each. Set a slot count, run that many requests
at once, and read the `x` figure the server prints for each one.

**Memory.** Three streams at Q8_0 used about 4.3 GB of video memory, and at Q4_K about
3.2 GB.

**Requests that start together.** Starting a request pauses the other streams for its
prompt, so the server gives a stream that is about to run out of audio up to 400 ms to
get ahead first. Simultaneous requests still play cleanly, but the last one can wait
about 1 s for its first audio. With guidance in the mix, a burst was the one case that
needed a client buffer: up to about 1 s of gaps with nothing buffered. The bundled UI
already buffers 1.25 s.

**When the slots are full.** The HTTP routes answer `409` straight away, and a
WebSocket session sends `queued` and waits. Each WebSocket piece takes its slots
separately, so sessions take turns between pieces. Voice conversion takes every slot,
so it needs the server idle and blocks new streams until it finishes. Registering a
voice runs between frames, so every stream waits for that encode, about 650 ms.

**The same seed gives different audio in a batch.** Each stream keeps its own random
numbers, but GPU rounding differs slightly with the batch size and that can change what
gets sampled. The speech is just as good, it just is not the same take. If you need
exactly repeatable output, use `--parallel 1` and generate one request at a time.

## Tuning settings

### `--chunk-max`

The largest chunk, default 25 frames (2 s). This used to be the main thing to tune,
because every flush threw away a fixed amount of vocoder work. That cost is gone, so
now it only decides how often audio arrives. An 18 second clip at Q8_0:

| `--chunk-max` | Flushes | Vocoder per frame |
| --- | --- | --- |
| 8 | 31 | 2.90 ms |
| 25 | 15 | 2.17 ms |
| 50 | 13 | 2.16 ms |

- **Raising it above 25 gains nothing** and makes a client wait longer between chunks.
  If you set `--chunk-max 40` on an older version, take it off.
- **Lowering it is cheap.** Going down to 8 costs about 2% of total speed, and a client
  then needs far less buffer, because the longest wait for a chunk drops from about 1 s
  to about 300 ms.

### `--chunk-first`

The smallest allowed first chunk, default 1 frame (80 ms). The server picks the real
size from how long frames are taking at the current load:

| Recent speed at this load | First chunk |
| --- | --- |
| 1.9x or better | 1 frame |
| 1.5x to 1.9x | 2 frames |
| 1.15x to 1.5x | 4 frames |
| Below 1.15x | 8 frames |

Raise it only if a client cannot handle very small packets, because it delays the first
sound. In `breeze-cli` at Q8_0, a value of 4 had first audio at 394 ms against 278 ms for
the default, and saved only 0.14 ms per frame. Setting both flags to the same number
turns the ramp off and streams one fixed size.

### `cfg_scale`

Guidance costs about 25% of the speed for one stream. The depth decoder runs both
versions of a frame in one graph, so it is much less than double. Values of 2 and 3 cost
the same as each other, and only 1 is free.

### `--split-chars`

This is about long text, not speed. Asked for several minutes in one pass the model loses
the text, so long text gets broken up, see [cli.md](cli.md). Each piece starts a new
prompt, and with `--parallel` that pauses the other streams briefly, so very small values
make those pauses more frequent. The default is 600.

### `--no-warmup`

The warm up runs every mix of streams your slot count allows, so it takes longer with
more slots. Time until the server was ready at Q8_0:

| Setting | Ready after |
| --- | --- |
| `--no-warmup` | about 3 s |
| `--parallel 1` | about 6 to 7 s |
| `--parallel 3` | about 9 s |
| `--parallel 4` | about 11 s |

With the warm up on, the first request after startup reaches first audio in about 200 to
350 ms, and later ones in about 80 ms. Without it, the first request and the first time
each mix of streams shows up all build their GPU pipelines while someone is listening.

### Client buffer

With one stream, or a few started apart, the server now sends audio early when a client
is about to run out, and the tests above had no gaps with nothing buffered. A buffer is
still the cheapest protection against other GPU use and against bursts. Fill it to a
level instead of delaying the start, see
[Streaming without stutter](server.md#streaming-without-stutter).

## The first run is slower

The GPU driver compiles a new pipeline the first time it sees a new tensor shape and
type, then keeps it in a disk cache. That shows up as a one off stall, and it can be
large: the vocoder measured 44 ms per frame at Q4_K and 177 ms at F16 on their first
runs, against 2.28 ms and 2.49 ms on the second. Updating the GPU driver can bring it
back. `breeze-server` does most of this during its warm up, but `breeze-cli` has no warm
up, so run it twice and go by the second one.

## Measuring

`breeze-cli --timings` prints one request stage by stage:

```
time to first audio 190 ms over 9 flushes
  reference encode       0.0 ms
  prompt build          42.0 ms
  backbone prefill      39.5 ms
  first vocoder         45.9 ms  (1 frames)
  backbone decode      712.3 ms  (9.37 ms/frame)
  depth decode        1924.3 ms  (25.32 ms/frame)
  vocoder              279.0 ms  (3.67 ms/frame)
  76 frames, 6.08 s audio
```

The real time factor is `80 / (backbone + depth + vocoder per frame)`. Vocoder time per
frame looks higher on short clips like this one, since a flush has a fixed cost of about
8.5 ms and a short clip has more small flushes.

`breeze-server --verbose` prints the same per frame numbers after each request, and the
line above them has the speed that request got, like `1.29x`. Look at that one under
load, since per frame stage numbers do not show time spent waiting on other streams.

If a run looks much slower than usual, check what else is on the GPU before changing
anything, and run it again. Bad runs here, with Q8_0 at 1.53x and F16 at 0.95x, came back
to normal on a rerun with the same settings, and the GPU has shown 46% load here with
the server idle.

## When it stutters

| What you see | Likely cause | What to do |
| --- | --- | --- |
| Gaps, and the speed is under `1.00x` | The GPU cannot keep up | Use a smaller model, turn off guidance, lower `--parallel`, or close other GPU programs |
| Gaps, and the speed is well over `1.00x` | The client buffer is too small for the chunk size | Buffer more, or lower `--chunk-max` |
| Gaps only when several streams run | Too many streams for this card | Lower `--parallel` or use a smaller model |
| Gaps only when requests start together | The burst case above | Buffer about 1 s, or space out the requests |
| Slow first request, fine after that | Pipelines being built | Leave the warm up on, see above |
| `409` from the HTTP routes | Every slot is taken | Raise `--parallel`, retry, or use the WebSocket, which waits |
| A run much slower than the last one | Other programs on the GPU | Check GPU load and run it again |
