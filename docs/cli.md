# CLI

`breeze-cli` runs one request and writes a WAV file.

```
breeze-cli <model.gguf> --text <text> [options]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--text <s>` | required | Text to speak, UTF-8. |
| `--instruction <s>` | `Speak clearly and naturally.` | Voice description or delivery direction. |
| `--ref-audio <wav>` | none | Reference audio for cloning. Resampled to mono at the model rate. |
| `--ref-text <s>` | none | Exact transcript of the reference audio. Required with `--ref-audio`. |
| `--voice <name>` | none | Use a saved voice instead of `--ref-audio`. See [voices.md](voices.md). |
| `--save-voice <name>` | none | Encode `--ref-audio` into a reusable voice file and exit. |
| `--description <s>` | none | Optional description stored with `--save-voice`. Edit it later in the web UI or over HTTP, see [voices.md](voices.md#descriptions-and-metadata). |
| `--list-voices` | | List the saved voices and exit, without loading the model. |
| `--voices-dir <path>` | `voices` | Where saved voices live. |
| `--cfg-scale <f>` | `1.0` | Classifier free guidance. `1.0` disables it. |
| `--seed <n>` | `42` | RNG seed. |
| `--temp <f>` | model default | Sampling temperature. `0` keeps the GGUF's own value. |
| `--top-k <n>` | model default | Sampling top-k. `0` keeps the model default. |
| `--top-p <f>` | model default | Sampling top-p. `0` keeps the model default. |
| `--rep-penalty <f>` | model default | Repetition penalty. `0` keeps the model default. |
| `--max-new <n>` | model default (750) | Frame cap. One frame is 80 ms. |
| `--split-chars <n>` | `600` | Split long text into pieces of about this many characters. `0` generates in one pass. |
| `--output <wav>` | `output.wav` | Output path. |
| `--chunk-first <n>` | `4` | Frames in the first streamed chunk. |
| `--chunk-max <n>` | `25` | Frames the chunk ramps up to. |
| `--timings` | off | Print a stage by stage latency breakdown. |
| `--cpu` | off | Force the CPU backend. |
| `-h`, `--help` | | Print usage. |

Progress prints as `generated N.NN s` while the audio streams in.

On Windows the binaries run in UTF-8 mode, so Chinese or any other non-ASCII
text and file paths on the command line reach the model intact whatever the
system language is set to. That needs Windows 10 version 1903 or newer. Older
versions hand the program text in the system code page instead, and anything it
cannot represent turns into `?`, so on those send non-ASCII text through the
[server](server.md) instead, where it arrives as UTF-8.

## Latency

`--timings` reports where the time goes:

```
time to first audio 364 ms over 7 flushes
  reference encode       0.0 ms
  prompt build          28.3 ms
  backbone prefill     109.8 ms
  first vocoder         51.8 ms  (4 frames)
  backbone decode      941.8 ms  (8.12 ms/frame)
  depth decode        4710.5 ms  (40.61 ms/frame)
  vocoder             1316.7 ms  (11.35 ms/frame)
  116 frames, 9.28 s audio
```

The depth decoder dominates because it runs 15 sequential single token passes
per frame. They go to the GPU as one graph, but each pass still waits on the code
the one before it picked. Everything else is small by comparison.

Audio is flushed in growing chunks, starting at 4 frames so playback can begin
early and growing to 25 frames so the vocoder stays efficient. That keeps time
to first audio near 350 ms while generation as a whole runs comfortably faster
than realtime.

`--chunk-first` and `--chunk-max` tune that ramp, and pairing them with
`--timings` is the easiest way to find good values for a given device before
passing the same numbers to `breeze-server`. Raising `--chunk-max` mostly buys
back vocoder time, since every flush re-decodes a fixed window of left context
that gets discarded. See [server.md](server.md) for the measurements and for
what the client has to do with the result.

## Recipes

Plain synthesis:

```
breeze-cli breeze-tts-2-q4_k.gguf \
  --text "The build finished without a single warning." \
  --output out.wav
```

Design a voice from a description:

```
breeze-cli breeze-tts-2-f16.gguf \
  --text "Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear, calm delivery." \
  --cfg-scale 1 \
  --output design.wav
```

Clone a voice. The transcript has to match the reference audio exactly,
including punctuation, or the clone degrades badly:

```
breeze-cli breeze-tts-2-f16.gguf \
  --text "It is good to hear your voice again after all this time." \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --output clone.wav
```

Clone a voice and direct the delivery:

```
breeze-cli breeze-tts-2-f16.gguf \
  --text "We need to discuss what happened last night." \
  --instruction "Speak slowly with a restrained, serious tone." \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --cfg-scale 1 \
  --output direction.wav
```

## Notes on the options

**`--cfg-scale`** runs the pipeline twice, once conditioned and once
unconditioned, then combines the logits as
`uncond + scale * (cond - uncond)`. The depth decoder batches both branches into
one graph, so the real cost is about 23% rather than the doubling you would
expect, and generation still outruns playback. The reference implementation
defaults to `1.0`, which skips the second pass entirely. Higher values push
harder toward the instruction; past about 2 the output picks up an audible
harshness, so raise it only when the voice is ignoring the description.

**`--seed`** fully determines the output for a given model and input. Sampling
uses temperature 0.9 and top-k 50, so different seeds give genuinely different
takes. Generating a few and picking the best one is normal.

**`--max-new`** caps frames, not characters. At 12.5 frames per second the
default 750 is 60 seconds. The model stops on its own at an end of speech
token, so this is a safety net for runaway generation rather than a length
control. It applies per piece, not to the whole request.

**`--split-chars`** is the real length control. Text longer than the budget is
broken on sentence boundaries and generated a piece at a time, with every piece
conditioned on the first one so the voice stays put. Chinese characters count
heavier than latin ones because they take longer to say.

Splitting is not cosmetic. Asked for several minutes in one pass the model holds
its voice but loses the text, drifting into garbled words somewhere past a
minute and a half. Raise the budget if pieces sound disconnected, lower it if
sentences go missing, and set `0` to turn splitting off.

**`--ref-audio`** accepts any WAV the reader understands and converts it to
mono at the model sample rate. Five to fifteen seconds of clean speech works
best. Background noise and music get cloned along with the voice.

## Reference audio quality

Cloning copies whatever it hears. A reference recorded on a headset in a quiet
room clones well. One with room echo, a fan, or music underneath produces a
voice that carries those artefacts into every generation, and no amount of
`--cfg-scale` tuning fixes it.
