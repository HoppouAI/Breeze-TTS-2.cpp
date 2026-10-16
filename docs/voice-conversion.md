# Voice conversion

Conversion respeaks an existing recording in another voice. The words and their
timing come from the source recording, the voice comes from the reference. No
transcript of the source is needed.

This is a different operation from cloning. Cloning reads text you supply in a
cloned voice. Conversion takes audio you already have and changes who is saying
it, keeping the original delivery's timing.

## How it works

The codec splits into two stages that are trained together but stored
separately: a semantic codebook that carries what is being said, and fifteen
acoustic codebooks that carry how it sounds.

Conversion encodes the source, forces the backbone to emit the source's semantic
codes frame by frame, and lets the depth decoder regenerate all fifteen acoustic
codebooks conditioned on the reference voice. Content is preserved because it is
copied, identity changes because it is regenerated.

The backbone still needs text to read. With an empty prompt it goes degenerate
and forcing codes against that state comes out mumbled, so a filler passage sized
to the clip is generated automatically. What the filler says does not matter,
only that there is roughly a clip's worth of it. Measured on a 14 word sentence:

| Text supplied | Words intact |
| --- | --- |
| None at all | 7 / 14 |
| `Hello.` | 10 / 14 |
| Generic filler, right length | 13 / 14 |
| Deliberately wrong text, right length | 13 / 14 |
| The correct transcript | 14 / 14 |
| Automatic filler | 14 / 14 |

Wrong text scoring the same as generic filler is the useful result: the words
come from the forced codes, not from the prompt. Supplying the real transcript
with `--text` still helps slightly on difficult material.

## Command line

```
breeze-convert <model.gguf> --source <in.wav> --ref-audio <voice.wav> --ref-text <s> --output <out.wav>
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--source <wav>` | required | Recording to convert. Any rate or channel count. |
| `--output <wav>` | `convert.wav` | Output path. |
| `--ref-audio <wav>` | required | Target voice. |
| `--ref-text <s>` | required | Exact transcript of the reference. |
| `--text <s>` | filler | Transcript of the source, if you have it. |
| `--temp <f>` | `0.3` | Depth sampling temperature. |
| `--top-k <n>` | `1` | Depth top-k. |
| `--keep-acoustic <n>` | `0` | Acoustic codebooks taken from the source instead of regenerated. |
| `--cfg-scale <f>` | `1.0` | Guidance toward the target voice. |
| `--seed <n>` | `42` | RNG seed. |
| `--cpu` | off | Force the CPU backend. |

Two more exist for digging into what the codec is doing rather than for normal
use. `--feed-source` feeds the original frames back to the backbone instead of
the converted ones, which leaks the source voice back in. `--keep <n>` rebuilds
the audio from only the first `n` codebooks with no voice change at all.

## `POST /v1/audio/convert`

Same operation over HTTP. Takes `multipart/form-data`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `source` | file | required | Recording to convert. |
| `ref_audio` | file | required unless `voice_id` | Target voice WAV. |
| `ref_text` | string | required unless `voice_id` | Exact transcript of the reference. |
| `voice_id` | string | none | Saved or cached voice to use instead of uploading one. |
| `text` | string | filler | Transcript of the source, if known. |
| `temperature` | float | `0.3` | Depth sampling temperature. |
| `top_k` | int | `1` | Depth top-k. |
| `cfg_scale` | float | `1.0` | Guidance toward the target voice. |
| `keep_acoustic` | int | `0` | Acoustic codebooks kept from the source. |
| `seed` | int | `42` | RNG seed. |

Unlike `/v1/audio/speech` this returns the whole clip in one response rather than
streaming, because conversion needs the entire source encoded before it can
start. Headers are the same: `X-Sample-Rate` and `X-Sample-Format`, raw s16le
PCM in the body.

```
curl -X POST http://127.0.0.1:8137/v1/audio/convert \
  -F "source=@recording.wav" \
  --form-string "voice_id=harbour" \
  -o converted.pcm
```

Note the asymmetry: the HTTP endpoint accepts a saved `voice_id`, but
`breeze-convert` only takes a clip with `--ref-audio`. Converting from the
command line always re-encodes the reference.

## Sampling is deliberately narrow

The defaults are close to greedy, which is unusual for this model. Opening
sampling up lets the source voice bleed back through, because speaker identity
lives in the acoustic residual that sampling randomises.

Measured as similarity of the converted output to each speaker, converting a male
source into a female target:

| Setting | To target | To source | Separation |
| --- | --- | --- | --- |
| `temp 0.3, top_k 1` | 0.9760 | 0.7648 | **0.2112** |
| `temp 0.3, top_k 10` | 0.9840 | 0.7978 | 0.1862 |
| `temp 0.6, top_k 20` | 0.9815 | 0.7852 | 0.1963 |
| `temp 0.9, top_k 50` | 0.9761 | 0.8263 | 0.1498 |

Leakage climbs steadily as sampling opens. The default is the cleanest
separation. If a difficult source drops words, `--temp 0.6 --top-k 20` is a
gentler middle ground than going all the way to the model's normal settings.

## Melody is hit and miss

Pitch is regenerated along with everything else, so the target voice supplies its
own intonation rather than copying the source's. On a sung source the result is
re-sung in the target's natural register instead of at the original pitch.

Whether the tune actually survives varies a lot between clips, so test yours
rather than trusting a rule. Two sources measured against their originals:

A 12.7 s singing clip, where the tune was lost without help:

| Path | Median F0 | Contour correlation |
| --- | --- | --- |
| Source through the codec and back | 275.9 Hz | +0.901 |
| Converted, `keep_acoustic 0` | 258.1 Hz | -0.200 |
| Converted, `keep_acoustic 1` | 266.7 Hz | +0.580 |
| Converted, `keep_acoustic 2` | 272.7 Hz | +0.725 |

A 5.3 s pop vocal, where it came out clearly sung either way:

| Path | Median F0 | Contour correlation |
| --- | --- | --- |
| Source | 229.7 Hz | |
| Converted, `keep_acoustic 0` | 187.5 Hz | +0.065 |
| Converted, `keep_acoustic 1` | 269.7 Hz | -0.208 |

Note the second case: both takes were unmistakably singing on listening, yet the
correlation numbers are near zero or negative. That is the transposition showing
up. The take follows the shape of the tune in a different register, which a
frame by frame pitch correlation scores as unrelated. **Judge this feature by
ear. The correlation number alone will mislead you.**

`keep_acoustic` copies that many low acoustic codebooks straight from the source
instead of regenerating them. Pitch lives in the low ones, so 1 or 2 pulls more
of the original contour through while the voice still comes out as the target.

Past 2 it gets unstable, and on ordinary speech it is worse than useless. The
same setting applied to a spoken clip widened the pitch spread from 107 Hz to
320 Hz and produced audible warbling. **Leave it at 0 for speech. For singing,
try 0 first and reach for 1 or 2 if the tune flattens out.**

Supplying `--text` matters more on singing than anywhere else. The same clip
converted textless came back as "Does it pinch I? Send my tears", and with the
lyrics passed in became "drink it up, I have no fear" exactly.

## Length

Long sources work. A 141 s recording converts in about 137 s on an RTX 3060 at
Q8_0, roughly realtime, and holds pitch across the whole thing.

The vocoder decodes the result in windows rather than all at once, because a
single graph over a long clip asks for gigabytes of memory in one allocation.
The source encode is still done in one pass, so a recording several times longer
than that may still run out of memory on a busy GPU.

## The web UI

The VOICE CHANGER tab exposes the same thing, with `keep_acoustic` as the KEEP
MELODY control. It stays on HTTP rather than the WebSocket, since conversion
returns the whole clip in one piece and has nothing to stream.
