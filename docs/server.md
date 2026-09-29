# Server API

`breeze-server` is a single binary HTTP server built on cpp-httplib. It loads
one model, serves it over HTTP, and streams raw PCM as the audio is generated
rather than waiting for the whole clip.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness, sample rate, websocket port |
| `POST /v1/audio/speech` | Generate speech, streamed. Also takes OpenAI style JSON, see [OpenAI compatible requests](#openai-compatible-requests) |
| `POST /v1/audio/convert` | Respeak a recording in another voice, see [voice-conversion.md](voice-conversion.md) |
| `POST /v1/voices` | Register or save a reference voice, see [voices.md](voices.md) |
| `GET /v1/voices` | List cached and saved voices |
| `DELETE /v1/voices/<id>` | Drop a voice from memory |

A separate WebSocket listener handles incremental text, mid stream direction
changes and interruption. It is documented in [websocket.md](websocket.md).

## Starting the server

```
breeze-server <model.gguf> [--host H] [--port P] [--webui] [--cpu]
                           [--chunk-first N] [--chunk-max N] [--verbose]
                           [--voices-dir PATH] [--ws-port P] [--split-chars N]
                           [--cors [ORIGINS]]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Interface to bind. Use `0.0.0.0` to accept remote connections. |
| `--port` | `8080` | TCP port. |
| `--webui` | off | Also serve the browser UI at `/`. |
| `--cpu` | off | Force the CPU backend instead of Vulkan. |
| `--chunk-first` | `4` | Frames in the first streamed chunk. |
| `--chunk-max` | `25` | Frames the chunk ramps up to. |
| `--verbose` | off | Add a per stage timing breakdown after each request. |
| `--voices-dir` | `voices` | Folder of saved `.breeze` voices to load at startup. See [voices.md](voices.md). |
| `--ws-port` | HTTP port + 1 | Port for streaming sessions. `-1` disables it. See [websocket.md](websocket.md). |
| `--split-chars` | `600` | Default length long text is broken up at. `0` sends the whole thing through in one pass. A request can still override it. |
| `--cors` | off | Let browser pages on other origins call the API. On its own it allows any origin; give a comma separated list to allow only those. See [Cross origin requests](#cross-origin-requests). |

```
breeze-server breeze-tts-2-q4_k.gguf --port 8137 --webui
```

## Console output

Each request prints a progress bar that fills as audio is streamed, then a
summary.

```
gen  design, 291 chars, cfg 1.0, seed 7
 87%|████████████████████▉   | 15.1/17.4s [00:11<00:01, 16.2 fps, 1.30x]
100%|████████████████████████| 19.4/19.4s [00:15, 16.1 fps, 1.29x]
     242 frames in 11 flushes, first audio 588 ms
```

The total is estimated from the text, since the model decides for itself when to
stop, so the bar can reach 100% early or run past its estimate. The last line
rewrites it with the real figure. `fps` is frames generated per second against
12.5 frames of audio per second, so the trailing number is the real time factor
and anything above `1.00x` is faster than playback.

There is no authentication and no rate limiting, on either the HTTP port or the
WebSocket one. Do not expose them directly to the internet; put them behind a
reverse proxy that handles both.

## Cross origin requests

The bundled web UI is served from the same origin as the API, so it needs
nothing. A page served from anywhere else, a Vite dev server or a separate front
end for instance, is blocked by the browser unless the server sends CORS
headers. It does not by default. Turn them on with `--cors`:

```
breeze-server model.gguf --cors
breeze-server model.gguf --cors http://localhost:5173,https://app.example.com
```

On its own the flag answers every origin with `Access-Control-Allow-Origin: *`.
With a list it reflects the request's `Origin` only when it matches an entry
exactly, scheme and port included, and adds `Vary: Origin` so caches keep the
answers apart. An origin that is not on the list gets no CORS headers and the
browser refuses the response.

`X-Sample-Rate` and `X-Sample-Format` are listed in
`Access-Control-Expose-Headers`, so a cross origin page can read the sample
rate the same way the bundled UI does. Preflight `OPTIONS` requests, which the
browser sends before `DELETE /v1/voices/<id>`, are answered with `204` and
cached for a day.

The WebSocket port is not affected. Browsers do not apply CORS to WebSockets,
and the socket accepts a connection from any page regardless of this flag, as
[websocket.md](websocket.md) notes.

## Streaming without stutter

Two numbers decide whether a stream plays cleanly, and they fail in different
ways. Getting one right does not save you from the other.

### Real time factor

The real time factor is seconds of audio produced per second of wall clock. Above
1.0 the model outruns playback, below it no amount of buffering will help because
the client drains faster than the server fills.

Measure it against your own hardware rather than assuming:

```
curl -s -o out.pcm -w "%{time_total}\n" \
  --form-string "text=<a sentence long enough to take several seconds>" \
  http://127.0.0.1:8137/v1/audio/speech
```

Audio seconds are `bytes / 2 / 24000`, so the factor is that divided by
`time_total`. On an RTX 3060 a long clip measures about 1.29x at Q4_K and 1.18x
at Q8_0 with the defaults. Those are thin margins. A busy GPU eats them.

Guidance costs less than it looks like it should. `cfg_scale` above 1 adds a
second forward pass, but the depth decoder batches both branches into one graph,
so voice direction with a cloned reference measures 1.49x at `cfg_scale` 1 and
1.20x at 4, still comfortably above realtime. Cloning also adds a one off
reference encode of roughly 650 ms, which shows up in time to first audio rather
than in the rate.

If you land near or below 1.0, no client setting will fix it. Drop to a smaller
quantisation, raise `--chunk-max`, or use a faster device. Quantising the depth
decoder with `--depth` also helps, but it damages the high end and the damage
grows over a long passage, so weigh it against the numbers in
[models.md](models.md).

### Queue depth

This is the one that actually causes stutter on a machine whose real time factor
looks fine.

The client's queue drains continuously but refills in one lump per chunk, so the
queue has to be deeper than the time it takes to produce a whole chunk. At
`--chunk-max 40` a chunk is 3.2 s of audio that takes roughly 2.5 s to generate,
so a client holding only 0.5 s of audio runs dry waiting for it, even though the
average rate is comfortably ahead.

The rule: **buffer more audio than the slowest chunk takes to produce.** For the
defaults that means at least 1 s, and the bundled UI prebuffers 1.25 s.

Note the difference between prebuffering and delaying. Starting the first chunk
half a second late does not create a cushion, it only shifts the start; the queue
still holds one chunk. Accumulate chunks until you are actually holding N seconds
of audio, then play them back to back. The bundled UI does this and exposes N as
the buffer slider.

### Tuning the chunk ramp

Audio is vocoded in chunks of whole frames at 12.5 frames per second. The first
chunk sets how long the client waits for sound, so the chunk starts small and
grows by a third each flush until it reaches `--chunk-max`.

Every flush also re-decodes `sliding_window + 16` frames of left context and
throws that audio away, so small chunks are expensive. At 25 frames the vocoder
decodes 113 frames to emit 25. Raising the ceiling reclaims most of it, measured
on an RTX 3060 at Q8_0:

| `--chunk-max` | Vocoder per frame | Total per frame | Real time factor |
| --- | --- | --- | --- |
| 25 | 13.15 ms | 66.11 ms | 1.21x |
| 40 | 10.87 ms | 62.10 ms | 1.29x |
| 60 | 10.89 ms | 62.25 ms | 1.29x |

Past about 40 the context is amortised and there is nothing left to win, while
chunks keep getting slower to produce and demand a deeper client queue.

Lower `--chunk-first` for a faster start. Four frames is about 320 ms of audio
and lands near 400 ms on an RTX 3060; one frame gets there in roughly 220 ms but
flushes far more often. It only affects the first chunk, so it costs nothing in
throughput.

Setting both flags to the same value disables the ramp and streams a fixed size.

### Summary

- Aim for `--chunk-max 40` unless you are on a fast device and want a finer stream.
- Buffer at least 1 s on the client, more if you raised `--chunk-max`.
- Prebuffer by fill level, not by delay.
- Check the real time factor first if it stutters no matter what you buffer.

## `GET /health`

Liveness probe. Returns once the model is loaded and ready.

```
curl http://127.0.0.1:8137/health
```

```json
{"status":"ok","sample_rate":24000,"ws_port":8081}
```

The server does not accept connections until loading finishes, so a successful
response also means the model is warm. Poll this after startup instead of
guessing at a delay.

`ws_port` is where the streaming socket ended up, or `0` when it is disabled, so
clients can discover it rather than being configured with it.

## `POST /v1/audio/speech`

Generates speech and streams it back.

Accepts `multipart/form-data` (needed for the reference audio upload) or
`application/x-www-form-urlencoded`.

### Fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `text` | string | required | Text to speak, UTF-8. |
| `instruction` | string | `Speak clearly and naturally.` | Voice description or delivery direction. |
| `ref_audio` | file | none | Reference WAV for cloning. Any sample rate or channel count; resampled to mono at the model rate. |
| `ref_text` | string | empty | Exact transcript of `ref_audio`. Required whenever `ref_audio` is present. |
| `voice_id` | string | none | A saved or cached voice to clone instead of uploading a clip. Skips the reference encode. See [voices.md](voices.md). |
| `cfg_scale` | float | `1.0` | Classifier free guidance. `1.0` disables it. |
| `seed` | int | `42` | RNG seed. |
| `temperature` | float | model default | Sampling temperature. `0` keeps whatever the GGUF was built with. |
| `top_k` | int | model default | Sampling top-k. `0` keeps the model default. |
| `top_p` | float | model default | Sampling top-p. `0` keeps the model default. |
| `repetition_penalty` | float | model default | Repetition penalty. `0` keeps the model default. |
| `split_chars` | int | `--split-chars` | Long text is split on sentence boundaries into pieces of about this size and generated one at a time. `0` generates in one pass. |
| `max_new_tokens` | int | model default | Frame cap per piece, 12.5 frames per second. `0` uses the model default of 750. |

Every sampling field treats `0` as "use the model default", so leaving them out
behaves exactly as before rather than forcing a zero.

There is no length limit on `text`. Anything past the budget is split and
generated piece by piece, each conditioned on the first so the voice does not
change partway. Generating minutes in a single pass is what `split_chars 0`
does, and the model loses track of the text well before it runs out of frames,
so leave splitting on unless you have a reason not to. Starting the server with
`--split-chars 0` turns it off for every request that does not set its own.

### Response

`200 OK` with a chunked body.

| Header | Value |
| --- | --- |
| `Content-Type` | `audio/pcm` |
| `Transfer-Encoding` | `chunked` |
| `X-Sample-Rate` | `24000` |
| `X-Sample-Format` | `s16le` |
| `Cache-Control` | `no-store` |

The body is **headerless** signed 16 bit little endian mono PCM. It is not a
WAV file. Wrap it yourself if you need one, or use the CLI which writes WAV
directly.

Chunks arrive roughly every 2 seconds of generated audio, so playback can start
long before generation finishes.

### Errors

| Status | Body | Cause |
| --- | --- | --- |
| `400` | `{"error":"text is required"}` | `text` missing or empty. |
| `404` | `{"error":"unknown voice_id"}` | No cached or saved voice by that name. |
| `409` | `{"error":"busy"}` | Another generation is already running. |

The server holds one model and serves one request at a time. A second
concurrent request is rejected immediately with `409` rather than queued, so
clients should retry with backoff or run several servers behind a load
balancer. The WebSocket endpoint queues instead of refusing, which is one reason
to prefer it for anything conversational.

Because the response is streamed, a failure that happens **after** the first
chunk cannot change the status code. The connection is closed early instead, so
treat a truncated body as an error.

### Examples

Send text fields with `--form-string`, not `-F`. `curl` reads a value that starts
with `(` as the opening of a multipart group, so `-F "text=(sigh) ..."` posts an
empty field and the model ends up reading the multipart boundary out loud.

Voice design:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  --form-string "text=Welcome aboard. Your journey begins now." \
  --form-string "instruction=A warm, thoughtful young woman with a clear, calm delivery." \
  --form-string "cfg_scale=1" \
  --form-string "seed=42" \
  -o speech.pcm
```

Voice clone:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  --form-string "text=It is good to hear your voice again." \
  -F "ref_audio=@reference.wav" \
  --form-string "ref_text=This is the exact transcript of the reference audio." \
  -o clone.pcm
```

Voice direction, which is a clone plus an instruction:

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  --form-string "text=We need to discuss what happened last night." \
  --form-string "instruction=Speak slowly with a restrained, serious tone." \
  -F "ref_audio=@reference.wav" \
  --form-string "ref_text=This is the exact transcript of the reference audio." \
  --form-string "cfg_scale=1" \
  -o direction.pcm
```

Play the raw stream straight out of `curl`:

```
curl -sN -X POST http://127.0.0.1:8137/v1/audio/speech --form-string "text=Hello there." \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

Convert to WAV:

```
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.pcm speech.wav
```

## OpenAI compatible requests

`POST /v1/audio/speech` also takes the JSON body of OpenAI's speech API, so
apps built for it, like SillyTavern or the official OpenAI SDKs, can point at
the server as they are. The `Content-Type` header decides which one you get:
`application/json` gets the behaviour described here, a form post gets the
native behaviour above.

```
curl http://127.0.0.1:8137/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello from Breeze.","voice":"narrator","response_format":"mp3"}' \
  -o speech.mp3
```

With the Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8137/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(
    model="tts-1", voice="narrator", input="Streaming from the OpenAI SDK.",
) as r:
    r.stream_to_file("speech.mp3")
```

### Fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `input` | string | required | Text to speak, UTF-8. |
| `voice` | string or `{"id": ...}` | none | A saved or cached voice, the same ids `voice_id` takes. Leave it out to design a voice from `instructions` instead. |
| `instructions` | string | `Speak clearly and naturally.` | Voice description when designing, delivery direction when cloning. |
| `response_format` | string | `mp3` | `mp3`, `wav` or `pcm`. `opus`, `aac` and `flac` are accepted but sent as WAV. |
| `stream_format` | string | `audio` | `audio` streams the file itself, `sse` wraps it in server sent events. |
| `model` | string | ignored | Accepted so clients that always send one keep working. |
| `speed` | number | ignored | The model has no speed control. |

The native fields `cfg_scale`, `seed`, `temperature`, `top_k`, `top_p`,
`repetition_penalty`, `max_new_tokens` and `split_chars` are accepted as extra
JSON fields, with the same meaning and defaults as above.

OpenAI's own voice names such as `alloy` or `nova` are not built in, and a name
the server does not know is a `404`. If a client insists on those names, save
voices under them with `POST /v1/voices` (see [voices.md](voices.md)) and they
work like any other.

### Formats

| `response_format` | Sent as | `Content-Type` |
| --- | --- | --- |
| `mp3` | MP3, 64 kbps mono at 24 kHz | `audio/mpeg` |
| `wav` | 16 bit mono WAV at 24 kHz | `audio/wav` |
| `pcm` | Headerless s16le mono at 24 kHz, the same as the native endpoint and OpenAI's own `pcm` | `audio/pcm` |
| `opus`, `aac`, `flac` | WAV | `audio/wav` |

Every format streams as it is generated, so playback can start on the first
chunk. The WAV header goes out first with its length fields set to the maximum,
because the length is not known yet. Browsers and ffmpeg read that to the end
of the stream.

MP3 is encoded with [shine](https://github.com/toots/shine), which builds from
the `third_party/shine` submodule. A build without it, or one configured with
`-DBREEZE_MP3=OFF`, sends WAV when asked for `mp3` as well. The console line for
each request says what actually went out, for example `openai wav for opus`.

### Server sent events

With `"stream_format": "sse"` the response is `text/event-stream`. Each chunk
arrives as a `speech.audio.delta` event carrying the next bytes of the file in
`response_format`, base64 encoded, and a `speech.audio.done` event ends it.

```
data: {"type":"speech.audio.delta","audio":"//NIxAAAAAAAAAAAAFhpbmcAAAAPAAAA..."}

data: {"type":"speech.audio.done","usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}
```

Decoding the deltas and joining them gives the same file a plain request
returns. The token counts are always zero.

### Errors

Errors on this path use OpenAI's error shape, so clients show the message
instead of a bare status code.

```json
{"error":{"message":"input is required","type":"invalid_request_error","param":null,"code":null}}
```

| Status | Cause |
| --- | --- |
| `400` | The body is not a JSON object, `input` is missing, or a field has the wrong type or an unknown value. |
| `404` | `voice` is not a saved or cached voice. |
| `409` | Another generation is already running. The OpenAI SDKs retry this on their own. |

The `Authorization` header is ignored, so any API key works. That also means
there is still no authentication, see the warning above.

### SillyTavern

Open the TTS extension, pick **OpenAI Compatible** as the provider and fill in:

| Setting | Value |
| --- | --- |
| Provider Endpoint | `http://127.0.0.1:8137/v1/audio/speech` |
| API Key | Anything, it is ignored. |
| Model | Anything, it is ignored. |
| Available Voices | The voice ids you want, comma separated, e.g. `narrator,harbour`. The list from `GET /v1/voices` has them all. |

The default voice list is OpenAI's names, which will all come back `404` until
you replace it. Speed does nothing. SillyTavern asks for MP3, which is what it
gets.

## Streaming client sketch

```python
import struct
import urllib.request

body, boundary = build_multipart({"text": "Streaming from python."})
req = urllib.request.Request(
    "http://127.0.0.1:8137/v1/audio/speech",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

with urllib.request.urlopen(req) as r:
    rate = int(r.headers["X-Sample-Rate"])
    tail = b""
    while chunk := r.read(8192):
        chunk = tail + chunk
        n = len(chunk) // 2
        tail = chunk[n * 2:]
        samples = struct.unpack(f"<{n}h", chunk[:n * 2])
        play(samples, rate)
```

Buffer any odd trailing byte between reads, since a chunk boundary can land in
the middle of a sample.

## Web UI

`--webui` mounts a dark brutalist single page UI that mirrors the reference
Gradio demo.

| Route | Serves |
| --- | --- |
| `GET /` | `index.html` |
| `GET /style.css` | stylesheet |
| `GET /app.js` | client script |

It has three tabs matching the three generation modes, and it posts to the same
`/v1/audio/speech` endpoint.

**Stream while generating** is on by default. The UI reads the response body
incrementally and schedules each PCM chunk through the Web Audio API as it
arrives, so playback starts about a second in rather than after the whole clip is
rendered. A running counter shows how much audio has been produced, and appends a
rebuffer count if the queue ever ran dry. When generation finishes the same PCM
is wrapped into a WAV blob for the player and the download link.

**Buffer** is how much audio the UI holds before it starts playing, not how long
it waits. Chunks accumulate until that much is queued, then play back to back.
The default of 1.25 s covers the defaults here; raise it if you raised
`--chunk-max` or see rebuffers, lower it if you want a faster start and your real
time factor has room. If the queue does run dry the UI widens its own buffer for
the rest of that clip rather than stuttering repeatedly.

Turn the toggle off to buffer the whole response first and play it back from the
`<audio>` element, which is the better choice if you want to scrub the result
rather than hear it as early as possible.

The assets are compiled into the binary at build time from the
[webui](../webui) folder by `cmake/embed_webui.cmake`, so there is no static
directory to deploy. Editing a file there and rebuilding is enough to pick up
changes.
