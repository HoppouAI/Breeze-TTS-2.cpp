# C API

The shared library exposes a small C ABI declared in
[include/breeze/breeze.h](../include/breeze/breeze.h). It is the intended entry
point for bindings in any language.

Build it with `-DBREEZE_BUILD_SHARED=ON` (the default). The output is
`breeze.dll` on MSVC, `libbreeze.dll` on mingw, `libbreeze.so` on Linux and
`libbreeze.dylib` on macOS.

On mingw the library is linked statically against the compiler runtime, so the
only non system dependency is `vulkan-1.dll`, which ships with the GPU driver.

## Header

```c
#include "breeze/breeze.h"
```

Everything is `extern "C"` and safe to include from C or C++.

## Types

### `breeze_context`

An opaque handle holding the loaded model, the compute backend and the
tokenizer. Creating one allocates several gigabytes, so create it once and keep
it for the lifetime of the process.

### `breeze_audio_cb`

```c
typedef int (*breeze_audio_cb)(const float * samples, int n_samples, void * user);
```

Called on the calling thread every time a chunk of audio is ready. `samples`
points to `n_samples` mono floats in the range -1 to 1 at the model sample
rate. The buffer is only valid for the duration of the call, so copy it if you
need to keep it.

Return `0` to keep generating. Return anything else to stop early; the
in flight frame finishes and `breeze_generate` returns normally.

`user` is passed straight through from `breeze_generate`.

### `breeze_request`

```c
typedef struct {
    const char * text;
    const char * instruction;
    const char * ref_text;
    const float * ref_audio;
    int   ref_audio_len;
    float cfg_scale;
    int   seed;
    int   max_new_tokens;
    int   split_chars;
    float temperature;
    int   top_k;
    float top_p;
    float repetition_penalty;
} breeze_request;
```

| Field | Meaning |
| --- | --- |
| `text` | The text to speak. Required. UTF-8. |
| `instruction` | Voice description or delivery direction. `NULL` uses `Speak clearly and naturally.` |
| `ref_text` | Exact transcript of `ref_audio`. Required when cloning, otherwise `NULL`. |
| `ref_audio` | Mono reference samples at the model sample rate, or `NULL`. |
| `ref_audio_len` | Number of floats in `ref_audio`. |
| `cfg_scale` | Classifier free guidance. `1.0` disables it and skips the second pass. |
| `seed` | RNG seed. The same seed and inputs reproduce the same audio. |
| `max_new_tokens` | Frame cap per piece. `0` uses the model default of 750 frames, about 60 seconds. |
| `split_chars` | Size of the pieces long text is broken into. `0` uses the default of 600. Negative generates in one pass, which caps you at `max_new_tokens` and degrades past a minute or so. |
| `temperature` | Sampling temperature. `0` keeps whatever the GGUF was built with. |
| `top_k` | Sampling top-k. `0` keeps the model default. |
| `top_p` | Sampling top-p. `0` keeps the model default. |
| `repetition_penalty` | Repetition penalty. `0` keeps the model default. |

Zero initialise the struct before filling it so future fields default sanely:

```c
breeze_request req = {0};
req.text = "Hello there.";
req.cfg_scale = 1.0f;
req.seed = 42;
```

Supplying `ref_audio` without `ref_text` (or the reverse) is rejected.

## Functions

### `breeze_init`

```c
breeze_context * breeze_init(const char * gguf_path, int use_gpu);
```

Loads a GGUF model. Pass `1` for `use_gpu` to try Vulkan first and fall back to
CPU if no device is usable, or `0` to force CPU.

Returns `NULL` on failure; call `breeze_last_error` for the reason.

### `breeze_free`

```c
void breeze_free(breeze_context * ctx);
```

Releases the model and backend. Safe to call with `NULL`.

### `breeze_sample_rate`

```c
int breeze_sample_rate(breeze_context * ctx);
```

Output sample rate in Hz, read from the model metadata. Currently 24000. Use
this rather than hardcoding, and resample reference audio to match before
passing it in.

### `breeze_generate`

```c
int breeze_generate(breeze_context * ctx, const breeze_request * req,
                    breeze_audio_cb cb, void * user);
```

Runs the full pipeline and streams audio to `cb` as it is produced. Returns `0`
on success, non zero on failure.

The call blocks until generation finishes. Audio arrives in chunks that start at 1
frame (80 ms) and grow to 25 frames (2 seconds), except for the final chunk, which
holds whatever is left.

### `breeze_generate_wav`

```c
int breeze_generate_wav(breeze_context * ctx, const breeze_request * req,
                        const char * out_path);
```

Same as `breeze_generate` but buffers everything and writes a 16 bit PCM WAV.
Returns `0` on success.

### `breeze_last_error`

```c
const char * breeze_last_error(void);
```

Message for the most recent failure on the current thread. The pointer stays
valid until the next failing call on that thread.

## Thread safety

A `breeze_context` is **not** reentrant. One generation at a time per context.
Guard it with a mutex, or create one context per worker if you have the memory
for it, since each holds its own copy of the weights.

`breeze_init` and `breeze_free` are safe to call concurrently for different
contexts. `breeze_last_error` is thread local.

## Streaming to a live output

`breeze_generate` calls back with chunks as they are vocoded, so it can drive a
sound device directly. Two things matter if you do that.

The callback runs on the calling thread and generation is blocked until it
returns, so do not do anything slow in it. Copy into a ring buffer and let the
audio device drain that on its own thread.

Size that ring buffer by production time, not by chunk count. Chunks arrive in
growing sizes, and the queue drains continuously while the next one is being
generated, so it needs to hold more audio than one chunk takes to produce. On an
RTX 3060 at Q8_0 that is upwards of a second. The reasoning and the measurements
are in [server.md](server.md#streaming-without-stutter), and they apply to any
client, not just the browser one.

Chunk sizes are the library defaults here. `breeze_request` has no field for
them, so a C consumer cannot tune the ramp the way `breeze-cli` and
`breeze-server` can; compensate on the client buffer instead.

## Complete example

```c
#include "breeze/breeze.h"
#include <stdio.h>

static int on_audio(const float * s, int n, void * user) {
    FILE * f = (FILE *) user;
    for (int i = 0; i < n; i++) {
        float v = s[i] < -1.0f ? -1.0f : (s[i] > 1.0f ? 1.0f : s[i]);
        short pcm = (short) (v * 32767.0f);
        fwrite(&pcm, sizeof pcm, 1, f);
    }
    return 0;
}

int main(void) {
    breeze_context * ctx = breeze_init("breeze-tts-2-q4_k.gguf", 1);
    if (!ctx) {
        fprintf(stderr, "%s\n", breeze_last_error());
        return 1;
    }

    breeze_request req = {0};
    req.text = "The build finished without a single warning.";
    req.instruction = "A calm narrator with a warm tone.";
    req.cfg_scale = 1.0f;
    req.seed = 42;

    FILE * raw = fopen("out.pcm", "wb");
    int rc = breeze_generate(ctx, &req, on_audio, raw);
    fclose(raw);

    if (rc != 0) fprintf(stderr, "%s\n", breeze_last_error());
    breeze_free(ctx);
    return rc;
}
```

Compile against the static core instead if you would rather not ship a DLL:

```
gcc example.c -Iinclude -Lbuild -lbreeze_core -lggml -lstdc++ -o example
```
