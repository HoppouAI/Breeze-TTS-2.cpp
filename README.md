<div align="center">

# Breeze-TTS-2.cpp

**Bilingual instruction following text to speech in C++ and GGUF.**
Voice design, voice cloning, voice direction and experimental voice conversion,
streaming at better than realtime on a mid range GPU.

<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-1f6feb?style=for-the-badge" alt="License"></a>
<img src="https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++17">
<img src="https://img.shields.io/badge/Vulkan-AC162C?style=for-the-badge&logo=vulkan&logoColor=white" alt="Vulkan">
<a href="https://huggingface.co/HoppouAI/Breeze-TTS-2.cpp"><img src="https://img.shields.io/badge/GGUF_weights-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="GGUF weights"></a>

</div>

A C++ / GGUF reimplementation of [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2),
running on [ggml](https://github.com/ggml-org/ggml). The Vulkan backend means it works across NVIDIA,
AMD and Intel GPUs, and it falls back to CPU. Ships a CLI, a streaming HTTP and WebSocket server with
a web UI, and a plain C shared library for bindings.

English and Mandarin, 24 kHz, roughly 1.2x realtime at Q8_0 on an RTX 3060.

## Documentation

<div align="center">

<a href="docs/build.md"><img src="https://img.shields.io/badge/Build-2b3137?style=for-the-badge" alt="Build"></a>
<a href="docs/models.md"><img src="https://img.shields.io/badge/Models-2b3137?style=for-the-badge" alt="Models"></a>
<a href="docs/cli.md"><img src="https://img.shields.io/badge/CLI-2b3137?style=for-the-badge" alt="CLI"></a>
<a href="docs/server.md"><img src="https://img.shields.io/badge/HTTP_Server-2b3137?style=for-the-badge" alt="Server"></a>
<a href="docs/websocket.md"><img src="https://img.shields.io/badge/WebSocket-2b3137?style=for-the-badge" alt="WebSocket"></a>
<br>
<a href="docs/voices.md"><img src="https://img.shields.io/badge/Saved_Voices-2b3137?style=for-the-badge" alt="Voices"></a>
<a href="docs/voice-conversion.md"><img src="https://img.shields.io/badge/Voice_Conversion-6e2b3a?style=for-the-badge" alt="Voice conversion"></a>
<a href="docs/c-api.md"><img src="https://img.shields.io/badge/C_API-2b3137?style=for-the-badge" alt="C API"></a>
<a href="docs/ctypes.md"><img src="https://img.shields.io/badge/Python-2b3137?style=for-the-badge" alt="Python"></a>
<a href="docs/architecture.md"><img src="https://img.shields.io/badge/Architecture-2b3137?style=for-the-badge" alt="Architecture"></a>

</div>

## What it does

| Mode | You give it | You get |
| --- | --- | --- |
| **Voice design** | A text description of a voice | A voice invented to match the description |
| **Voice cloning** | A reference clip and its transcript | New speech in that voice |
| **Voice direction** | A reference clip plus an instruction | That voice, steered in tone or pace |
| **Voice conversion** | A recording to respeak, and a target voice | The same performance in a different voice |

## Build

```
git clone --recursive https://github.com/HoppouAI/Breeze-TTS-2.cpp
cd Breeze-TTS-2.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Add `-DBREEZE_VULKAN=OFF` for a CPU only build. Outputs are `breeze-cli`, `breeze-convert`,
`breeze-server`, `breeze-quantize` and the shared library. See [docs/build.md](docs/build.md).

### Nix

If you have [Nix](https://nixos.org/) with flake support, the dev shell provides all build
dependencies (CMake, Ninja, Vulkan SDK, Python for conversion) and a few convenience commands:

To enter the dev shell, run:

```
nix develop          # or: nix develop .#cpu
```

After entering the dev shell, run the following commands to build and run the CLI/server:

```
breeze-build         # configure + build (Vulkan)
breeze-build --cpu   # configure + build (CPU only)
breeze-cli           # run the CLI
breeze-server        # run the server
```

## Convert the weights

## Weights


<div align="center">

<a href="https://huggingface.co/HoppouAI/Breeze-TTS-2.cpp"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg.svg" alt="Model on Hugging Face"></a>

</div>

Prebuilt GGUFs are on the Hub at
[HoppouAI/Breeze-TTS-2.cpp](https://huggingface.co/HoppouAI/Breeze-TTS-2.cpp), or convert them
yourself. The download must include the `audio_tokenizer/` directory, which holds the vocoder.

```
pip install -r scripts/requirements.txt
python scripts/convert_hf_to_gguf.py /path/to/Breeze-TTS-2 -o breeze-tts-2-f16.gguf --dtype f16
build/breeze-quantize breeze-tts-2-f16.gguf breeze-tts-2-q4_k.gguf q4_k
```

| Variant | Size | Approximate VRAM | Notes |
| --- | --- | --- | --- |
| F16 | 5.9 GB | ~7 GB | Reference quality |
| Q8_0 | 3.3 GB | ~4 GB | **Recommended** |
| Q6_K | 2.9 GB | ~3.5 GB | |
| Q4_K | 2.4 GB | ~3 GB | Smallest safe choice |
| Q8_0-dd4, Q8_0-dd2, Q4_K-dd2 | 2.3 to 3.4 GB | | Experimental, see below |

The `-dd` variants quantize the depth decoder as well, which the others leave at higher precision.
They can be meaningfully faster on hardware where the depth decoder is the bottleneck, since it runs
15 sequential steps for every single frame of audio. The catch is that depth codes feed back into the
backbone each frame, so error compounds with length: output holds up early and then drifts muffled and
thin past roughly 45 seconds of continuous generation. Fine for short lines, not for narration.

## CLI

```
# voice design
build/breeze-cli breeze-tts-2-q8_0.gguf \
  --text "(sigh) Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear, calm delivery." \
  --output design.wav

# voice clone
build/breeze-cli breeze-tts-2-q8_0.gguf \
  --text "It is good to hear your voice again." \
  --ref-audio reference.wav --ref-text "Exact transcript of the reference." \
  --output clone.wav

# voice direction
build/breeze-cli breeze-tts-2-q8_0.gguf \
  --text "(clears throat) We need to talk." \
  --instruction "Speak slowly with a restrained, serious tone." \
  --ref-audio reference.wav --ref-text "Exact transcript of the reference." \
  --output direction.wav
```

Encoding a reference clip is the slowest part of a clone and it is deterministic, so save it once with
`--save-voice name` and use `--voice name` from then on. Saved voices load instantly, work in the CLI,
the server and the web UI, and cut time to first audio from around 900 ms to around 280 ms. See
[docs/voices.md](docs/voices.md).

## Vocal events

Inline tags in round brackets produce non speech sounds. `(laugh)`, `(sigh)`, `(cough)` and
`(clears throat)` are the reliable ones, with `[笑]` and `[叹气]` on the Chinese side, but the tag
vocabulary is **free form**. The model was trained on descriptive tags rather than a fixed token list,
so things like `(whispering)`, `(gasp)` or `(nervous chuckle)` will often work.

The catch is that at the default `--cfg-scale 1.0` the model treats a tag as a suggestion and usually
ignores anything outside the common set. **Vocal events generally need `--cfg-scale 2` to `3` to
actually fire**, because guidance is what pushes the output away from plain neutral reading. Expect
some added harshness at that range, so raise it for lines that need the event and drop back to 1.0 for
ordinary speech.

```
build/breeze-cli breeze-tts-2-q8_0.gguf \
  --text "(nervous chuckle) I am sure it is nothing to worry about." \
  --instruction "An anxious man trying to sound casual." \
  --cfg-scale 2.5 --output event.wav
```

## Voice conversion (experimental)

`breeze-convert` respeaks an existing recording in another voice. Unlike cloning, which reads new text,
this keeps the original performance: the timing, the phrasing, the rhythm and the emphasis all survive,
and only the speaker identity changes. This is not part of the upstream model. It falls out of the way
the codec splits semantic content from acoustic detail, so it is unique to this implementation and it is
genuinely experimental.

```
build/breeze-convert breeze-tts-2-q8_0.gguf \
  --source recording.wav --text "Transcript of the recording." \
  --ref-audio target_voice.wav --ref-text "Exact transcript of the target clip." \
  --output converted.wav
```

Worth knowing before you rely on it:

- It defaults to near greedy sampling (`--temp 0.3 --top-k 1`) on purpose. Opening sampling up lets
  source timbre leak back into the result, which measurably weakens the target identity.
- **Pitch and melody do not survive.** The converted take follows the target voice's natural pitch, not
  the source's, so singing comes out spoken. `--keep-acoustic 1` or `2` retains the lowest acoustic
  codebooks from the source and recovers a usable amount of melody, at some cost to how cleanly the
  target voice comes through. Higher values fall apart.
- Runs at roughly realtime, so a three minute recording takes about three minutes.
- `--text` is optional. Giving the source transcript lines the backbone up with the forced codes and
  generally helps, but leaving it out runs textless and still works.

See [docs/voice-conversion.md](docs/voice-conversion.md).

## Server

```
build/breeze-server breeze-tts-2-q8_0.gguf --host 127.0.0.1 --port 8080 --webui
```

Open http://localhost:8080/ for the web UI. The HTTP API streams mono 24 kHz signed 16 bit little
endian PCM, matching the reference server.

```
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  --form-string "text=(clears throat) We need to talk." \
  --form-string "instruction=Speak slowly with a restrained, serious tone." \
  --form-string "voice_id=narrator" \
  --output out.pcm
```

Use `--form-string` for text fields, not `-F`. A value starting with `(` makes `curl` build a
multipart group instead of sending the text, which bites the moment you use a vocal event tag.

A WebSocket server also comes up on the HTTP port plus one. It takes text incrementally, streams audio
back as it is produced, and supports changing the delivery instruction or cancelling mid sentence,
which is what you want when driving it from a chat model. See [docs/websocket.md](docs/websocket.md).

> **Neither port has authentication or rate limiting.** Keep them bound to `127.0.0.1` unless something
> in front of them is handling that.

## Bindings

The shared library exposes a plain C ABI, so any language with an FFI can drive it.

```c
breeze_context * ctx = breeze_init("breeze-tts-2-q8_0.gguf", 1);
breeze_request req = {0};
req.text = "Hello there.";
req.cfg_scale = 1.0f;
req.seed = 42;
breeze_generate_wav(ctx, &req, "out.wav");
breeze_free(ctx);
```

Every field treats `0` as "use the model default", so zero initialising the struct is safe and stays
safe as fields are added. See [docs/c-api.md](docs/c-api.md) and [docs/ctypes.md](docs/ctypes.md).

On mingw the library links the compiler runtime statically, so `vulkan-1.dll` from the GPU driver is
the only external dependency.

## Notes

- `--cfg-scale` defaults to 1.0, matching the reference. Values above 1 run the whole pipeline twice
  and push harder toward the instruction. Past about 3 it turns harsh.
- The vocoder comes from the bundled `audio_tokenizer/`, not the Mimi codec sitting in the main
  checkpoint. The reference never uses that one at inference time.
- Streaming decodes in 2 second chunks with 72 frames of left context, which matches the vocoder
  transformer's attention window, so chunks decode the same as they would in a single pass.

## License

Source code is [Apache 2.0](LICENSE). The Breeze TTS 2 model weights are governed by the BreezeBlue
Research and Non-Commercial License. You are responsible for complying with the weight license and for
obtaining consent for any reference audio or voices you use.
