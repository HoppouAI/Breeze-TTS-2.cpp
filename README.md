# Breeze-TTS-2.cpp

A C++ / GGUF reimplementation of [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2),
running on [ggml](https://github.com/ggml-org/ggml) with a Vulkan backend so it works across GPUs and
falls back to CPU. Ships a CLI, a streaming HTTP server with an OpenAI style endpoint, a dark
brutalist web UI, and a plain C shared library for bindings.

Breeze TTS 2 is a bilingual (English / Chinese) instruction following speech model. It does voice
design (make a voice from a text description), voice cloning (copy a voice from a reference clip), and
voice direction (clone a voice and steer tone or pace). It supports inline vocal events like
`(laugh)`, `(sigh)`, `(cough)` in English and `[笑]`, `[叹气]` in Chinese.

## Documentation

| Guide | Contents |
| --- | --- |
| [docs/build.md](docs/build.md) | Toolchain, CMake options, troubleshooting |
| [docs/models.md](docs/models.md) | Weights, GGUF conversion, quantization |
| [docs/cli.md](docs/cli.md) | `breeze-cli` reference |
| [docs/server.md](docs/server.md) | HTTP endpoints and streaming |
| [docs/websocket.md](docs/websocket.md) | Streaming sessions, incremental text, interruption |
| [docs/voices.md](docs/voices.md) | Caching and saving reference voices |
| [docs/voice-conversion.md](docs/voice-conversion.md) | Respeaking a recording in another voice |
| [docs/c-api.md](docs/c-api.md) | The C API in `breeze.h` |
| [docs/ctypes.md](docs/ctypes.md) | Binding the library from Python |
| [docs/architecture.md](docs/architecture.md) | How the four stages work |

## Build

```
git clone --recursive <this repo>
cd Breeze-TTS-2.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Add `-DBREEZE_VULKAN=OFF` for a CPU only build. Outputs are `breeze-cli`, `breeze-server`,
`breeze-quantize` and the shared library.

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

The runtime loads a single GGUF file. The download must include the `audio_tokenizer/` directory,
which holds the vocoder.

```
pip install -r scripts/requirements.txt
python scripts/convert_hf_to_gguf.py /path/to/Breeze-TTS-2 -o breeze-tts-2-f16.gguf --dtype f16
build/breeze-quantize breeze-tts-2-f16.gguf breeze-tts-2-q4_k.gguf q4_k
```

| Variant | Size | Approximate VRAM |
| --- | --- | --- |
| F16 | 5.9 GB | ~7 GB |
| Q8_0 | 3.3 GB | ~4 GB |
| Q6_K | 2.9 GB | ~3.5 GB |
| Q4_K | 2.4 GB | ~3 GB |

## CLI

```
# voice design
build/breeze-cli breeze-tts-2-q4_k.gguf \
  --text "(sigh) Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear, calm delivery." \
  --output design.wav

# voice clone
build/breeze-cli breeze-tts-2-q4_k.gguf \
  --text "It is good to hear your voice again." \
  --ref-audio reference.wav --ref-text "Exact transcript of the reference." \
  --output clone.wav

# voice direction
build/breeze-cli breeze-tts-2-q4_k.gguf \
  --text "(clears throat) We need to talk." \
  --instruction "Speak slowly with a restrained, serious tone." \
  --ref-audio reference.wav --ref-text "Exact transcript of the reference." \
  --output direction.wav
```

## Server

```
build/breeze-server breeze-tts-2-q4_k.gguf --host 0.0.0.0 --port 8080 --webui
```

Open http://localhost:8080/ for the web UI, or call the API directly. The response is streaming mono
24 kHz signed 16 bit little endian PCM, matching the reference server.

```
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  --form-string "text=(clears throat) We need to talk." \
  --form-string "instruction=Speak slowly with a restrained, serious tone." \
  -F "ref_audio=@reference.wav" \
  --form-string "ref_text=Exact transcript of the reference." \
  --output out.pcm
```

Use `--form-string` for text fields, not `-F`. A value starting with `(` makes `curl` build a
multipart group instead of sending the text, which matters the moment you use a vocal event tag.

## Bindings

The shared library exposes a plain C ABI, so any language with an FFI can drive it. See
[docs/c-api.md](docs/c-api.md) for the interface and [docs/ctypes.md](docs/ctypes.md) for a working
Python wrapper.

```c
breeze_context * ctx = breeze_init("breeze-tts-2-q4_k.gguf", 1);
breeze_request req = {0};
req.text = "Hello there.";
req.cfg_scale = 1.0f;
req.seed = 42;
breeze_generate_wav(ctx, &req, "out.wav");
breeze_free(ctx);
```

On mingw the library links the compiler runtime statically, so `vulkan-1.dll` from the GPU driver is
the only external dependency.

## Notes

- `--cfg-scale` defaults to 1.0, matching the reference. Values above 1 run the whole pipeline twice
  and push harder toward the instruction, at the cost of some harshness past about 2.
- The vocoder comes from the bundled `audio_tokenizer/`, not the Mimi codec sitting in the main
  checkpoint. The reference never uses that one at inference time.
- Streaming decodes in 2 second chunks with 72 frames of left context, which matches the vocoder
  transformer's attention window, so chunks decode the same as they would in a single pass.

## License

Source code here is provided under Apache 2.0. The Breeze TTS 2 model weights are governed by the
BreezeBlue Research and Non-Commercial License. You are responsible for complying with the weight
license and for obtaining consent for any reference audio or voices you use.
