# Building

## Requirements

| Tool | Notes |
| --- | --- |
| CMake | 3.19 or newer |
| C++17 compiler | mingw-w64 (winlibs UCRT), MSVC, gcc or clang |
| Vulkan SDK | Only for the GPU backend. Provides `glslc` for shader compilation. |
| Ninja | Optional but much faster than the default generator |
| Python 3.9+ | Only for converting weights to GGUF |

The Vulkan backend works on any driver exposing Vulkan 1.2, which covers
NVIDIA, AMD and Intel on both Windows and Linux.

## Clone

The ggml backend is a submodule, so clone recursively:

```
git clone --recursive https://github.com/<owner>/Breeze-TTS-2.cpp
cd Breeze-TTS-2.cpp
```

If you already cloned without `--recursive`:

```
git submodule update --init --recursive
```

## Configure and build

GPU (Vulkan):

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CPU only:

```
cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DBREEZE_VULKAN=OFF
cmake --build build-cpu -j
```

## CMake options

| Option | Default | Effect |
| --- | --- | --- |
| `BREEZE_VULKAN` | `ON` | Build ggml with the Vulkan backend. |
| `BREEZE_CUDA` | `OFF` | Build ggml with the CUDA backend. Needs the CUDA toolkit, and MSVC on Windows. |
| `BREEZE_BUILD_CLI` | `ON` | Build `breeze-cli`. |
| `BREEZE_BUILD_SERVER` | `ON` | Build `breeze-server`. |
| `BREEZE_BUILD_SHARED` | `ON` | Build the shared C library. |
| `BREEZE_MP3` | `ON` | Let the server encode MP3 for OpenAI style requests, using the `third_party/shine` submodule. Off, or with the submodule missing, those requests get WAV instead. |

### Vulkan or CUDA

Vulkan stays the default. It is a little ahead at Q4_K and Q8_0, level at F16, and
it needs no toolkit. Measured on an RTX 3060, per frame of audio (80 ms):

| Model | `cfg_scale` | Backend | Backbone | Depth | Vocoder | Total per frame | Real time factor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q4_K | 1 | Vulkan | 6.50 ms | 22.16 ms | 6.21 ms | 34.87 ms | 2.29x |
| Q4_K | 1 | CUDA | 9.37 ms | 23.28 ms | 6.85 ms | 39.50 ms | 2.03x |
| Q4_K | 3 | Vulkan | 9.72 ms | 30.69 ms | 5.79 ms | 46.20 ms | 1.73x |
| Q4_K | 3 | CUDA | 19.31 ms | 28.47 ms | 7.31 ms | 55.09 ms | 1.45x |
| F16 | 1 | Vulkan | 12.66 ms | 40.62 ms | 6.12 ms | 59.40 ms | 1.35x |
| F16 | 1 | CUDA | 12.70 ms | 42.36 ms | 5.75 ms | 60.81 ms | 1.32x |

CUDA used to be far behind, about 0.6x, for two reasons, and both are fixed:

- **CUDA graphs were never compiled in.** ggml leaves them off unless it is built
  by llama.cpp, so `BREEZE_CUDA` now turns them on. The depth decoder replays one
  fixed graph per frame, which took it from 55 ms to 23 ms. Set
  `GGML_CUDA_DISABLE_GRAPHS=1` to compare.
- **The CUDA `conv_transpose_1d` kernel is a per sample loop.** It cost about 1.6 s
  on a 100 frame decode. The vocoder now upsamples with a matmul and overlap add,
  which took it from 62 ms to 7 ms per frame on CUDA, and from 12 ms to 6 ms on
  Vulkan.

The backbone is what still separates them, because it builds a new graph every
frame and CUDA cannot replay that. Only the depth decoder gets the full benefit
of graphs today.

Tensor core speed, including an unlocked card, only reaches the large batched
matmuls: backbone prefill and the vocoder. Depth and backbone decode multiply one
or two rows at a time on a path that skips the tensor cores, and they are limited
by reading the weights from memory. That is why a smaller depth type is the
fastest option on either backend: F16 spends about 42 ms per frame in depth, and
F16 with a Q4_K depth decoder spends about 25 ms.

## Outputs

| Target | Description |
| --- | --- |
| `breeze-cli` | One shot synthesis to WAV |
| `breeze-server` | HTTP server with optional web UI |
| `breeze-quantize` | Converts an F16 GGUF to Q8_0, Q6_K, Q4_K, Q3_K or Q2_K |
| `libbreeze` | Shared C library, see [c-api.md](c-api.md) |
| `libbreeze_core` | Static library if you would rather link directly |

The web UI is compiled into `breeze-server` at build time by
`cmake/embed_webui.cmake`, which turns the files in [webui](../webui) into a
generated header. Edit the sources and rebuild to pick up changes.

## Backend selection at runtime

Both apps ask ggml for any GPU backend it was built with and fall back to CPU if
there is none. Pass `--cpu` to skip the GPU entirely. The startup line names the
backend that was picked:

```
backend: Vulkan0, sample rate: 24000
```

## mingw specifics

mingw builds link the compiler runtime statically, so the binaries depend only
on system DLLs plus `vulkan-1.dll`. You can copy them to a machine with no
mingw installed and they will run.

OpenMP is disabled for mingw because there is no static `libgomp`, and linking
it dynamically would leak a `libgomp-1.dll` dependency into the shared library.
The ggml CPU backend falls back to its own thread pool, so it still runs
multithreaded.

## Troubleshooting

**`third_party/ggml is missing`** means the submodule was not initialised. Run
`git submodule update --init --recursive`.

**`third_party/shine is missing so mp3 requests will get wav`** is the same thing
for the MP3 encoder, which was added after ggml. A clone from before then needs
`git submodule update --init --recursive` once to pick it up.

**Vulkan shader generation crashes with `0xC0000139`** on Windows when another
program has put an incompatible `libstdc++-6.dll` earlier on `PATH`. Tesseract
OCR is a common culprit. Put your mingw `bin` directory first:

```powershell
$env:PATH = "C:\path\to\mingw64\bin;" + $env:PATH
```

**No Vulkan device found** usually means an outdated GPU driver, or running
inside a container or remote session without GPU passthrough. It is not fatal;
the model loads on CPU instead, just slowly.

**Out of video memory** during generation: use a smaller quantization. Q4_K
needs roughly 3 GB of VRAM against about 7 GB for F16. See
[models.md](models.md).
