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

### CUDA is slower here, and it is worth knowing why

CUDA looks like the obvious win, because the depth decoder is bound by dispatch
overhead and CUDA graphs exist to fix exactly that. Measured on an RTX 3060 it is
not, by a wide margin:

| Backend | Backbone | Depth | Vocoder | Total per frame | Real time factor |
| --- | --- | --- | --- | --- | --- |
| Vulkan | 7.69 ms | 34.03 ms | 10.44 ms | 52.16 ms | 1.53x |
| CUDA | 10.32 ms | 69.64 ms | 57.45 ms | 137.41 ms | 0.58x |
| CUDA, `GGML_CUDA_DISABLE_GRAPHS=1` | 11.45 ms | 77.23 ms | 60.27 ms | 148.95 ms | 0.54x |

The split shows what is happening. Timing one large graph against the small per
step ones:

| Stage | Vulkan | CUDA |
| --- | --- | --- |
| Backbone prefill, one large graph | 118.0 ms | **31.7 ms** |
| First vocoder, small graph | **58.1 ms** | 141.3 ms |

CUDA is nearly four times faster on the big graph and roughly half the speed on
the small ones. Generation is thousands of small graphs, so Vulkan wins overall.

The cause is on our side, not CUDA's. Every step builds a brand new `ggml_cgraph`
in a fresh context, so ggml cannot recognise it as the same graph and CUDA graph
capture never pays off; disabling capture entirely only makes it slightly worse,
which confirms it is doing very little. On top of that `ggml_backend_tensor_set`
is a synchronous copy on CUDA, and we do several per graph.

So CUDA is left off by default. If the runtime is ever reworked to reuse graphs
across steps, this is the first thing worth re measuring, because the prefill
number says the headroom is real.

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
