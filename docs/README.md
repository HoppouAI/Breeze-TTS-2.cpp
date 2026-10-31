# Documentation

| Document | Contents |
| --- | --- |
| [build.md](build.md) | Toolchain requirements, CMake options, backend selection, troubleshooting |
| [models.md](models.md) | Downloading weights, GGUF conversion, quantization, tensor layout |
| [cli.md](cli.md) | `breeze-cli` flags and usage recipes |
| [server.md](server.md) | HTTP endpoints, request and response formats, streaming, WebUI |
| [websocket.md](websocket.md) | Streaming sessions, incremental text, live direction changes, interruption |
| [voices.md](voices.md) | Caching and saving reference voices, the `.breeze` format |
| [voice-conversion.md](voice-conversion.md) | Respeaking a recording in another voice, `breeze-convert` |
| [performance.md](performance.md) | What's new, picking a model, running several streams, tuning and measuring |
| [c-api.md](c-api.md) | The `breeze.h` C API, lifecycle, callbacks, thread safety |
| [ctypes.md](ctypes.md) | Binding the shared library from Python with `ctypes` |
| [architecture.md](architecture.md) | The four model stages and how the C++ mirrors the reference |

## Quick orientation

Breeze-TTS-2 generates speech in four stages:

1. A T5Gemma2 **text encoder** turns the prompt into a conditioning sequence.
2. A Qwen3 **backbone** autoregressively predicts codebook 0, one frame at a
   time, at 12.5 frames per second.
3. A small **depth decoder** fills in codebooks 1 to 15 for each frame.
4. A **vocoder** turns the 16 codebooks back into a 24 kHz waveform.

Each frame is 1920 samples, so one frame of codes becomes 80 ms of audio.

## Three generation modes

| Mode | Reference audio | Instruction | Template |
| --- | --- | --- | --- |
| Voice design | no | describes the voice | `tts_instruction` |
| Voice clone | yes | `Speak clearly and naturally.` | `ref_edit_tata` |
| Voice direction | yes | describes the delivery | `ref_edit_tata` |

The mode is picked automatically from whether reference audio was supplied,
so there is no mode flag anywhere in the API.

There is also **voice conversion**, which is a different operation rather than a
fourth mode. It takes a recording you already have instead of text, keeps its
words and timing, and replaces the speaker. See
[voice-conversion.md](voice-conversion.md).
