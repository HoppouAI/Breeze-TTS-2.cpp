# Binding from Python with ctypes

There is no Python package. The shared library exposes a plain C ABI, so
`ctypes` is enough. This page is a working reference you can paste into a
project.

Read [c-api.md](c-api.md) first for what each function actually does.

## Loading the library

Since Python 3.8, Windows ignores `PATH` when resolving DLL dependencies. Use
`os.add_dll_directory` for the folder holding the library.

```python
import ctypes
import os
import platform
from pathlib import Path


def load(lib_dir):
    lib_dir = Path(lib_dir).resolve()
    system = platform.system()
    if system == "Windows":
        names = ["breeze.dll", "libbreeze.dll"]
    elif system == "Darwin":
        names = ["libbreeze.dylib"]
    else:
        names = ["libbreeze.so"]

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(lib_dir))

    for name in names:
        path = lib_dir / name
        if path.exists():
            return ctypes.CDLL(str(path))
    raise FileNotFoundError(f"no breeze library in {lib_dir}")
```

mingw builds link the compiler runtime statically, so the only external
dependency is `vulkan-1.dll` from the GPU driver.

## Declarations

```python
class BreezeRequest(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("instruction", ctypes.c_char_p),
        ("ref_text", ctypes.c_char_p),
        ("ref_audio", ctypes.POINTER(ctypes.c_float)),
        ("ref_audio_len", ctypes.c_int),
        ("cfg_scale", ctypes.c_float),
        ("seed", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int),
        ("split_chars", ctypes.c_int),
    ]


AUDIO_CB = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p
)


def bind(lib):
    lib.breeze_init.restype = ctypes.c_void_p
    lib.breeze_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.breeze_free.argtypes = [ctypes.c_void_p]
    lib.breeze_sample_rate.restype = ctypes.c_int
    lib.breeze_sample_rate.argtypes = [ctypes.c_void_p]
    lib.breeze_generate.restype = ctypes.c_int
    lib.breeze_generate.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(BreezeRequest), AUDIO_CB, ctypes.c_void_p
    ]
    lib.breeze_generate_wav.restype = ctypes.c_int
    lib.breeze_generate_wav.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(BreezeRequest), ctypes.c_char_p
    ]
    lib.breeze_last_error.restype = ctypes.c_char_p
    return lib
```

`breeze_init` must be declared `c_void_p` rather than left at the default
`c_int`, otherwise the 64 bit handle gets truncated and the process crashes on
the first call that uses it.

## Minimal wrapper

```python
import numpy as np


class Breeze:
    def __init__(self, model, lib_dir, use_gpu=True):
        self.lib = bind(load(lib_dir))
        self.ctx = self.lib.breeze_init(str(model).encode(), int(use_gpu))
        if not self.ctx:
            raise RuntimeError(self.lib.breeze_last_error().decode(errors="replace"))
        self.sample_rate = self.lib.breeze_sample_rate(self.ctx)

    def close(self):
        if self.ctx:
            self.lib.breeze_free(self.ctx)
            self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _request(self, text, instruction, ref_audio, ref_text, cfg_scale, seed, max_new, split_chars):
        req = BreezeRequest()
        req.text = text.encode()
        req.instruction = (instruction or "Speak clearly and naturally.").encode()
        req.ref_text = (ref_text or "").encode()
        req.cfg_scale = float(cfg_scale)
        req.seed = int(seed)
        req.max_new_tokens = int(max_new)
        req.split_chars = int(split_chars)
        keep = None
        if ref_audio is not None:
            keep = np.ascontiguousarray(ref_audio, dtype=np.float32)
            req.ref_audio = keep.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            req.ref_audio_len = len(keep)
        return req, keep

    def generate(self, text, instruction=None, ref_audio=None, ref_text=None,
                 cfg_scale=1.0, seed=42, max_new=0, split_chars=0, on_chunk=None):
        req, keep = self._request(text, instruction, ref_audio, ref_text,
                                  cfg_scale, seed, max_new, split_chars)
        out = []

        def cb(samples, n, _user):
            block = np.ctypeslib.as_array(samples, shape=(n,)).copy()
            out.append(block)
            if on_chunk is not None:
                on_chunk(block)
            return 0

        c_cb = AUDIO_CB(cb)
        rc = self.lib.breeze_generate(self.ctx, ctypes.byref(req), c_cb, None)
        del keep
        if rc != 0:
            raise RuntimeError(self.lib.breeze_last_error().decode(errors="replace"))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
```

Two things that bite:

- Keep a reference to the `AUDIO_CB` object for the whole call. If Python
  collects it while C still holds the pointer, the process dies.
- Keep a reference to the reference audio array too. `ctypes` stores a raw
  pointer and does not own the buffer.

## Usage

```python
import soundfile as sf

with Breeze("breeze-tts-2-q4_k.gguf", "build") as tts:
    audio = tts.generate(
        "The build finished without a single warning.",
        instruction="A calm narrator with a warm tone.",
        seed=42,
    )
    sf.write("out.wav", audio, tts.sample_rate)
```

Streaming to a sound device as chunks arrive:

```python
import sounddevice as sd

with Breeze("breeze-tts-2-q4_k.gguf", "build") as tts:
    stream = sd.OutputStream(samplerate=tts.sample_rate, channels=1, dtype="float32")
    stream.start()
    tts.generate("Streaming straight to the speakers.", on_chunk=stream.write)
    stream.stop()
```

Voice cloning, resampling the reference to the model rate first:

```python
import librosa

ref, _ = librosa.load("reference.wav", sr=tts.sample_rate, mono=True)
audio = tts.generate(
    "It is good to hear your voice again.",
    ref_audio=ref,
    ref_text="This is the exact transcript of the reference audio.",
)
```

## Stopping early

Return non zero from the callback to stop generation. The current frame
finishes and `breeze_generate` returns normally.

```python
def cb(samples, n, _user):
    out.append(np.ctypeslib.as_array(samples, shape=(n,)).copy())
    return 1 if sum(len(b) for b in out) >= limit else 0
```

## Threading

A context is not reentrant, so serialise calls with a lock. `breeze_generate`
blocks and holds the GIL only while your callback runs, so other Python threads
keep making progress during the compute.

If you want real parallelism, run one `breeze-server` with `--parallel`, which
shares the weights and one GPU batch between streams, see
[performance.md](performance.md#running-several-streams). Each extra context here
loads its own copy of the weights.

## Other languages

The same ABI works anywhere with an FFI. The pieces to translate are: an opaque
pointer handle, the eight field request struct with its natural C alignment,
and a callback taking `(const float *, int, void *)` and returning `int`.
