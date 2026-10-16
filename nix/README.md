# Nix

The flake ships three things: a dev shell for building the tree, ready-built packages, and an
importable NixOS module that runs `breeze-server` as a systemd service.

## Packages

| Package | Backend | Output |
| --- | --- | --- |
| `breeze-tts` (default) | Vulkan | `breeze-cli`, `breeze-convert`, `breeze-quantize`, `breeze-server` |
| `breeze-tts-cpu` | CPU | the same four binaries, no Vulkan bits |

Both link ggml statically (`-DBUILD_SHARED_LIBS=OFF`, OpenMP off) against a pinned ggml revision,
so the build-tree rpaths and shared ggml libraries never leave the sandbox. The Vulkan variant
wraps its binaries with the loader on `LD_LIBRARY_PATH` at runtime.

Build a package directly from the flake, or use `breeze-build` inside the dev shell:

```
nix build .#breeze-tts
nix build .#breeze-tts-cpu
```

## Dev shell

```
nix develop          # Vulkan-capable: CMake, Ninja, Vulkan SDK, Python conversion deps
nix develop .#cpu    # same, without the Vulkan SDK

breeze-build         # configure + build (Vulkan); --cpu for CPU only
breeze-cli           # run build/breeze-cli
breeze-server        # run build/breeze-server
```

## NixOS module

`nixosModules.breeze-tts` (aliased as `nixosModules.default`) wires `breeze-server` into systemd.
With a catalog `model` set, the unit fetches the checkpoint from the Hub on the first start and
re-verifies it against the pinned sha256 on every start after that — an already-correct file turns
the download step into a no-op, a mismatched one is discarded and re-fetched, and downloads resume
through a `.part` file with an atomic rename when interrupted.

```nix
{
  inputs.breeze-tts.url = "github:HoppouAI/Breeze-TTS-2.cpp/feat/nix";
  # ...
  modules = [
    breeze-tts.nixosModules.default
    {
      hardware.graphics.enable = true;  # Vulkan driver bits at /run/opengl-driver
      services.breeze-tts = {
        enable = true;
        webui = true;
        model = "q8_0";   # or: customModel = "/path/to/your.gguf";
      };
    }
  ];
}
```

The unit is `breeze-tts.service` (`journalctl -u breeze-tts` for logs), restarted on failure, and
nothing under the state dir is deleted by the service.

### Options

| Option | Type | Default | Notes |
| --- | --- | --- | --- |
| `enable` | bool | `false` | Turn the service on. |
| `stateDir` | str | `"/var/lib/breeze-tts"` | The single state knob: saved voices in `<stateDir>/voices`, the served model (catalog downloads and bare `customModel` names) in `<stateDir>/models`. Under `/var/lib` systemd creates and owns both children; anywhere else the operator owns the directory. |
| `package` | package | `packages.<system>.breeze-tts` | The derivation in use; point at `packages.breeze-tts-cpu` for a CPU-only build. Its binaries also land in `environment.systemPackages`. |
| `model` | enum or null | `null` | Hub catalog pick: `f16` (reference), `q8_0` (recommended), `q6_k`, `q4_k`, `q8_0-dd2`, `q8_0-dd4`, `q4_k-dd2`. Downloaded into `<stateDir>/models` under its Hub name. Ignored when `customModel` is set. |
| `customModel` | str or null | `null` | A GGUF outside the catalog: a bare filename resolved inside `<stateDir>/models`, or any path (absolute, store path, HF checkout). Wins over `model` — nothing is downloaded or verified, so the file's correctness is yours. No `audio_tokenizer/` folder needed; the converter bakes the vocoder into the GGUF. |
| `host` | str | `"127.0.0.1"` | Bind interface. The server authenticates nothing, so keep it on loopback unless something in front handles access. |
| `port` | port | `1923` | HTTP port. |
| `wsPort` | port or null | `null` | WebSocket listener; `null` means HTTP port + 1. |
| `cpuOnly` | bool | `false` | Adds `--cpu` at runtime, so a Vulkan-built package can still serve CPU-only sessions. For a build without the Vulkan bits at all, use `packages.breeze-tts-cpu` as the `package`. |
| `webui` | bool | `false` | Serve the bundled web UI at `/`. |
| `extraArgs` | list of str | `[ ]` | Extra server flags, e.g. `[ "--chunk-max" "40" ]` or `[ "--verbose" ]`. |

### Operational notes

- **GPU machines** need `hardware.graphics.enable = true` so the Vulkan ICDs exist at
  `/run/opengl-driver`; the unit sets `VK_ICD_FILENAMES` for both the 64 and 32 bit locations.
- **Saved voices** (`.breeze` files in `<stateDir>/voices`) survive upgrades and reboots and are
  loaded automatically at startup — no extra setup for speak/listVoices clients.
- **Model pins** live in `nix/module.nix` (`modelCatalog`). To re-pin after a Hub update, compare
  hashes against https://huggingface.co/api/models/HoppouAI/Breeze-TTS-2.cpp/tree/main.
- **Neither port authenticates or rate-limits**; defaults bind to loopback only, and widening
  `host` puts trust in whatever fronts it.