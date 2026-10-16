{
  description = "C++ / GGUF reimplementation of Breeze-TTS-2 running on ggml with a Vulkan backend";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    {
      # Importable module: runs breeze-server as a systemd service.
      # Set services.breeze-tts.model (a Hub catalog pick, downloaded to
      # <stateDir>/models) or services.breeze-tts.customModel to hand it a
      # file you already have.
      nixosModules.breeze-tts = import ./nix/module.nix self;
      nixosModules.default = self.nixosModules.breeze-tts;
    }
    // flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;

        # Build-time dependencies shared by all configurations
        buildDeps = with pkgs; [
          cmake
          ninja
          gcc
          pkg-config
        ];

        # Vulkan SDK dependencies (glslc from shaderc is needed for shader compilation)
        vulkanDeps = with pkgs; [
          vulkan-headers
          vulkan-loader
          vulkan-tools
          shaderc
          vulkan-validation-layers
        ];

        # Python toolchain for the weight conversion scripts
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          numpy
          gguf
        ]);

        # The ggml revision the project's submodule pins (checked at
        # v0.22.0-1-g36da5713). Flakes cannot see submodule content, so the
        # package fetches this exact commit itself.
        ggmlSrc = pkgs.fetchFromGitHub {
          owner = "ggml-org";
          repo = "ggml";
          rev = "36da57138425487184aa1da2eee2cde155909c6f";
          hash = "sha256-D9D/attiTe9MFT3sHbwTzPni3LxEVVGYsgKCb7V7qtE=";
        };

        # The main derivation. Builds against the pinned ggml.
        mkBreezeTts = vulkan:
          pkgs.stdenv.mkDerivation {
            pname = "breeze-tts-2" + lib.optionalString (!vulkan) "-cpu";
            version = "0.1.0";
            src = self;

            preConfigure = ''
              # the git-archive/flake source carries the submodule as an
              # empty directory; ln -sfn cannot replace a real directory (it
              # nests the link inside it), so clear the stub first
              rm -rf third_party/ggml
              ln -sfn ${ggmlSrc} third_party/ggml
              # a dirty worktree path input snapshots ignored directories too
              # (build/, models/, result); a stale CMakeCache.txt from the
              # tree's previous location would break configure
              rm -rf build
            '';

            # the project ships no install() rules for its own targets; they
            # are built into build/ upstream, so copy them into place here
            postInstall = ''
              # the phases run from inside the cmake build dir
              mkdir -p $out/bin
              for exe in breeze-cli breeze-server breeze-convert breeze-quantize; do
                if [ -f "$exe" ]; then
                  install -m755 "$exe" "$out/bin/$exe"
                fi
              done
            '';

            nativeBuildInputs = buildDeps
              ++ lib.optionals vulkan [ pkgs.shaderc pkgs.makeWrapper ];
            buildInputs = lib.optionals vulkan
              [ pkgs.vulkan-headers pkgs.vulkan-loader ];

            cmakeFlags = [
              "-DCMAKE_BUILD_TYPE=Release"
              # static ggml: keeps the binaries free of build-tree rpaths and
              # the store free of ggml shared libs
              "-DBUILD_SHARED_LIBS=OFF"
              "-DGGML_OPENMP=OFF"
            ] ++ lib.optionals (!vulkan) [ "-DBREEZE_VULKAN=OFF" ];

            # The Vulkan loader is dlopened at runtime, not linked
            postFixup = lib.optionalString vulkan ''
              for exe in breeze-cli breeze-server breeze-convert breeze-quantize; do
                if [ -f "$out/bin/$exe" ]; then
                  wrapProgram "$out/bin/$exe" \
                    --prefix LD_LIBRARY_PATH : "${pkgs.vulkan-loader}/lib"
                fi
              done
            '';

            meta = {
              license = lib.licenses.asl20;
              platforms = lib.platforms.linux;
              mainProgram = "breeze-server";
            };
          };

        breeze-tts = mkBreezeTts true;
        breeze-tts-cpu = mkBreezeTts false;

        # Convenience build command, only available inside nix develop
        breeze-build = pkgs.writeShellScriptBin "breeze-build" ''
          set -euo pipefail

          # Ensure the ggml and shine submodules are present
          if [ ! -f "third_party/ggml/CMakeLists.txt" ] || [ ! -f "third_party/shine/src/lib/layer3.c" ]; then
            echo "Fetching submodules..."
            git submodule update --init --recursive
          fi

          # Pass --cpu to disable Vulkan, extra args after that go to cmake configure
          cmakeArgs=(-DCMAKE_BUILD_TYPE=Release)
          if [ "''${1:-}" = "--cpu" ]; then
            cmakeArgs+=("-DBREEZE_VULKAN=OFF")
            shift
          fi

          echo "Configuring..."
          cmake -B build -G Ninja "''${cmakeArgs[@]}" "$@"

          echo "Building..."
          cmake --build build -j

          echo "\nOutputs in build/"
        '';

        # Run the CLI from the build output
        breeze-cli = pkgs.writeShellScriptBin "breeze-cli" ''
          exec "$(git rev-parse --show-toplevel)/build/breeze-cli" "$@"
        '';

        # Run the server from the build output
        breeze-server = pkgs.writeShellScriptBin "breeze-server" ''
          exec "$(git rev-parse --show-toplevel)/build/breeze-server" "$@"
        '';

        # Printed when entering the dev shell
        welcomeMsg = ''
          \033[1mBreeze-TTS-2.cpp\033[0m

          \033[1mTo get started, run the following:\033[0m

          breeze-build          # configure + build (Vulkan)
          breeze-build --cpu    # configure + build (CPU only)
          breeze-cli            # run the CLI (after building)
          breeze-server         # run the server (after building)
        '';
      in
      {
        packages = {
          inherit breeze-tts breeze-tts-cpu;
          default = breeze-tts;
        };

        devShells = {
          # Full development shell with Vulkan support (default)
          default = pkgs.mkShell rec {
            packages = buildDeps ++ vulkanDeps ++ [ pythonEnv breeze-build breeze-cli breeze-server ];

            # Make Vulkan layers and loaders discoverable
            VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
            LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [
              vulkan-loader
            ];

            shellHook = ''echo -e "${welcomeMsg}"'';
          };

          # CPU-only development shell (no Vulkan SDK)
          cpu = pkgs.mkShell {
            packages = buildDeps ++ [ pythonEnv breeze-build breeze-cli breeze-server ];

            shellHook = ''echo -e "${welcomeMsg}"'';
          };
        };
      });
}