{
  description = "C++ / GGUF reimplementation of Breeze-TTS-2 running on ggml with a Vulkan backend";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

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

        # Convenience build command, only available inside nix develop
        breeze-build = pkgs.writeShellScriptBin "breeze-build" ''
          set -euo pipefail

          # Ensure the ggml submodule is present
          if [ ! -f "third_party/ggml/CMakeLists.txt" ]; then
            echo "Fetching ggml submodule..."
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