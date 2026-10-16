# NixOS module: breeze-server as a systemd service.
#
# Usage:
#   imports = [ github:HoppouAI/Breeze-TTS-2.cpp ];
#   services.breeze-tts = {
#     enable = true;
#     model = "q8_0";   # picked from the Hub catalog, downloaded into <stateDir>/models
#   };
#
# customModel hands the server a file you already have — it wins over model
# and skips downloading. The model folder does not need audio_tokenizer/: the
# vocoder is baked into the GGUF by the converter. The server is stateless
# for generations — what it persists are saved .breeze voices, which live
# under the state dir so they survive upgrades and reboots (loaded
# automatically at startup: pass them to speak/listVoices clients with no
# extra setup).

self:
{ lib, config, pkgs, ... }:
let
	cfg = config.services.breeze-tts;

	# The shipped checkpoints, from https://huggingface.co/HoppouAI/Breeze-TTS-2.cpp.
	# LFS oids from the repo's tree API, so a download that does not match the
	# pin cannot be started by the server. To re-pin after a hub update:
	#   https://huggingface.co/api/models/HoppouAI/Breeze-TTS-2.cpp/tree/main
	modelCatalog = {
		"q8_0" = { file = "breeze-tts-2-q8_0.gguf"; sha256 = "a02bcc4b69b0601032727f8040c4942149b1b73aa0f69022fe5aaa6a8f0ef879"; };
		"q8_0-dd2" = { file = "breeze-tts-2-q8_0-dd2.gguf"; sha256 = "c43155fdd783e459f84c78b88f6cd91e833cf8b54649f289de7825139ee2d1c9"; };
		"q8_0-dd4" = { file = "breeze-tts-2-q8_0-dd4.gguf"; sha256 = "5248f348388cb7fd904a8bde3e65383a0fbc562d2fe48d7c375cf17372d3751b"; };
		"q6_k" = { file = "breeze-tts-2-q6_k.gguf"; sha256 = "0eb51de6cbccd54b1c408198797254ac46b3cbd55ca1174c630eaac6d01064d3"; };
		"q4_k" = { file = "breeze-tts-2-q4_k.gguf"; sha256 = "483418fbbb438f5f1c08dbe2b017e42db0f1d126765cd3f772d8622a30a4915c"; };
		"q4_k-dd2" = { file = "breeze-tts-2-q4_k-dd2.gguf"; sha256 = "6d675f8c1ce43bdd368489e9c9f52a725e9ede23e3bbf338ec5c9c31d8169cf0"; };
		"f16" = { file = "breeze-tts-2-f16.gguf"; sha256 = "db740cd464d75f2635a891586f083c5b466fd63542bebd957fe7890324b04c35"; };
	};

	# the whole on-disk layout derives from stateDir — voices and models are
	# its children, not independently configurable
	modelsDir = "${cfg.stateDir}/models";
	voicesDir = "${cfg.stateDir}/voices";

	custom = cfg.customModel != null;
	pickedModel = if custom || cfg.model == null then null else modelCatalog.${cfg.model};
	modelPath =
		if custom then
			(if lib.hasInfix "/" cfg.customModel
				then cfg.customModel
				else "${modelsDir}/${cfg.customModel}")
		else if pickedModel != null then "${modelsDir}/${pickedModel.file}"
		else
			throw "services.breeze-tts: set model to one of ${
				lib.concatStringsSep ", " (builtins.attrNames modelCatalog)
			}, or point customModel at a GGUF";

	# catalog picks are fetched into modelsDir by the unit itself: an existing
	# file that passes the pin makes this a no-op, so on every start the pin is
	# simply re-verified; download or resume goes into a .part file with an
	# atomic rename, and a mismatched file is discarded for re-download
	fetchModel = pkgs.writeShellScript "breeze-fetch-model" ''
		set -euo pipefail
		target="${modelsDir}/${pickedModel.file}"
		hash="${pickedModel.sha256}"
		url="https://huggingface.co/HoppouAI/Breeze-TTS-2.cpp/resolve/main/${pickedModel.file}"
		verify() { echo "$hash  $1" | sha256sum --status -c -; }

		if [ -f "$target" ]; then
			if verify "$target"; then exit 0; fi
			printf 'model %s failed its sha256 check, discarding for re-download\n' "$target"
			rm -f "$target"
		fi

		part="$target.part"
		if [ -f "$part" ]; then then_resume="-C -"; else then_resume=""; fi
		printf 'fetching %s into %s\n' "${pickedModel.file}" "${modelsDir}"
		curl -fL --retry 5 --retry-delay 3 --connect-timeout 15 $then_resume -o "$part" "$url"
		verify "$part"
		mv -f "$part" "$target"
	'';

	serverArgs = [
		"--host"
		cfg.host
		"--port"
		(toString cfg.port)
		"--voices-dir"
		voicesDir
	] ++ lib.optionals cfg.webui [ "--webui" ] ++ cfg.extraArgs;

	# systemd manages StateDirectory= entries only under /var/lib; anywhere
	# else the operator owns the directory
	stateSubdir = lib.removePrefix "/var/lib/" cfg.stateDir;
	stateManaged = stateSubdir != cfg.stateDir;
in
{
	options.services.breeze-tts = {
		enable = lib.mkEnableOption "the Breeze-TTS-2.cpp streaming speech server";

		stateDir = lib.mkOption {
			type = lib.types.str;
			default = "/var/lib/breeze-tts";
			description = "The single state-directory knob: saved voices live in <stateDir>/voices and the served model (catalog downloads, bare customModel names) in <stateDir>/models. Under /var/lib both children are created and kept by systemd; anywhere else the operator owns the directory.";
		};

		package = lib.mkOption {
			type = lib.types.package;
			default = self.packages.${pkgs.system}.breeze-tts;
			defaultText = "packages.<system>.breeze-tts";
			description = "The breeze-tts-2.cpp derivation. Use the CPU only package with the cpuOnly option or pick packages.breeze-tts-cpu here.";
		};

		model = lib.mkOption {
			type = lib.types.nullOr (lib.types.enum (builtins.attrNames modelCatalog));
			default = null;
			description = ''
				Which shipped checkpoint to serve: f16 (reference quality) or a quantised variant (q8_0 recommended, the -dd ones quantise the depth decoder too and are leaner in VRAM but drift past ~45 s of narration). The file is downloaded from the Hub into <stateDir>/models on the first start and named for its Hub file; afterwards, as long as the expected file is there and passes its sha256 pin, the download step is a no-op. Ignored when customModel is set.
			'';
		};

		customModel = lib.mkOption {
			type = lib.types.nullOr lib.types.str;
			default = null;
			description = "Escape hatch for a GGUF outside the catalog: a bare filename resolved inside <stateDir>/models, or any path (absolute, store path, Hugging Face checkout). When set it wins over model — nothing is downloaded or verified, so the operator owns the file's correctness.";
		};

		host = lib.mkOption {
			type = lib.types.str;
			default = "127.0.0.1";
			description = "Interface to bind. The server does not authenticate; anything beyond 127.0.0.1 needs something in front handling access control.";
		};

		port = lib.mkOption {
			type = lib.types.port;
			default = 1923;
			description = "HTTP port. The WebSocket streaming listener lands on this port plus one unless wsPort is set.";
		};

		wsPort = lib.mkOption {
			type = lib.types.nullOr lib.types.port;
			default = null;
			description = "Fixed WebSocket port. null lets the server pick the HTTP port plus one; set it with a fixed companion module on other hosts.";
		};

		cpuOnly = lib.mkOption {
			type = lib.types.bool;
			default = false;
			description = "Force the CPU backend instead of Vulkan for this instance.";
		};

		webui = lib.mkOption {
			type = lib.types.bool;
			default = false;
			description = "Serve the bundled web UI at / (opt-in: the flag is --webui).";
		};

		extraArgs = lib.mkOption {
			type = lib.types.listOf lib.types.str;
			default = [];
			description = "Extra command line flags, e.g. [ \"--chunk-max\" \"40\" ] or [ \"--verbose\" ].";
		};
	};

	config = lib.mkIf cfg.enable {
		warnings = lib.mkIf (custom && cfg.model != null) [
			"services.breeze-tts: customModel is set, so model = \"${cfg.model}\" is ignored and no download will run"
		];

		environment.systemPackages = [ cfg.package ];

		systemd.services.breeze-tts = {
			description = "Breeze-TTS-2.cpp speech generation server";
			wantedBy = [ "multi-user.target" ];
			after = [ "network.target" ];

			serviceConfig = {
				Type = "simple";
				ExecStartPre = lib.mkIf (pickedModel != null) [
					fetchModel
				];
				ExecStart = lib.concatStringsSep " " (
					[ "${lib.getExe cfg.package}" modelPath ]
					++ lib.optionals cfg.cpuOnly [ "--cpu" ]
					++ lib.optionals (cfg.wsPort != null) [
						"--ws-port"
						(toString cfg.wsPort)
					]
					++ serverArgs
				);
				StateDirectory = lib.optionals stateManaged [
					stateSubdir
					"${stateSubdir}/voices"
					"${stateSubdir}/models"
				];
				# Voice files are user content; keep them on restarts
				Restart = "on-failure";
				RestartSec = 5;
			};

			path = [ pkgs.curl pkgs.coreutils ];

			# Hardware-accelerated instances need the Vulkan ICDs nixos exposes at
			# /run/opengl-driver once hardware graphics (hardware.graphics.* via the
			# opengl option in older releases) is enabled.
			environment.VK_ICD_FILENAMES =
				"/run/opengl-driver/share/vulkan/icd.d:/run/opengl-driver-32/share/vulkan/icd.d";
		};
	};
}