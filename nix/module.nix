# NixOS module: breeze-server as a systemd service.
#
# Usage:
#   imports = [ github:HoppouAI/Breeze-TTS-2.cpp ];
#   services.breeze-tts = {
#     enable = true;
#     modelPath = "breeze-tts-2-q8_0.gguf";   # bare filename, resolved inside modelsDir
#   };
#
# The model folder does not need audio_tokenizer/: the vocoder is baked into
# the GGUF by the converter. The server is stateless for generations — what it
# persists are saved .breeze voices, which live under the state dir so they
# survive upgrades and reboots (loaded automatically at startup: pass them to
# speak/listVoices clients with no extra setup).

self:
{ lib, config, pkgs, ... }:
let
	cfg = config.services.breeze-tts;

	# the null options each derive from the state dir unless set explicitly
	voicesDir = if cfg.voicesDir != null then cfg.voicesDir else "${cfg.stateDir}/voices";
	modelsDir = if cfg.modelsDir != null then cfg.modelsDir else "${cfg.stateDir}/models";
	modelPath =
		if lib.hasInfix "/" cfg.modelPath
		then cfg.modelPath
		else "${modelsDir}/${cfg.modelPath}";

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
			description = "State directory for the service: saved voices (<stateDir>/voices) and, implicitly, the served models (<stateDir>/models). Under /var/lib both are created and kept by systemd; point them at a mounted directory with stateDir or the per-item options.";
		};

		package = lib.mkOption {
			type = lib.types.package;
			default = self.packages.${pkgs.system}.breeze-tts;
			defaultText = "packages.<system>.breeze-tts";
			description = "The breeze-tts-2.cpp derivation. Use the CPU only package with the cpuOnly option or pick packages.breeze-tts-cpu here.";
		};

		modelPath = lib.mkOption {
			type = lib.types.str;
			description = "GGUF checkpoint to serve: a bare filename resolved inside modelsDir, or any path (absolute, store path, Hugging Face cache path).";
		};

		modelsDir = lib.mkOption {
			type = lib.types.nullOr lib.types.str;
			default = null;
			description = "Folder of GGUF checkpoints; a bare modelPath filename resolves within it. null means <stateDir>/models, so the models folder lives inside the state dir unless pointed elsewhere (a Hugging Face local checkout, a shared models directory).";
		};

		voicesDir = lib.mkOption {
			type = lib.types.nullOr lib.types.str;
			default = null;
			description = "Folder of saved .breeze voices, passed to the server as --voices-dir. null means <stateDir>/voices; override only to move voices off the state dir.";
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
		environment.systemPackages = [ cfg.package ];

		systemd.services.breeze-tts = {
			description = "Breeze-TTS-2.cpp speech generation server";
			wantedBy = [ "multi-user.target" ];
			after = [ "network.target" ];

			serviceConfig = {
				Type = "simple";
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
				];
				# Voice files are user content; keep them on restarts
				Restart = "on-failure";
				RestartSec = 5;
			};

			# Hardware-accelerated instances need the Vulkan ICDs nixos exposes at
			# /run/opengl-driver once hardware graphics (hardware.graphics.* via the
			# opengl option in older releases) is enabled.
			environment.VK_ICD_FILENAMES =
				"/run/opengl-driver/share/vulkan/icd.d:/run/opengl-driver-32/share/vulkan/icd.d";
		};
	};
}