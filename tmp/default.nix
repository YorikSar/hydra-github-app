{
  lib,
  stdenv,
  fetchFromGitHub,
  nix-update-script,
  fetchPnpmDeps,
  pnpm,
  pnpmConfigHook,
  pnpmBuildHook,
  nodejs-slim,
  makeWrapper,
}:

stdenv.mkDerivation (
  finalAttrs:
  let
    foundation = stdenv.mkDerivation {
      pname = "foundation-simulator";
      __structuredAttrs = true;
      strictDeps = true;
      inherit (finalAttrs)
        version
        src
        pnpmDeps
        ;
      pnpmWorkspaces = [
        "@simulacrum/foundation-simulator"
      ];
      pnpmRoot = "packages/foundation";
      nativeBuildInputs = [
        nodejs-slim
        pnpmConfigHook
        pnpmBuildHook
        pnpm
      ];
      installPhase = ''
        runHook preInstall

        pushd packages/foundation

        CI=true pnpm prune --prod --ignore-scripts

        mkdir -p $out/share/node_modules
        cp -r -t $out/share dist package.json
        cp -rL -t $out/share/node_modules node_modules/*

        runHook postInstall
      '';
    };
  in
  {
    pname = "github-api-simulator";
    version = "0.8.0";
    __structuredAttrs = true;
    strictDeps = true;

    src = fetchFromGitHub {
      owner = "thefrontside";
      repo = "simulacrum";
      tag = "@simulacrum/github-api-simulator-v${finalAttrs.version}";
      hash = "sha256-jGi5nODAvCviC6IfQgdVxypC3kd+FVoH4ivYxN/7PZM=";
    };

    nativeBuildInputs = [
      makeWrapper
      nodejs-slim
      pnpmConfigHook
      pnpmBuildHook
      pnpm
    ];
    pnpmWorkspaces = [
      "@simulacrum/foundation-simulator"
      "@simulacrum/github-api-simulator"
    ];
    pnpmRoot = "packages/github-api";
    pnpmDeps = fetchPnpmDeps {
      inherit (finalAttrs)
        pname
        version
        src
        pnpmWorkspaces
        ;
      inherit pnpm;
      fetcherVersion = 4;
      hash = "sha256-ik8cRAtnCbpD8pHc/GsSPeWERDdXL0m7/U3ho3nVeSQ=";
    };

    pnpmBuildScript = "prepack";

    installPhase = ''
      runHook preInstall

      mkdir -p $out
      pnpm --filter=@simulacrum/github-api-simulator deploy --legacy $out --ignore-scripts --prod

      mkdir -p $out/bin
      makeWrapper '${nodejs-slim}/bin/node' "$out/bin/${finalAttrs.pname}" \
        --add-flags "$out/share/packages/github-api/bin/start.mjs" \
        --chdir "$out/share/packages/github-api" \
        --set NODE_ENV production

      runHook postInstall
    '';

    # installPhase = ''
    #   runHook preInstall
    #
    #   pushd packages/github-api
    #
    #   CI=true pnpm prune --prod --ignore-scripts
    #   # Clean up broken symlinks left behind by `pnpm prune`
    #   # https://github.com/pnpm/pnpm/issues/3645
    #
    #   popd
    #
    #
    #   find node_modules/.pnpm/node_modules -maxdepth 2 -type l -not -xtype d -delete
    #   mkdir -p $out/share/packages/{github-api,foundation}
    #   cp -r -t $out/share node_modules
    #   cp -r -t $out/share/packages/github-api packages/github-api/{dist,bin,package.json,node_modules}
    #   cp -r -t $out/share/packages/foundation packages/foundation/{dist,package.json,node_modules}
    #
    #   mkdir -p $out/bin
    #   makeWrapper '${nodejs-slim}/bin/node' "$out/bin/${finalAttrs.pname}" \
    #     --add-flags "$out/share/packages/github-api/bin/start.mjs" \
    #     --chdir "$out/share/packages/github-api" \
    #     --set NODE_ENV production
    #
    #   runHook postInstall
    # '';
    #
    passthru.updateScript = nix-update-script { };
    passthru.foundation = foundation;

    meta = {
      description = "A simulation platform for use during testing, during development and for high-fidelity application previews";
      homepage = "https://github.com/thefrontside/simulacrum";
      changelog = "https://github.com/thefrontside/simulacrum/releases/tag/${finalAttrs.src.tag}";
      license = lib.licenses.asl20;
      mainProgram = finalAttrs.pname;
      platforms = lib.platforms.all;
    };
  }
)
