{
  lib,
  rustPlatform,
}:
rustPlatform.buildRustPackage {
  name = "gh-app";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./Cargo.toml
      ./Cargo.lock
      ./src
    ];
  };

  cargoLock.lockFile = ./Cargo.lock;
}
