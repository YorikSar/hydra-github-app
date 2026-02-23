{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    git-hooks-nix = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      git-hooks-nix,
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      inherit (nixpkgs) lib;
      forAllSystems =
        f:
        lib.genAttrs systems (
          system:
          f {
            inherit system;
            pkgs = nixpkgs.legacyPackages.${system};
          }
        );
    in
    {
      packages = forAllSystems (
        { system, pkgs }:
        {
          default = pkgs.callPackage ./package.nix { };
        }
      );
      devShells = forAllSystems (
        { system, pkgs }:
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.rustc
              pkgs.cargo
              pkgs.clippy
              pkgs.rustfmt
            ]
            ++ lib.optional pkgs.stdenv.hostPlatform.isDarwin pkgs.libiconv;
          };
        }
      );
      checks = forAllSystems (
        { system, pkgs }:
        lib.mapAttrs' (name: lib.nameValuePair "packages-${name}") self.packages.${system}
        // {
          git-hooks = git-hooks-nix.lib.${system}.run {
            src = self;
            hooks = {
              actionlint.enable = true;
              nixfmt.enable = true;
              rustfmt.enable = true;
            };
          };
        }
      );
    };
}
