{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
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
      packages =
        let
          modulesEval = nixpkgs.lib.evalModules {
            modules = [
              self.nixosModules.default
              {
                _module.check = false;
                _module.args.pkgs = nixpkgs.legacyPackages.x86_64-linux;
              }
            ];
          };
        in
        forAllSystems (
          { system, pkgs }:
          {
            default = pkgs.callPackage ./package.nix { };
            nixosModuleDoc =
              (pkgs.nixosOptionsDoc {
                options = nixpkgs.lib.removeAttrs modulesEval.options [ "_module" ];
              }).optionsCommonMark;
            updateNixOSModuleDoc = pkgs.writeScriptBin "replace-doc.sed" ''
              #!${pkgs.lib.getExe pkgs.gnused} -nf
              # Print all lines
              p
              # When encounter the start line
              /^<!--begin generated NixOS module documentation-->$/ {
                # It will already be printed
                # Append contents of the generated doc file
                r ${self.packages.${system}.nixosModuleDoc}
                # Set a label for iteration
                :0
                # Until we find the end line
                /^<!--end generated NixOS module documentation-->$/! {
                  # Advance to the next lien
                  n
                  # Jump to the label
                  b0
                }
                # Print the end line
                p
              }
              # We consumed all lines between start and end lines
            '';
          }
        );
      nixosModules.default = import ./nixosModule.nix;
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
              module-doc-in-readme = {
                enable = true;
                name = "Check that module doc is up to date in README";
                files = "README\\.md$";
                entry = "${pkgs.lib.getExe self.packages.${system}.updateNixOSModuleDoc} -i";
              };
            };
          };
        }
      );
    };
}
