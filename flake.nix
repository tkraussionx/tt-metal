{
  description = "C++ project with CMake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            cmake
            gcc
            clang_17
            ninja
            gdb
            lldb
            libcxx
            numactl # for libnuma
          ];

          shellHook = ''
            echo "Welcome to the C++ development environment!"
            echo "CMake and necessary build tools are available."
            export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:${pkgs.libcxx}/lib/cmake
          '';
        };
      }
    );
}
