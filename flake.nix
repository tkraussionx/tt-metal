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
        devShells.default = pkgs.mkShell.override { stdenv = pkgs.clang17Stdenv; } {
          buildInputs = with pkgs; [
            cmake
            ninja
            gdb
            lldb
            libcxx
            numactl # for libnuma
            glib
            glibc
          ];

          # export CMAKE_LIBRARY_PATH=$CMAKE_LIBRARY_PATH:${pkgs.libcxx}/lib
          # export CMAKE_CXX_FLAGS="-stdlib=libc++"
          # export CMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++abi"
          shellHook = ''
            echo "Welcome to the C++ development environment!"
            echo "CMake and necessary build tools are available."
          '';
        };
      }
    );
}
