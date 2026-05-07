{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    nativeBuildInputs = with pkgs.buildPackages; [ uv statix basedpyright ollama cargo ];
  shellHook = ''
    source ./scripts/.env
    '';
}

