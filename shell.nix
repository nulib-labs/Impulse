{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    nativeBuildInputs = with pkgs.buildPackages; [ uv statix basedpyright ollama ];
  shellHook = ''
    source ./scripts/.env
    '';
}

