{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    nativeBuildInputs = with pkgs.buildPackages; [ uv statix basedpyright ollama cargo wayclip ];
  shellHook = ''
    source ./scripts/.env
    '';
}

