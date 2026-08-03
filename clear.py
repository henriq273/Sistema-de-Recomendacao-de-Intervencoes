"""
Remove dados residuais gerados por execuções/testes do sistema de recomendação:
checkpoint(s) do agente (checkpoint*.pt), log de interações (interaction_log.jsonl)
e, opcionalmente, caches de bytecode (__pycache__).

Por padrão roda em modo dry-run (só lista o que seria removido). Passe --confirmar
para remover de fato — mesmo padrão do `git clean -n` / `git clean -f`.

Uso:
    python clear.py                        # dry-run: só mostra o que seria removido
    python clear.py --confirmar             # remove checkpoint(s) e log de interações
    python clear.py --confirmar --cache      # também remove __pycache__
    python clear.py --confirmar --manter-log # remove só o(s) checkpoint(s)
    python clear.py --confirmar --extra foo.pt bar.jsonl  # remove arquivos extras também
"""
import argparse
import glob
import os
import shutil

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Nomes vêm de CHECKPOINT_PATH/INTERACTION_LOG_PATH em sistema_de_recomendacao_v2.py e
# v3.py. Usa glob para "checkpoint*.pt" para não ficar preso a um nome exato se um
# futuro checkpoint (ex.: checkpoint_v4.pt) for adicionado.
CHECKPOINT_GLOB = "checkpoint*.pt"
INTERACTION_LOG_GLOB = "interaction_log*.jsonl"

# Diretórios que nunca devem ser varridos em busca de __pycache__ (ambiente virtual,
# metadados do git).
_SKIP_DIRS = {".venv", ".git"}


def _find_checkpoints():
    return sorted(glob.glob(os.path.join(REPO_DIR, CHECKPOINT_GLOB)))


def _find_interaction_logs():
    return sorted(glob.glob(os.path.join(REPO_DIR, INTERACTION_LOG_GLOB)))


def _find_pycache_dirs():
    found = []
    for root, dirs, _ in os.walk(REPO_DIR):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        if "__pycache__" in dirs:
            found.append(os.path.join(root, "__pycache__"))
    return sorted(found)


def _collect_targets(include_checkpoint, include_log, include_cache, extra):
    targets = []
    if include_checkpoint:
        targets += [("checkpoint", p) for p in _find_checkpoints()]
    if include_log:
        targets += [("log de interações", p) for p in _find_interaction_logs()]
    if include_cache:
        targets += [("cache (__pycache__)", p) for p in _find_pycache_dirs()]
    for rel_path in extra:
        path = os.path.join(REPO_DIR, rel_path)
        if os.path.exists(path):
            targets.append(("extra", path))
        else:
            print(f"Aviso: '{rel_path}' não existe, ignorando.")
    return targets


def _remove(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove dados residuais de execuções/testes do sistema de recomendação.",
    )
    parser.add_argument(
        "--confirmar", action="store_true",
        help="Remove de fato. Sem esta flag, só mostra o que seria removido (dry-run).",
    )
    parser.add_argument("--manter-checkpoint", action="store_true", help="Não remove checkpoint*.pt.")
    parser.add_argument("--manter-log", action="store_true", help="Não remove interaction_log*.jsonl.")
    parser.add_argument(
        "--cache", action="store_true",
        help="Também remove diretórios __pycache__ (desligado por padrão: neste repo eles "
             "estão versionados no git, então removê-los aparece como alteração para revisar).",
    )
    parser.add_argument(
        "--extra", nargs="*", default=[],
        help="Caminhos adicionais (relativos à raiz do repo) para remover junto.",
    )
    args = parser.parse_args()

    targets = _collect_targets(
        include_checkpoint=not args.manter_checkpoint,
        include_log=not args.manter_log,
        include_cache=args.cache,
        extra=args.extra,
    )

    if not targets:
        print("Nada para remover.")
        return

    print("Itens encontrados:")
    for kind, path in targets:
        rel = os.path.relpath(path, REPO_DIR)
        print(f"  [{kind}] {rel}")

    if not args.confirmar:
        print(f"\n{len(targets)} item(ns) seriam removidos. Rode com --confirmar para remover de fato.")
        return

    for _, path in targets:
        _remove(path)
    print(f"\n{len(targets)} item(ns) removido(s).")


if __name__ == "__main__":
    main()
