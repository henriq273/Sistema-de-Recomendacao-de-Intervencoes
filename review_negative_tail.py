"""
review_negative_tail.py — BANCADA DE CURADORIA MANUAL. Não faz parte do caminho
de produção. Ferramenta interativa para revisar a cauda de menor valência de
datasets sem taxonomia de categoria dedicada a conteúdo negativo (OASIS,
EmoMadrid) -- diferente do GAPED, que se resolve estruturalmente (aprovar só as
categorias N/P da taxonomia documentada, sem passar por revisão item a item; ver
README.md).

Por quê este script existe: a denylist de palavra-chave (safety.py, Camada 2) não
pega conteúdo perturbador sem palavra-gatilho na descrição, e mesmo a revisão
geométrica por valência (Camada 3) não pega tudo -- a própria literatura do GAPED
documenta dessensibilização de avaliador (imagens de categoria negativa podem ter
valência moderada, não extrema). Sem revisão humana item a item da cauda de menor
valência, qualquer APPROVED_CATEGORIES aprovada para OASIS/EmoMadrid corre o risco
de aprovar categorias que contêm itens perturbadores não capturados por nenhuma
das outras camadas.

Uso:
    python review_negative_tail.py OASIS
    python review_negative_tail.py EmoMadrid --n-items 100
    python review_negative_tail.py OASIS --no-resume   # ignora watermark, revisa do zero
"""
import argparse
import json
import os

import data_source
from data_source import _get

REVIEW_WATERMARK_PATH = "review_watermark.json"
BLOCKED_IDS_OUTPUT_PATH = "blocked_item_ids.txt"


def _load_watermark() -> dict:
    """Marca d'água de progresso por dataset, para retomar entre sessões sem
    revisar o mesmo item duas vezes."""
    if not os.path.exists(REVIEW_WATERMARK_PATH):
        return {}
    with open(REVIEW_WATERMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_watermark(watermark: dict) -> None:
    with open(REVIEW_WATERMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(watermark, f, indent=2)


def _append_blocked_id(item_id: str, path: str = BLOCKED_IDS_OUTPUT_PATH) -> None:
    """Grava incrementalmente -- decisão persiste mesmo se a sessão for
    interrompida no meio (ex.: Ctrl+C, queda de energia)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(item_id + "\n")


def review_negative_tail(dataset_name: str, n_items: int = 100,
                          resume: bool = True) -> tuple[int, bool]:
    """
    Itera os itens de menor valência de um dataset (via iter_raw_docs, dados
    brutos -- não precisa de aprovação prévia para ser revisado, é justamente a
    ferramenta que decide a aprovação), pede decisão humana item a item, grava
    bloqueios incrementalmente em BLOCKED_ID_OUTPUT_PATH (para colar depois em
    sysrec.BLOCKED_ITEM_IDS).

    Não fixa um número pequeno arbitrário de itens a revisar: n_items=100 por
    padrão, e o critério de parada por dataset (ver seção 4.2 do plano de
    consolidação) é rodar em lotes de 100 e parar quando um lote inteiro não gerar
    nenhum bloqueio -- não um número decidido de antemão. Esta função devolve
    (contagem de bloqueios do lote, se o lote foi de fato completado -- False
    quando a sessão foi interrompida via [q]uit antes de esgotar o lote), para o
    chamador (ou um loop externo) decidir se vale rodar outro lote e se o sinal de
    "lote limpo" é confiável.
    """
    watermark = _load_watermark()
    already_reviewed = set(watermark.get(dataset_name, []))

    # OASIS e EmoMadrid (os dois datasets a que este script se destina, ver
    # docstring do módulo) são datasets só de imagem -- restringir a modalidade
    # evita carregar videos.json/audios.json à toa em cada execução.
    docs = list(data_source.iter_raw_docs(modality="image", dataset_filter=dataset_name))

    def _valence(d):
        v = _get(d, "ratings.valenceNormalized")
        return v if v is not None else _get(d, "staticAnnotations.valenceNormalized")

    docs_with_va = [(d, _valence(d)) for d in docs]
    docs_with_va = [(d, v) for d, v in docs_with_va if v is not None]
    docs_with_va.sort(key=lambda x: x[1])  # mais negativo primeiro

    print(f"=== Revisão manual: {dataset_name} ({len(docs_with_va)} itens com V/A) ===")
    reviewed_count = 0
    blocked_count = 0
    quit_early = False
    try:
        for doc, valence in docs_with_va:
            item_id = str(doc["_id"])
            if resume and item_id in already_reviewed:
                continue
            if reviewed_count >= n_items:
                print(f"\nLimite de {n_items} itens revisados nesta sessão. "
                      f"Rode novamente para continuar de onde parou.")
                break

            arousal = _get(doc, "ratings.arousalNormalized")
            if arousal is None:
                arousal = _get(doc, "staticAnnotations.arousalNormalized")
            print(f"\n[{item_id}] {dataset_name} | category={doc.get('category')}")
            print(f"  V={valence:+.3f}  A={arousal if arousal is not None else 'N/A'}")
            print(f"  título: {doc.get('title', '(sem título)')}")
            url = doc.get("imageUrl") or doc.get("audioUrl") or doc.get("videoUrl")
            if url:
                print(f"  url: {url}")

            decision = input("  [a]provar / [b]loquear / [s]kip / [q]uit: ").strip().lower()
            if decision == "q":
                quit_early = True
                break
            if decision == "b":
                _append_blocked_id(item_id)
                blocked_count += 1
                print("  -> bloqueado.")
            elif decision == "a":
                print("  -> aprovado (nenhuma ação; item elegível pelas demais camadas).")
            # "s" (skip) não marca como revisado -- reaparece na próxima sessão

            if decision in ("a", "b"):
                already_reviewed.add(item_id)
                reviewed_count += 1
    finally:
        # salva o watermark mesmo em interrupção (Ctrl+C, EOFError etc.) -- do
        # contrário decisões já gravadas em BLOCKED_IDS_OUTPUT_PATH ficam
        # dessincronizadas do progresso e itens já revisados voltam a aparecer.
        watermark[dataset_name] = sorted(already_reviewed)
        _save_watermark(watermark)

    print(f"\nProgresso salvo: {len(already_reviewed)} itens revisados no total "
          f"para {dataset_name} ({blocked_count} bloqueado(s) neste lote).")
    full_batch_completed = not quit_early
    return blocked_count, full_batch_completed


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Bancada de curadoria manual: revisão humana item a item da "
                     "cauda de menor valência de um dataset (OASIS, EmoMadrid).",
    )
    parser.add_argument("dataset", help="Nome do dataset a revisar (ex.: OASIS, EmoMadrid).")
    parser.add_argument("--n-items", type=int, default=100,
                         help="Tamanho do lote revisado nesta execução (padrão: 100).")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignora o watermark e revisa desde o item mais negativo.")
    args = parser.parse_args()

    blocked, full_batch_completed = review_negative_tail(
        args.dataset, n_items=args.n_items, resume=not args.no_resume)
    if blocked == 0 and full_batch_completed:
        print(f"\nLote sem nenhum bloqueio -- candidato a ponto de corte para "
              f"{args.dataset} (ver seção 4.2 do plano de consolidação: parar quando "
              f"um lote inteiro não gerar bloqueio).")
    elif blocked == 0:
        print(f"\nSessão interrompida antes de completar o lote -- nenhum sinal de "
              f"ponto de corte ainda; rode novamente para revisar os itens restantes.")


if __name__ == "__main__":
    _main()
