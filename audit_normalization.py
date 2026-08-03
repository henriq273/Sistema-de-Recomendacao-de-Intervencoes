"""
Auditoria dos campos normalizados (*Normalized) do catálogo real.

Script standalone: não faz parte do caminho de produção, roda sob demanda para
diagnóstico, antes de qualquer treino sobre o dataset real (ver ordem de
implementação — passo 2). Só leitura: find()/aggregate() sem $out/$merge, nunca
escreve no banco.

Recalcula o valor esperado a partir do campo bruto e da escala de origem documentada
em NORMALIZATION_REFERENCE, e reporta discrepâncias. Datasets com scale=None são
pulados com aviso explícito — nunca com uma fórmula assumida silenciosamente.

Uso:
    python audit_normalization.py
"""
import numpy as np

import data_source
import sistema_de_recomendacao_v3 as sysrec
from data_source import _get, get_read_only_client


def audit_dataset(collection, dataset_name: str, tolerance: float = sysrec.NORMALIZATION_TOLERANCE) -> list:
    """Recalcula (raw - mid) / half a partir da escala documentada e compara com o
    valor já normalizado armazenado no banco. Só leitura."""
    ref = sysrec.NORMALIZATION_REFERENCE.get(dataset_name)
    if ref is None or ref["scale"] is None:
        print(f"[{dataset_name}] escala não confirmada na fonte -- PULAR até checar a "
              f"documentação original antes de auditar.")
        return []

    lo, hi = ref["scale"]
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    stored_field = ref["raw_field"].replace("Mean", "Normalized")
    discrepancias = []

    for doc in collection.find({"sourceMeta.dataset": dataset_name}):  # leitura apenas
        raw = _get(doc, ref["raw_field"])
        stored = _get(doc, stored_field)
        if raw is None or stored is None:
            discrepancias.append({"id": doc["_id"], "motivo": "campo ausente"})
            continue
        expected = (raw - mid) / half
        if abs(expected - stored) > tolerance:
            discrepancias.append({
                "id": doc["_id"], "raw": raw, "stored": stored, "expected": expected,
            })
    return discrepancias


def run_full_audit() -> dict:
    """Audita todos os datasets listados em NORMALIZATION_REFERENCE contra o valor
    normalizado já armazenado no banco."""
    collection = get_read_only_client()
    relatorio = {}
    for dataset_name in sysrec.NORMALIZATION_REFERENCE:
        relatorio[dataset_name] = audit_dataset(collection, dataset_name)
    print("\n=== Resumo: discrepâncias de normalização ===")
    for nome, discs in relatorio.items():
        print(f"{nome}: {len(discs)} discrepância(s)")
    return relatorio


def check_out_of_range(collection) -> None:
    """Checagem estatística complementar (independe do campo bruto): qualquer
    valenciaNorm/arousalNorm fora de [-1, 1] indica normalização quebrada, não uma
    escala de origem diferente -- não há escala de origem que produza isso se a
    fórmula estiver certa."""
    print("=== Faixa fora de [-1, 1] ===")
    for campo in ("ratings.valenceNormalized", "ratings.arousalNormalized"):
        fora = list(collection.find({
            "$or": [{campo: {"$lt": -1.0}}, {campo: {"$gt": 1.0}}],
        }))
        print(f"  {campo}: {len(fora)} documento(s) fora da faixa")


def check_zero_variance(collection) -> None:
    """Desvio-padrão próximo de zero por dataset é sinal de bug de normalização
    (todo item colapsando no mesmo valor) ou de um campo bruto constante na fonte --
    ambos merecem investigação antes de treinar sobre o dataset."""
    print("=== Desvio-padrão por dataset (valenciaNorm) ===")
    for dataset_name in sysrec.NORMALIZATION_REFERENCE:
        valores = [
            _get(doc, "ratings.valenceNormalized")
            for doc in collection.find({"sourceMeta.dataset": dataset_name})
        ]
        valores = [v for v in valores if v is not None]
        if not valores:
            print(f"  {dataset_name}: sem documentos/campo ausente")
            continue
        desvio = float(np.std(valores))
        alerta = " <- ATENÇÃO: desvio ~0" if desvio < 1e-3 else ""
        print(f"  {dataset_name}: n={len(valores)} desvio={desvio:.4f}{alerta}")


def check_octant_region_agreement(collection) -> None:
    """Compara, para um mesmo oitante rotulado, a região do plano (V, A) ocupada por
    itens de datasets diferentes. Datasets cuja normalização diverge tendem a produzir
    nuvens de pontos deslocadas entre si mesmo rotuladas com o mesmo oitante -- sinal
    indireto de escala mal calibrada que a comparação par a par revela mais cedo que
    esperar o modelo falhar em produção."""
    print("=== Região (V, A) por oitante, comparada entre datasets ===")
    por_oitante_dataset = {}
    for doc in collection.find({}):
        oitante = doc.get("videoOctant") or doc.get("soundOctant") or doc.get("imageOctant")
        dataset_name = _get(doc, "sourceMeta.dataset") or _get(doc, "source.dataset")
        v = (
            _get(doc, "ratings.valenceNormalized")
            or _get(doc, "staticAnnotations.valenceNormalized")
        )
        a = (
            _get(doc, "ratings.arousalNormalized")
            or _get(doc, "staticAnnotations.arousalNormalized")
        )
        if oitante is None or dataset_name is None or v is None or a is None:
            continue
        por_oitante_dataset.setdefault(oitante, {}).setdefault(dataset_name, []).append((v, a))

    for oitante, por_dataset in sorted(por_oitante_dataset.items(), key=str):
        print(f"  Oitante {oitante}:")
        for dataset_name, pontos in por_dataset.items():
            vs = [p[0] for p in pontos]
            as_ = [p[1] for p in pontos]
            print(
                f"    {dataset_name}: n={len(pontos)} "
                f"V médio={np.mean(vs):+.2f} A médio={np.mean(as_):+.2f}"
            )


def run_statistical_checks() -> None:
    """As três checagens complementares que não dependem do campo bruto/escala de
    origem documentada -- rodam mesmo para datasets com scale=None em
    NORMALIZATION_REFERENCE."""
    collection = get_read_only_client()
    check_out_of_range(collection)
    print()
    check_zero_variance(collection)
    print()
    check_octant_region_agreement(collection)


if __name__ == "__main__":
    run_full_audit()
    print()
    run_statistical_checks()
