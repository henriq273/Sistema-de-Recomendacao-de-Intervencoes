"""
Auditoria dos campos normalizados (*Normalized) do catálogo real.

Script standalone: não faz parte do caminho de produção, roda sob demanda para
diagnóstico, antes de qualquer treino sobre o dataset real (ver ordem de
implementação — passo 2). Backend-agnóstico: itera sempre via
data_source.iter_raw_docs() (mongo ao vivo ou json_export local, conforme
sysrec.DATA_BACKEND) — nunca acessa Mongo/arquivo diretamente. Só leitura no backend
Mongo: find() sem $out/$merge, nunca escreve no banco.

Recalcula o valor esperado a partir do campo bruto e da escala de origem documentada
em NORMALIZATION_REFERENCE, e reporta discrepâncias. Datasets com scale=None são
pulados com aviso explícito — nunca com uma fórmula assumida silenciosamente.

Uso:
    python normalization.py
"""
import re

import numpy as np

import data_source
import sistema_de_recomendacao_v3 as sysrec
from data_source import _get


def audit_identity_dataset(dataset_name: str) -> list:
    """
    Para datasets sem campo *Normalized separado (scale="IDENTITY" em
    NORMALIZATION_REFERENCE, ex.: MuVi -- confirmado que a estrutura do documento não
    tem valenceNormalized/arousalNormalized, não é erro de fórmula): não há o que
    comparar fórmula-vs-armazenado. A checagem possível é de faixa -- o valor bruto já
    é usado diretamente (ver data_source.normalize_video_doc), então precisa estar
    dentro de [-1, 1] sem qualquer transformação.
    """
    ref = sysrec.NORMALIZATION_REFERENCE[dataset_name]
    valence_field = ref["raw_field"]
    arousal_field = valence_field.replace("valence", "arousal").replace("Valence", "Arousal")

    fora_da_faixa = []
    for doc in data_source.iter_raw_docs(dataset_filter=dataset_name):
        v = _get(doc, valence_field)
        a = _get(doc, arousal_field)
        if v is not None and not (-1 <= v <= 1):
            fora_da_faixa.append({"id": doc["_id"], "campo": "valencia", "valor": v})
        if a is not None and not (-1 <= a <= 1):
            fora_da_faixa.append({"id": doc["_id"], "campo": "arousal", "valor": a})
    return fora_da_faixa


def audit_dataset(dataset_name: str, tolerance: float = sysrec.NORMALIZATION_TOLERANCE) -> list:
    """Recalcula (raw - mid) / half a partir da escala documentada e compara com o
    valor já normalizado armazenado no catálogo bruto (data_source.iter_raw_docs)."""
    ref = sysrec.NORMALIZATION_REFERENCE.get(dataset_name)
    if ref is None:
        print(f"[{dataset_name}] não está em NORMALIZATION_REFERENCE -- nada a auditar.")
        return []

    if ref["scale"] == "IDENTITY":
        fora_da_faixa = audit_identity_dataset(dataset_name)
        status = f"{len(fora_da_faixa)} valor(es) fora de [-1, 1]" if fora_da_faixa else "todos os valores dentro de [-1, 1]"
        print(f"[{dataset_name}] sem campo *Normalized separado (IDENTITY) -- checagem de faixa: {status}.")
        return [
            {"id": item["id"], "motivo": f"{item['campo']} fora de [-1,1]: {item['valor']}"}
            for item in fora_da_faixa
        ]

    if ref["scale"] is None:
        print(f"[{dataset_name}] escala não confirmada na fonte -- PULAR até checar a "
              f"documentação original antes de auditar.")
        return []

    lo, hi = ref["scale"]
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    stored_field = ref["raw_field"].replace("Mean", "Normalized")
    discrepancias = []

    for doc in data_source.iter_raw_docs(dataset_filter=dataset_name):
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
    normalizado já armazenado no catálogo bruto do backend ativo."""
    relatorio = {}
    for dataset_name in sysrec.NORMALIZATION_REFERENCE:
        relatorio[dataset_name] = audit_dataset(dataset_name)
    print("\n=== Resumo: discrepâncias de normalização ===")
    for nome, discs in relatorio.items():
        print(f"{nome}: {len(discs)} discrepância(s)")
    return relatorio


def check_out_of_range() -> None:
    """Checagem estatística complementar (independe do campo bruto): qualquer
    valenciaNorm/arousalNorm fora de [-1, 1] indica normalização quebrada, não uma
    escala de origem diferente -- não há escala de origem que produza isso se a
    fórmula estiver certa."""
    print("=== Faixa fora de [-1, 1] ===")
    for campo in ("ratings.valenceNormalized", "ratings.arousalNormalized"):
        fora = [
            doc for doc in data_source.iter_raw_docs()
            if (v := _get(doc, campo)) is not None and (v < -1.0 or v > 1.0)
        ]
        print(f"  {campo}: {len(fora)} documento(s) fora da faixa")


def check_zero_variance() -> None:
    """Desvio-padrão próximo de zero por dataset é sinal de bug de normalização
    (todo item colapsando no mesmo valor) ou de um campo bruto constante na fonte --
    ambos merecem investigação antes de treinar sobre o dataset."""
    print("=== Desvio-padrão por dataset (valenciaNorm) ===")
    for dataset_name in sysrec.NORMALIZATION_REFERENCE:
        valores = [
            _get(doc, "ratings.valenceNormalized") or _get(doc, "staticAnnotations.valenceNormalized")
            for doc in data_source.iter_raw_docs(dataset_filter=dataset_name)
        ]
        valores = [v for v in valores if v is not None]
        if not valores:
            print(f"  {dataset_name}: sem documentos/campo ausente")
            continue
        desvio = float(np.std(valores))
        alerta = " <- ATENÇÃO: desvio ~0" if desvio < 1e-3 else ""
        print(f"  {dataset_name}: n={len(valores)} desvio={desvio:.4f}{alerta}")


def check_octant_region_agreement() -> None:
    """Compara, para um mesmo oitante rotulado, a região do plano (V, A) ocupada por
    itens de datasets diferentes. Datasets cuja normalização diverge tendem a produzir
    nuvens de pontos deslocadas entre si mesmo rotuladas com o mesmo oitante -- sinal
    indireto de escala mal calibrada que a comparação par a par revela mais cedo que
    esperar o modelo falhar em produção."""
    print("=== Região (V, A) por oitante, comparada entre datasets ===")
    por_oitante_dataset = {}
    for doc in data_source.iter_raw_docs():
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


_QUADRANT_TAG_RE = re.compile(r"quadrant_q([1-4])", re.IGNORECASE)
_QUADRANT_EXPECTED_SIGNS = {
    "quadrant_q1": (1, 1), "quadrant_q2": (-1, 1),
    "quadrant_q3": (-1, -1), "quadrant_q4": (1, -1),
}


def check_internal_consistency(row, octant_map=None, get_nearest_octant_fn=None) -> list:
    """
    Verifica concordância entre tags, oitante declarado, quadrante (se houver) e o
    sinal dos valores de valência/arousal de UM item já normalizado (esquema de
    data_source.build_raw_catalog: valencia_norm/arousal_norm/tags/octant_raw).
    Retorna lista de problemas (vazia se tudo concordar).

    Foi este checador que revelou a inconsistência do EMOPIA (seção 5 do plano de
    revisões): quadrante, tags, oitante declarado e o valor armazenado apontavam em
    quatro direções diferentes para o mesmo item -- não uma ambiguidade de fronteira
    (que gera no máximo 1 problema, ver meditation_local), e sim uma contradição
    sistemática (2+ problemas simultâneos).
    """
    octant_map = octant_map if octant_map is not None else sysrec.OCTANT_MAP
    get_nearest_octant_fn = get_nearest_octant_fn or (lambda v, a, _om: sysrec.nearest_octant(v, a))

    problems = []
    v, a = row.get("valencia_norm"), row.get("arousal_norm")
    if v is None or a is None:
        return ["V/A ausente -- não é possível checar consistência"]

    tags = set(t.lower() for t in (row.get("tags") or []))

    if "positive_valence" in tags and v < 0:
        problems.append("tag positive_valence contradiz valencia_norm negativa")
    if "negative_valence" in tags and v > 0:
        problems.append("tag negative_valence contradiz valencia_norm positiva")
    if "high_arousal" in tags and a < 0:
        problems.append("tag high_arousal contradiz arousal_norm negativo")
    if "low_arousal" in tags and a > 0:
        problems.append("tag low_arousal contradiz arousal_norm positivo")

    declared_octant = row.get("octant_raw")
    if declared_octant:
        geo_octant = get_nearest_octant_fn(v, a, octant_map)
        if geo_octant != declared_octant:
            problems.append(f"octant_raw={declared_octant} diverge do geométrico={geo_octant}")

    quadrant_tags = [t for t in tags if _QUADRANT_TAG_RE.match(t)]
    if quadrant_tags:
        exp_v, exp_a = _QUADRANT_EXPECTED_SIGNS.get(quadrant_tags[0], (0, 0))
        if exp_v and (v > 0) != (exp_v > 0):
            problems.append(f"{quadrant_tags[0]} contradiz sinal de valencia_norm")
        if exp_a and (a > 0) != (exp_a > 0):
            problems.append(f"{quadrant_tags[0]} contradiz sinal de arousal_norm")

    return problems


def run_consistency_audit(catalog_df, octant_map=None, get_nearest_octant_fn=None):
    """
    Roda check_internal_consistency sobre o catálogo BRUTO inteiro (antes da
    curadoria de segurança -- ver data_source.build_raw_catalog; rodar pós-curadoria
    esconderia justamente os itens de dataset ainda não aprovado, como EMOPIA, que
    são o motivo de existir esta checagem). Separa achados em severos (2+
    contradições simultâneas) e leves (1 contradição de fronteira), para priorizar a
    fila de revisão manual (Camada 4 da curadoria de segurança já implementada).
    """
    severos, leves = [], []
    for _, row in catalog_df.iterrows():
        problems = check_internal_consistency(row, octant_map, get_nearest_octant_fn)
        if len(problems) >= 2:
            severos.append((row["item_id"], row["dataset"], problems))
        elif len(problems) == 1:
            leves.append((row["item_id"], row["dataset"], problems))
    print(f"Consistência interna: {len(severos)} severos, {len(leves)} leves, "
          f"sobre {len(catalog_df)} itens.")
    return severos, leves


def run_statistical_checks() -> None:
    """As três checagens complementares que não dependem do campo bruto/escala de
    origem documentada -- rodam mesmo para datasets com scale=None em
    NORMALIZATION_REFERENCE."""
    check_out_of_range()
    print()
    check_zero_variance()
    print()
    check_octant_region_agreement()


if __name__ == "__main__":
    if sysrec.DATA_BACKEND == "csv":
        # "csv" é o padrão em sistema_de_recomendacao_v3.py para o catálogo sintético
        # (dataset.csv), que não tem datasets reais (DEAM/OASIS/GAPED/...) para
        # auditar -- normalization.py só faz sentido contra o catálogo real. Ao
        # rodar como script sem configuração prévia, cai para "json_export" (arquivos
        # locais em dbs/), que não exige Mongo nem pymongo. Para auditar o Mongo ao
        # vivo, defina DATA_BACKEND = "mongo" em sistema_de_recomendacao_v3.py antes.
        print("[normalization] DATA_BACKEND='csv' não se aplica à auditoria -- "
              "usando 'json_export' (dbs/) nesta execução.\n")
        sysrec.DATA_BACKEND = "json_export"

    run_full_audit()
    print()
    run_statistical_checks()
    print()
    # Catálogo BRUTO (pré-curadoria) -- ver docstring de run_consistency_audit sobre
    # por que não pode ser o catálogo já filtrado por safety.apply_safety_filter.
    raw_catalog = data_source.build_raw_catalog()
    run_consistency_audit(raw_catalog)
