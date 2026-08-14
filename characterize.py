"""
characterize.py — BANCADA DE DIAGNÓSTICO. Não faz parte do caminho de produção.

Roda contra o catálogo carregado via sysrec.load_active_catalog() (qualquer
DATA_BACKEND) e produz um relatório estatístico do dataset real, usado para
calibrar parâmetros que hoje estão herdados do dataset sintético
(CANDIDATE_POOL_SIZE, limiares do guardrail, buckets de tempo).

Adaptado às APIs de fato existentes em data_source.py/safety.py/normalization.py
(algumas divergem do desenho original destes planos -- ver notas inline onde isso
acontece):
  - load_active_catalog() já devolve, além de EXPECTED_COLUMNS, as colunas de
    diagnóstico (dataset, tipo_modalidade, category, tags, octant_raw,
    confidence_tier) — ver data_source._to_feature_space_schema.
  - "Oitante" no catálogo já é SEMPRE geométrico (nearest_octant sobre
    Valencia/Arousal), nunca o rótulo declarado — a comparação declarado vs.
    geométrico usa a coluna separada "octant_raw".
  - normalization.run_consistency_audit precisa do catálogo BRUTO (pré-curadoria,
    esquema data_source.UNIFIED_SCHEMA_FIELDS), não do catálogo já adaptado para
    FeatureSpace -- por isso a seção H reconstrói via data_source.build_raw_catalog(),
    em vez de reusar o df já curado.
  - safety.explore_taxonomy() agrega por (dataset, category), não por dataset --
    a seção E soma por dataset antes de comparar com a sobrevivência da curadoria.

Uso:
    python characterize.py
"""
from collections import Counter

import numpy as np
import pandas as pd

import data_source
import normalization
import safety
import sistema_de_recomendacao_v3 as sysrec


def report_va_distribution(df: pd.DataFrame) -> None:
    """A. Distribuição de Valência/Arousal — geral e por modalidade."""
    print("=== A. Distribuição Valência/Arousal ===")
    print(df[["Valencia", "Arousal"]].describe())
    print("\nPor modalidade (tipo_modalidade):")
    print(df.groupby("tipo_modalidade")[["Valencia", "Arousal"]].describe())


def report_octant_density(df: pd.DataFrame) -> None:
    """B. Densidade por oitante geométrico. df['Oitante'] já É geométrico por
    construção (ver data_source._to_feature_space_schema) -- a comparação com o
    rótulo declarado usa a coluna separada 'octant_raw', preservada à parte
    justamente para essa checagem, sem que o caminho de produção dependa dela."""
    print("\n=== B. Densidade por oitante geométrico ===")
    counts = df["Oitante"].value_counts().sort_index()
    print(counts)
    print(f"\nMenor densidade: oitante {counts.idxmin()} ({counts.min()} itens)")
    print(f"Maior densidade: oitante {counts.idxmax()} ({counts.max()} itens)")

    if "octant_raw" in df.columns:
        declared = pd.to_numeric(df["octant_raw"], errors="coerce")
        agreement = (declared == df["Oitante"]).mean()
        print(f"\nConcordância rótulo declarado (octant_raw) vs. geométrico (Oitante): "
              f"{agreement:.1%}")


def report_duration(df: pd.DataFrame) -> None:
    """D. Duração por modalidade -- identifica se algum bucket de tempo fica sem
    candidatos de alguma modalidade."""
    print("\n=== D. Duração por modalidade ===")
    print(df.groupby("tipo_modalidade")["Duracao"].describe())
    print("\nItens elegíveis por bucket de tempo e modalidade:")
    for t in sysrec.CONTEXT_DURATIONS:
        counts = df[df["Duracao"] <= t].groupby("tipo_modalidade").size()
        print(f"  <= {t:>3}min: {dict(counts)}")


def _raw_counts_by_dataset() -> dict:
    """Soma por dataset as contagens de safety.explore_taxonomy(), que agrega por
    (dataset, category) -- não por dataset sozinho. Reusa a função existente em vez
    de reimplementar a iteração sobre documentos brutos."""
    counts, _ = safety.explore_taxonomy()
    by_dataset: dict = {}
    for (dataset, _category), n in counts.items():
        by_dataset[dataset] = by_dataset.get(dataset, 0) + n
    return by_dataset


def report_curation_survival(df_curated: pd.DataFrame) -> None:
    """E. Quantos itens de cada dataset sobreviveram à curadoria (allowlist +
    denylist + blocked_ids + revisão geométrica), comparado ao total original bruto
    (safety.explore_taxonomy(), que contorna a curadoria de propósito)."""
    print("\n=== E. Sobrevivência da curadoria por dataset ===")
    raw_counts_by_dataset = _raw_counts_by_dataset()
    survived = df_curated.groupby("dataset").size().to_dict()
    for dataset, raw_n in sorted(raw_counts_by_dataset.items()):
        kept = survived.get(dataset, 0)
        pct = kept / raw_n if raw_n else 0.0
        print(f"  {dataset:16s} {kept:>5} / {raw_n:>5}  ({pct:.1%})")


def report_confidence_tier(df: pd.DataFrame) -> None:
    """F. Proporção do catálogo vinda de fonte psicométrica vs. heurística. Usa a
    coluna 'confidence_tier' já resolvida por item em data_source.py (via
    sysrec.CONFIDENCE_TIER), em vez de remapear a partir de 'dataset'."""
    print("\n=== F. Confiabilidade da origem (psychometric vs heuristic) ===")
    tiers = df["confidence_tier"].value_counts()
    total = len(df)
    for tier, n in tiers.items():
        print(f"  {tier:14s} {n:>5} ({n/total:.1%})")


def report_pool_size_baseline(df: pd.DataFrame, allowed_dest_octants) -> dict:
    """
    C. O mais importante: tamanho de pool (sem guardrail ainda -- só elegibilidade
    por tempo + distância no plano V-A) sobre TODA a grade de contextos plausíveis.
    Retorna os resultados brutos para uso na calibração do guardrail.
    """
    print("\n=== C. Tamanho de pool na grade completa (baseline, sem guardrail) ===")
    results = []
    for curr in range(1, 9):
        for dest in allowed_dest_octants:
            for t in sysrec.CONTEXT_DURATIONS:
                elig = df.index[df["Duracao"] <= t]
                if len(elig) == 0:
                    results.append((0, curr, dest, t))
                    continue
                pool_n = min(sysrec.CANDIDATE_POOL_SIZE, len(elig))
                results.append((pool_n, curr, dest, t))

    sizes = [r[0] for r in results]
    print(f"  Contextos avaliados: {len(results)}")
    print(f"  Pool mínimo: {min(sizes)}  |  Pool médio: {np.mean(sizes):.1f}")
    zero_pool = [r for r in results if r[0] == 0]
    print(f"  Contextos com pool ZERO: {len(zero_pool)}")
    for n, curr, dest, t in zero_pool[:10]:
        print(f"    curr={curr} dest={dest} t={t}min -> {n} itens elegíveis")
    return {"raw_results": results}


def report_pool_size_with_guardrail(df: pd.DataFrame, allowed_dest_octants) -> dict:
    """
    C'. Mesma varredura da seção C, mas aplicando o guardrail de 4 regras
    (Recommender._apply_safety_filter) com os limiares já calibrados/preenchidos em
    sysrec.SAFETY_AROUSAL_THRESHOLD / sysrec.SAFETY_AVERSIVE_VALENCE_THRESHOLD.
    Usado para reconfirmar, após implementar o guardrail, que o pool mínimo real
    bate com o previsto pela calibração (passo 5 da ordem de execução do plano).
    """
    print("\n=== C'. Tamanho de pool na grade completa (COM guardrail aplicado) ===")
    results = []
    for curr in range(1, 9):
        for dest in allowed_dest_octants:
            tv, ta = sysrec.OCTANT_MAP[dest]
            for t in sysrec.CONTEXT_DURATIONS:
                elig = df.index[df["Duracao"] <= t].to_numpy()
                if len(elig) == 0:
                    results.append((0, curr, dest, t))
                    continue

                A = df.loc[elig, "Arousal"].to_numpy(dtype=np.float32)
                V = df.loc[elig, "Valencia"].to_numpy(dtype=np.float32)
                r1 = (curr in sysrec.LOW_ENERGY_OCTANTS) & (A > sysrec.SAFETY_AROUSAL_THRESHOLD)
                r2 = (curr in sysrec.HIGH_ENERGY_OCTANTS) & (A > sysrec.SAFETY_AROUSAL_THRESHOLD)
                r3 = (ta < 0) & (A > sysrec.SAFETY_AROUSAL_THRESHOLD)
                r4 = (tv > 0) & (V < sysrec.SAFETY_AVERSIVE_VALENCE_THRESHOLD)
                blocked = r1 | r2 | r3 | r4
                safe = elig[~blocked] if blocked.any() and (~blocked).any() else elig

                pool_n = min(sysrec.CANDIDATE_POOL_SIZE, len(safe))
                results.append((pool_n, curr, dest, t))

    sizes = [r[0] for r in results]
    print(f"  Contextos avaliados: {len(results)}")
    print(f"  Pool mínimo: {min(sizes)}  |  Pool médio: {np.mean(sizes):.1f}")
    below_floor = [r for r in results if r[0] < sysrec.TOP_K]
    print(f"  Contextos abaixo do piso (TOP_K={sysrec.TOP_K}): {len(below_floor)}")
    for n, curr, dest, t in below_floor[:10]:
        print(f"    curr={curr} dest={dest} t={t}min -> {n} itens pós-guardrail")
    return {"raw_results": results}


def report_normalization_audit() -> None:
    """G. Reaudição em escala plena (reusa normalization.py já especificado)."""
    print("\n=== G. Reaudição de normalização (amostra completa por dataset) ===")
    for dataset_name in sysrec.NORMALIZATION_REFERENCE:
        discrepancias = normalization.audit_dataset(dataset_name)
        print(f"  {dataset_name:16s} {len(discrepancias)} discrepância(s)")


def report_consistency_audit() -> None:
    """H. Agregado do checador de consistência interna, por dataset. Precisa do
    catálogo BRUTO (pré-curadoria, esquema UNIFIED_SCHEMA_FIELDS) -- reconstrói via
    data_source.build_raw_catalog() em vez de reusar o df já adaptado para
    FeatureSpace (que não tem valencia_norm/arousal_norm/octant_raw no formato que
    normalization.run_consistency_audit espera)."""
    print("\n=== H. Consistência interna, agregada por dataset ===")
    raw_catalog = data_source.build_raw_catalog()
    severos, leves = normalization.run_consistency_audit(raw_catalog)
    sev_by_dataset = Counter(item_dataset for _, item_dataset, _ in severos)
    leve_by_dataset = Counter(item_dataset for _, item_dataset, _ in leves)
    for dataset in sorted(set(sev_by_dataset) | set(leve_by_dataset)):
        print(f"  {dataset:16s} severos={sev_by_dataset.get(dataset, 0):>4}  "
              f"leves={leve_by_dataset.get(dataset, 0):>4}")


def calibrate_guardrail_thresholds(df: pd.DataFrame, allowed_dest_octants,
                                    min_pool_floor: int | None = None) -> dict:
    """
    Varre candidatos de limiar de arousal (percentis da distribuição REAL, não
    valores fixos herdados do sintético) e escolhe o mais protetor (mais baixo)
    que ainda mantém o pool mínimo, em toda a grade, acima de min_pool_floor.

    Critério: entre os candidatos que NÃO produzem pool abaixo do piso em nenhum
    contexto da grade, escolher o de maior proteção (menor limiar de arousal).
    Como arousal_percentiles está em ordem crescente, o PRIMEIRO candidato válido já
    é o mais protetor -- não é preciso continuar a busca depois de achá-lo. Se
    nenhum candidato atender ao piso, reportar o melhor compromisso e sinalizar para
    revisão manual -- nunca escolher um valor silenciosamente inseguro.
    """
    min_pool_floor = sysrec.TOP_K if min_pool_floor is None else min_pool_floor
    arousal_percentiles = [50, 60, 70, 75, 80, 85, 90]
    candidates = [float(np.percentile(df["Arousal"], p)) for p in arousal_percentiles]

    print("=== Calibração do limiar de arousal (guardrail, R1) ===")
    print(f"{'percentil':>10} {'limiar':>8} {'bloqueados (baixa energia)':>28} "
          f"{'pool mínimo pós-guardrail':>28} {'contextos < piso':>18}")

    best = None
    for pct, threshold in zip(arousal_percentiles, candidates):
        blocked_total, checked_total = 0, 0
        worst_pool = None
        below_floor = 0

        for curr in range(1, 9):
            for dest in allowed_dest_octants:
                for t in sysrec.CONTEXT_DURATIONS:
                    elig = df.index[df["Duracao"] <= t].to_numpy()
                    if len(elig) == 0:
                        continue
                    safe = elig
                    if curr in sysrec.LOW_ENERGY_OCTANTS:
                        arousal = df.loc[elig, "Arousal"].to_numpy()
                        mask = arousal <= threshold
                        checked_total += len(elig)
                        blocked_total += int((~mask).sum())
                        safe = elig[mask] if mask.any() else elig

                    n = min(sysrec.CANDIDATE_POOL_SIZE, len(safe))
                    if worst_pool is None or n < worst_pool:
                        worst_pool = n
                    if n < min_pool_floor:
                        below_floor += 1

        blocked_rate = blocked_total / checked_total if checked_total else 0.0
        print(f"{pct:>10} {threshold:>8.3f} {blocked_rate:>27.1%} "
              f"{worst_pool:>28} {below_floor:>18}")

        if below_floor == 0 and best is None:
            # Primeiro candidato válido na varredura ascendente = mais protetor.
            best = {"threshold": threshold, "percentile": pct, "worst_pool": worst_pool}

    if best is None:
        print("\n[ATENÇÃO] Nenhum candidato manteve o piso de pool em toda a grade.")
        print("Revisar manualmente: considerar aumentar CANDIDATE_POOL_SIZE, reduzir")
        print("min_pool_floor, ou aceitar que alguns contextos terão menos que TOP_K opções.")
    else:
        print(f"\nLimiar de arousal escolhido: {best['threshold']:.3f} "
              f"(percentil {best['percentile']}, pool mínimo garantido {best['worst_pool']})")

    return best


def calibrate_valence_threshold(df: pd.DataFrame, allowed_dest_octants,
                                 min_pool_floor: int | None = None) -> dict:
    """
    Varredura análoga a calibrate_guardrail_thresholds, para o limiar de valência
    aversiva (regra R4: bloqueia item com Valencia < limiar quando o destino tem
    valência positiva). Percentis BAIXOS da distribuição real de valência (5, 10,
    15, 20) -- ao contrário do arousal, aqui um limiar MAIOR (menos negativo) é mais
    protetor (bloqueia mais itens aversivos), então o critério de seleção é
    invertido: entre os candidatos válidos, o ÚLTIMO da varredura ascendente é o
    mais protetor (não o primeiro, como em calibrate_guardrail_thresholds).
    """
    min_pool_floor = sysrec.TOP_K if min_pool_floor is None else min_pool_floor
    valence_percentiles = [5, 10, 15, 20]
    candidates = [float(np.percentile(df["Valencia"], p)) for p in valence_percentiles]

    print("\n=== Calibração do limiar de valência aversiva (guardrail, R4) ===")
    print(f"{'percentil':>10} {'limiar':>8} {'bloqueados (destino positivo)':>30} "
          f"{'pool mínimo pós-guardrail':>28} {'contextos < piso':>18}")

    best = None
    for pct, threshold in zip(valence_percentiles, candidates):
        blocked_total, checked_total = 0, 0
        worst_pool = None
        below_floor = 0

        for curr in range(1, 9):
            for dest in allowed_dest_octants:
                tv = sysrec.OCTANT_MAP[dest][0]
                for t in sysrec.CONTEXT_DURATIONS:
                    elig = df.index[df["Duracao"] <= t].to_numpy()
                    if len(elig) == 0:
                        continue
                    safe = elig
                    if tv > 0:
                        valencia = df.loc[elig, "Valencia"].to_numpy()
                        mask = valencia >= threshold
                        checked_total += len(elig)
                        blocked_total += int((~mask).sum())
                        safe = elig[mask] if mask.any() else elig

                    n = min(sysrec.CANDIDATE_POOL_SIZE, len(safe))
                    if worst_pool is None or n < worst_pool:
                        worst_pool = n
                    if n < min_pool_floor:
                        below_floor += 1

        blocked_rate = blocked_total / checked_total if checked_total else 0.0
        print(f"{pct:>10} {threshold:>8.3f} {blocked_rate:>29.1%} "
              f"{worst_pool:>28} {below_floor:>18}")

        if below_floor == 0:
            # Varredura ascendente (limiar cada vez menos negativo, mais protetor):
            # sobrescrever a cada candidato válido para manter o ÚLTIMO (mais alto).
            best = {"threshold": threshold, "percentile": pct, "worst_pool": worst_pool}

    if best is None:
        print("\n[ATENÇÃO] Nenhum candidato manteve o piso de pool em toda a grade.")
        print("Revisar manualmente: considerar aumentar CANDIDATE_POOL_SIZE, reduzir")
        print("min_pool_floor, ou aceitar que alguns contextos terão menos que TOP_K opções.")
    else:
        print(f"\nLimiar de valência escolhido: {best['threshold']:.3f} "
              f"(percentil {best['percentile']}, pool mínimo garantido {best['worst_pool']})")

    return best


def run_full_characterization() -> pd.DataFrame:
    df = sysrec.load_active_catalog()
    if len(df) == 0:
        raise RuntimeError(
            "Catálogo vazio -- confirme se APPROVED_CATEGORIES já foi populada (ver "
            "safety.explore_taxonomy); caso contrário, confirme se os exports em dbs/ "
            "(ou a conexão Mongo) têm itens com Valência/Arousal resolvíveis."
        )

    report_va_distribution(df)
    report_octant_density(df)
    report_duration(df)
    report_curation_survival(df)
    report_confidence_tier(df)
    report_pool_size_baseline(df, sysrec.ALLOWED_DEST_OCTANTS)
    report_normalization_audit()
    report_consistency_audit()

    return df  # devolvido para uso posterior na calibração do guardrail


if __name__ == "__main__":
    catalog = run_full_characterization()
    print()
    calibrate_guardrail_thresholds(catalog, sysrec.ALLOWED_DEST_OCTANTS)
    calibrate_valence_threshold(catalog, sysrec.ALLOWED_DEST_OCTANTS)
