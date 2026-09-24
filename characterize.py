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
from contextlib import contextmanager

import numpy as np
import pandas as pd

import data_source
import normalization
import safety
import sistema_de_recomendacao_v3 as sysrec


def report_catalog_composition(df: pd.DataFrame) -> None:
    """
    Composição do catálogo pós-curadoria, por dataset e por modalidade. Responde
    se o catálogo está dominado por uma única fonte -- o que distorceria cobertura,
    diversidade, baselines e a calibração do guardrail (percentis de uma
    distribuição dominada por um dataset não representam o catálogo como um todo).

    Complementa (não duplica) a seção E (report_curation_survival), que já compara
    bruto vs. curado por dataset usando safety.explore_taxonomy() -- aqui o foco é
    a composição do catálogo JÁ curado, incluindo o cruzamento (dataset, category)
    que a seção E não quebra.
    """
    print("=== Composição do catálogo pós-curadoria ===")
    total = len(df)
    print(f"Total: {total} itens\n")

    print("Por dataset:")
    by_dataset = df.groupby("dataset").size().sort_values(ascending=False)
    for dataset, n in by_dataset.items():
        print(f"  {dataset:20s} {n:>6} ({n/total:.1%})")

    print("\nPor modalidade:")
    by_modality = df.groupby("tipo_modalidade").size().sort_values(ascending=False)
    for modality, n in by_modality.items():
        print(f"  {modality:20s} {n:>6} ({n/total:.1%})")

    print("\nPor (dataset, category):")
    by_pair = df.groupby(["dataset", "category"]).size().sort_values(ascending=False)
    for (dataset, category), n in by_pair.items():
        print(f"  {dataset:16s} / {category:24s} {n:>6}")

    # Sinalizar concentração excessiva explicitamente, não deixar para o leitor notar
    top_share = by_dataset.iloc[0] / total if total else 0.0
    if top_share > 0.5:
        print(f"\n[ATENÇÃO] '{by_dataset.index[0]}' concentra {top_share:.1%} do "
              f"catálogo. Cobertura, diversidade, baselines e a calibração do "
              f"guardrail refletirão majoritariamente esse dataset, não o conjunto.")


def report_simulator_vocabulary(df: pd.DataFrame) -> None:
    """
    Verifica se os valores referenciados pelos bônus do simulador de feedback
    (simulate_feedback_holdout, único perfil existente -- o perfil "principal"
    anterior foi removido por não ser chamado em nenhum lugar do código) existem
    de fato no catálogo carregado. Um bônus que nunca dispara significa que o
    simulador se comporta diferente do que o código sugere -- não é bug de
    execução (nada quebra), é divergência silenciosa entre intenção e efeito.

    Adaptado à API real: _MODALIDADE_TO_TIPO (data_source.py) mapeia o catálogo
    real para Tipo em PORTUGUÊS ("Vídeo"/"Áudio"/"Imagem"), não para
    "video"/"audio"/"image" como uma leitura literal de tipo_modalidade sugeriria.
    simulate_feedback_holdout já bifurca pela presença da coluna 'category' e usa
    bônus por category (não por Tipo/Indoor) contra o catálogo real -- ver a
    docstring da função em sistema_de_recomendacao_v3.py. Esta checagem confirma
    empiricamente o que aquele comentário já descreve, em vez de reafirmá-lo às
    cegas.
    """
    print("=== Vocabulário de Tipo/category: simulador vs. catálogo real ===")
    real_types = set(df["Tipo"].unique())
    print(f"Tipos presentes no catálogo: {sorted(real_types)}")

    print("\n  simulate_feedback_holdout (único perfil, avaliação/baselines/regret/heatmap):")
    has_category = "category" in df.columns and df["category"].notna().any()
    if has_category:
        real_categories = set(df["category"].dropna().unique())
        expected_cat = {"positive", "neutral"}
        matched_cat = expected_cat & real_categories
        missing_cat = expected_cat - real_categories
        print("    coluna 'category' presente -> usa bônus por category, não por Tipo/Indoor")
        print(f"    categorias no catálogo:    {sorted(real_categories)}")
        print(f"    referenciadas pelos bônus: {sorted(expected_cat)}")
        print(f"    presentes:                 {sorted(matched_cat) or '(nenhum)'}")
        if missing_cat:
            print(f"    [ATENÇÃO] ausentes (bônus nunca dispara para estes valores): "
                  f"{sorted(missing_cat)}")
    else:
        expected_holdout = {"Mindfulness", "Imagem"}
        matched_holdout = expected_holdout & real_types
        missing_holdout = expected_holdout - real_types
        print("    coluna 'category' ausente -> usa bônus por Tipo/Indoor (perfil sintético)")
        print(f"    referenciados pelos bônus: {sorted(expected_holdout)}")
        print(f"    presentes:                 {sorted(matched_holdout) or '(nenhum)'}")
        if missing_holdout:
            print(f"    [ATENÇÃO] ausentes (bônus nunca dispara para estes valores): "
                  f"{sorted(missing_holdout)}")

    if "Indoor" in df.columns:
        indoor_values = set(df["Indoor"].unique())
        print(f"\n  Valores distintos de Indoor: {sorted(indoor_values)}")
        if len(indoor_values) == 1:
            print(f"    [ATENÇÃO] Indoor é constante ({indoor_values.pop()}) -- o bônus "
                  f"de Indoor do holdout (ramo sintético) aplica-se uniformemente a "
                  f"todos os itens, ou a nenhum. Não discrimina. Irrelevante para o "
                  f"catálogo real, que usa o ramo por category quando presente.")


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


def _apply_safety_filter_standalone(df: pd.DataFrame, eligible: np.ndarray,
                                     curr_oct: int, dest_oct: int) -> np.ndarray:
    """Réplica pura de Recommender._apply_safety_filter, sem precisar de uma
    instância de Recommender/Agent -- usado só por ferramentas de diagnóstico.
    Respeita sysrec.USE_SAFETY_FILTER, como o método real, para que
    safety_filter_disabled() (Parte C) tenha efeito sobre ela."""
    if not sysrec.USE_SAFETY_FILTER:
        return eligible

    A = df.loc[eligible, "Arousal"].to_numpy(dtype=np.float32)
    V = df.loc[eligible, "Valencia"].to_numpy(dtype=np.float32)
    tv, ta = sysrec.OCTANT_MAP[dest_oct]

    r1 = (curr_oct in sysrec.LOW_ENERGY_OCTANTS) & (A > sysrec.SAFETY_AROUSAL_THRESHOLD)
    r2 = (curr_oct in sysrec.HIGH_ENERGY_OCTANTS) & (A > sysrec.SAFETY_AROUSAL_THRESHOLD)
    r3 = (ta < 0) & (A > sysrec.SAFETY_AROUSAL_THRESHOLD)
    r4 = (tv > 0) & (V < sysrec.SAFETY_AVERSIVE_VALENCE_THRESHOLD)
    blocked = r1 | r2 | r3 | r4
    safe = eligible[~blocked]
    return safe if len(safe) > 0 else eligible


def report_guardrail_breakdown(df: pd.DataFrame) -> None:
    """
    Decompõe o efeito do guardrail regra por regra, e separadamente por par
    (curr, dest) via _apply_safety_filter_standalone, para localizar exatamente
    qual regra é responsável por um pool pequeno observado em algum contexto.
    Roda sobre o CATÁLOGO INTEIRO (não um contexto só), para responder de onde
    vem o número que está sendo observado como "sobraram poucos itens".
    """
    total = len(df)
    print(f"=== Decomposição do guardrail (catálogo com {total} itens) ===\n")

    valencia = df["Valencia"].to_numpy(dtype=np.float32)
    arousal = df["Arousal"].to_numpy(dtype=np.float32)

    def _count(mask):
        return int(mask.sum())

    print("Regras avaliadas isoladamente sobre o catálogo inteiro:")
    r1 = arousal > sysrec.SAFETY_AROUSAL_THRESHOLD  # aplicável só se curr in LOW_ENERGY
    print(f"  R1 (arousal > {sysrec.SAFETY_AROUSAL_THRESHOLD:.3f}): "
          f"{_count(r1)}/{total} itens ({_count(r1)/total:.1%}) -- "
          f"só afeta curr in {sysrec.LOW_ENERGY_OCTANTS}")
    print(f"  R2 (mesmo limiar de arousal): idêntico a R1 -- "
          f"só afeta curr in {sysrec.HIGH_ENERGY_OCTANTS}")

    for dest in sysrec.ALLOWED_DEST_OCTANTS:
        tv, ta = sysrec.OCTANT_MAP[dest]
        r3 = (ta < 0) & r1  # nenhum ALLOWED_DEST_OCTANTS tem ta<0 hoje -- deve dar 0
        r4 = (tv > 0) & (valencia < sysrec.SAFETY_AVERSIVE_VALENCE_THRESHOLD)
        print(f"  dest={dest} (V={tv:+.2f},A={ta:+.2f}): "
              f"R3={_count(r3)} (esperado 0 se ta>=0)  "
              f"R4={_count(r4)}/{total} ({_count(r4)/total:.1%})")

    print(f"\n  [CHAVE] R4 é a única regra sem restrição de curr_oct -- ela se "
          f"aplica em TODA chamada de recommend() com destino em "
          f"{sysrec.ALLOWED_DEST_OCTANTS}, diferente de R1/R2 que só afetam estados "
          f"específicos. Se R4 sozinha já exclui uma fração grande do catálogo, "
          f"ela é a candidata principal para recalibração.")

    print("\nPor par (curr, dest) -- via _apply_safety_filter_standalone:")
    eligible_all = df.index.to_numpy(dtype=int)
    worst = None
    for curr in range(1, 9):
        for dest in sysrec.ALLOWED_DEST_OCTANTS:
            safe = _apply_safety_filter_standalone(df, eligible_all, curr, dest)
            n_safe = len(safe)
            if worst is None or n_safe < worst[0]:
                worst = (n_safe, curr, dest)
            print(f"  curr={curr} dest={dest}: {n_safe}/{total} sobrevivem")
    print(f"\n  Pior caso: curr={worst[1]} dest={worst[2]} -> {worst[0]} itens")


def report_calibration_staleness(df: pd.DataFrame) -> None:
    """Compara a distribuição ATUAL de valência/arousal com o que os limiares
    configurados implicam -- se o catálogo mudou de composição depois da
    calibração (ex.: safety.auto_approve_clean_categories rodou depois), os
    limiares podem não corresponder mais aos percentis que motivaram sua escolha."""
    v_percentile = float((df["Valencia"] < sysrec.SAFETY_AVERSIVE_VALENCE_THRESHOLD).mean() * 100)
    a_percentile = float((df["Arousal"] > sysrec.SAFETY_AROUSAL_THRESHOLD).mean() * 100)
    print("=== Atualidade da calibração ===")
    print(f"  SAFETY_AVERSIVE_VALENCE_THRESHOLD={sysrec.SAFETY_AVERSIVE_VALENCE_THRESHOLD:.3f} "
          f"hoje corresponde ao percentil {v_percentile:.1f} da valência real "
          f"(bloqueia {v_percentile:.1f}% do catálogo via R4 sozinha)")
    print(f"  SAFETY_AROUSAL_THRESHOLD={sysrec.SAFETY_AROUSAL_THRESHOLD:.3f} hoje "
          f"corresponde ao percentil {100-a_percentile:.1f} do arousal real")
    print("  Se estes percentis divergem muito do que foi PRETENDIDO na "
          "calibração original, o catálogo mudou de composição desde então -- "
          "recalibrar (Parte B), não ajustar os números na mão.")


@contextmanager
def safety_filter_disabled():
    """
    Desliga sysrec.USE_SAFETY_FILTER dentro do bloco `with`, restaurando o valor
    original ao sair -- inclusive em caso de exceção. Uso exclusivo de bancada de
    diagnóstico (characterize.py, comparações com/sem guardrail). NUNCA envolver
    o loop interativo de produção (main()) com isto -- ver teste de aceitação 6
    do plano (busca textual: nunca usado em torno de main()/loop interativo).
    """
    original = sysrec.USE_SAFETY_FILTER
    sysrec.USE_SAFETY_FILTER = False
    try:
        yield
    finally:
        sysrec.USE_SAFETY_FILTER = original


def compare_pool_with_without_guardrail(df: pd.DataFrame, allowed_dest_octants) -> None:
    """
    Compara, célula a célula da grade curr x allowed_dest_octants, o tamanho do
    pool elegível COM e SEM o guardrail geométrico -- quantifica exatamente
    quanto ele está custando em tamanho de pool.

    Adaptado do pseudocódigo original do plano: report_pool_size_baseline (seção
    C) reimplementa a elegibilidade sem checar sysrec.USE_SAFETY_FILTER, então
    envolvê-la em safety_filter_disabled() não teria efeito algum. Em vez disso,
    esta função chama _apply_safety_filter_standalone diretamente -- que respeita
    a flag -- para que o context manager realmente tenha efeito na comparação.
    """
    print("=== Pool COM guardrail vs. SEM guardrail (diagnóstico) ===")
    eligible_all = df.index.to_numpy(dtype=int)
    worst_cost = None
    for curr in range(1, 9):
        for dest in allowed_dest_octants:
            with_guard = len(_apply_safety_filter_standalone(df, eligible_all, curr, dest))
            with safety_filter_disabled():
                without_guard = len(_apply_safety_filter_standalone(df, eligible_all, curr, dest))
            cost = without_guard - with_guard
            if worst_cost is None or cost > worst_cost[0]:
                worst_cost = (cost, curr, dest, with_guard, without_guard)
            print(f"  curr={curr} dest={dest}: com={with_guard:>5}  sem={without_guard:>5}  custo={cost:>5}")
    print(f"\n  Maior custo: curr={worst_cost[1]} dest={worst_cost[2]} -> guardrail remove "
          f"{worst_cost[0]} itens ({worst_cost[3]} restantes de {worst_cost[4]})")


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


@contextmanager
def _override(module, **kwargs):
    """Sobrescreve temporariamente atributos de módulo, restaurando os valores
    originais ao sair -- inclusive em caso de exceção. Generaliza
    safety_filter_disabled (que só cobria USE_SAFETY_FILTER) para qualquer
    combinação de constantes -- mesmo padrão já usado em
    fatigue_diagnostics._override. Uso exclusivo de bancada de diagnóstico."""
    original = {k: getattr(module, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in original.items():
            setattr(module, k, v)


def _worst_pool_with_threshold(df: pd.DataFrame, arousal_threshold: float | None = None,
                                valence_threshold: float | None = None) -> int:
    """Pior pool pós-guardrail (regras R1-R4 completas, via
    _apply_safety_filter_standalone -- não uma reimplementação parcial) na grade
    curr x ALLOWED_DEST_OCTANTS, com um dos limiares hipoteticamente sobrescrito.
    Usado só para checar viabilidade de um candidato em report_threshold_tradeoff;
    nunca aplica o limiar de fato (restaura ao sair, via _override)."""
    overrides = {}
    if arousal_threshold is not None:
        overrides["SAFETY_AROUSAL_THRESHOLD"] = float(arousal_threshold)
    if valence_threshold is not None:
        overrides["SAFETY_AVERSIVE_VALENCE_THRESHOLD"] = float(valence_threshold)

    eligible_all = df.index.to_numpy(dtype=int)
    worst = None
    with _override(sysrec, **overrides):
        for curr in range(1, 9):
            for dest in sysrec.ALLOWED_DEST_OCTANTS:
                safe = _apply_safety_filter_standalone(df, eligible_all, curr, dest)
                n = min(sysrec.CANDIDATE_POOL_SIZE, len(safe))
                if worst is None or n < worst:
                    worst = n
    return worst


def report_threshold_tradeoff(df: pd.DataFrame) -> None:
    """
    Ferramenta de calibração atual (substitui uma varredura por percentil usada
    antes, removida por ser circular -- ver histórico). NÃO escolhe nada --
    apresenta a curva limiar x consequência para uma decisão humana informada, e
    checa viabilidade (pior pool na grade real). O limiar é uma decisão de
    segurança ancorada na semântica da escala afetiva ([-1, 1], psicometricamente
    validada), não um percentil da distribuição corrente do catálogo: calibrar por
    percentil é circular (o limiar passa a ser definido pelos dados que deveria
    filtrar, e se move sempre que a composição do catálogo muda -- foi exatamente
    isso que invalidou a calibração anterior quando
    safety.auto_approve_clean_categories rodou).
    """
    print("=== Limiar de arousal: trade-off ===")
    print(f"{'limiar':>8} {'% catálogo bloqueado':>22} {'itens restantes':>18} "
          f"{'pior pool na grade':>20}")
    for threshold in np.arange(-0.4, 0.85, 0.1):
        threshold = float(threshold)
        blocked = df["Arousal"] > threshold
        remaining = len(df) - int(blocked.sum())
        worst_pool = _worst_pool_with_threshold(df, arousal_threshold=threshold)
        viable = "" if worst_pool >= sysrec.TOP_K else "   <-- INVIÁVEL"
        print(f"{threshold:>8.2f} {blocked.mean():>21.1%} {remaining:>18} "
              f"{worst_pool:>20}{viable}")
    print("\n  Escolha o limiar pela semântica afetiva pretendida (que nível de "
          "ativação é excessivo para alguém em baixa energia?), usando a coluna "
          "de viabilidade apenas para descartar valores impraticáveis.")

    print("\n=== Limiar de valência aversiva: trade-off ===")
    print(f"{'limiar':>8} {'% catálogo bloqueado':>22} {'itens restantes':>18} "
          f"{'pior pool na grade':>20}")
    for threshold in np.arange(-0.9, 0.05, 0.1):
        threshold = float(threshold)
        blocked = df["Valencia"] < threshold
        remaining = len(df) - int(blocked.sum())
        worst_pool = _worst_pool_with_threshold(df, valence_threshold=threshold)
        viable = "" if worst_pool >= sysrec.TOP_K else "   <-- INVIÁVEL"
        print(f"{threshold:>8.2f} {blocked.mean():>21.1%} {remaining:>18} "
              f"{worst_pool:>20}{viable}")
    print("\n  Escolha o limiar pela semântica afetiva pretendida (que grau de "
          "valência aversiva contradiz o objetivo de um destino de valência "
          "positiva?), usando a coluna de viabilidade apenas para descartar "
          "valores impraticáveis.")


def verify_guardrail_effective(df: pd.DataFrame, recommender) -> None:
    """
    Confirma que o guardrail de fato bloqueia algo nos estados que deveria
    proteger. Um guardrail que bloqueia 0 itens não está protegendo -- foi
    exatamente esse o sintoma observado com os limiares herdados do sintético
    (0/210 bloqueados para o oitante 5, ver sanity check [4] de sanity_checks).
    """
    print("=== Verificação de efetividade do guardrail ===")
    for curr in list(sysrec.LOW_ENERGY_OCTANTS) + list(sysrec.HIGH_ENERGY_OCTANTS):
        for dest in sysrec.ALLOWED_DEST_OCTANTS:
            eligible = df.index.to_numpy(dtype=int)
            before = len(eligible)
            after = len(recommender._apply_safety_filter(eligible, curr, dest))
            blocked = before - after
            flag = "" if blocked > 0 else "   <-- NADA BLOQUEADO"
            print(f"  curr={curr} dest={dest}: {blocked:>4}/{before} bloqueados{flag}")


def report_feature_degeneracy(df: pd.DataFrame, feature_space) -> None:
    """
    Mede quanto cada bloco de features de fato distingue itens. Uma feature
    constante contribui zero para a similaridade de cosseno usada pelo MMR --
    se a maioria dos blocos for constante, o MMR não tem o que diversificar, e
    reponderar MMR_LAMBDA não resolveria nada. Diagnóstico para a diversidade
    intra-lista observada em coverage_and_diversity (--eval): decide entre
    reformular features (se os vetores forem majoritariamente idênticos) e
    reponderar MMR/ampliar CANDIDATE_POOL_SIZE (se os vetores forem distintos mas
    o pool típico ainda for muito homogêneo) -- ver Tarefa 6 do plano.
    """
    print("=== Degenerescência das features ===")
    matrix = feature_space.item_matrix
    n_types, n_tags = len(feature_space.all_types), len(feature_space.all_tags)

    blocks = {
        "one-hot Tipo":    (0, n_types),
        "Indoor":          (n_types, n_types + 1),
        "one-hot Tag":     (n_types + 1, n_types + 1 + n_tags),
        "Duracao norm":    (n_types + 1 + n_tags, n_types + 2 + n_tags),
        "Valencia norm":   (n_types + 2 + n_tags, n_types + 3 + n_tags),
        "Arousal norm":    (n_types + 3 + n_tags, n_types + 4 + n_tags),
    }
    for name, (lo, hi) in blocks.items():
        block = matrix[:, lo:hi]
        n_distinct = len(np.unique(block, axis=0))
        variance = float(block.var(axis=0).mean())
        flag = "  <-- CONSTANTE" if n_distinct <= 1 else ""
        print(f"  {name:16s} dim={hi-lo:>3}  valores distintos={n_distinct:>5}  "
              f"variância média={variance:.5f}{flag}")

    print(f"\n  Vetores de item completamente idênticos: "
          f"{len(matrix) - len(np.unique(matrix, axis=0))} de {len(matrix)}")

    # Similaridade média entre pares dentro de um pool típico
    sample_idx = np.random.choice(len(matrix), size=min(sysrec.CANDIDATE_POOL_SIZE, len(matrix)),
                                   replace=False)
    sims = [
        sysrec._cosine_similarity(matrix[i], matrix[j])
        for i in sample_idx for j in sample_idx if i < j
    ]
    print(f"  Similaridade cosseno média num pool aleatório de "
          f"{len(sample_idx)} itens: {np.mean(sims):.4f}")
    print("  (próximo de 1.0 = itens quase indistinguíveis; MMR não tem o que fazer)")


def run_full_characterization() -> pd.DataFrame:
    df = sysrec.load_active_catalog()
    if len(df) == 0:
        raise RuntimeError(
            "Catálogo vazio -- confirme se APPROVED_CATEGORIES já foi populada (ver "
            "safety.explore_taxonomy); caso contrário, confirme se os exports em dbs/ "
            "(ou a conexão Mongo) têm itens com Valência/Arousal resolvíveis."
        )

    report_catalog_composition(df)
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
    report_simulator_vocabulary(catalog)
    print()
    report_guardrail_breakdown(catalog)
    print()
    report_calibration_staleness(catalog)
    print()
    compare_pool_with_without_guardrail(catalog, sysrec.ALLOWED_DEST_OCTANTS)
    print()
    # report_threshold_tradeoff é a ferramenta de calibração atual: apresenta a
    # curva limiar x consequência para decisão humana ancorada na semântica da
    # escala afetiva, não escolhe nada sozinha (a varredura por percentil antiga
    # foi removida por ser circular -- ver histórico do repositório).
    report_threshold_tradeoff(catalog)
    print()

    # Verificação de efetividade contra os limiares ATUAIS de
    # sysrec.SAFETY_AROUSAL_THRESHOLD/SAFETY_AVERSIVE_VALENCE_THRESHOLD (não os
    # candidatos calibrados acima -- aplicar os valores escolhidos no código e
    # rodar de novo para confirmar).
    _feature_space = sysrec.FeatureSpace(catalog)
    _agent = sysrec.Agent(catalog, _feature_space)
    _recommender = sysrec.Recommender(catalog, _feature_space, _agent)
    verify_guardrail_effective(catalog, _recommender)
    print()
    report_feature_degeneracy(catalog, _feature_space)
    print()

    # Confirmação final (ordem de execução, passo 4 do plano de diagnóstico do
    # guardrail): reconfere a decomposição contra os limiares ATUAIS já aplicados
    # no código, não os candidatos calibrados acima.
    report_guardrail_breakdown(catalog)
