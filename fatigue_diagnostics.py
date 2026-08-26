"""
fatigue_diagnostics.py — BANCADA DE DIAGNÓSTICO. Não faz parte do caminho de
produção. Mede o comportamento real de espaçamento de recomendações -- para saber
o que de fato está acontecendo (gaps entre reaparições do mesmo item, força
relativa da penalidade de fadiga frente ao gap de score real), não supor.

Roda contra o catálogo carregado via sysrec.load_active_catalog() (qualquer
DATA_BACKEND). Usado antes e depois da correção de espaçamento com garantia
mínima (FATIGUE_MIN_GAP, ver FatigueTracker.blocked_mask) para confirmar, com
números reais, o problema e a correção.

IMPORTANTE: o cooldown de fadiga só se aplica a itens que o usuário de fato
EXECUTA (FatigueTracker.mark_executed) -- recommend() sozinho não marca mais nada.
Este script simula que o usuário sempre executa a recomendação do slot 1 (a
"melhor", já que agora é sempre o argmax determinístico -- ver
Recommender._select_slots), chamando mark_executed nela após cada rodada. Por
isso "gaps_top1" reflete o comportamento real do espaçamento; "gaps_any_slot"
inclui itens de slots 2/3 que nunca foram "executados" nesta simulação e por
construção não têm cooldown nenhum -- útil para contraste, não para medir o
mecanismo de espaçamento em si.

Uso:
    python fatigue_diagnostics.py
"""
import random
from collections import defaultdict

import numpy as np

import sistema_de_recomendacao_v3 as sysrec


def trace_repeats(recommender: sysrec.Recommender, contexts: list, k: int = None) -> dict:
    """
    Roda recommend() para uma sequência de contextos (fixa ou variada), simulando
    que o usuário sempre executa a recomendação do slot 1 (mark_executed -- mesmo
    caminho que _run_interaction usa de verdade). Registra em que índice de
    interação cada item apareceu -- em qualquer slot da lista, e separadamente só
    no slot 1 (o mais consequente, e o único "executado" nesta simulação) -- e
    calcula os intervalos (gaps) entre aparições consecutivas do mesmo item.
    """
    k = sysrec.TOP_K if k is None else k
    appearances = defaultdict(list)
    appearances_top1 = defaultdict(list)

    for i, (curr, dest, t) in enumerate(contexts):
        results = recommender.recommend(curr, dest, t, k=k)
        for r in results:
            appearances[r["item_idx"]].append(i)
        if results:
            appearances_top1[results[0]["item_idx"]].append(i)
            recommender.fatigue.mark_executed(results[0]["item_idx"])

    def _gaps(appearance_dict):
        out = []
        for idxs in appearance_dict.values():
            out.extend(b - a for a, b in zip(idxs, idxs[1:]))
        return out

    return {
        "gaps_any_slot": _gaps(appearances),
        "gaps_top1": _gaps(appearances_top1),
    }


def _print_gap_histogram(label: str, gaps: list) -> None:
    print(f"  {label}:")
    if not gaps:
        print("    (sem repetições observadas na amostra)")
        return
    gaps = np.array(gaps)
    print(f"    n={len(gaps)}  min={gaps.min()}  mediana={np.median(gaps):.1f}  "
          f"média={gaps.mean():.1f}")
    print(f"    gap=1 (reaparece na PRÓXIMA interação): {(gaps == 1).mean():.1%}")
    print(f"    gap<=2: {(gaps <= 2).mean():.1%}   gap<=3: {(gaps <= 3).mean():.1%}")


def score_gap_analysis(recommender: sysrec.Recommender, contexts: list, n_samples: int = 100) -> None:
    """
    Compara a penalidade MÁXIMA possível de fadiga (FATIGUE_LAMBDA, em delta=0)
    contra o gap real de score entre o 1º e o 2º colocado do pool, em contextos
    reais. Responde: a penalidade de fadiga é, sequer em tese, grande o
    suficiente para mudar o item escolhido? Se o gap de score for tipicamente
    maior que FATIGUE_LAMBDA, a resposta é não -- e isso explica por que a
    fadiga suave pode não impedir repetição imediata.
    """
    gaps = []
    for curr, dest, t in contexts[:n_samples]:
        eligible = recommender.df.index[recommender.df["Duracao"] <= t].to_numpy(dtype=int)
        if len(eligible) == 0:
            continue
        eligible = recommender._apply_safety_filter(eligible, curr, dest)
        pool_idx, pool_dist = recommender._candidate_pool(eligible, curr, dest)
        if len(pool_idx) < 2:
            continue
        user_state = recommender.feature_space.user_state(curr, dest, t)
        q_dist = recommender.agent.q_distribution(user_state, pool_idx)
        q_values = (q_dist * recommender.agent.support_np).sum(axis=1)
        score = recommender._hybrid_score(pool_dist, q_values, q_dist)
        top2 = np.sort(score)[::-1][:2]
        gaps.append(float(top2[0] - top2[1]))

    gaps = np.array(gaps)
    if len(gaps) == 0:
        print("  (nenhum contexto com pool >= 2 -- sem dados para comparar)")
        return
    print(f"  Gap de score 1º-2º colocado: média={gaps.mean():.4f}  "
          f"mediana={np.median(gaps):.4f}")
    print(f"  FATIGUE_LAMBDA (penalidade máxima possível): {sysrec.FATIGUE_LAMBDA}")
    print(f"  Fração de contextos onde a penalidade máxima NÃO supera o gap de "
          f"score (ou seja, a fadiga não teria força para trocar o 1º colocado): "
          f"{(gaps >= sysrec.FATIGUE_LAMBDA).mean():.1%}")


def run_fatigue_diagnostics(recommender: sysrec.Recommender, df) -> None:
    print("=== Diagnóstico 1: curva de decaimento da penalidade suave ===")
    for delta in range(0, 16):
        p = sysrec.FATIGUE_LAMBDA * 0.5 ** (delta / sysrec.FATIGUE_HALFLIFE)
        print(f"  delta={delta:>2}: penalidade={p:.4f}")

    fixed_ctx = [(5, 7, 30)] * 50
    print("\n=== Diagnóstico 2: contexto FIXO repetido (pior caso), 50 chamadas ===")
    result = trace_repeats(recommender, fixed_ctx)
    _print_gap_histogram("top-1", result["gaps_top1"])
    _print_gap_histogram("qualquer slot", result["gaps_any_slot"])

    random_ctx = [(random.randint(1, 8), random.choice(sysrec.ALLOWED_DEST_OCTANTS),
                   random.choice([15, 30, 60])) for _ in range(200)]
    print("\n=== Diagnóstico 3: contextos VARIADOS (caso médio), 200 chamadas ===")
    result = trace_repeats(recommender, random_ctx)
    _print_gap_histogram("top-1", result["gaps_top1"])
    _print_gap_histogram("qualquer slot", result["gaps_any_slot"])

    print("\n=== Diagnóstico 4: penalidade de fadiga vs. gap real de score ===")
    score_gap_analysis(recommender, fixed_ctx + random_ctx)


if __name__ == "__main__":
    sysrec.set_seed(sysrec.SEED)
    df = sysrec.load_active_catalog()
    feature_space = sysrec.FeatureSpace(df)
    agent = sysrec.Agent(df, feature_space)
    recommender = sysrec.Recommender(df, feature_space, agent)
    run_fatigue_diagnostics(recommender, df)
