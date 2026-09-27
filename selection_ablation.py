"""
selection_ablation.py — BANCADA DE AVALIAÇÃO da seleção dos slots da lista. Não
faz parte do caminho de produção: nunca é chamado por main(), nunca chama
Agent.save(), nunca escreve no log de interações e nunca treina o agente durante a
comparação (o agente fica congelado -- ver _frozen_agent).

Acionado por:
    python sistema_de_recomendacao_v3.py --eval-selection

O que compara (Recommender._select_slots, SELECTION_MODE):
  greedy        -> top-k determinístico por `adjusted`
  no_mmr        -> slot 1 argmax + slot exploratório + guloso nos demais
  full@<lambda> -> slot 1 argmax + slot exploratório + MMR nos demais, com a
                   varredura MMR_LAMBDA_SWEEP (full@MMR_LAMBDA é o modo de produção)
Todos compartilham o pipeline inteiro até o `adjusted` score -- a única variável
é a seleção. Os baselines e o multi-seed medem o argmax de UM item; o MMR só atua
na lista de 3, então é aqui que ele é medido.

Divergências em relação à especificação (adaptadas à interface real):
  - O slot 1 NÃO é sorteado por softmax: desde o commit que fixou o melhor item em
    primeiro lugar, é sempre o argmax de `adjusted` (ver _select_slots). Por isso
    "greedy" e "no_mmr" só diferem quando o slot exploratório é sorteado.
  - Só a CAUDA (posições 1..k-1) é embaralhada quando há slot exploratório; o slot
    1 fica sempre na posição 0. ev_slot1 usa o slot_type para localizá-lo mesmo
    assim.
  - `deterministic=True` continua ortogonal ao modo (desliga só a exploração) em
    vez de virar atalho para "greedy" -- o sanity check [3] de --eval depende dessa
    semântica, e trocá-la mudaria a saída de referência.

Protocolo P1 (principal): contextos, afinidades latentes, uniformes de escolha e
uma seed de seleção por contexto são pré-gerados por réplica (números aleatórios
comuns). Cada configuração tem sua própria instância de Recommender e cada
chamada roda em seeded_scope(seed de seleção do contexto), com
register_fatigue=False: partindo do mesmo estado do gerador, full e no_mmr sorteiam
o MESMO slot exploratório -- só os slots que o MMR decide podem diferir.
O protocolo P2 (sessão sequencial com fadiga) não foi implementado: a
especificação o condiciona ao resultado de P1.
"""
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import multiseed
import sistema_de_recomendacao_v3 as sysrec

AFFINITY_SCALES = (0.0, 0.1, 0.25, 0.5)   # 0.0 = sem heterogeneidade de preferência
CHOOSER_TEMPERATURE = 0.2
MMR_LAMBDA_SWEEP = (1.0, 0.9, 0.8, 0.7, 0.5, 0.3)
LIST_EVAL_N_CONTEXTS = 200            # contextos por réplica no protocolo P1
# Regime quente: episódios de treino do agente (mesmo procedimento de
# multiseed.run_replicate), antes de congelá-lo. Abaixo de N_EPISODES_PER_REPLICATE
# por custo; bem acima de WARMUP_INTERACTIONS, então a DQN já domina o score
# híbrido e o guardrail probabilístico já está ativo.
LIST_EVAL_WARM_TRAIN_EPISODES = 1000
INERTIA_CALLS_PER_CONTEXT = 5
SELECTION_ABLATION_RESULTS_DIR = "results/selection_ablation"

REGIMES = ("cold", "warm")
_SEL_SPAWN_KEY = 1000   # fluxo das seeds de seleção por contexto
_CTX_SPAWN_KEY = 1001   # fluxo dos contextos/afinidades/uniformes de escolha


def selection_configs(lambdas=MMR_LAMBDA_SWEEP) -> list[tuple[str, str, float | None]]:
    """(nome, modo, lambda). full@<MMR_LAMBDA> é o modo de produção."""
    configs = [("greedy", "greedy", None), ("no_mmr", "no_mmr", None)]
    configs += [(f"full@{lam:g}", "full", lam) for lam in lambdas]
    return configs


def _modality_col(df: pd.DataFrame) -> str:
    return "tipo_modalidade" if "tipo_modalidade" in df.columns else "Tipo"


def check_range_index(df: pd.DataFrame) -> None:
    """A tabela de valor esperado e FeatureSpace.item_matrix são indexadas por
    posição; item_idx só coincide com posição se df.index == RangeIndex."""
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError(
            "selection_ablation exige df.index == RangeIndex(len(df)): a tabela de "
            "valor esperado (multiseed.build_expected_value_table) e "
            "FeatureSpace.item_matrix são indexadas por posição, e item_idx só "
            "coincide com a posição nesse caso. Rode df.reset_index(drop=True) na carga."
        )


# ---------- Métricas de lista ----------

def list_metrics(items: list[int], slot_types: list[str], df: pd.DataFrame, feature_space,
                 ev_row: np.ndarray, pool_idx: np.ndarray) -> dict:
    """
    items: item_idx da lista retornada por recommend(); ev_row: valor esperado do
    contexto por posição em df (build_expected_value_table()[(curr, dest)]);
    pool_idx: pool de candidatos antes da seleção -- o teto de diversidade possível
    (se o pool só tem uma modalidade, nenhum modo pode passar de 1).
    """
    vecs = feature_space.item_matrix[items]
    pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]
    cos = [sysrec._cosine_similarity(vecs[i], vecs[j]) for i, j in pairs]
    va = df.loc[items, ["Valencia", "Arousal"]].to_numpy()
    mod_col = _modality_col(df)
    ev = ev_row[items]
    slot1 = slot_types.index("greedy") if "greedy" in slot_types else 0
    return {
        # diversidade no espaço de features que o MMR usa (mesmo cálculo de
        # coverage_and_diversity)
        "ild_features": 1.0 - float(np.mean(cos)) if cos else 0.0,
        # diversidade interpretável, independente do espaço de features
        "n_modalities": int(df.loc[items, mod_col].nunique()),
        "n_datasets": int(df.loc[items, "dataset"].nunique()) if "dataset" in df.columns else 0,
        "n_categories": int(df.loc[items, "category"].nunique()) if "category" in df.columns else 0,
        "va_spread": float(np.mean([np.linalg.norm(va[i] - va[j]) for i, j in pairs])) if pairs else 0.0,
        # relevância, pelo valor esperado do simulador holdout
        "ev_mean": float(ev.mean()),
        "ev_slot1": float(ev[slot1]),
        "ev_best": float(ev.max()),
        # teto de diversidade existente no pool
        "pool_n_modalities": int(df.loc[pool_idx, mod_col].nunique()),
    }


# ---------- Modelo de escolha com preferência latente ----------

def choice_metrics(items: list[int], df: pd.DataFrame, ev_row: np.ndarray,
                   affinity_by_modality: dict, u_choice: float) -> dict:
    """
    Usuário anônimo com afinidade latente por modalidade: utilidade de cada item =
    valor esperado + afinidade pela modalidade dele; escolha por softmax
    (CHOOSER_TEMPERATURE) via transformação inversa de um uniforme pré-gerado -- o
    mesmo uniforme para todos os modos (CRN). É uma HIPÓTESE sobre o usuário, não
    um dado: reportar sempre com a varredura completa de AFFINITY_SCALES.
    """
    mods = df.loc[items, _modality_col(df)].to_numpy()
    utility = ev_row[items] + np.array([affinity_by_modality[m] for m in mods])
    p = sysrec._softmax(utility, CHOOSER_TEMPERATURE)
    chosen = int(np.searchsorted(np.cumsum(p), u_choice, side="right"))
    chosen = min(chosen, len(items) - 1)
    return {"chosen_utility": float(utility[chosen]), "best_utility": float(utility.max())}


# ---------- Agente congelado ----------

@contextmanager
def _frozen_agent(agent):
    """Durante a comparação o agente NÃO aprende: se aprendesse, cada modo o
    alimentaria com dados diferentes e a comparação mediria dinâmica de treino em
    vez de seleção. Qualquer chamada a learn_from_feedback levanta."""
    def _refuse(*args, **kwargs):
        raise RuntimeError("learn_from_feedback chamado durante a ablação de seleção "
                           "-- o agente deveria estar congelado.")
    agent.learn_from_feedback = _refuse
    try:
        yield agent
    finally:
        del agent.learn_from_feedback   # remove a sobrescrita de instância


# ---------- Contador de inércia sobre a configuração atual ----------

def run_inertia_check(df: pd.DataFrame, feature_space, agent,
                      mmr_lambda: float | None = None,
                      n_calls: int = INERTIA_CALLS_PER_CONTEXT) -> dict:
    """Grade curr (1..8) x dest (ALLOWED_DEST_OCTANTS), n_calls por contexto, modo
    "full", register_fatigue=False. Responde a pergunta mais barata: o MMR está
    fazendo alguma coisa hoje? Também mede o teto de diversidade do pool."""
    time_avail = feature_space.max_duration
    rec = sysrec.Recommender(df, feature_space, agent)
    mod_col = _modality_col(df)
    pool_mods = []
    with _frozen_agent(agent):
        for curr in range(1, 9):
            for dest in sysrec.ALLOWED_DEST_OCTANTS:
                for _ in range(n_calls):
                    rec.recommend(curr, dest, time_avail, register_fatigue=False,
                                  selection_mode="full", mmr_lambda=mmr_lambda)
                    pool_mods.append(int(df.loc[rec.last_pool_idx, mod_col].nunique()))
    report = rec.mmr_report()
    report["pool_n_modalities_mean"] = float(np.mean(pool_mods))
    report["pool_single_modality_rate"] = float(np.mean(np.array(pool_mods) == 1))
    return report


# ---------- Protocolo P1 ----------

def pregenerate_contexts(root_entropy: int, replicate_id: int, n_contexts: int,
                         modalities: list[str]) -> dict:
    """Streams próprios via SeedSequence(root_entropy, spawn_key=...) -- NÃO usa
    spawn() sobre objetos do plano de seeds do multi-seed (spawn() altera o
    contador interno do objeto e acoplaria os dois módulos)."""
    ctx_rng = np.random.default_rng(
        np.random.SeedSequence(root_entropy, spawn_key=(replicate_id, _CTX_SPAWN_KEY)))
    sel_ss = np.random.SeedSequence(root_entropy, spawn_key=(replicate_id, _SEL_SPAWN_KEY))
    return {
        "curr": ctx_rng.integers(1, 9, size=n_contexts),
        "dest": ctx_rng.choice(np.array(sysrec.ALLOWED_DEST_OCTANTS), size=n_contexts),
        # z ~ N(0,1) por (contexto, modalidade); afinidade na escala s = s * z --
        # as escalas são versões da MESMA preferência latente (CRN entre escalas).
        "z_affinity": ctx_rng.standard_normal((n_contexts, len(modalities))),
        "u_choice": ctx_rng.random(n_contexts),
        "selection_seed": sel_ss.generate_state(n_contexts, dtype=np.uint32),
        "modalities": list(modalities),
    }


def run_protocol_p1(df: pd.DataFrame, feature_space, agent, contexts: dict,
                    ev_table: dict, configs: list) -> dict:
    """
    Para cada contexto e cada configuração: recommend() em
    seeded_scope(seed de seleção do contexto), register_fatigue=False, uma
    instância de Recommender por configuração. Retorna, por configuração, arrays
    por contexto de cada métrica e o relatório do contador de inércia.
    """
    time_avail = feature_space.max_duration
    n = len(contexts["curr"])
    recommenders = {name: sysrec.Recommender(df, feature_space, agent) for name, _, _ in configs}
    rows = {name: [] for name, _, _ in configs}
    slot_traces = {name: [] for name, _, _ in configs}

    with _frozen_agent(agent):
        for i in range(n):
            curr, dest = int(contexts["curr"][i]), int(contexts["dest"][i])
            ev_row = ev_table[(curr, dest)]
            affinities = {
                scale: {m: scale * float(contexts["z_affinity"][i, j])
                        for j, m in enumerate(contexts["modalities"])}
                for scale in AFFINITY_SCALES
            }
            for name, mode, lam in configs:
                rec = recommenders[name]
                with multiseed.seeded_scope(int(contexts["selection_seed"][i])):
                    results = rec.recommend(curr, dest, time_avail, register_fatigue=False,
                                            selection_mode=mode, mmr_lambda=lam)
                items = [r["item_idx"] for r in results]
                slot_types = [r["slot_type"] for r in results]
                metrics = list_metrics(items, slot_types, df, feature_space, ev_row,
                                       rec.last_pool_idx)
                for scale, aff in affinities.items():
                    cm = choice_metrics(items, df, ev_row, aff, float(contexts["u_choice"][i]))
                    metrics[f"chosen_utility@{scale:g}"] = cm["chosen_utility"]
                    metrics[f"best_utility@{scale:g}"] = cm["best_utility"]
                rows[name].append(metrics)
                slot_traces[name].append(list(zip(items, slot_types,
                                                  [r["propensity"] for r in results])))

    out = {}
    for name, _, _ in configs:
        keys = rows[name][0].keys()
        out[name] = {
            "metrics": {k: np.array([r[k] for r in rows[name]], dtype=np.float64) for k in keys},
            "mmr": recommenders[name].mmr_report(),
            "slots": slot_traces[name],
        }
    return out


# ---------- Estatística ----------

_MAIN_METRICS = ("ild_features", "n_modalities", "n_datasets", "va_spread",
                 "ev_mean", "ev_slot1", "pool_n_modalities")


def _paired(per_rep_a: list, per_rep_b: list, label_a: str, label_b: str) -> dict:
    return multiseed.paired_comparison(np.array(per_rep_a), np.array(per_rep_b), label_a, label_b)


def summarize(replicates: list[dict], configs: list, n_replicates: int) -> dict:
    """A unidade de análise é a réplica: por réplica, a média de cada métrica por
    configuração; entre réplicas, média/desvio e comparações pareadas."""
    prod = f"full@{sysrec.MMR_LAMBDA:g}"
    summary = {}
    for regime in REGIMES:
        per_rep = {}   # config -> metric -> [média por réplica]
        for rep in replicates:
            for name, _, _ in configs:
                for metric, arr in rep[regime][name]["metrics"].items():
                    per_rep.setdefault(name, {}).setdefault(metric, []).append(float(arr.mean()))
        agg = {name: {m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                      for m, v in metrics.items()}
               for name, metrics in per_rep.items()}

        comparisons = {}
        if n_replicates > 1 and prod in per_rep:
            metrics_to_compare = list(_MAIN_METRICS) + [f"chosen_utility@{s:g}" for s in AFFINITY_SCALES]
            for a, b in ((prod, "no_mmr"), ("no_mmr", "greedy"), (prod, "greedy")):
                comparisons[f"{a} - {b}"] = {
                    m: _paired(per_rep[a][m], per_rep[b][m], a, b) for m in metrics_to_compare
                }

        mmr = {}
        for name, mode, _ in configs:
            if mode != "full":
                continue
            reports = [rep[regime][name]["mmr"] for rep in replicates]
            mmr[name] = {k: (float(np.mean([r[k] for r in reports if r[k] is not None]))
                             if any(r[k] is not None for r in reports) else None)
                         for k in reports[0]}

        equivalence_ok = all(
            np.array_equal(rep[regime]["full@1"]["metrics"][m], rep[regime]["no_mmr"]["metrics"][m])
            for rep in replicates for m in rep[regime]["no_mmr"]["metrics"]
        ) if "full@1" in per_rep and "no_mmr" in per_rep else None

        summary[regime] = {"aggregate": agg, "comparisons": comparisons, "mmr_inertia": mmr,
                           "lambda_1_equals_no_mmr": equivalence_ok}
    return summary


# ---------- Relatório ----------

def _print_report(summary: dict, inertia_now: dict, configs: list, n_replicates: int) -> None:
    prod = f"full@{sysrec.MMR_LAMBDA:g}"
    print("\n=== Contador de inércia sobre a configuração atual "
          f"(grade curr x dest, {INERTIA_CALLS_PER_CONTEXT} chamadas/contexto, réplica 0) ===")
    for regime, rep in inertia_now.items():
        print(f"  [{regime}] inércia={_fmt(rep['inertia_rate'])}  "
              f"por pool homogêneo={_fmt(rep['inert_by_homogeneous_pool'])}  "
              f"por lambda={_fmt(rep['inert_by_lambda'])}  "
              f"spread médio={_fmt(rep['mean_similarity_spread'])}  "
              f"pool com 1 modalidade={rep['pool_single_modality_rate']:.1%}")

    for regime in REGIMES:
        s = summary[regime]
        print(f"\n=== Regime {regime!r}: métricas de lista por configuração "
              f"(média entre {n_replicates} réplica(s)) ===")
        cols = ("ild_features", "n_modalities", "n_categories", "va_spread",
                "ev_mean", "ev_slot1", "pool_n_modalities")
        print(f"{'config':>10} " + " ".join(f"{c:>14}" for c in cols))
        for name, _, _ in configs:
            agg = s["aggregate"][name]
            print(f"{name:>10} " + " ".join(f"{agg[c]['mean']:>14.4f}" for c in cols))

        print(f"\n  Utilidade escolhida sob afinidade latente (escala -> média por configuração):")
        print(f"{'config':>10} " + " ".join(f"{'@' + format(sc, 'g'):>10}" for sc in AFFINITY_SCALES))
        for name, _, _ in configs:
            agg = s["aggregate"][name]
            print(f"{name:>10} " + " ".join(
                f"{agg[f'chosen_utility@{sc:g}']['mean']:>10.4f}" for sc in AFFINITY_SCALES))

        print("\n  Fronteira lambda (modo full): diversidade x relevância x inércia")
        print(f"{'config':>10} {'ild':>9} {'n_mod':>7} {'ev_mean':>9} {'inércia':>9} "
              f"{'homog.':>8} {'lambda':>8}")
        for name, mode, _ in configs:
            if mode != "full":
                continue
            agg, mmr = s["aggregate"][name], s["mmr_inertia"][name]
            print(f"{name:>10} {agg['ild_features']['mean']:>9.4f} "
                  f"{agg['n_modalities']['mean']:>7.3f} {agg['ev_mean']['mean']:>9.4f} "
                  f"{_fmt(mmr['inertia_rate']):>9} {_fmt(mmr['inert_by_homogeneous_pool']):>8} "
                  f"{_fmt(mmr['inert_by_lambda']):>8}")
        eq = s["lambda_1_equals_no_mmr"]
        print(f"  Verificação embutida: full@1 idêntico a no_mmr em todas as métricas: {eq}")

        if s["comparisons"]:
            print(f"\n  Comparações pareadas por réplica ({regime}):")
            for label, per_metric in s["comparisons"].items():
                print(f"    {label}:")
                for metric, c in per_metric.items():
                    print(f"      {metric:>22}: diff={c['mean_diff']:+.4f} (dp={c['std_diff']:.4f}) "
                          f"V/D/E={c['wins']}/{c['losses']}/{c['ties']} "
                          f"p={c['p_value']:.4f} [{c['test']}]")
        else:
            print("\n  (uma réplica só -- SEED_MODE='single': valores descritivos, sem teste)")

    print(f"\n  Modo de produção comparado: {prod}. Ler pela tabela da seção 6 da "
          "especificação: inércia ~1 por pool homogêneo -> o MMR não tem com o que "
          "trabalhar; inércia alta por lambda -> baixar lambda; pool_n_modalities ~1 -> "
          "a diversidade é limitada pelo pool, não pela seleção.")


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


# ---------- Orquestração ----------

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run_selection_ablation(n_contexts: int = LIST_EVAL_N_CONTEXTS,
                           warm_train_episodes: int = LIST_EVAL_WARM_TRAIN_EPISODES,
                           lambdas=MMR_LAMBDA_SWEEP,
                           results_dir: str = SELECTION_ABLATION_RESULTS_DIR) -> dict:
    sysrec.set_seed(sysrec.SEED)
    df = sysrec.load_active_catalog()
    if len(df) == 0:
        raise RuntimeError("Catálogo vazio -- a ablação de seleção não pode rodar.")
    check_range_index(df)
    feature_space = sysrec.FeatureSpace(df)
    eligible = df.index.to_numpy(dtype=int)
    modalities = sorted(df[_modality_col(df)].unique().tolist())
    configs = selection_configs(lambdas)

    mode = sysrec.SEED_MODE
    if mode == "single":
        # Uma réplica, com a raiz derivada de SEED -- valores descritivos, sem teste.
        root_entropy, plan = multiseed.build_seed_plan("multi_fixed", 1, sysrec.SEED)
    else:
        root_entropy, plan = multiseed.build_seed_plan(mode, sysrec.N_REPLICATES, sysrec.SEED,
                                                      sysrec.REPLAY_ENTROPY)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(results_dir, f"{ts}_{mode}")
    os.makedirs(run_dir, exist_ok=True)
    manifest = {
        "seed_mode": mode,
        "root_entropy": str(root_entropy),
        "replicates": [{"replicate_id": r.replicate_id, "global_seed": r.global_seed} for r in plan],
        "configs": [{"name": n, "mode": m, "mmr_lambda": lam} for n, m, lam in configs],
        "production_selection_mode": sysrec.SELECTION_MODE,
        "production_mmr_lambda": sysrec.MMR_LAMBDA,
        "affinity_scales": list(AFFINITY_SCALES),
        "chooser_temperature": CHOOSER_TEMPERATURE,
        "regimes": list(REGIMES),
        "n_contexts_per_replicate": n_contexts,
        "warm_train_episodes": warm_train_episodes,
        "config": multiseed.config_snapshot(),
        "catalog_fingerprint": multiseed.catalog_fingerprint(df),
        "git_commit": multiseed.git_commit_or_none(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[selection_ablation] modo={mode}  entropia raiz={root_entropy}  "
          f"réplicas={len(plan)}  contextos/réplica={n_contexts}  -> {run_dir}")

    ev_table = multiseed.build_expected_value_table(df, eligible)

    replicates = []
    inertia_now = {}
    for rseed in plan:
        print(f"[selection_ablation] réplica {rseed.replicate_id}/{len(plan) - 1}...")
        with multiseed.seeded_scope(rseed.global_seed):
            cold_agent = sysrec.Agent(df, feature_space)
        warm_agent = multiseed.run_replicate(df, feature_space, rseed, warm_train_episodes,
                                             return_agent=True)["agent"]
        agents = {"cold": cold_agent, "warm": warm_agent}

        if rseed.replicate_id == 0:
            for regime, agent in agents.items():
                inertia_now[regime] = run_inertia_check(df, feature_space, agent)

        contexts = pregenerate_contexts(root_entropy, rseed.replicate_id, n_contexts, modalities)
        rep = {regime: run_protocol_p1(df, feature_space, agent, contexts, ev_table, configs)
               for regime, agent in agents.items()}
        replicates.append(rep)

        arrays = {f"{regime}__{name}__{metric}": arr
                  for regime in REGIMES for name in rep[regime]
                  for metric, arr in rep[regime][name]["metrics"].items()}
        np.savez(os.path.join(run_dir, f"replicate_{rseed.replicate_id:03d}.npz"), **arrays)
        with open(os.path.join(run_dir, f"replicate_{rseed.replicate_id:03d}_mmr.json"),
                  "w", encoding="utf-8") as f:
            json.dump(_jsonable({regime: {name: rep[regime][name]["mmr"] for name in rep[regime]}
                                 for regime in REGIMES}), f, indent=2)

    summary = summarize(replicates, configs, len(plan))
    out = {"manifest": manifest, "inertia_current_config": inertia_now, "summary": summary}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(out), f, indent=2, ensure_ascii=False)
    _print_report(summary, inertia_now, configs, len(plan))
    return out
