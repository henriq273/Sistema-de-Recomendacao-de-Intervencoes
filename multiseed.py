"""
multiseed.py — BANCADA DE AVALIAÇÃO multi-seed. Não faz parte do caminho de
produção: nunca chama Agent.save(), nunca escreve em INTERACTION_LOG_PATH, nunca
instancia Recommender nem FatigueTracker, nunca é chamado por main().

Três modos, selecionados por sysrec.SEED_MODE:
  single       -> caminho legado (sistema_de_recomendacao_v3._run_offline_evaluation),
                  intocado -- este módulo nem é importado nesse caso.
  multi_fixed  -> N_REPLICATES réplicas, seeds derivadas deterministicamente de SEED.
  multi_random -> N_REPLICATES réplicas, seed raiz obtida de entropia do SO (ou de
                  REPLAY_ENTROPY, para reproduzir uma execução anterior), registrada
                  no manifesto ANTES da primeira réplica rodar.

Terminologia: RÉPLICA (N_REPLICATES) é uma repetição independente do experimento
inteiro (Agent novo, do zero); EPISÓDIO (N_EPISODES_PER_REPLICATE) é uma interação
simulada dentro de uma réplica. Evitar "época" (treino) e "geração" (evolutivo).

Números aleatórios comuns (CRN): toda a aleatoriedade de cada episódio (contexto,
teste de execução, ruído do score, sorteio do braço aleatório) é pré-gerada uma vez
por réplica e dada a TODOS os braços igualmente -- a única coisa que difere entre
braços é o item escolhido. Isso torna os braços comparáveis entre si e estáveis à
adição de novos braços (ver pregenerate_randomness/run_replicate).

Uso:
    # em sistema_de_recomendacao_v3.py: SEED_MODE = "multi_fixed" (ou "multi_random")
    python sistema_de_recomendacao_v3.py --eval
"""
import hashlib
import json
import math
import os
import random
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

import sistema_de_recomendacao_v3 as sysrec

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------- Seeds ----------

@dataclass(frozen=True)
class ReplicateSeed:
    replicate_id: int
    global_seed: int                  # uint32 -> seeded_scope (random/np legado/torch)
    ctx_ss: np.random.SeedSequence    # gerador local de contextos
    noise_ss: np.random.SeedSequence  # gerador local de ruído


def build_seed_plan(mode: str, n_replicates: int, master_seed: int = sysrec.SEED,
                    replay_entropy: int | None = None):
    """
    Retorna (entropia_raiz, lista de ReplicateSeed).
      multi_fixed:  raiz = SeedSequence(master_seed) -> sempre as mesmas seeds.
      multi_random: raiz = SeedSequence() (entropia do SO), ou
                    SeedSequence(replay_entropy) para reproduzir execução anterior.
    Seeds sequenciais (SEED+i) podem produzir fluxos correlacionados;
    SeedSequence.spawn() aplica uma função de mistura que garante independência
    estatística entre filhas da mesma raiz.
    """
    if mode == "multi_fixed":
        root = np.random.SeedSequence(master_seed)
    elif mode == "multi_random":
        root = (np.random.SeedSequence(replay_entropy)
                if replay_entropy is not None else np.random.SeedSequence())
    else:
        raise ValueError(f"build_seed_plan não se aplica ao modo {mode!r}")

    plan = []
    for i, child in enumerate(root.spawn(n_replicates)):
        model_ss, ctx_ss, noise_ss = child.spawn(3)
        # np.random.seed (API legada, estado global) só aceita valores < 2**32. A
        # entropia raiz tem 128 bits -- generate_state trunca para uint32.
        global_seed = int(model_ss.generate_state(1, dtype=np.uint32)[0])
        plan.append(ReplicateSeed(i, global_seed, ctx_ss, noise_ss))
    return root.entropy, plan


# ---------- Escopo de seed ----------

@contextmanager
def seeded_scope(seed: int):
    """
    Aplica `seed` ao estado global de random/numpy/torch dentro do bloco e
    restaura o estado ANTERIOR ao sair, inclusive em caso de exceção -- restaurar
    o estado, não apenas re-chamar set_seed(SEED) ao final, que reiniciaria o
    fluxo do começo em vez de retomar de onde estava. Também fixa o cuDNN em modo
    determinístico dentro do bloco, restaurando as flags depois. Não altera
    sysrec.set_seed() -- o caminho legado (SEED_MODE="single") continua idêntico.

    Consequência útil: como cada réplica começa aplicando sua própria seed e todo
    o resto é estado novo (Agent recém-instanciado), o resultado de uma réplica
    não depende da ordem em que as réplicas rodam -- seguro paralelizar depois.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cudnn_det = torch.backends.cudnn.deterministic
    cudnn_bench = torch.backends.cudnn.benchmark
    try:
        sysrec.set_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.backends.cudnn.deterministic = cudnn_det
        torch.backends.cudnn.benchmark = cudnn_bench


# ---------- Números aleatórios comuns (CRN) ----------

@dataclass
class EpisodeRandomness:
    curr: np.ndarray          # oitante atual
    dest: np.ndarray          # oitante desejado, de ALLOWED_DEST_OCTANTS
    u_exec: np.ndarray        # uniforme [0,1): teste de p_execution do simulador
    z_noise: np.ndarray       # normal(0,1): ruído do score do simulador
    u_random_arm: np.ndarray  # uniforme [0,1): escolha do braço aleatório


def pregenerate_randomness(n_episodes: int, ctx_ss, noise_ss) -> EpisodeRandomness:
    ctx_rng = np.random.default_rng(ctx_ss)
    noise_rng = np.random.default_rng(noise_ss)
    return EpisodeRandomness(
        curr=ctx_rng.integers(1, 9, size=n_episodes),
        dest=ctx_rng.choice(np.array(sysrec.ALLOWED_DEST_OCTANTS), size=n_episodes),
        u_exec=noise_rng.random(n_episodes),
        z_noise=noise_rng.standard_normal(n_episodes),
        u_random_arm=noise_rng.random(n_episodes),
    )


# ---------- Oráculo determinístico (pseudo-regret, opcional) ----------

def build_expected_value_table(df: pd.DataFrame, eligible: np.ndarray) -> dict:
    """
    {(curr, dest): np.ndarray de valor esperado, na MESMA ordem de `eligible`}.
    Sem filtro de tempo, o valor esperado de cada item depende só de (curr, dest)
    -- 8 x 4 = 32 combinações. Determinística e independente de réplica/seed:
    calculada uma única vez por execução (não por réplica).

    Usa expected_reward_proxy/_holdout_bonus (já existentes, do oráculo de regret
    de sistema_de_recomendacao_v3.py) -- reaproveitados, não reimplementados, para
    o oráculo aqui ser fiel ao MESMO simulador que run_replicate usa.
    """
    records = df.loc[eligible].to_dict("records")
    table = {}
    for curr in range(1, 9):
        for dest in sysrec.ALLOWED_DEST_OCTANTS:
            table[(curr, dest)] = np.array([
                sysrec.expected_reward_proxy(curr, dest, pd.Series(r), sysrec._holdout_bonus)
                for r in records
            ], dtype=np.float32)
    return table


# ---------- Execução de uma réplica ----------

ARMS = ("aleatorio", "mais_popular", "conteudo_puro", "agent_online")


def run_replicate(df: pd.DataFrame, feature_space, rseed: ReplicateSeed, n_episodes: int,
                  oracle_table: dict | None = None, return_agent: bool = False) -> dict:
    """
    Roda N_EPISODES_PER_REPLICATE episódios independentes (sem estado entre eles,
    mesmo padrão de baselines()/regret_curve -- NÃO usa Recommender nem
    FatigueTracker) para os quatro braços de ARMS, sob números aleatórios comuns.

    Ressalvas que devem acompanhar os resultados:
    - agent_online mede o núcleo de aprendizado (argmax de Q sobre o catálogo já
      curado), não o pipeline de produção -- sem guardrail geométrico em tempo
      real, sem fadiga, sem MMR, sem score híbrido.
    - MULTISEED_INCLUDE_WARMSTART deve espelhar o que a produção de fato faz.

    return_agent=True: inclui o Agent treinado no retorno (chave "agent") -- usado
    pelo regime "quente" de selection_ablation.py, que precisa de um agente
    treinado exatamente por este procedimento. O agente só depende das próprias
    escolhas/recompensas (CRN: os outros braços não o afetam).
    """
    rnd = pregenerate_randomness(n_episodes, rseed.ctx_ss, rseed.noise_ss)
    eligible = df.index.to_numpy(dtype=int)           # TIME_FILTER_ENABLED: ver time_avail abaixo
    time_avail = feature_space.max_duration            # "tempo pleno" -- nenhum item é filtrado
    pos_by_item = {int(idx): i for i, idx in enumerate(eligible)}

    popular_item = int(eligible[np.argmax(df.loc[eligible, "Valencia"].to_numpy())])
    content_cache: dict[tuple[int, int], int] = {}   # só 8 x 4 = 32 pares possíveis

    rewards = {arm: np.empty(n_episodes, dtype=np.float32) for arm in ARMS}
    pseudo_regret = (np.empty(n_episodes, dtype=np.float32)
                     if oracle_table is not None else None)

    with seeded_scope(rseed.global_seed):
        agent = sysrec.Agent(df, feature_space)
        if sysrec.MULTISEED_INCLUDE_WARMSTART:
            raise NotImplementedError(
                "MULTISEED_INCLUDE_WARMSTART=True, mas main()/produção não "
                "implementa warm-start hoje -- não há warm_start() para chamar. "
                "Deixe False (espelha a produção real, Agent sempre novo) ou "
                "implemente warm_start() antes de ativar esta flag."
            )

        for ep in range(n_episodes):
            curr, dest = int(rnd.curr[ep]), int(rnd.dest[ep])
            user_state = feature_space.user_state(curr, dest, time_avail)

            if (curr, dest) not in content_cache:
                d = sysrec.distance_to_point(df.loc[eligible], sysrec.target_point(curr, dest))
                content_cache[(curr, dest)] = int(eligible[np.argmin(d)])

            random_pos = min(int(rnd.u_random_arm[ep] * len(eligible)), len(eligible) - 1)
            picks = {
                "aleatorio": int(eligible[random_pos]),
                "mais_popular": popular_item,
                "conteudo_puro": content_cache[(curr, dest)],
                "agent_online": int(eligible[np.argmax(agent.q_values(user_state, eligible))]),
            }

            for arm, item_idx in picks.items():
                reward, next_oct = sysrec.simulate_feedback_holdout(
                    curr, dest, df.loc[item_idx],
                    u_exec=rnd.u_exec[ep], z_noise=rnd.z_noise[ep],
                )
                rewards[arm][ep] = reward
                if arm == "agent_online":
                    next_state = feature_space.user_state(next_oct, dest, time_avail)
                    agent.learn_from_feedback(user_state, item_idx, reward, next_state)
                    if pseudo_regret is not None:
                        table = oracle_table[(curr, dest)]
                        pseudo_regret[ep] = float(table.max() - table[pos_by_item[item_idx]])

    result = {"rewards": rewards, "pseudo_regret": pseudo_regret}
    if return_agent:
        result["agent"] = agent
    return result


# ---------- Persistência e rastreabilidade ----------

def catalog_fingerprint(df: pd.DataFrame) -> str:
    """Muda sempre que a curadoria mudar -- denuncia comparações entre execuções
    feitas sobre catálogos diferentes."""
    key_col = "item_id" if "item_id" in df.columns else "Nome"
    ids = "\n".join(sorted(df[key_col].astype(str)))
    return hashlib.sha256(ids.encode()).hexdigest()[:16]


def git_commit_or_none() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# Constantes que alteram o resultado -- sem isso, dois diretórios de resultado não
# podem ser comparados com confiança.
_CONFIG_SNAPSHOT_KEYS = (
    "BATCH_SIZE", "WARMUP_INTERACTIONS", "LEARNING_RATE", "WEIGHT_DECAY", "GAMMA",
    "UPDATES_PER_FEEDBACK", "ISO_ALPHA", "CANDIDATE_POOL_SIZE",
    "SAFETY_AROUSAL_THRESHOLD", "SAFETY_AVERSIVE_VALENCE_THRESHOLD",
    "TIME_FILTER_ENABLED", "FATIGUE_MIN_GAP", "TOP_K",
    "MULTISEED_INCLUDE_WARMSTART", "MULTISEED_INCLUDE_REGRET",
    "N_EPISODES_PER_REPLICATE", "N_REPLICATES", "FINAL_WINDOW_FRACTION",
)


def config_snapshot() -> dict:
    return {k: getattr(sysrec, k) for k in _CONFIG_SNAPSHOT_KEYS}


def _run_dir(mode: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = os.path.join(sysrec.MULTISEED_RESULTS_DIR, f"{ts}_{mode}")
    os.makedirs(path, exist_ok=True)
    return path


def write_manifest(run_dir: str, mode: str, root_entropy: int, plan: list,
                   df: pd.DataFrame) -> dict:
    """Gravado ANTES da primeira réplica rodar -- se a execução falhar no meio, a
    entropia raiz (e as seeds já derivadas) não se perdem."""
    manifest = {
        "mode": mode,
        "root_entropy": str(root_entropy),   # string: 128 bits sem perda de precisão em JSON
        "replicates": [
            {"replicate_id": r.replicate_id, "global_seed": r.global_seed}
            for r in plan
        ],
        "n_episodes_per_replicate": sysrec.N_EPISODES_PER_REPLICATE,
        "config": config_snapshot(),
        "catalog_fingerprint": catalog_fingerprint(df),
        "git_commit": git_commit_or_none(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[multiseed] modo={mode}  entropia raiz={root_entropy}")
    print(f"[multiseed] para reproduzir: SEED_MODE='multi_random', "
          f"REPLAY_ENTROPY={root_entropy}")
    return manifest


def save_replicate(run_dir: str, result: dict, replicate_id: int) -> None:
    path = os.path.join(run_dir, f"replicate_{replicate_id:03d}.npz")
    arrays = {f"reward_{arm}": result["rewards"][arm] for arm in ARMS}
    if result["pseudo_regret"] is not None:
        arrays["pseudo_regret"] = result["pseudo_regret"]
    np.savez(path, **arrays)


def _load_replicate(run_dir: str, replicate_id: int) -> dict:
    data = np.load(os.path.join(run_dir, f"replicate_{replicate_id:03d}.npz"))
    return {
        "rewards": {arm: data[f"reward_{arm}"] for arm in ARMS},
        "pseudo_regret": data["pseudo_regret"] if "pseudo_regret" in data.files else None,
    }


def _load_and_verify_resume(resume_dir: str, mode: str, df: pd.DataFrame) -> dict:
    """Confere que a impressão digital do catálogo e a configuração coincidem com
    as atuais antes de reaproveitar qualquer réplica já rodada -- retomar sobre um
    catálogo/config diferentes produziria resultados incomparáveis, sem aviso."""
    manifest_path = os.path.join(resume_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Diretório de retomada sem manifest.json: {resume_dir}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest["mode"] != mode:
        raise ValueError(f"Retomada abortada: modo do manifesto ({manifest['mode']!r}) "
                          f"diverge do SEED_MODE atual ({mode!r}).")
    current_fp = catalog_fingerprint(df)
    if manifest["catalog_fingerprint"] != current_fp:
        raise ValueError("Retomada abortada: catalog_fingerprint diverge do manifesto -- "
                          "o catálogo mudou desde a execução original.")
    if manifest["config"] != config_snapshot():
        raise ValueError("Retomada abortada: configuração atual diverge do manifesto.")
    print(f"[multiseed] retomando de {resume_dir} (entropia raiz={manifest['root_entropy']})")
    return manifest


# ---------- Agregação e estatística ----------

def _final_window_mean(arr: np.ndarray, fraction: float) -> float:
    n = len(arr)
    start = max(0, int(n * (1 - fraction)))
    return float(np.mean(arr[start:]))


def aggregate_results(all_rewards: dict) -> dict:
    """all_rewards: {arm: [array por réplica]}. A unidade de análise é a RÉPLICA,
    não o episódio -- _bootstrap_ci legado reamostra episódios de uma única
    execução, o que ignora variância entre seeds e trata como independentes
    episódios correlacionados (o agente aprende). Os intervalos aqui são mais
    largos que os do _bootstrap_ci legado -- e são os honestos."""
    summary = {}
    for arm, per_replicate in all_rewards.items():
        run_means = np.array([float(np.mean(r)) for r in per_replicate])
        final_means = np.array([_final_window_mean(r, sysrec.FINAL_WINDOW_FRACTION)
                                for r in per_replicate])
        summary[arm] = {
            "run_mean": {"mean": float(run_means.mean()), "std": float(run_means.std()),
                        "min": float(run_means.min()), "max": float(run_means.max())},
            "final_window_mean": {"mean": float(final_means.mean()), "std": float(final_means.std()),
                                  "min": float(final_means.min()), "max": float(final_means.max())},
            "per_replicate_final_window_mean": final_means.tolist(),
        }
    return summary


def sign_test_p(wins: int, losses: int) -> float:
    """Teste do sinal exato, bilateral. Empates são descartados. Fallback sem
    dependência quando scipy não está instalado."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired_comparison(final_a: np.ndarray, final_b: np.ndarray, label_a: str, label_b: str) -> dict:
    """Comparação pareada por replicate_id, na janela final. A comparação
    principal do relatório é agent_online - conteudo_puro; aleatorio/mais_popular
    são descritivos (declarar a hipótese principal de antemão evita a questão de
    comparações múltiplas -- se as quatro forem testadas formalmente, aplicar
    correção de Holm)."""
    diffs = final_a - final_b
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())

    if _HAVE_SCIPY and (diffs != 0).any():
        try:
            _, p = wilcoxon(final_a, final_b)
            test_name = "wilcoxon"
        except ValueError:
            p = sign_test_p(wins, losses)
            test_name = "sign_test (wilcoxon não aplicável a esta amostra)"
    else:
        p = sign_test_p(wins, losses)
        test_name = "sign_test" if _HAVE_SCIPY else "sign_test (scipy indisponível)"

    return {
        "label": f"{label_a} - {label_b}",
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std()),
        "wins": wins, "losses": losses, "ties": ties,
        "p_value": float(p), "test": test_name,
    }


def learning_curve(rewards_per_replicate: list, smoothing: int, n_points: int = 10) -> dict:
    """Média entre réplicas por episódio, suavizada por média móvel de `smoothing`,
    com desvio-padrão entre réplicas na mesma janela. `n_points` valores ao longo
    dos episódios para impressão em texto; os arrays completos vão para
    summary.npz para plotagem posterior (matplotlib, se usado, é dependência só
    de bancada)."""
    stacked = np.stack(rewards_per_replicate)  # (n_replicates, n_episodes)
    mean_per_episode = stacked.mean(axis=0)
    std_per_episode = stacked.std(axis=0)
    smoothing = max(1, min(smoothing, len(mean_per_episode)))
    kernel = np.ones(smoothing) / smoothing
    smoothed_mean = np.convolve(mean_per_episode, kernel, mode="valid")
    smoothed_std = np.convolve(std_per_episode, kernel, mode="valid")

    idx = np.linspace(0, len(smoothed_mean) - 1, num=min(n_points, len(smoothed_mean)), dtype=int)
    return {
        "full_mean": smoothed_mean.tolist(),
        "full_std": smoothed_std.tolist(),
        "sample_points": [(int(i), float(smoothed_mean[i])) for i in idx],
    }


# ---------- Relatório ----------

def _print_report(summary: dict, comparison: dict, curve_agent: dict, curve_content: dict,
                  all_pseudo_regret: list) -> None:
    print(f"\n=== Avaliação multi-seed: resumo por braço (janela final = "
          f"{sysrec.FINAL_WINDOW_FRACTION:.0%} dos episódios, {sysrec.N_REPLICATES} réplicas) ===")
    for arm in ARMS:
        s = summary[arm]["final_window_mean"]
        print(f"  {arm:16s} média={s['mean']:+.4f}  desvio_entre_réplicas={s['std']:.4f}  "
              f"min={s['min']:+.4f}  max={s['max']:+.4f}")

    print("\n=== Comparação pareada: agent_online vs. conteudo_puro (janela final) ===")
    print(f"  diferença média entre réplicas: {comparison['mean_diff']:+.4f} "
          f"(desvio={comparison['std_diff']:.4f})")
    print(f"  vitórias/derrotas/empates do agente: "
          f"{comparison['wins']}/{comparison['losses']}/{comparison['ties']}")
    print(f"  teste: {comparison['test']}  p-valor={comparison['p_value']:.4f}")

    print("\n=== Curva de aprendizado (média entre réplicas, agent_online vs. conteudo_puro) ===")
    print(f"{'episódio':>10} {'agent_online':>14} {'conteudo_puro':>14}")
    for (i_a, v_a), (_, v_c) in zip(curve_agent["sample_points"], curve_content["sample_points"]):
        print(f"{i_a:>10} {v_a:>14.4f} {v_c:>14.4f}")
    print("  Curva subindo ao longo dos episódios = aprendizado. Curva plana = sem "
          "aprendizado, mesmo que a média final seja boa.")

    if all_pseudo_regret:
        all_pr = np.concatenate(all_pseudo_regret)
        n_neg = int((all_pr < -1e-6).sum())
        print("\n=== Pseudo-regret (agent_online vs. oráculo determinístico) ===")
        print(f"  média: {all_pr.mean():.4f}  desvio: {all_pr.std():.4f}")
        print(f"  episódios com pseudo-regret negativo (deveria ser 0, por construção): "
              f"{n_neg}/{len(all_pr)}")

    print("\n  Limite: multi-seed responde se a vantagem é robusta a inicialização e "
          "amostragem -- não responde se ela vai além do alinhamento geométrico "
          "(depende do simulador de perfis heterogêneos e da verificação de "
          "vocabulário de Tipo nos bônus do holdout, ambas ainda pendentes).")
    print("  Nota de reprodutibilidade: seeds raiz obtidas de entropia do sistema "
          "operacional (128 bits) no modo multi_random; fluxos derivados via "
          "numpy.random.SeedSequence.")


# ---------- Orquestração ----------

def run_multiseed_evaluation(resume_dir: str | None = None) -> dict:
    """
    Ponto de entrada, chamado por sistema_de_recomendacao_v3.py quando
    SEED_MODE != "single". Fluxo: 1) carrega catálogo/FeatureSpace; 2) resolve o
    plano de seeds (ou retoma um diretório existente, verificando compatibilidade);
    3) grava o manifesto ANTES de qualquer réplica; 4) roda (ou reaproveita) cada
    réplica; 5) agrega e reporta.
    """
    mode = sysrec.SEED_MODE
    if mode == "single":
        raise ValueError("run_multiseed_evaluation não se aplica a SEED_MODE='single'.")

    sysrec.set_seed(sysrec.SEED)
    df = sysrec.load_active_catalog()
    if len(df) == 0:
        raise RuntimeError(
            "Catálogo vazio -- confirme APPROVED_CATEGORIES/DATA_BACKEND antes de "
            "rodar a avaliação multi-seed."
        )
    feature_space = sysrec.FeatureSpace(df)
    eligible = df.index.to_numpy(dtype=int)

    if resume_dir is not None:
        manifest = _load_and_verify_resume(resume_dir, mode, df)
        replay_entropy = int(manifest["root_entropy"])
        root_entropy, plan = build_seed_plan(mode, sysrec.N_REPLICATES, sysrec.SEED, replay_entropy)
        run_dir = resume_dir
    else:
        root_entropy, plan = build_seed_plan(mode, sysrec.N_REPLICATES, sysrec.SEED,
                                             sysrec.REPLAY_ENTROPY)
        run_dir = _run_dir(mode)
        manifest = write_manifest(run_dir, mode, root_entropy, plan, df)

    oracle_table = (build_expected_value_table(df, eligible)
                    if sysrec.MULTISEED_INCLUDE_REGRET else None)

    all_rewards = {arm: [] for arm in ARMS}
    all_pseudo_regret = []
    for rseed in plan:
        npz_path = os.path.join(run_dir, f"replicate_{rseed.replicate_id:03d}.npz")
        if os.path.exists(npz_path):
            print(f"[multiseed] réplica {rseed.replicate_id}: .npz já existe, "
                  f"pulando (retomada)")
            result = _load_replicate(run_dir, rseed.replicate_id)
        else:
            print(f"[multiseed] réplica {rseed.replicate_id}/{len(plan) - 1} "
                  f"(seed={rseed.global_seed})...")
            result = run_replicate(df, feature_space, rseed,
                                   sysrec.N_EPISODES_PER_REPLICATE, oracle_table)
            save_replicate(run_dir, result, rseed.replicate_id)

        for arm in ARMS:
            all_rewards[arm].append(result["rewards"][arm])
        if result["pseudo_regret"] is not None:
            all_pseudo_regret.append(result["pseudo_regret"])

    summary = aggregate_results(all_rewards)
    final_agent = np.array(summary["agent_online"]["per_replicate_final_window_mean"])
    final_content = np.array(summary["conteudo_puro"]["per_replicate_final_window_mean"])
    comparison = paired_comparison(final_agent, final_content, "agent_online", "conteudo_puro")

    curve_agent = learning_curve(all_rewards["agent_online"], sysrec.LEARNING_CURVE_SMOOTHING)
    curve_content = learning_curve(all_rewards["conteudo_puro"], sysrec.LEARNING_CURVE_SMOOTHING)

    summary_out = {
        "manifest": manifest,
        "summary_by_arm": summary,
        "comparison_agent_vs_content": comparison,
        "n_replicates": len(plan),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    np.savez(
        os.path.join(run_dir, "summary.npz"),
        learning_curve_agent_mean=np.array(curve_agent["full_mean"]),
        learning_curve_agent_std=np.array(curve_agent["full_std"]),
        learning_curve_content_mean=np.array(curve_content["full_mean"]),
        learning_curve_content_std=np.array(curve_content["full_std"]),
    )

    _print_report(summary, comparison, curve_agent, curve_content, all_pseudo_regret)
    return summary_out
