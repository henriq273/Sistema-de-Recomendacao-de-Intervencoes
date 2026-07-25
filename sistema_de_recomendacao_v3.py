"""
Sistema de Recomendação de Intervenções — v3

  - Configuração central (constantes de rede, RL, PER, guardrail, etc.)
  - Núcleo: carga do dataset, geometria dos oitantes (Circumplexo) e
    engenharia de features (FeatureSpace)
  - Agente: rede ContentAwareDQN, replay priorizado (PER) e o Agent que
    calcula Q-values e aprende com feedback real
  - Política de recomendação: guardrail de segurança, pool de candidatos,
    score híbrido heurística/DQN, fadiga e anti-monotonia (softmax/MMR)
  - Simulador e avaliação offline — bancada de teste, não fazem parte do
    caminho de produção (nunca usados para treinar o modelo real)
  - CLI interativo e ponto de entrada
"""
import json
import os
import random
import sys
from collections import namedtuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Dataset 
DATASET_PATH = "dataset.csv"   # nome simples ou caminho absoluto (ex.: caminho do Drive)

# Reprodutibilidade
SEED = 42

# Rede e otimização
HIDDEN_DIMS = (128, 64, 32)
DROPOUT = 0.2
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0
TARGET_TAU = 0.005

# RL
BANDIT_MODE = True          # True: alvo = recompensa (cada recomendação é episódio fechado)
UPDATES_PER_FEEDBACK = 3    # passos de replay por feedback recebido
GAMMA = 0.99                   # fator de desconto do futuro (0 = só recompensa imediata)

# Prioritized Experience Replay
MEMORY_CAPACITY = 5000
PER_ALPHA = 0.6              # 0 = uniforme, 1 = priorização total
PER_BETA = 0.4                # correção de viés de importance sampling (cresce até 1)
PER_BETA_INCREMENT = 1e-4
PER_EPSILON = 1e-5           # evita prioridade zero

# Alvo afetivo
# Ponto-alvo no plano (V,A) = atual + ISO_ALPHA * (desejado - atual)
#   ISO_ALPHA = 1.0 -> mira exatamente o oitante desejado
#   ISO_ALPHA < 1.0 -> princípio-iso: mira um ponto intermediário, aproximação gradual
ISO_ALPHA = 1.0

# Conjunto de candidatos
CANDIDATE_POOL_SIZE = 12    # M itens mais próximos do ponto-alvo entram no pool

# Guardrail de segurança
USE_SAFETY_FILTER = True
LOW_ENERGY_OCTANTS = (5, 6)      # Triste/Deprimido, Entediado/Cansado
SAFETY_AROUSAL_THRESHOLD = 0.6   # itens acima disso são bloqueados nesses estados

# Cold-start (heurística -> DQN)
WARMUP_INTERACTIONS = 50    # num. de feedbacks reais até confiar totalmente no DQN 

# Anti-monotonia
SOFTMAX_TEMPERATURE = 0.3   # temperatura do sorteio do slot 1 (menor = mais guloso)
EXPLORE_TEMPERATURE = 1.0   # temperatura do slot exploratório (maior = mais diverso)
P_EXPLORE_SLOT = 0.5        # probabilidade de um dos slots ser exploratório
MMR_LAMBDA = 0.7            # 0.7*relevância - 0.3*similaridade (diversidade da lista)
FATIGUE_LAMBDA = 0.5        # peso máximo da penalidade de fadiga
FATIGUE_HALFLIFE = 10       # em nº de interações; meia-vida do decaimento da penalidade
TOP_K = 3                   # itens recomendados por vez

# Persistência - conservar pesos, histórico de treino e log de interações
CHECKPOINT_PATH = "checkpoint_v3.pt"
INTERACTION_LOG_PATH = "interaction_log.jsonl"

# Centroides dos oitantes no plano (Valência, Arousal) do Modelo Circumplexo de Emoções.
OCTANT_MAP = {
    1: (0.8, 0.8),     # Excitado/Eufórico
    2: (0.3, 0.9),     # Tenso/Alerta
    3: (-0.3, 0.9),    # Estressado/Ansioso
    4: (-0.8, 0.8),    # Irritado/Raiva
    5: (-0.8, -0.2),   # Triste/Deprimido
    6: (-0.3, -0.8),   # Entediado/Cansado
    7: (0.3, -0.8),    # Calmo/Relaxado
    8: (0.8, -0.2),    # Sereno/Contente
}

OCTANT_LABELS = {
    1: "Excitado/Eufórico",
    2: "Tenso/Alerta",
    3: "Estressado/Ansioso",
    4: "Irritado/Raiva",
    5: "Triste/Deprimido",
    6: "Entediado/Cansado",
    7: "Calmo/Relaxado",
    8: "Sereno/Contente",
}

EXPECTED_COLUMNS = ["Nome", "Tipo", "Valencia", "Arousal", "Duracao", "Indoor", "Tag", "Oitante"]


def set_seed(seed: int = SEED) -> None:
    """Fixa a semente de todas as fontes de aleatoriedade usadas no projeto 
    (para fins de reprodutibilidade)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_path(path: str = DATASET_PATH) -> str:
    """
    Localiza o CSV do catálogo testando, em ordem: o caminho dado; a variável de
    ambiente INTERVENTIONS_DATASET; a pasta do próprio módulo; o diretório de
    trabalho atual; e o caminho padrão do Google Drive (Colab) (fallback final).
    """
    filename = os.path.basename(path)
    candidates = [path]

    env_path = os.environ.get("INTERVENTIONS_DATASET")
    if env_path:
        candidates.append(env_path)

    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(module_dir, filename))
    except NameError:
        pass  # __file__ não existe em notebook/Colab

    candidates.append(os.path.join(os.getcwd(), filename))
    candidates.append(
        os.path.join("/content/drive/MyDrive/Sistema de Recomendação", filename)
    )

    tried = []
    for candidate in candidates:
        tried.append(candidate)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Dataset não encontrado. Caminhos testados:\n" + "\n".join(tried)
    )


def load_dataset(path: str = None) -> pd.DataFrame:
    """Carrega e valida o dataset.csv."""
    resolved = resolve_dataset_path(path or DATASET_PATH)
    df = pd.read_csv(resolved).reset_index(drop=True)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no dataset: {missing}")
    if not df["Valencia"].between(-1.0, 1.0).all():
        raise ValueError("Valores de Valencia fora do intervalo [-1, 1].")
    if not df["Arousal"].between(-1.0, 1.0).all():
        raise ValueError("Valores de Arousal fora do intervalo [-1, 1].")

    return df


def target_point(curr_oct: int, dest_oct: int, iso_alpha: float = ISO_ALPHA) -> np.ndarray:
    """
    Ponto-alvo no plano (V, A): atual + iso_alpha * (desejado - atual).
    Com iso_alpha=1.0 (padrão), o alvo é exatamente o centroide do oitante desejado;
    valores menores miram um ponto intermediário (princípio-iso, aproximação gradual).
    """
    current = np.array(OCTANT_MAP[curr_oct], dtype=np.float32)
    desired = np.array(OCTANT_MAP[dest_oct], dtype=np.float32)
    return current + iso_alpha * (desired - current)


def distance_to_point(df: pd.DataFrame, point: np.ndarray) -> np.ndarray:
    """Distância euclidiana de cada item do DataFrame ao ponto dado, no plano (V, A)."""
    values = df[["Valencia", "Arousal"]].to_numpy(dtype=np.float32)
    return np.linalg.norm(values - point.reshape(1, 2), axis=1)


def nearest_octant(v: float, a: float) -> int:
    """
    Oitante cujo centroide é mais próximo de (v, a).
    """
    point = np.array([v, a], dtype=np.float32)
    return min(
        OCTANT_MAP,
        key=lambda octant: float(np.linalg.norm(point - np.array(OCTANT_MAP[octant], dtype=np.float32))),
    )


class FeatureSpace:
    """Engenharia de features dos itens e do estado do usuário, com mapeamento estável."""

    def __init__(self, df: pd.DataFrame):
        """sorted() torna o mapeamento independente da ordem das linhas do CSV: sem isso,
        reordenar o dataset invalidaria silenciosamente checkpoints salvos."""
        self.all_types = sorted(df["Tipo"].unique().tolist())
        self.all_tags = sorted(df["Tag"].unique().tolist())
        self.max_duration = float(df["Duracao"].max())
        self._type_index = {t: i for i, t in enumerate(self.all_types)}
        self._tag_index = {t: i for i, t in enumerate(self.all_tags)}
        self.item_matrix = self._build_item_matrix(df)

    def _build_item_matrix(self, df: pd.DataFrame) -> np.ndarray:
        # Vetor de features do item, ordem fixa (features novas vão no fim):
        # [ one-hot(Tipo) | Indoor | one-hot(Tag) | Duracao/max_duration | (V+1)/2 | (A+1)/2 ]
        n_types, n_tags = len(self.all_types), len(self.all_tags)
        dim = n_types + 1 + n_tags + 3
        matrix = np.zeros((len(df), dim), dtype=np.float32)
        rows = np.arange(len(df))

        type_idx = df["Tipo"].map(self._type_index).to_numpy()
        tag_idx = df["Tag"].map(self._tag_index).to_numpy()
        matrix[rows, type_idx] = 1.0
        matrix[:, n_types] = df["Indoor"].to_numpy(dtype=np.float32)
        matrix[rows, n_types + 1 + tag_idx] = 1.0
        matrix[:, n_types + 1 + n_tags] = df["Duracao"].to_numpy(dtype=np.float32) / self.max_duration
        matrix[:, n_types + 1 + n_tags + 1] = (df["Valencia"].to_numpy(dtype=np.float32) + 1.0) / 2.0
        matrix[:, n_types + 1 + n_tags + 2] = (df["Arousal"].to_numpy(dtype=np.float32) + 1.0) / 2.0
        return matrix

    def user_state(self, curr_oct: int, dest_oct: int, time_avail: float) -> np.ndarray:
        # [ one-hot(oitante atual) (8) | one-hot(oitante desejado) (8) | tempo normalizado (1) ]
        state = np.zeros(17, dtype=np.float32)
        state[curr_oct - 1] = 1.0
        state[8 + dest_oct - 1] = 1.0
        state[16] = min(time_avail / self.max_duration, 1.0)
        return state

    @property
    def item_feat_dim(self) -> int:
        return self.item_matrix.shape[1]

    def item_features(self, item_indices: np.ndarray) -> np.ndarray:
        return self.item_matrix[item_indices]

    def signature(self) -> dict:
        """Assinatura do espaço de features, persistida no checkpoint junto dos pesos."""
        return {
            "all_types": list(self.all_types),
            "all_tags": list(self.all_tags),
            "max_duration": self.max_duration,
        }

    def assert_compatible(self, signature: dict) -> None:
        """
        Levanta ValueError se a assinatura de um checkpoint salvo não bate com este
        dataset. Sem essa checagem, pesos carregados apontariam silenciosamente para
        dimensões com significado trocado.
        """
        if signature["all_types"] != self.all_types:
            raise ValueError("Checkpoint incompatível: mapeamento de Tipo divergente do dataset atual.")
        if signature["all_tags"] != self.all_tags:
            raise ValueError("Checkpoint incompatível: mapeamento de Tag divergente do dataset atual.")
        if float(signature["max_duration"]) != self.max_duration:
            raise ValueError("Checkpoint incompatível: max_duration divergente do dataset atual.")


# Agente: rede neural, replay priorizado e DQN content-aware

Transition = namedtuple(
    "Transition", ("user_state", "item_features", "reward", "next_user_state", "done")
)


class ContentAwareDQN(nn.Module):
    """
    Recebe a concatenação [estado_do_usuário || features_do_item] e devolve um único
    Q-value para esse par — não há uma saída por ação como no DQN clássico. Para
    recomendar, calcula-se Q para todos os itens candidatos e ranqueia-se.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        h1, h2, h3 = HIDDEN_DIMS
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            # LayerNorm: normaliza por amostra, é indiferente ao tamanho 
            # do batch e se comporta igual em treino e inferência. Faz-se necessaśrio
            # porque o sistema alterna entre replay em lote e atualização online.
            nn.LayerNorm(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h3),
            nn.ReLU(inplace=True),
            nn.Linear(h3, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrioritizedReplayBuffer:
    """
    Replay priorizado proporcional.

    Com poucos itens no catálogo e feedback humano caro de coletar, não se pode
    desperdiçar atualizações em transições já aprendidas: o PER concentra o esforço
    onde o erro de TD é maior, e os pesos de importance sampling corrigem o viés
    dessa amostragem não uniforme.
    """

    def __init__(self, capacity: int = MEMORY_CAPACITY):
        self.capacity = capacity
        self.memory: list[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.beta = PER_BETA

    def push(self, transition: Transition) -> None:
        # Prioridade máxima corrente garante que toda transição nova seja amostrada ao menos uma vez.
        max_priority = self.priorities[: len(self.memory)].max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        n = len(self.memory)
        probs = self.priorities[:n] ** PER_ALPHA
        probs /= probs.sum()
        indices = np.random.choice(n, batch_size, p=probs, replace=False)

        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + PER_BETA_INCREMENT)

        batch = Transition(*zip(*(self.memory[i] for i in indices)))
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.priorities[indices] = np.abs(td_errors) + PER_EPSILON

    def __len__(self) -> int:
        return len(self.memory)


class Agent:
    # Agente DQN content-aware: calcula Q-values e aprende com feedback real.

    def __init__(self, df: pd.DataFrame, feature_space: FeatureSpace):
        if len(df) != feature_space.item_matrix.shape[0]:
            raise ValueError("df e feature_space não têm o mesmo número de itens.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_space = feature_space

        input_dim = 17 + feature_space.item_feat_dim
        self.policy_net = ContentAwareDQN(input_dim).to(self.device)
        self.target_net = ContentAwareDQN(input_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.requires_grad_(False)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        # reduction="none" é obrigatório: cada perda precisa ser ponderada pelo peso de
        # importance sampling do PER antes de ser mediada.
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.memory = PrioritizedReplayBuffer()
        self.n_feedbacks = 0
        self.history = {"loss": [], "reward": []}

    @torch.no_grad()
    def q_values(self, user_state: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        # Q-value da rede de política para cada item candidato, dado o estado do usuário.
        self.policy_net.eval()
        user_repeat = np.repeat(user_state.reshape(1, -1), len(item_indices), axis=0)
        item_feats = self.feature_space.item_features(item_indices)
        x = torch.from_numpy(np.concatenate([user_repeat, item_feats], axis=1)).to(self.device)
        return self.policy_net(x).squeeze(-1).cpu().numpy()

    def store(self, user_state: np.ndarray, item_idx: int, reward: float,
              next_user_state: np.ndarray, done: bool = True) -> None:
        # Empilha uma transição no buffer de replay.
        if BANDIT_MODE:
            done = True
        item_features = self.feature_space.item_features(np.array([item_idx]))[0]
        self.memory.push(Transition(user_state, item_features, reward, next_user_state, done))

    def replay(self) -> float:
        # Um passo de treino a partir de um minibatch priorizado. 0.0 se o buffer ainda é insuficiente.
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        self.policy_net.train()
        batch, indices, is_weights = self.memory.sample(BATCH_SIZE)

        user_states = torch.from_numpy(np.stack(batch.user_state)).to(self.device)
        item_feats = torch.from_numpy(np.stack(batch.item_features)).to(self.device)
        rewards = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array(batch.done, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.stack(batch.next_user_state)).to(self.device)
        is_weights = torch.from_numpy(is_weights).to(self.device)

        curr_q = self.policy_net(torch.cat([user_states, item_feats], dim=1)).squeeze(-1)

        with torch.no_grad():
            if BANDIT_MODE:
                # Cada recomendação é um episódio fechado: o alvo é a própria recompensa.
                target_q = rewards
            else:
                """Double DQN (inativo em BANDIT_MODE, mantido como salvaguarda): a
                política escolhe a melhor ação, a target network a avalia. Separar
                quem escolhe de quem avalia elimina o viés de superestimação do
                max() do DQN padrão."""
                target_q = self._double_dqn_target(next_states, rewards, dones)

        td_errors = (curr_q - target_q).detach().cpu().numpy()
        loss = (self.loss_fn(curr_q, target_q) * is_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Limita o dano de um feedback atípico isolado: cada feedback real dispara
        # replay imediatamente.
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)
        self._soft_update_target()
        return float(loss.item())

    def _double_dqn_target(self, next_states: torch.Tensor, rewards: torch.Tensor,
                            dones: torch.Tensor) -> torch.Tensor:
        n_items = self.feature_space.item_matrix.shape[0]
        batch_size = next_states.shape[0]
        all_items = torch.from_numpy(self.feature_space.item_matrix).to(self.device)

        expanded_states = next_states.unsqueeze(1).expand(batch_size, n_items, -1)
        expanded_items = all_items.unsqueeze(0).expand(batch_size, n_items, -1)
        candidates = torch.cat([expanded_states, expanded_items], dim=2).reshape(batch_size * n_items, -1)

        policy_q = self.policy_net(candidates).view(batch_size, n_items)
        best_actions = policy_q.argmax(dim=1)

        best_item_feats = all_items[best_actions]
        next_q = self.target_net(torch.cat([next_states, best_item_feats], dim=1)).squeeze(-1)
        return rewards + GAMMA * next_q * (1.0 - dones)

    def _soft_update_target(self) -> None:
        """Soft update (Polyak): evita o 'alvo móvel' sem a descontinuidade de cópias periódicas abruptas."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.mul_(1.0 - TARGET_TAU).add_(policy_param.data, alpha=TARGET_TAU)

    def learn_from_feedback(self, user_state: np.ndarray, item_idx: int, reward: float,
                             next_user_state: np.ndarray) -> float:
        """Registra um feedback real e roda UPDATES_PER_FEEDBACK passos de replay. Retorna a perda média."""
        self.n_feedbacks += 1
        self.store(user_state, item_idx, reward, next_user_state)

        losses = [loss for loss in (self.replay() for _ in range(UPDATES_PER_FEEDBACK)) if loss > 0.0]
        avg_loss = float(np.mean(losses)) if losses else 0.0

        self.history["loss"].append(avg_loss)
        self.history["reward"].append(float(reward))
        return avg_loss

    def save(self, path: str = CHECKPOINT_PATH) -> None:
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "n_feedbacks": self.n_feedbacks,
            "history": self.history,
            "feature_signature": self.feature_space.signature(),
        }, path)

    def load(self, path: str, feature_space: FeatureSpace) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        feature_space.assert_compatible(data["feature_signature"])
        self.feature_space = feature_space
        self.policy_net.load_state_dict(data["policy_state"])
        self.target_net.load_state_dict(data["target_state"])
        self.optimizer.load_state_dict(data["optimizer_state"])
        self.n_feedbacks = int(data.get("n_feedbacks", 0))
        self.history = data.get("history", {"loss": [], "reward": []})
        self.target_net.eval()


# Política de recomendação

"""Princípio de separação: tudo nesta seção é pós-processamento. Fadiga,
softmax e MMR atuam depois que a rede calculou os Q-values, apenas na seleção do que
exibir. Nada aqui altera pesos, gradientes ou o que o modelo aprende.
O treino (Agent.replay()) continua enxergando os Q-values puros."""

class FatigueTracker:
    # Penaliza itens recomendados recentemente, para evitar repetição e monotonia.

    def __init__(self):
        self.last_seen: dict[int, int] = {}
        self.counter = 0

    def penalty(self, item_indices: np.ndarray) -> np.ndarray:
        penalties = np.zeros(len(item_indices), dtype=np.float32)
        for i, item_idx in enumerate(item_indices):
            if item_idx in self.last_seen:
                delta = self.counter - self.last_seen[item_idx]
                penalties[i] = FATIGUE_LAMBDA * 0.5 ** (delta / FATIGUE_HALFLIFE)
        return penalties

    def register(self, item_indices: np.ndarray) -> None:
        for item_idx in item_indices:
            self.last_seen[int(item_idx)] = self.counter
        self.counter += 1

    def state_dict(self) -> dict:
        return {"last_seen": dict(self.last_seen), "counter": self.counter}

    def load_state_dict(self, state: dict) -> None:
        self.last_seen = {int(k): int(v) for k, v in state.get("last_seen", {}).items()}
        self.counter = int(state.get("counter", 0))


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    x = (values - values.max()) / temperature
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)


class Recommender:
    # Pipeline completo: elegibilidade -> guardrail -> pool -> score híbrido -> fadiga -> seleção.

    def __init__(self, df: pd.DataFrame, feature_space: FeatureSpace, agent: Agent):
        self.df = df
        self.feature_space = feature_space
        self.agent = agent
        self.fatigue = FatigueTracker()
        self.safety_checked = 0
        self.safety_blocked = 0
        self.recommended_items: set[int] = set()

    @property
    def safety_violation_rate(self) -> float:
        return self.safety_blocked / self.safety_checked if self.safety_checked else 0.0

    @property
    def catalog_coverage(self) -> float:
        return len(self.recommended_items) / len(self.df)

    def recommend(self, curr_oct: int, dest_oct: int, time_avail: float, k: int = TOP_K) -> list[dict]:
        eligible = self.df.index[self.df["Duracao"] <= time_avail].to_numpy(dtype=int)
        if len(eligible) == 0:
            return []

        eligible = self._apply_safety_filter(eligible, curr_oct)
        pool_idx, pool_dist = self._candidate_pool(eligible, curr_oct, dest_oct)

        user_state = self.feature_space.user_state(curr_oct, dest_oct, time_avail)
        q_values = self.agent.q_values(user_state, pool_idx)
        score = self._hybrid_score(pool_dist, q_values)
        adjusted = score - self.fatigue.penalty(pool_idx)

        pool_vectors = self.feature_space.item_features(pool_idx)
        positions, slot_types, propensities = self._select_slots(adjusted, pool_vectors, k)

        results = []
        for pos, slot_type, propensity in zip(positions, slot_types, propensities):
            item_idx = int(pool_idx[pos])
            row = self.df.loc[item_idx]
            results.append({
                "item_idx": item_idx,
                "nome": row["Nome"],
                "tipo": row["Tipo"],
                "tag": row["Tag"],
                "duracao": int(row["Duracao"]),
                "valencia": float(row["Valencia"]),
                "arousal": float(row["Arousal"]),
                "indoor": int(row["Indoor"]),
                "q_value": float(q_values[pos]),        # Q bruto da rede (diagnóstico)
                "score": float(adjusted[pos]),           # score híbrido pós-fadiga
                "propensity": propensity,                # π(a|x) para avaliação off-policy
                "slot_type": slot_type,                  # "softmax" | "explore" | "mmr"
            })
            self.recommended_items.add(item_idx)

        self.fatigue.register(np.array([r["item_idx"] for r in results], dtype=int))
        return results

    def _apply_safety_filter(self, eligible: np.ndarray, curr_oct: int) -> np.ndarray:
        """
        Guardrail de segurança: bloqueia itens de alta ativação para usuários em estados
        de baixa energia. Regra dura, independente do que a DQN aprendeu. Um DQN
        otimiza apenas a recompensa recebida, sem noção de que recomendar atividade de
        alta ativação física a alguém em estado depressivo pode agravar o quadro.
        """
        if not USE_SAFETY_FILTER or curr_oct not in LOW_ENERGY_OCTANTS:
            return eligible

        self.safety_checked += len(eligible)
        arousal = self.df.loc[eligible, "Arousal"].to_numpy(dtype=np.float32)
        safe = eligible[arousal <= SAFETY_AROUSAL_THRESHOLD]
        self.safety_blocked += len(eligible) - len(safe)

        # Nunca travar: se o filtro esvaziar o conjunto, reverter para o conjunto anterior.
        return safe if len(safe) > 0 else eligible

    def _candidate_pool(self, eligible: np.ndarray, curr_oct: int, dest_oct: int):
        """
        Pool dos CANDIDATE_POOL_SIZE itens mais próximos do ponto-alvo, por distância
        contínua no plano (V, A), nao por igualdade de oitante discreto.
        """
        point = target_point(curr_oct, dest_oct, ISO_ALPHA)
        distances = distance_to_point(self.df.loc[eligible], point)
        order = np.argsort(distances)[:CANDIDATE_POOL_SIZE]
        return eligible[order], distances[order]

    def _hybrid_score(self, pool_dist: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        """
        Mistura heurística de proximidade afetiva com a DQN. Enquanto a rede está fria
        (poucos feedbacks) quem governa é a heurística, segura e interpretável; conforme
        feedback real se acumula, o peso migra para a DQN. Em WARMUP_INTERACTIONS
        feedbacks, a rede assume integralmente.
        """
        heuristic = 1.0 - _minmax(pool_dist)
        q_norm = _minmax(q_values)
        w = min(1.0, self.agent.n_feedbacks / WARMUP_INTERACTIONS)
        return (1.0 - w) * heuristic + w * q_norm

    def _select_slots(self, adjusted: np.ndarray, pool_vectors: np.ndarray, k: int):
        """
        Slot 1: amostragem estocástica por softmax, para que o "melhor" item varie entre
        interações em vez de ser sempre o argmax. Slot exploratório (opcional): softmax
        com temperatura mais alta sobre o restante, em posição sorteada na lista final
        (não previsível ao usuário). Slots restantes: MMR (Maximal Marginal Relevance),
        para garantir que as opções sejam qualitativamente distintas entre si.

        Retorna (posições no pool, tipo de cada slot, propensão de cada slot). A
        propensão é exata para os slots sorteados e 1.0 para os slots determinísticos
        do MMR, condicionados aos sorteios anteriores. O campo slot_type permite que a
        análise off-policy futura (IPS/SNIPS/Doubly Robust) decida quais registros usar.
        """
        n = len(adjusted)
        k = min(k, n)
        remaining = list(range(n))
        chosen, slot_types, propensities = [], [], []

        probs = _softmax(adjusted[remaining], SOFTMAX_TEMPERATURE)
        pos = int(np.random.choice(remaining, p=probs))
        chosen.append(pos)
        slot_types.append("softmax")
        propensities.append(float(probs[remaining.index(pos)]))
        remaining.remove(pos)

        has_explore = False
        if remaining and len(chosen) < k and random.random() < P_EXPLORE_SLOT:
            probs = _softmax(adjusted[remaining], EXPLORE_TEMPERATURE)
            pos = int(np.random.choice(remaining, p=probs))
            chosen.append(pos)
            slot_types.append("explore")
            propensities.append(float(probs[remaining.index(pos)]))
            remaining.remove(pos)
            has_explore = True

        while len(chosen) < k and remaining:
            best_pos, best_value = None, -np.inf
            for candidate in remaining:
                similarity = max(
                    _cosine_similarity(pool_vectors[candidate], pool_vectors[s]) for s in chosen
                )
                value = MMR_LAMBDA * adjusted[candidate] - (1.0 - MMR_LAMBDA) * similarity
                if value > best_value:
                    best_pos, best_value = candidate, value
            chosen.append(best_pos)
            slot_types.append("mmr")
            propensities.append(1.0)
            remaining.remove(best_pos)

        if has_explore and len(chosen) > 1:
            order = list(range(len(chosen)))
            random.shuffle(order)
            chosen = [chosen[i] for i in order]
            slot_types = [slot_types[i] for i in order]
            propensities = [propensities[i] for i in order]

        return chosen, slot_types, propensities


# Simulador de feedback - Bancada de teste

"""Usado exclusivamente para testes offline, baselines e verificação do pipeline.
NUNCA deve ser usado para treinar o modelo que interage com usuários reais: treinar
e avaliar contra a mesma heurística faz o modelo apenas imitá-la, introduzindo viés
e circularidade metodológica."""

HIGH_ENERGY_OCTANTS = (3, 4)


def _simulate(curr_oct: int, dest_oct: int, item: pd.Series, bonus_fn) -> tuple[float, int]:
    curr_v, curr_a = OCTANT_MAP[curr_oct]
    dest_v, dest_a = OCTANT_MAP[dest_oct]
    item_v, item_a = float(item["Valencia"]), float(item["Arousal"])
    duration = float(item["Duracao"])

    # Probabilidade de execução: baixa energia tende a rejeitar itens ativadores/longos.
    p_execution = 1.0
    if curr_oct in LOW_ENERGY_OCTANTS:
        if item_a > 0.5:
            p_execution -= 0.3
        if duration > 30:
            p_execution -= 0.2
    p_execution = max(0.05, p_execution)
    if random.random() > p_execution:
        return -0.5, curr_oct

    # Alinhamento entre a mudança desejada (dest - curr) e o vetor (V, A) do item.
    delta_target = np.array([dest_v - curr_v, dest_a - curr_a])
    item_vec = np.array([item_v, item_a])
    norm = np.linalg.norm(delta_target) * np.linalg.norm(item_vec)
    alignment = float(np.dot(delta_target, item_vec) / norm) if norm > 1e-8 else 0.0

    score = 0.5 + 0.4 * alignment
    score += bonus_fn(curr_oct, item)
    score += random.gauss(0, 0.05)
    score = max(0.0, min(1.0, score))

    reward = -1.0 + 3.0 * score   # contínuo em [-1, 2], preserva granularidade
    next_oct = dest_oct if score > 0.65 else curr_oct
    return reward, next_oct


def simulate_feedback(curr_oct: int, dest_oct: int, item: pd.Series) -> tuple[float, int]:
    """
    Simula o feedback do usuário. Retorna (recompensa_continua, proximo_oitante).
    Perfil "principal", usado nos testes offline padrão.
    """
    def bonus(octant: int, item_: pd.Series) -> float:
        b = 0.0
        if octant == 4 and item_["Tipo"] == "Corporal":
            b += 0.1
        if octant == 3 and item_["Tipo"] == "Áudio":
            b += 0.1
        if octant in LOW_ENERGY_OCTANTS and item_["Tipo"] in ("Vídeo", "Jogo"):
            b += 0.05
        return b

    return _simulate(curr_oct, dest_oct, item, bonus)


def simulate_feedback_holdout(curr_oct: int, dest_oct: int, item: pd.Series) -> tuple[float, int]:
    """
    Perfil alternativo de feedback, com bônus contextuais diferentes de simulate_feedback.
    Uso exclusivo em avaliação (baselines): treinar e avaliar contra perfis distintos é o
    que evita medir imitação em vez de generalização.
    """
    def bonus(octant: int, item_: pd.Series) -> float:
        b = 0.0
        if octant in LOW_ENERGY_OCTANTS and item_["Tipo"] == "Mindfulness":
            b += 0.1
        if octant in HIGH_ENERGY_OCTANTS and item_["Tipo"] == "Imagem":
            b += 0.1
        if octant == 2 and item_["Indoor"] == 1:
            b += 0.05
        return b

    return _simulate(curr_oct, dest_oct, item, bonus)


# Avaliação offline - Bancada de teste

"""Verificações offline: sanity checks estruturais, baselines contra o simulador holdout
e métricas de cobertura/diversidade. Usa o simulador apenas para gerar feedback
sintético em avaliação — nunca para treinar o modelo de produção (ver seção acima)."""

CONTEXT_DURATIONS = (5, 15, 30, 60)


def sanity_checks(recommender: Recommender, df: pd.DataFrame) -> None:
    # Bateria de checagens estruturais sobre o pipeline de recomendação.
    print("- Sanity checks")

    # 1. Nenhuma recomendação excede o tempo disponível.
    violations = sum(
        1
        for time_avail in CONTEXT_DURATIONS
        for curr in range(1, 9)
        for dest in range(1, 9)
        for item in recommender.recommend(curr, dest, time_avail)
        if item["duracao"] > time_avail
    )
    print(f"[1] Recomendações excedendo o tempo disponível: {violations} (esperado: 0)")

    # 2. curr=5, dest=7: valência média das recomendações supera a média do catálogo.
    catalog_mean = df["Valencia"].mean()
    recs = recommender.recommend(5, 7, 60)
    rec_mean = np.mean([item["valencia"] for item in recs]) if recs else float("nan")
    status = "OK" if recs and rec_mean > catalog_mean else "FALHOU"
    print(f"[2] Valência média recomendada={rec_mean:.3f} vs catálogo={catalog_mean:.3f} ({status})")

    # 3. Mudar curr_oct (mantendo o resto) altera a lista em fração significativa dos casos.
    changed, total = 0, 0
    for dest in range(1, 9):
        base = {item["item_idx"] for item in recommender.recommend(1, dest, 60)}
        for curr in range(2, 9):
            other = {item["item_idx"] for item in recommender.recommend(curr, dest, 60)}
            total += 1
            changed += other != base
    print(f"[3] Listas alteradas ao mudar o oitante atual: {changed}/{total}")

    # 4. Guardrail bloqueia exatamente 22 itens para oitantes de baixa energia (valor medido no dataset atual).
    checked_before, blocked_before = recommender.safety_checked, recommender.safety_blocked
    recommender.recommend(5, 7, 120)
    checked = recommender.safety_checked - checked_before
    blocked = recommender.safety_blocked - blocked_before
    print(f"[4] Guardrail: {blocked}/{checked} itens bloqueados para o oitante 5 (esperado: 22)")

    # 5. Duas chamadas idênticas produzem listas diferentes em fração razoável dos casos.
    trials = 20
    differing = sum(
        1
        for _ in range(trials)
        if [i["item_idx"] for i in recommender.recommend(5, 7, 30)]
        != [i["item_idx"] for i in recommender.recommend(5, 7, 30)]
    )
    print(f"[5] Chamadas idênticas com listas diferentes: {differing}/{trials}")

    # 6. Cobertura de catálogo sobre a grade completa de contextos.
    seen = {
        item["item_idx"]
        for time_avail in CONTEXT_DURATIONS
        for curr in range(1, 9)
        for dest in range(1, 9)
        for item in recommender.recommend(curr, dest, time_avail)
    }
    print(f"[6] Cobertura de catálogo na grade completa: {len(seen)}/{len(df)} itens (TOP_K={TOP_K})")


def _bootstrap_ci(rewards: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    means = [np.mean(np.random.choice(rewards, size=len(rewards), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(rewards)), float(lo), float(hi)


def _random_context():
    curr = int(np.random.randint(1, 9))
    dest = int(np.random.randint(1, 9))
    time_avail = int(np.random.choice([5, 15, 30, 60, 120]))
    return curr, dest, time_avail


def baselines(df: pd.DataFrame, feature_space: FeatureSpace, n_episodes: int = 1000) -> None:
    """
    Compara: Aleatório; Mais popular (maior valência); Conteúdo puro (mais próximo do
    ponto-alvo, sem aprendizado); e um Agent aprendendo online — todos avaliados contra
    o simulador holdout (nunca o principal, para evitar circularidade). Reporta
    recompensa média com intervalo de confiança por bootstrap.

    Um sistema neural que não supera "conteúdo puro" não está agregando valor — este é
    o baseline crítico.
    """
    agent = Agent(df, feature_space)

    def pick_random(elig, curr, dest, time_avail):
        return int(np.random.choice(elig))

    def pick_popular(elig, curr, dest, time_avail):
        return int(elig[np.argmax(df.loc[elig, "Valencia"].to_numpy())])

    def pick_content(elig, curr, dest, time_avail):
        distances = distance_to_point(df.loc[elig], target_point(curr, dest))
        return int(elig[np.argmin(distances)])

    def pick_agent(elig, curr, dest, time_avail):
        user_state = feature_space.user_state(curr, dest, time_avail)
        return int(elig[np.argmax(agent.q_values(user_state, elig))])

    policies = {
        "aleatorio": pick_random,
        "mais_popular": pick_popular,
        "conteudo_puro": pick_content,
        "agent_online": pick_agent,
    }
    rewards = {name: [] for name in policies}

    print("- Baselines (simulador holdout)")
    for _ in range(n_episodes):
        curr, dest, time_avail = _random_context()
        elig = df.index[df["Duracao"] <= time_avail].to_numpy(dtype=int)
        if len(elig) == 0:
            continue

        for name, pick in policies.items():
            item_idx = pick(elig, curr, dest, time_avail)
            reward, next_oct = simulate_feedback_holdout(curr, dest, df.loc[item_idx])
            rewards[name].append(reward)
            if name == "agent_online":
                user_state = feature_space.user_state(curr, dest, time_avail)
                next_state = feature_space.user_state(next_oct, dest, time_avail)
                agent.learn_from_feedback(user_state, item_idx, reward, next_state)

    for name, values in rewards.items():
        mean, lo, hi = _bootstrap_ci(np.array(values))
        print(f"  {name:16s} recompensa média = {mean:+.3f}  IC95%=[{lo:+.3f}, {hi:+.3f}]  (n={len(values)})")


def coverage_and_diversity(recommender: Recommender, df: pd.DataFrame) -> None:
    """Sobre a grade de contextos: cobertura de catálogo, diversidade intra-lista média
    (1 - similaridade cosseno média entre pares) e frequência de recomendação por item."""
    counts = {idx: 0 for idx in df.index}
    diversities = []

    for time_avail in CONTEXT_DURATIONS:
        for curr in range(1, 9):
            for dest in range(1, 9):
                items = recommender.recommend(curr, dest, time_avail)
                for item in items:
                    counts[item["item_idx"]] += 1
                if len(items) > 1:
                    vectors = [recommender.feature_space.item_matrix[i["item_idx"]] for i in items]
                    sims = [
                        _cosine_similarity(vectors[i], vectors[j])
                        for i in range(len(vectors))
                        for j in range(i + 1, len(vectors))
                    ]
                    diversities.append(1.0 - float(np.mean(sims)))

    coverage = sum(1 for c in counts.values() if c > 0) / len(df)
    avg_diversity = float(np.mean(diversities)) if diversities else float("nan")
    top_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]

    print("- Cobertura e diversidade")
    print(f"Cobertura de catálogo: {coverage:.3f}")
    print(f"Diversidade intra-lista média: {avg_diversity:.3f}")
    print("Itens mais recomendados:")
    for item_idx, count in top_items:
        print(f"  {df.loc[item_idx, 'Nome']}: {count}")


# CLI interativo e ponto de entrada

def _print_octant_map() -> None:
    print("Mapa de oitantes:")
    for octant in range(1, 9):
        v, a = OCTANT_MAP[octant]
        print(f"  {octant}: {OCTANT_LABELS[octant]} (V={v:.2f}, A={a:.2f})")


def _read_octant(prompt: str) -> int:
    while True:
        value = input(prompt).strip().lower()
        if value in {"sair", "q"}:
            raise KeyboardInterrupt
        if value.isdigit() and 1 <= int(value) <= 8:
            return int(value)
        print("Entrada inválida. Digite um número entre 1 e 8, ou 'sair'.")


def _read_time(prompt: str) -> int:
    while True:
        value = input(prompt).strip().lower()
        if value in {"sair", "q"}:
            raise KeyboardInterrupt
        if value.isdigit() and 1 <= int(value) <= 120:
            return int(value)
        print("Entrada inválida. Digite um número entre 1 e 120, ou 'sair'.")


def _read_choice(prompt: str, options: list[str]) -> str:
    while True:
        value = input(prompt).strip().upper()
        if value in {"SAIR", "Q"}:
            raise KeyboardInterrupt
        if value in options:
            return value
        print(f"Escolha inválida. Use uma das opções: {', '.join(options)}.")


def _read_feedback() -> float:
    levels = {"1": -1.0, "2": 0.0, "3": 1.0, "4": 2.0}
    print("Feedback:")
    print("  [1] Não gostei / não funcionou")
    print("  [2] Indiferente / parcial")
    print("  [3] Gostei / funcionou")
    print("  [4] Gostei muito / muito eficaz")
    while True:
        value = input("Sua avaliação: ").strip()
        if value in levels:
            return levels[value]
        print("Feedback inválido. Escolha 1, 2, 3 ou 4.")


def _persist_interaction(record: dict, path: str = INTERACTION_LOG_PATH) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_interaction(recommender: Recommender, agent: Agent, feature_space: FeatureSpace) -> bool:
    """Executa uma rodada de recomendação + feedback. Retorna False para encerrar o loop."""
    try:
        curr_oct = _read_octant("Oitante atual (1-8): ")
        dest_oct = _read_octant("Oitante desejado (1-8): ")
        time_avail = _read_time("Tempo disponível (1-120 minutos): ")
    except KeyboardInterrupt:
        return False

    recommendations = recommender.recommend(curr_oct, dest_oct, time_avail)
    if not recommendations:
        print("Nenhuma intervenção disponível para o tempo informado.\n")
        return True

    labels = ["A", "B", "C"][: len(recommendations)]
    print("\nRecomendações:")
    for label, item in zip(labels, recommendations):
        print(
            f"  [{label}] {item['nome']} | {item['tipo']} | {item['tag']} | {item['duracao']} min | "
            f"V={item['valencia']:.2f} | A={item['arousal']:.2f} | indoor={item['indoor']} | "
            f"Q={item['q_value']:.3f} | π={item['propensity']:.4f}"
        )
    print("  [0] Nenhuma das opções")

    try:
        choice = _read_choice("Escolha uma opção: ", labels + ["0"])
    except KeyboardInterrupt:
        return False

    if choice == "0":
        print("Nenhuma intervenção selecionada. Sem aprendizado para esta rodada.\n")
        return True

    item = recommendations[labels.index(choice)]
    try:
        reward = _read_feedback()
    except KeyboardInterrupt:
        return False

    next_oct = dest_oct if reward > 0 else curr_oct
    user_state = feature_space.user_state(curr_oct, dest_oct, time_avail)
    next_state = feature_space.user_state(next_oct, dest_oct, time_avail)
    loss = agent.learn_from_feedback(user_state, item["item_idx"], reward, next_state)

    _persist_interaction({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "curr_oct": curr_oct,
        "dest_oct": dest_oct,
        "time_avail": time_avail,
        "item_idx": item["item_idx"],
        "reward": reward,
        "propensity": item["propensity"],
        "slot_type": item["slot_type"],
        "q_value": item["q_value"],
        "score": item["score"],
        "n_feedbacks": agent.n_feedbacks,
    })
    print(f"Interação registrada. Perda média: {loss:.4f}\n")
    return True


def main(dataset_path: str = None, carregar: str = None, salvar: str = CHECKPOINT_PATH,
         interativo: bool = True) -> Recommender:
    """
    Chamável sem argumentos. Em notebook/Colab:
        from sistema_de_recomendacao_v2 import main
        main()
        main(dataset_path="/content/drive/MyDrive/.../dataset.csv")
    """
    set_seed(SEED)
    df = load_dataset(dataset_path)
    feature_space = FeatureSpace(df)
    agent = Agent(df, feature_space)

    checkpoint = carregar if carregar is not None else salvar
    if checkpoint and os.path.exists(checkpoint):
        agent.load(checkpoint, feature_space)

    recommender = Recommender(df, feature_space, agent)
    try:
        if interativo:
            _print_octant_map()
            while _run_interaction(recommender, agent, feature_space):
                pass
    finally:
        agent.save(salvar)
        if interativo:
            print("\nSessão finalizada.")
            print(f"Interações com aprendizado: {agent.n_feedbacks}")
            print(f"Taxa de violação de segurança: {recommender.safety_violation_rate:.3f}")
            print(f"Cobertura do catálogo: {recommender.catalog_coverage:.3f}")

    return recommender


def _run_offline_evaluation() -> None:
    # Bancada de teste completa: sanity checks, cobertura/diversidade e baselines.
    set_seed(SEED)
    df = load_dataset()
    feature_space = FeatureSpace(df)
    agent = Agent(df, feature_space)
    recommender = Recommender(df, feature_space, agent)

    sanity_checks(recommender, df)
    print()
    coverage_and_diversity(recommender, df)
    print()
    baselines(df, feature_space, n_episodes=1000)


if __name__ == "__main__":
    # python sistema_de_recomendacao_v2.py --eval roda a bancada de teste offline
    # (simulador + baselines) em vez do loop interativo de produção.
    if "--eval" in sys.argv:
        _run_offline_evaluation()
    else:
        main()
