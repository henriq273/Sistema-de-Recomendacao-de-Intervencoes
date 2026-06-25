from __future__ import annotations
 
import argparse
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from collections import namedtuple

DEFAULT_DATASET_PATH = os.environ.get(
    "INTERVENTIONS_DATASET",
    str(Path(__file__).parent / "dataset.csv"),
)

items_df = pd.read_csv(DEFAULT_DATASET_PATH)

# Mapeamento aproximado dos Oitantes para coordenadas (Valência, Arousal)
# Supondo distribuição circular anti-horária começando da "Alegria/Excitação"

OCTANT_MAP = {
    1: (0.8, 0.8),    # Excitado / Eufórico
    2: (0.3, 0.9),    # Tenso / Alerta
    3: (-0.3, 0.9),   # Estressado / Ansioso
    4: (-0.8, 0.8),   # Irritado / Raiva
    5: (-0.8, -0.2),  # Triste / Deprimido
    6: (-0.3, -0.8),  # Entediado / Cansado
    7: (0.3, -0.8),   # Calmo / Relaxado
    8: (0.8, -0.2),   # Sereno / Contente
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
 
Transition = namedtuple(
    "Transition",
    ("user_state", "item_features", "reward", "next_user_state", "done"),
)
 
def set_seed(seed: int) -> None:
    """Garante reprodutibilidade entre execuções."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

MAX_DURATION = float(items_df['Duracao'].max())
NUM_ITEMS = len(items_df)
EPISODES = 1000

# Mapeamentos para One-Hot Encoding
ALL_TYPES = items_df['Tipo'].unique().tolist()
ALL_TAGS = items_df['Tag'].unique().tolist()

class ContentAwareDQN(nn.Module):
    def __init__(self, user_state_dim, item_features_dim):
        super(ContentAwareDQN, self).__init__()

        # A entrada é a concatenação do Estado do Usuário + Features do Item
        input_dim = user_state_dim + item_features_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128) # Batch Norm ajuda na estabilidade
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1) # Saída: Q-Value único para este par (Usuário, Item)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Evitar overfitting em dados pequenos

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.output(x)

class RLRecommender:
    def __init__(self, df):
        self.items_df = df

        # Dimensões dos vetores
        # User State: Oitante Atual (8) + Oitante Desejado (8) + Tempo Disp (1) = 17
        self.user_state_dim = 17

        # Item Features: One-Hot Tipo (len(types)) + Indoor (1) + One-Hot Tag (len(tags)) + Norm Duracao (1) + Norm Valencia (1) + Norm Arousal (1)
        self.item_feat_dim = len(ALL_TYPES) + 1 + len(ALL_TAGS) + 1 + 1 + 1

        # Hiperparâmetros
        self.gamma = 0.90
        self.epsilon = 1.0
        self.epsilon_min = 0.05 # Mantem-se 5% de exploração sempre para descobrir novas prefs
        self.epsilon_decay = 0.997
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.memory = deque(maxlen=5000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inicializa Rede Neural
        self.policy_net = ContentAwareDQN(self.user_state_dim, self.item_feat_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Cache de features dos itens para não recalcular sempre
        self.item_features_cache = self._precompute_item_features()

    def _precompute_item_features(self):
        """Gera a matriz de features de todos os itens antecipadamente."""
        cache = {}
        for idx, row in self.items_df.iterrows():
            # One-hot Tipo
            type_vec = np.zeros(len(ALL_TYPES))
            if row['Tipo'] in ALL_TYPES:
                type_vec[ALL_TYPES.index(row['Tipo'])] = 1

            # One-hot Tag
            tag_vec = np.zeros(len(ALL_TAGS))
            if row['Tag'] in ALL_TAGS:
                tag_vec[ALL_TAGS.index(row['Tag'])] = 1

            # Outros atributos normalizados
            indoor = [row['Indoor']]
            dur_norm = [row['Duracao'] / MAX_DURATION]
            val_norm = [(row['Valencia'] - 1) / 8] # Normalizando 1-9 para 0-1
            ar_norm = [(row['Arousal'] - 1) / 8]

            # Concatena tudo
            features = np.concatenate((type_vec, indoor, tag_vec, dur_norm, val_norm, ar_norm))
            cache[idx] = features
        return cache

    def get_user_state(self, curr_oct, dest_oct, time_avail):
        s_curr = np.zeros(8)
        s_curr[curr_oct - 1] = 1
        s_dest = np.zeros(8)
        s_dest[dest_oct - 1] = 1
        time_norm = [time_avail / MAX_DURATION]
        return np.concatenate((s_curr, s_dest, time_norm))

    def select_action_training(self, user_state_vec, valid_indices):
        
        # Método usado apenas no treinamento simulado (seleciona 1 item).
        
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_indices)

        return self._get_best_indices(user_state_vec, valid_indices, k=1)[0]

    def recommend_top_k(self, user_state_vec, valid_indices, k=3):
        
        # Método para inferência real. Retorna os Top-K índices e seus Q-values.
        
        # Aqui não usa-se Epsilon-Greedy puro, pois se quer mostrar as melhores.
        # A exploração acontece se o usuário escolher a opção 2 ou 3 da lista,
        # ou se for adicionado ruído aos Q-values (opcional).

        top_indices = self._get_best_indices(user_state_vec, valid_indices, k)
        return top_indices

    def _get_best_indices(self, user_state_vec, valid_indices, k=1):
        # Função auxiliar interna para calcular Q-values e ordenar
        self.policy_net.eval()
        with torch.no_grad():
            batch_inputs = []
            for idx in valid_indices:
                item_feat = self.item_features_cache[idx]
                combined = np.concatenate((user_state_vec, item_feat))
                batch_inputs.append(combined)

            tensor_inputs = torch.FloatTensor(np.array(batch_inputs)).to(self.device)
            q_values = self.policy_net(tensor_inputs).cpu().numpy().flatten()

            # Pega os índices dos K maiores Q-values
            # argsort ordena crescente, pega os ultimos k e inverte
            if len(q_values) < k: k = len(q_values)

            top_k_local_indices = np.argsort(q_values)[-k:][::-1]

            # Mapeia de volta para os índices reais do dataframe
            return [valid_indices[i] for i in top_k_local_indices]

    def store_transition(self, u_state, item_idx, reward, next_u_state, done):
        item_feat = self.item_features_cache[item_idx]
        self.memory.append(Transition(u_state, item_feat, reward, next_u_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size: return 0.0

        self.policy_net.train()
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        u_states = torch.FloatTensor(np.array(batch.user_state)).to(self.device)
        i_feats = torch.FloatTensor(np.array(batch.item_features)).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_u_states = torch.FloatTensor(np.array(batch.next_user_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        current_inputs = torch.cat((u_states, i_feats), dim=1)
        curr_Q = self.policy_net(current_inputs)

        next_inputs = torch.cat((next_u_states, i_feats), dim=1)
        with torch.no_grad():
            next_Q = self.policy_net(next_inputs)

        target_Q = rewards + (self.gamma * next_Q * (1 - dones))
        loss = self.loss_fn(curr_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

# Simulador

def simulate_environment_feedback(curr_oct, dest_oct, item):
    # Lógica de simulação
    curr_v, curr_a = OCTANT_MAP[curr_oct]
    dest_v, dest_a = OCTANT_MAP[dest_oct]

    p_execution = 1.0
    if curr_oct in [5, 6]:
        if item['Arousal'] > 7: p_execution -= 0.3
        if item['Duracao'] > 30: p_execution -= 0.2

    if random.random() > max(0, p_execution): return 0

    delta_v = dest_v - curr_v
    item_v = (item['Valencia'] - 5) / 4
    score = 0.5

    if delta_v > 0 and item_v > 0: score += 0.3
    if delta_v > 0 and item_v < 0: score -= 0.4
    if curr_oct == 4 and item['Tipo'] == 'Atividade': score += 0.3
    if curr_oct == 3 and item['Tipo'] == 'Audio': score += 0.3

    final_score = score + random.uniform(-0.1, 0.1)

    # Retorna recompensa simplificada para o treino simulado
    if final_score < 0.4: return -1
    elif final_score < 0.8: return 1
    else: return 2

# Loop Principal

if __name__ == "__main__":
    agent = RLRecommender(items_df)

    print("=== Fase de Pré-Treinamento do Modelo (Simulação) ===")
    print(f"Treinando em {EPISODES} episódios...")

    for e in range(EPISODES):
        curr = random.randint(1, 8)
        dest = random.randint(1, 8)
        time = random.choice([5, 10, 15, 30, 45, 60])

        u_state = agent.get_user_state(curr, dest, time)
        valid_items = items_df[items_df['Duracao'] <= time]
        if valid_items.empty: continue

        # No treino simulado, escolhe apenas 1 ação (epsilon-greedy)
        action_idx = agent.select_action_training(u_state, valid_items.index.tolist())
        selected_item = items_df.iloc[action_idx]

        reward = simulate_environment_feedback(curr, dest, selected_item)

        next_curr = random.randint(1, 8)
        next_state = agent.get_user_state(next_curr, dest, time)

        agent.store_transition(u_state, action_idx, reward, next_state, False)
        agent.replay()

        if e % 250 == 0:
            print(f"Progresso: {e}/{EPISODES} | Epsilon: {agent.epsilon:.2f}")

    print("\n=== Fase de Interação com o Usuário (Top-3 Recomendações) ===")
    agent.epsilon = 0 # Desativa epsilon para usar a ordenação da rede, exploração será via escolha do usuário

    '''
    Explorar alternativa em que foi habilitada a exploração mesmo na fase real, para descobrir novas preferências,
    mas isso pode confundir o usuário se as recomendações mudarem drasticamente. Por isso, optou-se por mostrar
    sempre as melhores segundo o modelo e deixar a exploração acontecer via feedback do usuário
    (escolhendo opções 2 ou 3, ou dando feedback positivo para itens menos recomendados).
    '''

    while True:
        try:
            print("\n" + "="*60)
            i_curr = input("Oitante Atual (1-8) [sair]: ")
            if i_curr.lower() == 'sair': break
            i_dest = input("Oitante Desejado (1-8): ")
            i_time = input("Tempo disponível (min): ")

            curr, dest, time = int(i_curr), int(i_dest), float(i_time)

            # 1. Preparar Estado
            u_state = agent.get_user_state(curr, dest, time)

            # 2. Filtrar
            valid_items = items_df[items_df['Duracao'] <= time]
            if valid_items.empty:
                print(">> Nenhuma intervenção disponível para este tempo.")
                continue

            # 3. Obter Top-3
            valid_idx = valid_items.index.tolist()
            # Se houver menos de 3 itens, recomenda o que tiver
            top_k = agent.recommend_top_k(u_state, valid_idx, k=3)

            # 4. Exibição Formatada
            print(f"\nPara sair do Oitante {curr} e ir para {dest} em {time}min, sugerimos:")

            print(f"\nMelhor Opção:")
            best_item = items_df.iloc[top_k[0]]
            print(f"   [A] {best_item['Nome']}")
            print(f"       ( {best_item['Tipo']} - {best_item['Tag']} - {best_item['Duracao']}min - Val: ({best_item['Valencia']}) - Arous: ({best_item['Arousal']}) )")

            print(f"\nOutras Recomendações:")
            other_options = []
            if len(top_k) > 1:
                item_2 = items_df.iloc[top_k[1]]
                print(f"   [B] {item_2['Nome']} ( {item_2['Tipo']} - {item_2['Tag']} - Val: ({item_2['Valencia']}) - Arous: ({item_2['Arousal']}) )")
                other_options.append('B')
            if len(top_k) > 2:
                item_3 = items_df.iloc[top_k[2]]
                print(f"   [C] {item_3['Nome']} ( {item_3['Tipo']} - {item_3['Tag']} - Val: ({item_3['Valencia']}) - Arous: ({item_3['Arousal']}) )")
                other_options.append('C')

            # 5. Coleta de Escolha e Feedback
            print("\nQual intervenção você realizou?")
            print("   Digite A, B, C ou 0 (Nenhuma)")
            choice_char = input("Sua escolha: ").upper()

            chosen_idx = -1
            if choice_char == 'A': chosen_idx = top_k[0]
            elif choice_char == 'B' and len(top_k) > 1: chosen_idx = top_k[1]
            elif choice_char == 'C' and len(top_k) > 2: chosen_idx = top_k[2]
            elif choice_char == '0':
                print("Entendido. Nenhuma ação registrada.")
                continue
            else:
                print("Escolha inválida.")
                continue

            # 6. Avaliação
            print(f"\nComo foi a experiência com '{items_df.iloc[chosen_idx]['Nome']}'?")
            print("  [1] Não gostei/Não funcionou (-1.0)")
            print("  [2] Gostei/Funcionou (+1.0)")
            print("  [3] Gostei muito! (+2.0)")

            fb_val = input("Avaliação (1-3): ")
            rewards_map = {'1': -1.0, '2': 1.0, '3': 2.0}

            if fb_val in rewards_map:
                reward = rewards_map[fb_val]

                # Aprendizado Online
                # Assume-se sucesso -> Estado Desejado, Falha -> Estado Atual
                next_curr_real = dest if reward > 0 else curr
                next_state = agent.get_user_state(next_curr_real, dest, 0)

                agent.store_transition(u_state, chosen_idx, reward, next_state, True)

                loss_val = 0
                for _ in range(3): loss_val = agent.replay()
                print(f">> Modelo atualizado! (Loss: {loss_val:.4f})")
            else:
                print("Feedback inválido.")

        except ValueError:
            print("Erro de entrada.")
        except Exception as e:
            print(f"Erro: {e}")