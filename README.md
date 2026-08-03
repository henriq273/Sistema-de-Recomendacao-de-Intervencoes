# Sistema de Recomendação de Intervenções
Projeto de iniciação científica para o desenvolvimento de um sistema de recomendação de intervenções para o bem-estar, baseado em aprendizado por reforço.

## Visão geral do pipeline

```
fonte de dados (CSV ou Mongo)
        │
        ▼
carga + normalização (load_active_catalog)
        │
        ▼
curadoria de segurança (safety.py) ── só quando o backend é Mongo
        │
        ▼
FeatureSpace + Agent (DQN categórico) + Recommender
        │
        ▼
CLI interativo (main)  |  bancada de teste offline (--eval)
```

Todo o núcleo do sistema (config, features, agente, política de recomendação, CLI) vive
em `sistema_de_recomendacao_v3.py`. Os módulos `data_source.py`, `safety.py` e
`normalization.py` são companheiros, usados quando a fonte de dados é um MongoDB real
em vez do `dataset.csv` sintético.

## Requisitos

- Python 3.10+
- `numpy`, `pandas`, `torch` (núcleo do sistema)
- `pymongo` (só necessário se `DATA_BACKEND = "mongo"`)

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas torch pymongo
```

## 1. Carregar o dataset

A fonte de dados é escolhida pela constante `DATA_BACKEND` no topo de
`sistema_de_recomendacao_v3.py`:

- **`"csv"` (padrão)** — lê `dataset.csv` via `load_dataset()`. Não depende de rede nem
  de `pymongo`. É o backend usado tanto pelo CLI interativo (com `DATA_BACKEND="csv"`)
  quanto, **sempre**, pela bancada de teste offline (`--eval`), independentemente do
  valor de `DATA_BACKEND` — a bancada pressupõe esse dataset sintético específico
  (ex.: o sanity check de guardrail espera bloquear exatamente 22/100 itens).

- **`"mongo"`** — lê um catálogo real via `data_source.load_catalog()`, com documentos
  multimodais (vídeo/áudio/imagem) vindos de datasets afetivos públicos (DEAM, MuVi,
  OASIS, EmoMadrid, GAPED, EMOPIA). **O sistema nunca escreve no banco**: toda leitura é
  via `find()`/`aggregate()` somente leitura; não há nenhuma chamada de
  `update`/`insert`/`delete` em nenhum módulo (verificado automaticamente, ver seção 3).

  Antes de trocar para `"mongo"`, ajustar em `sistema_de_recomendacao_v3.py`:
  ```python
  DATA_BACKEND = "mongo"
  MONGO_URI = "mongodb://<host>/<db>?readPreference=secondary"  # idealmente um usuário read-only
  MONGO_DB = "nome_do_banco"
  MONGO_COLLECTION = "interventions"
  ```

  `load_catalog()` normaliza cada documento (por modalidade) para o mesmo esquema de
  colunas que `FeatureSpace` já espera (`Nome`, `Tipo`, `Valencia`, `Arousal`,
  `Duracao`, `Indoor`, `Tag`, `Oitante`) — é um substituto direto de `load_dataset()`,
  sem tocar em `FeatureSpace`/`Agent`/`Recommender`. Documentos com modalidade
  desconhecida ou sem valência/arousal resolvível são descartados e contados por
  motivo, nunca incluídos com valor nulo ou inventado.

  Ponto único de carga (dá para chamar diretamente, sem se preocupar com o backend):
  ```python
  from sistema_de_recomendacao_v3 import load_active_catalog
  df = load_active_catalog()
  ```

## 2. Limpar / curar os dados (backend Mongo)

Curadoria de segurança em `safety.py`, **100% em memória, nunca escreve no banco** —
chamada automaticamente por `data_source.load_catalog()`. Quatro camadas, aplicadas
nessa ordem:

1. **Allowlist de categoria** — só entram itens de `(dataset, category)` explicitamente
   aprovados em `APPROVED_CATEGORIES` (em `sistema_de_recomendacao_v3.py`). Começa
   **vazia** de propósito: com a allowlist vazia, o catálogo carregado fica vazio — é
   o comportamento seguro por padrão, não um bug.
2. **Denylist de palavra-chave** — bloqueia itens cujo nome/tags/categoria batam com
   `SAFETY_DENYLIST_KEYWORDS` (termos como "abuse", "violence", "gore", "phobia" etc.).
3. **Revisão geométrica** — valência abaixo de `SAFETY_MIN_VALENCE_REVIEW` exige
   aprovação explícita de categoria (defesa em profundidade, redundante com a Camada 1
   de propósito).
4. **Bloqueio individual por ID** — `BLOCKED_ITEM_IDS`, sempre aplicado por último,
   nunca sobrescrito pelas camadas anteriores.

Antes de popular `APPROVED_CATEGORIES`, levantar a taxonomia real do banco (consulta
só leitura, salvar a saída localmente para revisão manual):

```python
from data_source import get_read_only_client
from safety import explore_taxonomy

resultado = explore_taxonomy(get_read_only_client())
```

`APPROVED_CATEGORIES`/`BLOCKED_ITEM_IDS` são editados manualmente em
`sistema_de_recomendacao_v3.py`, como qualquer alteração de código — revisado e
versionado em git, nunca escrito de volta no banco.

## 3. Testar

### Auditoria de normalização (`normalization.py`)

Script standalone, roda sob demanda contra o Mongo real (só leitura), fora do caminho
de produção — verifica se os campos `*Normalized` batem com o valor esperado a partir
do campo bruto e da escala de origem documentada em `NORMALIZATION_REFERENCE`:

```bash
python normalization.py
```

Datasets com escala não confirmada (`scale=None` em `NORMALIZATION_REFERENCE`, ex.:
MuVi, EmoMadrid) são pulados com aviso explícito — nunca com uma fórmula assumida
silenciosamente. Também roda três checagens estatísticas complementares (faixa fora de
`[-1,1]`, desvio-padrão por dataset, comparação de região V-A entre datasets para o
mesmo oitante rotulado).

### Testes de aceitação (`test_data_pipeline.py`)

Cobre `data_source.py`/`safety.py`/`normalization.py` de ponta a ponta com uma
`FakeCollection` em memória — **não precisa de Mongo real**:

```bash
python test_data_pipeline.py
```

### Bancada de teste offline do modelo (`--eval`)

Sanity checks estruturais, cobertura/diversidade, uso da escala de feedback,
baselines (aleatório / mais popular / conteúdo puro / agente online) e calibração da
cabeça categórica — sempre sobre `dataset.csv`, nunca usada para treinar o modelo real:

```bash
python sistema_de_recomendacao_v3.py --eval
```

## 4. Rodar o programa

Loop interativo de recomendação + feedback (carrega/salva `checkpoint_v3.pt` e grava
cada interação em `interaction_log.jsonl`):

```bash
python sistema_de_recomendacao_v3.py
```

Em notebook/Colab:

```python
from sistema_de_recomendacao_v3 import main
main()
main(dataset_path="/content/drive/MyDrive/.../dataset.csv")
```

A cada recomendação, o feedback é dado em 5 níveis (muito ruim → muito bom), mais a
opção de não executar nenhuma intervenção (sem aprendizado nessa rodada).

## Limpeza de dados residuais

`clear.py` remove o checkpoint e o log de interações gerados por execuções/testes
(dry-run por padrão — só lista o que seria removido):

```bash
python clear.py                    # dry-run
python clear.py --confirmar         # remove checkpoint*.pt e interaction_log*.jsonl
python clear.py --confirmar --cache  # também remove __pycache__
python clear.py --help              # demais opções (--manter-log, --manter-checkpoint, --extra)
```
