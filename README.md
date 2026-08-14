# Sistema de Recomendação de Intervenções
Projeto de iniciação científica para o desenvolvimento de um sistema de recomendação de intervenções para o bem-estar, baseado em aprendizado por reforço.

## Visão geral do pipeline

```
fonte de dados (CSV | export JSON local em dbs/ | Mongo ao vivo)
        │
        ▼
carga + normalização (load_active_catalog)
        │
        ▼
curadoria de segurança (safety.py) ── backends json_export e mongo
        │
        ▼
FeatureSpace + Agent (DQN categórico) + Recommender
        │
        ▼
CLI interativo (main)  |  bancada de teste offline (--eval)
```

Todo o núcleo do sistema (config, features, agente, política de recomendação, CLI) vive
em `sistema_de_recomendacao_v3.py`. Os módulos `data_source.py`, `safety.py` e
`normalization.py` são companheiros, usados quando a fonte de dados é um catálogo real
(export JSON local ou MongoDB ao vivo) em vez do `dataset.csv` sintético.
`characterize.py` e `review_negative_tail.py` são bancadas de diagnóstico/curadoria
manual, fora do caminho de produção (ver seções 2 e 3 abaixo).

## Requisitos

- Python 3.10+
- `numpy`, `pandas`, `torch` (núcleo do sistema)
- `pymongo` (só necessário se `DATA_BACKEND = "mongo"` — **não** é necessário para
  `"csv"` nem para `"json_export"`; o import de `pymongo` em `data_source.py` é local à
  função que de fato conecta no banco)

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas torch pymongo
```

## 1. Carregar o dataset

A fonte de dados é escolhida pela constante `DATA_BACKEND` no topo de
`sistema_de_recomendacao_v3.py`:

- **`"csv"`** — lê `dataset.csv` via `load_dataset()`. Não depende de rede nem de
  `pymongo`. É o backend usado **sempre** pela bancada de teste offline (`--eval`),
  independentemente do valor de `DATA_BACKEND` — a bancada pressupõe esse dataset
  sintético específico.

- **`"json_export"` (padrão)** — lê um catálogo real a partir de arquivos locais gerados por
  `mongoexport --jsonArray`, um por modalidade, **sem depender de conexão viva nem de
  `pymongo` instalado**. Backend pensado para testar a normalização/curadoria contra
  dados reais offline, antes de (ou sem) ter acesso ao Mongo ao vivo.

  **Diretório de referência: `dbs/`, na raiz do projeto.** É o único lugar
  pesquisado — os arquivos precisam se chamar exatamente `videos.json`, `audios.json`
  e `images.json` dentro dessa pasta (`sysrec.JSON_EXPORT_DIR` /
  `sysrec.JSON_EXPORT_PATHS`, em `sistema_de_recomendacao_v3.py`). Modalidade cujo
  arquivo ainda não existir é pulada com aviso, sem bloquear as demais — dá para testar
  hoje só com `dbs/videos.json`, antes de `audios.json`/`images.json` existirem:

  ```bash
  mongoexport --uri "<mongo-uri>" --collection videos --jsonArray --out dbs/videos.json
  mongoexport --uri "<mongo-uri>" --collection audios --jsonArray --out dbs/audios.json
  mongoexport --uri "<mongo-uri>" --collection images --jsonArray --out dbs/images.json
  ```

  ```python
  DATA_BACKEND = "json_export"
  ```

- **`"mongo"`** — lê o catálogo real ao vivo via `data_source.load_catalog()`, com
  documentos multimodais (vídeo/áudio/imagem) vindos de datasets afetivos públicos
  (DEAM, MuVi, OASIS, EmoMadrid, GAPED, EMOPIA), uma coleção por modalidade. **O
  sistema nunca escreve no banco**: toda leitura é via `find()` somente leitura; não há
  nenhuma chamada de `update`/`insert`/`delete` em nenhum módulo (verificado
  automaticamente, ver seção 3).

  Antes de trocar para `"mongo"`, ajustar em `sistema_de_recomendacao_v3.py`:
  ```python
  DATA_BACKEND = "mongo"
  MONGO_URI = "mongodb://<host>/<db>?readPreference=secondary"  # idealmente um usuário read-only
  MONGO_DB = "nome_do_banco"
  MONGO_COLLECTIONS = {"video": "videos", "audio": "audios", "image": "images"}
  ```

  Tanto `load_catalog()` (mongo) quanto `load_from_json_export()` (json_export)
  normalizam cada documento (por modalidade, via `normalize_video_doc` /
  `normalize_audio_doc` / `normalize_image_doc`) para o mesmo esquema de colunas que
  `FeatureSpace` já espera (`Nome`, `Tipo`, `Valencia`, `Arousal`, `Duracao`, `Indoor`,
  `Tag`, `Oitante`) — substitutos diretos de `load_dataset()`, sem tocar em
  `FeatureSpace`/`Agent`/`Recommender`. Documentos sem valência/arousal resolvível são
  descartados e contados por motivo, nunca incluídos com valor nulo ou inventado. Os
  dois backends compartilham o mesmo núcleo de normalização/curadoria — a única
  diferença é de onde os documentos brutos vêm (`data_source.iter_raw_docs()` é o
  ponto único de acesso a documentos brutos, usado também por `normalization.py` e
  `safety.py`, o que os torna agnósticos de qual dos dois backends está ativo).

  Ponto único de carga (dá para chamar diretamente, sem se preocupar com o backend):
  ```python
  from sistema_de_recomendacao_v3 import load_active_catalog
  df = load_active_catalog()
  ```

## 2. Limpar / curar os dados (backends json_export e mongo)

Curadoria de segurança em `safety.py`, **100% em memória, nunca escreve no banco** —
chamada automaticamente por `data_source.load_catalog()`/`load_from_json_export()`.
Quatro camadas, aplicadas nessa ordem:

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

Antes de popular `APPROVED_CATEGORIES`, levantar a taxonomia real do backend ativo
(`DATA_BACKEND`, seção 1) — consulta só leitura, salvar a saída localmente para
revisão manual:

```python
from safety import explore_taxonomy

contagens, subcategorias = explore_taxonomy()
```

`explore_taxonomy()` itera documentos brutos via `data_source.iter_raw_docs()`,
contornando a curadoria de propósito — é a ferramenta que informa a decisão de
aprovar uma categoria, não pode depender da aprovação já ter acontecido.

`APPROVED_CATEGORIES`/`BLOCKED_ITEM_IDS` são editados manualmente em
`sistema_de_recomendacao_v3.py`, como qualquer alteração de código — revisado e
versionado em git, nunca escrito de volta no banco.

### Revisão manual da cauda negativa (`review_negative_tail.py`)

A denylist de palavra-chave e a revisão geométrica por valência (Camadas 2 e 3) não
pegam tudo: a denylist não pega conteúdo perturbador sem palavra-gatilho na
descrição, e mesmo a Camada 3 não pega tudo, porque a literatura do GAPED documenta
dessensibilização de avaliador — imagens de categoria negativa podem ter valência
moderada, não extrema. Para datasets sem taxonomia de categoria dedicada a conteúdo
negativo (**OASIS**, **EmoMadrid** — diferente do **GAPED**, que tem taxonomia fixa
documentada e se resolve estruturalmente, aprovando só as categorias `N`/`P`), a
aprovação de `APPROVED_CATEGORIES` depende de revisão humana item a item da cauda
de menor valência:

```bash
python review_negative_tail.py OASIS                    # lote de 100 itens, do mais negativo
python review_negative_tail.py EmoMadrid --n-items 100
python review_negative_tail.py OASIS --no-resume          # ignora o watermark, revisa do zero
```

Para cada item: `[a]`provar, `[b]`loquear, `[s]`kip (reaparece na próxima sessão) ou
`[q]`uit. Bloqueios são gravados incrementalmente em `blocked_item_ids.txt` (colar
depois em `BLOCKED_ITEM_IDS`) — uma sessão interrompida no meio não perde decisões
já tomadas. Progresso por dataset fica em `review_watermark.json`, para retomar sem
revisar o mesmo item duas vezes. **Critério de parada por dataset:** não um número
fixo pequeno — rodar em lotes de 100 e parar quando um lote inteiro não gerar
nenhum bloqueio (o script já sinaliza isso ao final de cada execução).

## 3. Testar

### Auditoria de normalização (`normalization.py`)

Script standalone, roda sob demanda contra o Mongo real (só leitura), fora do caminho
de produção — verifica se os campos `*Normalized` batem com o valor esperado a partir
do campo bruto e da escala de origem documentada em `NORMALIZATION_REFERENCE`:

```bash
python normalization.py
```

Datasets com escala não confirmada (`scale=None` em `NORMALIZATION_REFERENCE`) são
pulados com aviso explícito — nunca com uma fórmula assumida silenciosamente. Datasets
sem campo `*Normalized` separado (`scale="IDENTITY"`, ex.: MuVi) viram uma checagem de
faixa (`audit_identity_dataset`) em vez da comparação fórmula-vs-armazenado. Também
roda três checagens estatísticas complementares (faixa fora de `[-1,1]`, desvio-padrão
por dataset, comparação de região V-A entre datasets para o mesmo oitante rotulado) e
um checador de consistência interna (`run_consistency_audit`) que cruza tags, oitante
declarado e quadrante (quando existir) contra o sinal de valência/arousal — foi esse
checador que revelou que o EMOPIA tinha um valor contínuo armazenado espúrio,
inconsistente com as próprias tags/oitante/quadrante do item.

### Recaracterização estatística e calibração do guardrail (`characterize.py`)

Script standalone, fora do caminho de produção — roda contra o catálogo carregado
via `load_active_catalog()` (qualquer `DATA_BACKEND`) e produz um relatório
estatístico do dataset real (distribuição de Valência/Arousal geral e por
modalidade, densidade por oitante geométrico, duração por modalidade/bucket de
tempo, sobrevivência da curadoria por dataset, confiabilidade da origem
psychometric/heuristic, tamanho de pool na grade completa de contextos, reaudição
de normalização e consistência interna agregada):

```bash
python characterize.py
```

Também calibra os limiares do guardrail (`calibrate_guardrail_thresholds` para
arousal, `calibrate_valence_threshold` para valência aversiva) a partir de
percentis da distribuição **real** do catálogo já curado — nunca herdados do
dataset sintético. Uma calibração por percentil só é válida para a distribuição que
ela de fato vai filtrar: **recalibrar sempre que `APPROVED_CATEGORIES` mudar**, já
que restringir/expandir a allowlist muda a distribuição do catálogo carregado. A
ferramenta escolhe o candidato mais protetor que ainda mantém o pool mínimo (≥
`TOP_K` em toda a grade `curr × ALLOWED_DEST_OCTANTS × tempo`); se nenhum candidato
atender ao piso, sinaliza para revisão manual em vez de escolher um valor
silenciosamente inseguro. Os valores calibrados vão manualmente em
`SAFETY_AROUSAL_THRESHOLD`/`SAFETY_AVERSIVE_VALENCE_THRESHOLD`
(`sistema_de_recomendacao_v3.py`), como qualquer alteração de código.

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

A cada recomendação, o feedback é dado em 5 níveis (muito ruim -> muito bom), mais a
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
