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
`characterize.py`, `review_negative_tail.py` e `fatigue_diagnostics.py` são bancadas
de diagnóstico/curadoria manual, fora do caminho de produção (ver seções 2 e 3
abaixo).

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
  `pymongo`. Dataset sintético de prototipagem. A bancada de teste offline
  (`--eval`, ver seção "Bancada de teste offline do modelo" abaixo) segue
  `DATA_BACKEND` como qualquer outro ponto de carga do catálogo — para rodá-la
  sobre este dataset sintético (comportamento fixo e conhecido), defina
  `DATA_BACKEND = "csv"` antes de chamar `--eval`.

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

Para datasets **sem** evidência de que as Camadas 2–4 (keyword/geométrica/item_id)
deixem passar conteúdo problemático, `safety.auto_approve_clean_categories()`
automatiza a aprovação em lote: aprova todo par `(dataset, category)`, restrito aos
datasets passados, cuja categoria/subcategorias não batam com nenhum termo de
`SAFETY_DENYLIST_KEYWORDS`. Não desativa nenhuma camada — Camadas 2/3/4 continuam
atuando item a item dentro das categorias aprovadas.

```python
from safety import auto_approve_clean_categories

auto_approve_clean_categories(dry_run=True)   # só imprime o que seria aprovado
aprovados = auto_approve_clean_categories(dry_run=False)  # aplica e retorna o conjunto
```

Datasets sem taxonomia de categoria dedicada a conteúdo negativo (hoje **OASIS** e
**EmoMadrid**, onde a revisão manual da cauda negativa já encontrou casos que a
denylist/filtro geométrico deixaram passar) ficam deliberadamente fora de
`AUTO_APPROVE_SAFE_DATASETS` — dependem de `review_negative_tail.py` (abaixo). GAPED
também fica fora: resolve-se por regra estrutural própria (só as categorias
neutra/positiva da taxonomia documentada), não por auto-aprovação.

`APPROVED_CATEGORIES`/`BLOCKED_ITEM_IDS` são editados manualmente em
`sistema_de_recomendacao_v3.py`, como qualquer alteração de código (com comentário
de proveniência por bloco) — revisado e versionado em git, nunca escrito de volta
no banco.

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
estatístico do dataset real: composição do catálogo por dataset/modalidade/
(dataset, category) com aviso automático se um dataset concentrar mais de 50% do
total (`report_catalog_composition`), distribuição de Valência/Arousal geral e por
modalidade, densidade por oitante geométrico, duração por modalidade/bucket de
tempo, sobrevivência da curadoria por dataset, confiabilidade da origem
psychometric/heuristic, tamanho de pool na grade completa de contextos, reaudição
de normalização, consistência interna agregada, vocabulário de `Tipo`/`category`
referenciado pelos bônus dos simuladores de feedback contra o catálogo real
(`report_simulator_vocabulary` — sinaliza bônus que nunca disparam) e
degenerescência dos blocos de feature usados pelo MMR (`report_feature_degeneracy`
— quanto cada bloco distingue itens; informa se baixa diversidade intra-lista vem
de features degeneradas ou de um pool naturalmente homogêneo):

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
`verify_guardrail_effective` reconfirma, depois de aplicar os novos limiares, que o
guardrail de fato bloqueia itens (>0) em todo par `(curr, dest)` de
`LOW_ENERGY_OCTANTS`/`HIGH_ENERGY_OCTANTS` × `ALLOWED_DEST_OCTANTS` — um guardrail
que bloqueia 0 itens não está protegendo.

`report_guardrail_breakdown` decompõe o efeito das 4 regras (R1–R4) sobre o
catálogo inteiro e por par `(curr, dest)`, via `_apply_safety_filter_standalone`
(réplica pura de `Recommender._apply_safety_filter`, sem precisar instanciar
`Recommender`/`Agent`). Achado relevante: **R3 e R4 não são filtradas por
`curr_oct`** — R3 (`ta < 0`) depende só do destino, e para os destinos com
arousal-alvo negativo (`7`, `8`) bloqueia o mesmo conjunto de itens para
**qualquer** `curr_oct`, não só `LOW_ENERGY_OCTANTS`/`HIGH_ENERGY_OCTANTS`; R4
nunca teve restrição de `curr_oct`. As duas juntas são as regras com maior
impacto agregado no catálogo, porque se aplicam em toda chamada de `recommend()`
para aquele destino, não só em estados específicos. `report_calibration_staleness`
reporta os percentis que os limiares atuais implicam contra a distribuição real —
útil para notar quando a calibração ficou desatualizada (ex.: depois de
`safety.auto_approve_clean_categories` mudar a composição do catálogo) sem
precisar rodar a varredura completa de novo.

`compare_pool_with_without_guardrail` e o context manager `safety_filter_disabled`
(que desliga `sysrec.USE_SAFETY_FILTER` dentro do bloco `with`, restaurando o
valor original mesmo em caso de exceção) quantificam, célula a célula, quanto o
guardrail custa em tamanho de pool — uso exclusivo desta bancada de diagnóstico,
**nunca** em torno do loop interativo de produção (`main()`). Não é uma flag nova:
`USE_SAFETY_FILTER` continua sendo a única fonte de verdade para ligar/desligar o
guardrail geométrico; o context manager só evita que alguém esqueça de religar
manualmente durante uma exploração.

### Diagnóstico de espaçamento de recomendações (`fatigue_diagnostics.py`)

Script standalone, fora do caminho de produção — mede o comportamento real de
espaçamento entre reaparições do mesmo item (gaps), tanto em um contexto fixo
repetido (pior caso) quanto em contextos variados (caso médio), e compara a
penalidade suave de fadiga (`FATIGUE_LAMBDA`/`FATIGUE_HALFLIFE`) contra o gap real
de score entre o 1º e o 2º colocado do pool — responde se a penalidade suave tem,
sequer em tese, força para trocar o item escolhido:

```bash
python fatigue_diagnostics.py
```

Foi rodando este script que se confirmou (contra o catálogo real) que a penalidade
suave sozinha nunca garantia espaçamento: em 0% dos contextos ela superava o gap de
score real, e o mesmo item reaparecia na interação imediatamente seguinte em ~33%
dos casos no pior cenário. Por isso `FatigueTracker` tem um bloqueio **rígido**
(`FATIGUE_MIN_GAP`, nº mínimo de rodadas executadas antes de um item poder
reaparecer, aplicado no pool antes da pontuação) além da penalidade suave (que
continua atuando sobre os itens já liberados do bloqueio). O script simula que o
usuário sempre executa a recomendação do slot 1 (`FatigueTracker.mark_executed`,
mesmo caminho que `_run_interaction` usa de verdade) para medir o cooldown real.
Rodar este script de novo depois de mudar
`FATIGUE_MIN_GAP`/`CANDIDATE_POOL_SIZE` confirma que nenhum gap abaixo do mínimo
configurado aparece mais, e que o pool não ficou degenerado em nenhum contexto.

### Testes de aceitação (`test_data_pipeline.py`)

Cobre `data_source.py`/`safety.py`/`normalization.py` de ponta a ponta com uma
`FakeCollection` em memória — **não precisa de Mongo real**:

```bash
python test_data_pipeline.py
```

### Bancada de teste offline do modelo (`--eval`)

Sanity checks estruturais, cobertura/diversidade, uso da escala de feedback,
baselines (aleatório / mais popular / conteúdo puro / agente online) e calibração da
cabeça categórica — nunca usada para treinar o modelo real. Segue `DATA_BACKEND`
como qualquer outro ponto de carga do catálogo (não é mais um caso especial): com o
padrão atual (`"json_export"`), roda contra o catálogo real (GAPED +
DEAM/EMOPIA/MEDITATION_LOCAL/MuVi auto-aprovados, ver `APPROVED_CATEGORIES`). Para
reproduzir a bancada fixa sobre o CSV sintético, defina `DATA_BACKEND = "csv"` antes
de rodar.

O check [2] mede proximidade ao ponto-alvo no plano V-A (`check_proximity_to_target`,
sobre toda a grade `curr × ALLOWED_DEST_OCTANTS`) em vez de valência bruta num único
par — a comparação antiga não discriminava nada perto da média do catálogo. O check
[3] (`check_state_sensitivity`) roda em modo `deterministic=True` (sem softmax
exploratório nem embaralhamento de cauda) para isolar sensibilidade real ao oitante
atual do ruído estocástico já medido pelo check [5]; usa `register_fatigue=False`
para não inflar o contador de fadiga da instância de `Recommender` compartilhada
pelos demais checks.

```bash
python sistema_de_recomendacao_v3.py --eval
```

### Regret cumulativo e heatmap 8×8 (parte da bancada `--eval`)

Duas métricas adicionais, com garantias de isolamento opostas por design:

- **Regret cumulativo** (`report_regret`/`regret_curve`) compara, episódio a
  episódio independente (sem estado de cooldown entre eles — mesmo padrão de
  `baselines()`), a recompensa do agente contra um **oráculo** determinístico
  (`oracle_pick`/`expected_reward_proxy`): o item de maior valor esperado
  aproximado segundo o MESMO bônus de `simulate_feedback_holdout`
  (`_holdout_bonus`, fatorado como função de módulo para os dois lados
  reaproveitarem). **Nunca instancia `Recommender` nem `FatigueTracker`** —
  garantia deliberada, para que o regret meça só o que o agente aprendeu, sem o
  guardrail/fadiga/MMR de produção no meio.
- **Heatmap 8×8** (`report_octant_heatmap`/`octant_heatmap`) mede o oposto: o
  comportamento real de produção (`Recommender.recommend()`, guardrail + fadiga +
  MMR inclusos) por par (oitante atual, oitante desejado), com uma instância nova
  de `Recommender` a cada célula — a fadiga não vaza entre células, mas
  **dentro** da mesma célula o cooldown é mantido de propósito, para refletir
  pedidos repetidos do mesmo par. Usa o agente já treinado por `baselines()`
  (`trained_agent`), não um agente recém-inicializado.

`_p_execution`/`_alignment_score` (fatorados de `_simulate`) expõem a parte
determinística do simulador holdout para o oráculo, sem precisar rodar Monte
Carlo a cada candidato — `expected_reward_proxy` é uma aproximação (ignora o
ruído gaussiano, que tem média zero, e quantiza o valor médio em vez de calcular
o valor esperado da quantização exatamente), então o regret medido é fiel ao
simulador holdout, não uma verdade de usuário real.

O treino de baselines (3000 episódios) domina o tempo total pelo número de
iterações, não pelo tamanho do catálogo — já levava minutos contra o CSV sintético
e continua na mesma ordem de grandeza contra o catálogo real; rode com um timeout
generoso ou em background.

O simulador de feedback usado aqui (`simulate_feedback_holdout`) usa bônus por
`category` contra o catálogo real (ex.: GAPED "positive"/"neutral") em vez de
Tipo/Indoor, que degeneravam lá — ver a docstring da função.

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

**Filtro por tempo disponível — EM STAND-BY (`TIME_FILTER_ENABLED = False`).** O CLI
não pergunta mais o tempo disponível, e `recommend()` não filtra mais por `Duracao`
nesta versão. O filtro e a pergunta continuam no código, só desativados por essa
flag — reativar trocando `TIME_FILTER_ENABLED` para `True` em
`sistema_de_recomendacao_v3.py`, sem precisar restaurar nada manualmente.

**Espaçamento de recomendações (`FATIGUE_MIN_GAP = 10`).** Além da penalidade suave
de fadiga (que só desestimula, nunca garante espaçamento — ver
`fatigue_diagnostics.py` acima), um item **executado** não pode reaparecer nas
`FATIGUE_MIN_GAP` rodadas seguintes (bloqueio rígido, aplicado no pool antes da
pontuação). O cooldown só vale para o item que o usuário de fato escolheu e
executou (`FatigueTracker.mark_executed`, chamado em `_run_interaction` quando a
escolha não é "[0] Não executei") — itens apenas exibidos em outros slots, ou
rodadas em que nada foi executado, não entram em cooldown. Ajustar `FATIGUE_MIN_GAP`
em `sistema_de_recomendacao_v3.py`; valores muito altos frente a
`CANDIDATE_POOL_SIZE` podem degenerar o pool em catálogos pequenos ou pouco
diversos — reconfirmar com `fatigue_diagnostics.py` depois de mudar.

## Limpeza de dados residuais

`clear.py` remove o checkpoint e o log de interações gerados por execuções/testes
(dry-run por padrão — só lista o que seria removido):

```bash
python clear.py                    # dry-run
python clear.py --confirm           # remove checkpoint*.pt e interaction_log*.jsonl
python clear.py --confirm --cache    # também remove __pycache__
python clear.py --help              # demais opções (--manter-log, --manter-checkpoint, --extra)
```
