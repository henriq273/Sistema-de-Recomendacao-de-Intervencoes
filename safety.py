"""
Curadoria de segurança do catálogo — 100% em memória, nunca escreve no banco.

Aplica até quatro camadas de filtro sobre o DataFrame já normalizado por
data_source.py, em memória, antes de qualquer uso do item pelo modelo:
  1. Allowlist de (dataset, category) — DESATIVADA por padrão (ver
     sysrec.SAFETY_CATEGORY_ALLOWLIST_ENABLED, decisão explícita do usuário: todo
     item passa por esta camada enquanto o flag for False). Quando reativada, só
     entra o que foi explicitamente aprovado em sysrec.APPROVED_CATEGORIES.
  2. Denylist de palavra-chave em nome/tags/category. Sempre ativa.
  3. Revisão geométrica: valência muito negativa exige aprovação explícita de
     categoria (redundante com a Camada 1 de propósito — defesa em profundidade).
     Com a Camada 1 desativada, esta camada também deixa de diferenciar por
     categoria (mesma razão de redundância, na direção oposta).
  4. Bloqueio individual por item_id, sempre por último, nunca sobrescrito pelas
     camadas anteriores. Sempre ativa.

A Camada 4 do processo original (revisão humana da fila prioritária + amostra
estratificada) não é código — é processo, executado pela equipe sobre os resultados
de explore_taxonomy() e sobre o que a Camada 3 sinalizar. O único artefato de código
correspondente é o preenchimento manual de APPROVED_CATEGORIES/BLOCKED_ITEM_IDS em
sistema_de_recomendacao_v3.py, feito como qualquer alteração de código — revisado e
versionado em git.
"""
import pandas as pd

import sistema_de_recomendacao_v3 as sysrec

# Import de data_source é local (dentro de explore_taxonomy), não no topo do arquivo:
# evita import circular (data_source importa safety.apply_safety_filter).


def apply_safety_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as camadas de curadoria ativas e devolve só os itens seguros."""
    before = len(df)
    if before == 0:
        print("Curadoria de segurança: 0 -> 0 itens (DataFrame de entrada já vazio).")
        return df.copy()

    # Camada 1: allowlist de categoria -- liga/desliga via
    # sysrec.SAFETY_CATEGORY_ALLOWLIST_ENABLED. Desativada (padrão atual, decisão
    # explícita do usuário): todo item passa por esta camada, sem consultar
    # APPROVED_CATEGORIES. Ativada: só (dataset, category) explicitamente aprovados
    # passam; com a allowlist vazia, bloqueia tudo -- comportamento seguro por padrão.
    if sysrec.SAFETY_CATEGORY_ALLOWLIST_ENABLED:
        mask_category = df.apply(
            lambda r: (r["dataset"], r["category"]) in sysrec.APPROVED_CATEGORIES, axis=1
        )
    else:
        mask_category = pd.Series(True, index=df.index)

    # Camada 2: denylist de palavra-chave em nome/tags/category.
    mask_keyword = ~df.apply(_matches_denylist_keyword, axis=1)

    # Camada 3: valência muito negativa exige aprovação explícita de categoria
    # (redundante com a Camada 1 de propósito -- defesa em profundidade).
    mask_geometric = df["valencia_norm"].apply(
        lambda v: v >= sysrec.SAFETY_MIN_VALENCE_REVIEW
    ) | mask_category

    # Camada 4: bloqueio individual por ID, sempre aplicado por último e nunca
    # sobrescrito pelas camadas anteriores.
    mask_not_blocked = ~df["item_id"].isin(sysrec.BLOCKED_ITEM_IDS)

    safe = df[mask_category & mask_keyword & mask_geometric & mask_not_blocked].copy()

    print(f"Curadoria de segurança: {before} -> {len(safe)} itens "
          f"({before - len(safe)} excluídos)")
    return safe


def _matches_denylist_keyword(row) -> bool:
    haystack = " ".join([
        str(row.get("nome", "")),
        " ".join(row.get("tags", []) or []),
        str(row.get("category", "")),
    ]).lower()
    return any(term in haystack for term in sysrec.SAFETY_DENYLIST_KEYWORDS)


def explore_taxonomy():
    """
    Roda uma vez, manualmente, para levantar todas as combinações
    (dataset, category, subcategories) existentes no backend ativo (mongo ao vivo ou
    json_export local, conforme sysrec.DATA_BACKEND). Usar o resultado para popular
    APPROVED_CATEGORIES em sistema_de_recomendacao_v3.py após revisão humana.

    Itera documentos BRUTOS via data_source.iter_raw_docs(), contornando a curadoria
    de segurança de propósito -- é assim que se decide o que a curadoria deve
    aprovar, não pode depender da aprovação já ter acontecido. No backend mongo, a
    leitura é somente find(), nunca escreve nada de volta no banco; salvar a saída
    localmente (arquivo texto/JSON) para revisão manual.
    """
    import data_source
    from collections import Counter, defaultdict

    counts: Counter = Counter()
    subcats_by_key: dict = defaultdict(set)

    for doc in data_source.iter_raw_docs():
        dataset = ((doc.get("sourceMeta") or {}).get("dataset")
                   or (doc.get("source") or {}).get("dataset") or "")
        category = doc.get("category", "")
        key = (dataset, category)
        counts[key] += 1
        subcats_by_key[key].update(doc.get("subcategories") or [])

    for key in sorted(counts, key=lambda k: (k[0], k[1])):
        dataset, category = key
        print(f"{dataset:16s} {category:24s} n={counts[key]:5d}  "
              f"subcats={sorted(subcats_by_key[key])}")
    return counts, subcats_by_key
