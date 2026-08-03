"""
Curadoria de segurança do catálogo — 100% em memória, nunca escreve no banco.

Aplica quatro camadas de filtro sobre o DataFrame já normalizado por data_source.py,
em memória, antes de qualquer uso do item pelo modelo:
  1. Allowlist de (dataset, category) — só entra o que foi explicitamente aprovado.
  2. Denylist de palavra-chave em nome/tags/category.
  3. Revisão geométrica: valência muito negativa exige aprovação explícita de
     categoria (redundante com a Camada 1 de propósito — defesa em profundidade).
  4. Bloqueio individual por item_id, sempre por último, nunca sobrescrito pelas
     camadas anteriores.

A Camada 4 do processo original (revisão humana da fila prioritária + amostra
estratificada) não é código — é processo, executado pela equipe sobre os resultados
de explore_taxonomy() e sobre o que a Camada 3 sinalizar. O único artefato de código
correspondente é o preenchimento manual de APPROVED_CATEGORIES/BLOCKED_ITEM_IDS em
sistema_de_recomendacao_v3.py, feito como qualquer alteração de código — revisado e
versionado em git.
"""
import pandas as pd

import sistema_de_recomendacao_v3 as sysrec


def apply_safety_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as quatro camadas de curadoria e devolve só os itens seguros."""
    before = len(df)
    if before == 0:
        print("Curadoria de segurança: 0 -> 0 itens (DataFrame de entrada já vazio).")
        return df.copy()

    # Camada 1: allowlist de categoria. Com a allowlist vazia (estado inicial), este
    # passo bloqueia TUDO — é o comportamento seguro por padrão, não um bug.
    mask_category = df.apply(
        lambda r: (r["dataset"], r["category"]) in sysrec.APPROVED_CATEGORIES, axis=1
    )

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


def explore_taxonomy(collection) -> list:
    """
    Roda uma vez, manualmente, para levantar todas as combinações
    (dataset, category, subcategories) existentes no banco. Usar o resultado para
    popular APPROVED_CATEGORIES em sistema_de_recomendacao_v3.py após revisão humana
    (ver seção 5.1 do plano).

    Consulta somente leitura (aggregate sem $out/$merge) — não escreve nada de volta
    no banco; salvar a saída localmente (arquivo texto/JSON) para revisão manual.
    """
    pipeline = [
        {"$group": {
            "_id": {"dataset": "$sourceMeta.dataset", "category": "$category"},
            "subcats": {"$addToSet": "$subcategories"},
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id.dataset": 1, "_id.category": 1}},
    ]
    return list(collection.aggregate(pipeline))
