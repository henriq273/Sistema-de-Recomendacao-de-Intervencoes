"""
Curadoria de segurança do catálogo — 100% em memória, nunca escreve no banco.

Aplica quatro camadas de filtro sobre o DataFrame já normalizado por data_source.py,
em memória, antes de qualquer uso do item pelo modelo:
  1. Allowlist de (dataset, category) — só entra o que foi explicitamente aprovado.
  2. Denylist de palavra-chave em nome/tags/category (correspondência por palavra
     inteira, não por subcadeia — ver _matches_denylist_keyword).
  3. Revisão geométrica: valência muito negativa exige aprovação explícita POR ITEM
     (REVIEWED_NEGATIVE_ITEM_IDS), independente da Camada 1 -- defesa em
     profundidade de verdade, não acoplada à allowlist de categoria (uma versão
     anterior tinha "| mask_category" aqui, que por absorção booleana anulava a
     camada inteira; ver histórico).
  4. Bloqueio individual por item_id, sempre por último, nunca sobrescrito pelas
     camadas anteriores.

A Camada 4 do processo original (revisão humana da fila prioritária + amostra
estratificada) não é código — é processo, executado pela equipe sobre os resultados
de explore_taxonomy() e sobre o que a Camada 3 sinalizar. O único artefato de código
correspondente é o preenchimento manual de APPROVED_CATEGORIES/BLOCKED_ITEM_IDS em
sistema_de_recomendacao_v3.py, feito como qualquer alteração de código — revisado e
versionado em git.
"""
import re

import pandas as pd

import sistema_de_recomendacao_v3 as sysrec

# Import de data_source é local (dentro de explore_taxonomy), não no topo do arquivo:
# evita import circular (data_source importa safety.apply_safety_filter).


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

    # Camada 3: valência muito negativa exige aprovação explícita POR ITEM
    # (REVIEWED_NEGATIVE_ITEM_IDS), independente da Camada 1 -- defesa em
    # profundidade de verdade, não acoplada à Camada 1. ANTES tinha "| mask_category"
    # aqui, o que por absorção booleana (A & (X | A) ≡ A, com A=mask_category) fazia
    # esta camada nunca excluir nada: a expressão final em `safe` já tem
    # `mask_category & ...`, então "| mask_category" a tornava redundante e a Camada 3
    # virava código morto para qualquer SAFETY_MIN_VALENCE_REVIEW. Uma categoria
    # aprovada em bloco não é evidência de que um item específico de valência muito
    # negativa dentro dela foi revisado -- por isso o escape hatch é por item, não
    # por categoria.
    mask_geometric = (
        (df["valencia_norm"] >= sysrec.SAFETY_MIN_VALENCE_REVIEW)
        | df["item_id"].isin(sysrec.REVIEWED_NEGATIVE_ITEM_IDS)
    )

    # Camada 4: bloqueio individual por ID, sempre aplicado por último e nunca
    # sobrescrito pelas camadas anteriores.
    mask_not_blocked = ~df["item_id"].isin(sysrec.BLOCKED_ITEM_IDS)

    safe = df[mask_category & mask_keyword & mask_geometric & mask_not_blocked].copy()

    print(f"Curadoria de segurança: {before} -> {len(safe)} itens "
          f"({before - len(safe)} excluídos)")
    return safe


def _normalize_haystack(row) -> str:
    """Junta os campos textuais (nome/tags/category -- o esquema bruto usado por
    apply_safety_filter não carrega subcategories, ver data_source.UNIFIED_SCHEMA_FIELDS)
    e normaliza separadores: tags usam underscore (ex.: "norm_violation"), e \\b não
    quebra em underscore porque \\w o inclui -- por isso _ e - viram espaço antes de
    aplicar fronteira de palavra em _matches_denylist_keyword."""
    parts = [
        str(row.get("nome", "")),
        " ".join(row.get("tags", []) or []),
        str(row.get("category", "")),
    ]
    text = " ".join(parts).lower()
    return re.sub(r"[_\-/]+", " ", text)


# Um padrão único com todos os termos em alternância -- compilado uma vez, não a
# cada linha do DataFrame. \b em volta de cada termo: correspondência por PALAVRA
# INTEIRA, não por subcadeia ("war" deixa de casar "warm"/"award"/"warehouse").
_DENYLIST_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sysrec.SAFETY_DENYLIST_KEYWORDS) + r")\b"
)


def _matches_denylist_keyword(row) -> bool:
    return bool(_DENYLIST_PATTERN.search(_normalize_haystack(row)))


def report_camada3_impact() -> None:
    """
    Lista os itens que a Camada 3 (revisão geométrica) agora bloqueia depois da
    correção do bug de absorção booleana em apply_safety_filter (ver comentário
    ali): itens de categoria aprovada (passariam pela Camada 1) mas com valência
    abaixo de SAFETY_MIN_VALENCE_REVIEW e ainda sem revisão individual
    (REVIEWED_NEGATIVE_ITEM_IDS). É o lote a rodar em review_negative_tail.py --
    cada um vira aprovado (entra em REVIEWED_NEGATIVE_ITEM_IDS, com comentário de
    proveniência) ou permanece bloqueado.
    """
    import data_source
    raw = data_source.build_raw_catalog()

    mask_category = raw.apply(
        lambda r: (r["dataset"], r["category"]) in sysrec.APPROVED_CATEGORIES, axis=1
    )
    below_threshold = raw["valencia_norm"] < sysrec.SAFETY_MIN_VALENCE_REVIEW
    not_reviewed = ~raw["item_id"].isin(sysrec.REVIEWED_NEGATIVE_ITEM_IDS)
    newly_blocked = raw[mask_category & below_threshold & not_reviewed]

    print("=== Itens bloqueados pela Camada 3 pós-correção (categoria aprovada, "
          "valência extrema, sem revisão individual) ===")
    print(f"  Total: {len(newly_blocked)}")
    for _, row in newly_blocked.sort_values("valencia_norm").iterrows():
        print(f"  {row['item_id']:>12}  valencia_norm={row['valencia_norm']:+.3f}  "
              f"dataset={row['dataset']:12s} category={row['category']}")


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


def audit_denylist_terms(n_samples: int = 3) -> None:
    """
    Para cada termo de SAFETY_DENYLIST_KEYWORDS: quantos itens do catálogo BRUTO
    (data_source.build_raw_catalog(), antes da curadoria) ele bloqueia sozinho, e
    uma amostra deles. Permite validar empiricamente cada termo em vez de adivinhar
    -- termos com 0 hits são candidatos a remoção; termos com hits inesperados
    revelam falso positivo. Usa a mesma normalização/correspondência de
    _matches_denylist_keyword (fronteira de palavra, separadores normalizados), não
    uma reimplementação -- os números aqui são exatamente o que a Camada 2 real
    bloqueia hoje.
    """
    import data_source
    raw = data_source.build_raw_catalog()
    haystacks = raw.apply(_normalize_haystack, axis=1)

    print("=== Auditoria da denylist, termo a termo (catálogo bruto) ===")
    for term in sysrec.SAFETY_DENYLIST_KEYWORDS:
        pattern = re.compile(rf"\b{re.escape(term)}\b")
        hits = haystacks.str.contains(pattern)
        n = int(hits.sum())
        flag = "   <-- 0 hits (candidato a remoção)" if n == 0 else ""
        print(f"  {term:16s} {n:>5} itens{flag}")
        if 0 < n <= 50:
            for nome in raw.loc[hits, "nome"].head(n_samples):
                print(f"      ex: {str(nome)[:60]}")

    total_blocked = int(haystacks.apply(lambda h: bool(_DENYLIST_PATTERN.search(h))).sum())
    print(f"\n  Total de itens bloqueados pela Camada 2 (denylist), catálogo bruto: "
          f"{total_blocked}/{len(raw)} ({total_blocked/len(raw):.1%})" if len(raw) else "")


# Datasets sem evidência de que as Camadas 2-4 (keyword/geométrica/item_id) falhem --
# diferente de OASIS e EmoMadrid, onde a revisão manual da cauda negativa
# (review_negative_tail.py) encontrou casos que a denylist e o filtro geométrico
# deixaram passar. GAPED fica de fora por ter regra estrutural própria (só
# categorias neutra/positiva da taxonomia documentada, ver APPROVED_CATEGORIES).
AUTO_APPROVE_SAFE_DATASETS = ("DEAM", "MEDITATION_LOCAL", "EMOPIA", "MuVi")


def auto_approve_clean_categories(safe_datasets=AUTO_APPROVE_SAFE_DATASETS,
                                   dry_run: bool = True) -> set:
    """
    Aprova em lote os pares (dataset, category) cujos nomes de categoria e
    subcategorias não contêm nenhum termo da denylist, restrito aos datasets
    passados. NÃO desativa nenhuma camada: as Camadas 2 (keyword por item),
    3 (valência extrema) e 4 (bloqueio por item_id) continuam atuando item a item
    dentro das categorias aprovadas -- aprovar uma categoria não libera itens
    individuais problemáticos dentro dela.

    dry_run=True (padrão): só imprime o que seria aprovado, não retorna decisão
    aplicada. Rodar primeiro em dry-run, revisar a saída, só então dry_run=False.
    """
    counts, subcats_by_key = explore_taxonomy()
    approved, skipped = set(), []

    for (dataset, category), n in sorted(counts.items()):
        if dataset not in safe_datasets:
            continue
        haystack = " ".join([
            category or "",
            " ".join(subcats_by_key.get((dataset, category), [])),
        ]).lower()
        hits = [t for t in sysrec.SAFETY_DENYLIST_KEYWORDS if t in haystack]
        if hits:
            skipped.append((dataset, category, n, hits))
            continue
        approved.add((dataset, category))

    print(f"\n=== auto_approve_clean_categories (dry_run={dry_run}) ===")
    print(f"Datasets considerados: {list(safe_datasets)}\n")
    print("Aprovados:")
    for dataset, category in sorted(approved):
        print(f"  ({dataset!r}, {category!r})  n={counts[(dataset, category)]}")
    if skipped:
        print("\nPulados (keyword na categoria/subcategoria):")
        for dataset, category, n, hits in skipped:
            print(f"  ({dataset!r}, {category!r})  n={n}  hits={hits}")

    print("\nDatasets NÃO cobertos por esta função, e por quê:")
    print("  OASIS, EmoMadrid -> revisão manual da cauda negativa pendente "
          "(Camadas 2/3 comprovadamente insuficientes nesses dois)")
    print("  GAPED            -> regra estrutural própria (só categorias neutra/positiva)")

    if dry_run:
        print("\n[dry-run] Nada foi aplicado. Revise a lista acima e rode com "
              "dry_run=False para obter o conjunto a colar em APPROVED_CATEGORIES.")
        return set()
    return approved
