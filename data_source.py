"""
Leitura read-only do catálogo real (MongoDB) e normalização por modalidade para o
esquema único usado pelo resto do sistema.

Restrição de infraestrutura, inegociável: este módulo nunca escreve no banco. Toda
leitura é via find()/aggregate() sem $out/$merge; nenhuma chamada de
update/insert/delete é aceitável aqui. Bloqueio de itens perigosos é feito
inteiramente em código versionado (ver safety_rules.py), como filtro em memória
aplicado depois da leitura e antes de qualquer uso do item pelo modelo.

load_catalog() é o substituto direto de sistema_de_recomendacao_v3.load_dataset(): o
ponto de entrada troca uma chamada pela outra conforme DATA_BACKEND (ver
sistema_de_recomendacao_v3.py), sem tocar em FeatureSpace/Agent/Recommender.
"""
import pandas as pd
from pymongo import MongoClient

import safety_rules
import sistema_de_recomendacao_v3 as sysrec

# Esquema unificado por item, antes da adaptação para as colunas do FeatureSpace.
UNIFIED_SCHEMA_FIELDS = [
    "item_id", "nome", "tipo_modalidade", "valencia_norm", "arousal_norm",
    "duracao_segundos", "tags", "category", "dataset", "octant_raw", "url",
]

# tipo_modalidade (vocabulário do banco) -> Tipo em português já usado pelo
# FeatureSpace/simulador. Os datasets afetivos de origem não têm as categorias mais
# ricas do catálogo sintético (Mindfulness, Corporal, Jogo) — a troca de fonte de dados
# perde essa granularidade por natureza, não é um bug de mapeamento.
_MODALIDADE_TO_TIPO = {"video": "Vídeo", "audio": "Áudio", "image": "Imagem"}


def get_read_only_client():
    """Cliente Mongo estritamente de leitura. Nunca chamar update/insert/delete em
    nenhum lugar do código que usa este cliente."""
    client = MongoClient(sysrec.MONGO_URI)
    return client[sysrec.MONGO_DB][sysrec.MONGO_COLLECTION]


def _get(doc: dict, path: str):
    """Acesso a campo aninhado por caminho tipo 'a.b.c'. Nunca lança exceção por campo
    ausente — devolve None, que o chamador trata (documento é descartado, ver
    load_catalog(), não quebra a carga inteira por um campo faltante)."""
    value = doc
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _resolve_non_continuous_va(doc: dict, dataset: str):
    """Datasets sem campo de V/A contínuo (ex.: EMOPIA, que só anota quadrante
    Q1-Q4). Resolve via centroide do quadrante quando possível; devolve (None, None)
    quando não — o documento é descartado em load_catalog() (contado em
    'va_ausente'), nunca incluído com valor nulo ou com fórmula inventada."""
    if dataset == "EMOPIA":
        quadrant = doc.get("quadrant")  # confirmar nome exato do campo no schema real
        if quadrant in sysrec.EMOPIA_QUADRANT_CENTROIDS:
            return sysrec.EMOPIA_QUADRANT_CENTROIDS[quadrant]
    return None, None


def normalize_video_doc(doc: dict) -> dict:
    return {
        "item_id": str(doc["_id"]),
        "nome": doc.get("title", ""),
        "tipo_modalidade": "video",
        # já costuma vir perto de [-1,1]; CONFIRMAR fonte antes de assumir (ver
        # audit_normalization.py e NORMALIZATION_REFERENCE em sistema_de_recomendacao_v3.py)
        "valencia_norm": _get(doc, "ratings.valenceMean"),
        "arousal_norm": _get(doc, "ratings.arousalMean"),
        "duracao_segundos": doc.get("durationSeconds"),
        "tags": doc.get("tags", []) or [],
        "category": doc.get("category", ""),
        "dataset": _get(doc, "sourceMeta.dataset") or "",
        "octant_raw": doc.get("videoOctant"),
        "url": doc.get("videoUrl", ""),
    }


def normalize_audio_doc(doc: dict) -> dict:
    dataset = _get(doc, "source.dataset") or ""
    if dataset == "DEAM":
        v = _get(doc, "staticAnnotations.valenceNormalized")
        a = _get(doc, "staticAnnotations.arousalNormalized")
    else:
        # EMOPIA e outros sem campo contínuo -> centroide de quadrante, se aplicável.
        v, a = _resolve_non_continuous_va(doc, dataset)
    return {
        "item_id": str(doc["_id"]),
        "nome": doc.get("title", ""),
        "tipo_modalidade": "audio",
        "valencia_norm": v,
        "arousal_norm": a,
        "duracao_segundos": doc.get("durationSeconds"),
        "tags": doc.get("tags", []) or [],
        "category": doc.get("category", ""),
        "dataset": dataset,
        "octant_raw": doc.get("soundOctant"),
        "url": doc.get("audioUrl", ""),
    }


def normalize_image_doc(doc: dict) -> dict:
    return {
        "item_id": str(doc["_id"]),
        "nome": doc.get("title", ""),
        "tipo_modalidade": "image",
        "valencia_norm": _get(doc, "ratings.valenceNormalized"),
        "arousal_norm": _get(doc, "ratings.arousalNormalized"),
        "duracao_segundos": doc.get("durationSeconds", 5),  # imagens têm duração arbitrada
        "tags": doc.get("tags", []) or [],
        "category": doc.get("category", ""),
        "dataset": _get(doc, "sourceMeta.dataset") or "",
        "octant_raw": doc.get("imageOctant"),
        "url": doc.get("imageUrl", ""),
    }


def _to_feature_space_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapta o esquema unificado (item_id/nome/tipo_modalidade/valencia_norm/...) para as
    colunas que FeatureSpace já espera (EXPECTED_COLUMNS em
    sistema_de_recomendacao_v3.py) — ponto de junção único, escolhido para não precisar
    tocar em FeatureSpace/Recommender.

    Três decisões de projeto que a fonte de dados não resolve sozinha, documentadas
    aqui para revisão:
      - Indoor: nenhum dos datasets afetivos de origem anota ambiente interno/externo.
        Fica fixo em 0 (não indoor) até que uma fonte real substitua isso — é um
        placeholder neutro, não uma inferência.
      - Tag: o esquema unificado guarda uma lista de tags; FeatureSpace espera uma
        única string por item. Usa a primeira tag; na ausência de tags, cai para
        category; na ausência de category, "Outro".
      - Oitante: recalculado geometricamente via nearest_octant(valencia_norm,
        arousal_norm), não a partir de octant_raw — o caminho de recomendação nunca
        depende dessa coluna, só da geometria (V, A) (mesmo motivo já documentado em
        nearest_octant()/distance_to_point() no núcleo do v3).
    """
    if len(df) == 0:
        return pd.DataFrame(columns=sysrec.EXPECTED_COLUMNS + ["item_id"])

    out = pd.DataFrame(index=df.index)
    out["Nome"] = df["nome"]
    out["Tipo"] = df["tipo_modalidade"].map(_MODALIDADE_TO_TIPO).fillna("Outro")
    out["Valencia"] = df["valencia_norm"].astype(float)
    out["Arousal"] = df["arousal_norm"].astype(float)
    # duracao_segundos -> minutos: o resto do sistema (elegibilidade por tempo
    # disponível, prompt do CLI) trabalha em minutos, não em segundos.
    out["Duracao"] = df["duracao_segundos"].fillna(0).astype(float) / 60.0
    out["Indoor"] = 0
    out["Tag"] = df.apply(
        lambda r: (r["tags"][0] if r["tags"] else None) or r["category"] or "Outro", axis=1
    )
    out["Oitante"] = [
        sysrec.nearest_octant(v, a) for v, a in zip(out["Valencia"], out["Arousal"])
    ]
    out["item_id"] = df["item_id"]  # preservado para rastreio (BLOCKED_ITEM_IDS, logs)
    return out.reset_index(drop=True)


def _validate_feature_space_schema(df: pd.DataFrame) -> None:
    missing = [c for c in sysrec.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no catálogo normalizado: {missing}")
    if len(df) == 0:
        return  # catálogo vazio é válido (allowlist ainda não populada) -- nada a checar
    if not df["Valencia"].between(-1.0, 1.0).all():
        raise ValueError("Valores de Valencia fora do intervalo [-1, 1] após normalização.")
    if not df["Arousal"].between(-1.0, 1.0).all():
        raise ValueError("Valores de Arousal fora do intervalo [-1, 1] após normalização.")


def load_catalog() -> pd.DataFrame:
    """
    Lê o catálogo completo do Mongo (somente leitura via find(), nunca escreve),
    normaliza por modalidade, aplica a curadoria de segurança
    (safety_rules.apply_safety_filter) e devolve um DataFrame com as mesmas colunas
    semânticas que FeatureSpace já espera — substituto direto de load_dataset() no
    ponto de entrada do sistema.
    """
    collection = get_read_only_client()
    rows = []
    discard_counts = {"modalidade_desconhecida": 0, "va_ausente": 0}

    for doc in collection.find({}):  # somente leitura
        modalidade = doc.get("mediaType")
        if modalidade == "video":
            row = normalize_video_doc(doc)
        elif modalidade == "audio":
            row = normalize_audio_doc(doc)
        elif modalidade == "image":
            row = normalize_image_doc(doc)
        else:
            discard_counts["modalidade_desconhecida"] += 1
            continue

        if row["valencia_norm"] is None or row["arousal_norm"] is None:
            discard_counts["va_ausente"] += 1
            continue
        rows.append(row)

    raw_df = pd.DataFrame(rows, columns=UNIFIED_SCHEMA_FIELDS)
    before_safety = len(raw_df)
    safe_df = safety_rules.apply_safety_filter(raw_df)
    discard_counts["curadoria_seguranca"] = before_safety - len(safe_df)

    catalog = _to_feature_space_schema(safe_df)
    _validate_feature_space_schema(catalog)

    total_descartado = sum(discard_counts.values())
    motivos = ", ".join(f"{motivo}={n}" for motivo, n in discard_counts.items())
    print(f"Catálogo carregado: {len(catalog)} itens seguros "
          f"({total_descartado} descartados: {motivos})")
    return catalog
