"""
Leitura read-only do catálogo real (MongoDB) e normalização por modalidade para o
esquema único usado pelo resto do sistema.

Restrição de infraestrutura, inegociável: este módulo nunca escreve no banco. Toda
leitura é via find()/aggregate() sem $out/$merge; nenhuma chamada de
update/insert/delete é aceitável aqui. Bloqueio de itens perigosos é feito
inteiramente em código versionado (ver safety.py), como filtro em memória
aplicado depois da leitura e antes de qualquer uso do item pelo modelo.

load_catalog() é o substituto direto de sistema_de_recomendacao_v3.load_dataset(): o
ponto de entrada troca uma chamada pela outra conforme DATA_BACKEND (ver
sistema_de_recomendacao_v3.py), sem tocar em FeatureSpace/Agent/Recommender.
"""
import re

import pandas as pd
from pymongo import MongoClient

import safety
import sistema_de_recomendacao_v3 as sysrec

# Esquema unificado por item, antes da adaptação para as colunas do FeatureSpace.
UNIFIED_SCHEMA_FIELDS = [
    "item_id", "nome", "tipo_modalidade", "valencia_norm", "arousal_norm",
    "duracao_segundos", "tags", "category", "dataset", "octant_raw", "url",
    "confidence_tier",
]

_QUADRANT_TAG_RE = re.compile(r"quadrant_q([1-4])", re.IGNORECASE)

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


def _extract_emopia_quadrant(doc: dict) -> str | None:
    """
    Extrai o quadrante Q1-Q4 de tags/subcategories (formato 'quadrant_qN') — único
    caminho confiável para o quadrante do EMOPIA.

    CORREÇÃO (ver seção 5 do plano de revisões pós-implementação): o campo contínuo
    armazenado em staticAnnotations para EMOPIA é espúrio — verificado empiricamente
    que contradiz as tags do próprio documento, o oitante declarado E o quadrante
    também declarado, todos ao mesmo tempo, para o mesmo item. O `annotationType`
    ("russell_4q_inferred_va_from_reference_octant_centroid") e uma `description`
    que referencia "audio_deam" dentro de um item do EMOPIA indicam bug de geração
    com parâmetro trocado na ingestão, não ambiguidade legítima de dado. Por isso
    staticAnnotations NUNCA é usado para EMOPIA, nem como fallback — só o quadrante.
    """
    candidatos = (doc.get("tags") or []) + (doc.get("subcategories") or [])
    for tag in candidatos:
        m = _QUADRANT_TAG_RE.match(tag)
        if m:
            return f"Q{m.group(1)}"
    return None


def normalize_video_doc(doc: dict) -> dict:
    dataset = _get(doc, "sourceMeta.dataset") or ""
    return {
        "item_id": str(doc["_id"]),
        "nome": doc.get("title", ""),
        "tipo_modalidade": "video",
        # Sem campo *Normalized separado para vídeo/imagem em geral — confirmado que
        # ratings.valenceMean/arousalMean já é o valor a usar diretamente (ver MuVi,
        # NORMALIZATION_REFERENCE em sistema_de_recomendacao_v3.py, scale="IDENTITY").
        "valencia_norm": _get(doc, "ratings.valenceMean"),
        "arousal_norm": _get(doc, "ratings.arousalMean"),
        "duracao_segundos": doc.get("durationSeconds"),
        "tags": doc.get("tags", []) or [],
        "category": doc.get("category", ""),
        "dataset": dataset,
        "octant_raw": doc.get("videoOctant"),
        "url": doc.get("videoUrl", ""),
        "confidence_tier": sysrec.CONFIDENCE_TIER.get(dataset, "unknown"),
    }


def normalize_audio_doc(doc: dict) -> dict:
    # Áudio usa source.dataset (não sourceMeta.dataset, usado por vídeo/imagem) --
    # confirmado por exemplos reais: documentos de áudio não têm chave sourceMeta, e
    # os de vídeo/imagem não têm source no mesmo sentido. Ver test_dataset_field_path
    # _por_modalidade em test_data_pipeline.py, que protege esta distinção de uma
    # futura refatoração que tente "simplificar" para um caminho único.
    dataset = _get(doc, "source.dataset") or ""

    if dataset.upper() == "EMOPIA":
        # CORREÇÃO CRÍTICA: nunca usar staticAnnotations aqui -- valor demonstravelmente
        # espúrio e inconsistente com tags/oitante/quadrante no próprio documento (ver
        # _extract_emopia_quadrant e seção 5 do plano de revisões).
        quadrant = _extract_emopia_quadrant(doc)
        if quadrant and quadrant in sysrec.EMOPIA_QUADRANT_CENTROIDS:
            v, a = sysrec.EMOPIA_QUADRANT_CENTROIDS[quadrant]
        else:
            v, a = None, None  # sem quadrante reconhecível -> descartado (va_ausente)
    elif dataset.upper() == "DEAM":
        v = _get(doc, "staticAnnotations.valenceNormalized")
        a = _get(doc, "staticAnnotations.arousalNormalized")
    elif dataset.upper() == "MEDITATION_LOCAL":
        v = _get(doc, "staticAnnotations.valenceNormalized")
        a = _get(doc, "staticAnnotations.arousalNormalized")
    else:
        v, a = None, None  # dataset de áudio não reconhecido -> descartar, não adivinhar

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
        "confidence_tier": sysrec.CONFIDENCE_TIER.get(dataset, "unknown"),
    }


def normalize_image_doc(doc: dict) -> dict:
    dataset = _get(doc, "sourceMeta.dataset") or ""
    return {
        "item_id": str(doc["_id"]),
        "nome": doc.get("title", ""),
        "tipo_modalidade": "image",
        "valencia_norm": _get(doc, "ratings.valenceNormalized"),
        "arousal_norm": _get(doc, "ratings.arousalNormalized"),
        "duracao_segundos": doc.get("durationSeconds", 5),  # imagens têm duração arbitrada
        "tags": doc.get("tags", []) or [],
        "category": doc.get("category", ""),
        "dataset": dataset,
        "octant_raw": doc.get("imageOctant"),
        "url": doc.get("imageUrl", ""),
        "confidence_tier": sysrec.CONFIDENCE_TIER.get(dataset, "unknown"),
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


def build_raw_catalog() -> tuple[pd.DataFrame, dict]:
    """
    Lê o catálogo completo do Mongo (somente leitura via find(), nunca escreve) e
    normaliza por modalidade para o esquema unificado (UNIFIED_SCHEMA_FIELDS) — SEM
    aplicar a curadoria de segurança.

    Usado por load_catalog() (que aplica a curadoria em seguida) e por
    normalization.run_consistency_audit(), que precisa ver os itens ANTES do filtro
    de segurança: com a allowlist vazia, itens de dataset ainda não revisado (ex.:
    EMOPIA) nunca apareceriam no catálogo final para serem sinalizados.
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
    return raw_df, discard_counts


def load_catalog() -> pd.DataFrame:
    """
    Lê e normaliza o catálogo (build_raw_catalog), aplica a curadoria de segurança
    (safety.apply_safety_filter) e devolve um DataFrame com as mesmas colunas
    semânticas que FeatureSpace já espera — substituto direto de load_dataset() no
    ponto de entrada do sistema.
    """
    raw_df, discard_counts = build_raw_catalog()

    before_safety = len(raw_df)
    safe_df = safety.apply_safety_filter(raw_df)
    discard_counts["curadoria_seguranca"] = before_safety - len(safe_df)

    catalog = _to_feature_space_schema(safe_df)
    _validate_feature_space_schema(catalog)

    total_descartado = sum(discard_counts.values())
    motivos = ", ".join(f"{motivo}={n}" for motivo, n in discard_counts.items())
    print(f"Catálogo carregado: {len(catalog)} itens seguros "
          f"({total_descartado} descartados: {motivos})")
    return catalog
