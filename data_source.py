"""
Leitura read-only do catálogo real (MongoDB, ao vivo, ou export local em JSON) e
normalização por modalidade para o esquema único usado pelo resto do sistema.

Dois backends, mesma forma de documento: mongoexport (--jsonArray) preserva a
estrutura exata dos documentos da coleção, só embrulhando _id/datas em Extended JSON.
Por isso normalize_video_doc/normalize_audio_doc/normalize_image_doc não sabem, nem
precisam saber, de qual backend um documento veio -- iter_raw_docs() (mais abaixo) é o
único ponto de acesso a documentos brutos, usado também por normalization.py e
safety.py, o que os torna igualmente agnósticos de backend.

Restrição de infraestrutura, inegociável para o backend Mongo: este módulo nunca
escreve no banco. Toda leitura é via find()/aggregate() sem $out/$merge; nenhuma
chamada de update/insert/delete é aceitável aqui. Bloqueio de itens perigosos é feito
inteiramente em código versionado (ver safety.py), como filtro em memória
aplicado depois da leitura e antes de qualquer uso do item pelo modelo.

load_catalog()/load_from_json_export() são substitutos diretos de
sistema_de_recomendacao_v3.load_dataset(): o ponto de entrada troca uma chamada pela
outra conforme DATA_BACKEND (ver sistema_de_recomendacao_v3.py), sem tocar em
FeatureSpace/Agent/Recommender.
"""
import json
import os
import re

import pandas as pd

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


def get_read_only_client(modality: str):
    """Cliente Mongo estritamente de leitura, para a coleção da modalidade dada
    ('video' | 'audio' | 'image' -- ver MONGO_COLLECTIONS). Nunca chamar
    update/insert/delete em nenhum lugar do código que usa este cliente.

    Import de pymongo é LOCAL de propósito: mantém `import data_source` funcionando
    sem pymongo instalado para quem só usa o backend 'csv' ou 'json_export' -- só
    quando esta função é de fato chamada (backend 'mongo') é que a dependência entra
    em jogo."""
    from pymongo import MongoClient

    client = MongoClient(sysrec.MONGO_URI)
    collection_name = sysrec.MONGO_COLLECTIONS[modality]
    return client[sysrec.MONGO_DB][collection_name]


def _unwrap_extended_json(obj):
    """Converte Extended JSON (modo relaxed, produzido por mongoexport --jsonArray)
    para tipos Python nativos: {"$oid": "..."} -> str; {"$date": "..."} -> str ISO
    (mantido como string; nenhum ponto do pipeline precisa de datetime). Recursivo
    sobre dicts e listas; qualquer outro valor passa direto."""
    if isinstance(obj, dict):
        if set(obj.keys()) == {"$oid"}:
            return obj["$oid"]
        if set(obj.keys()) == {"$date"}:
            return obj["$date"]
        return {k: _unwrap_extended_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap_extended_json(v) for v in obj]
    return obj


def _resolve_json_path(filename: str) -> str:
    """Resolve o caminho de um export dentro do diretório de referência ÚNICO onde as
    databases exportadas via mongoexport ficam armazenadas: sysrec.JSON_EXPORT_DIR
    (pasta dbs/ na raiz do projeto). Instrução explícita do usuário: este ponto NÃO
    segue a busca em múltiplos candidatos (diretório do módulo / cwd / etc.) usada em
    sistema_de_recomendacao_v3.resolve_dataset_path para o CSV -- dbs/ é a única
    referência aqui, mesmo que isso divirja do plano original.

    Um caminho absoluto já existente é aceito como está (ex.: uso avulso fora de
    dbs/); qualquer outro valor é tratado como nome de arquivo dentro de dbs/.
    """
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    candidate = os.path.join(sysrec.JSON_EXPORT_DIR, os.path.basename(filename))
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Export JSON não encontrado: '{candidate}' (diretório de referência: "
        f"{sysrec.JSON_EXPORT_DIR})."
    )


def _load_json_export(filename: str) -> list:
    """Lê um arquivo `mongoexport --jsonArray` de dbs/ e devolve uma lista de
    documentos com _id/datas desembrulhados -- mesmo formato que um documento de
    leitura ao vivo do Mongo."""
    resolved = _resolve_json_path(filename)
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(
            f"{resolved}: esperado um array JSON (gerar com mongoexport --jsonArray); "
            f"encontrado {type(raw).__name__}."
        )
    return [_unwrap_extended_json(doc) for doc in raw]


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

    Colunas extras preservadas (dataset, tipo_modalidade, category, tags, octant_raw,
    confidence_tier): não são consumidas por FeatureSpace/Recommender (que só olham
    EXPECTED_COLUMNS), mas são úteis para diagnóstico/auditoria sobre o catálogo já
    curado (ver characterize.py) -- sem elas, recaracterizar o catálogo real exigiria
    reconstruir o catálogo bruto separadamente. octant_raw é mantido À PARTE de
    Oitante (que continua puramente geométrico) justamente para permitir comparar
    rótulo declarado vs. geométrico sem reintroduzir dependência do rótulo declarado
    em nenhum caminho de produção.
    """
    extra_cols = ["dataset", "tipo_modalidade", "category", "tags", "octant_raw", "confidence_tier"]
    if len(df) == 0:
        return pd.DataFrame(columns=sysrec.EXPECTED_COLUMNS + ["item_id"] + extra_cols)

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

    # Colunas de diagnóstico, não usadas por FeatureSpace/Recommender -- ver docstring.
    for col in extra_cols:
        out[col] = df[col]

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


_NORMALIZERS_BY_MODALITY = {
    "video": normalize_video_doc, "audio": normalize_audio_doc, "image": normalize_image_doc,
}


def _append_if_valid(rows: list, row: dict, discarded: dict) -> None:
    if row.get("valencia_norm") is None or row.get("arousal_norm") is None:
        discarded["va_ausente"] += 1
        return
    rows.append(row)


def _build_catalog(video_docs, audio_docs, image_docs) -> pd.DataFrame:
    """Núcleo compartilhado pelos backends mongo e json_export: normaliza cada
    documento pelo adaptador de sua modalidade, descarta o que não tem V/A
    resolvível, e aplica a curadoria de segurança uma única vez no final."""
    rows: list = []
    discarded = {"va_ausente": 0}

    for doc in video_docs:
        if not doc.get("active", True):
            continue
        _append_if_valid(rows, normalize_video_doc(doc), discarded)
    for doc in audio_docs:
        if not doc.get("active", True):
            continue
        _append_if_valid(rows, normalize_audio_doc(doc), discarded)
    for doc in image_docs:
        if not doc.get("active", True):
            continue
        _append_if_valid(rows, normalize_image_doc(doc), discarded)

    raw_df = pd.DataFrame(rows, columns=UNIFIED_SCHEMA_FIELDS)
    before_safety = len(raw_df)
    safe_df = safety.apply_safety_filter(raw_df)
    discarded["curadoria_seguranca"] = before_safety - len(safe_df)

    catalog = _to_feature_space_schema(safe_df)
    _validate_feature_space_schema(catalog)

    total_descartado = sum(discarded.values())
    motivos = ", ".join(f"{motivo}={n}" for motivo, n in discarded.items())
    print(f"Catálogo carregado: {len(catalog)} itens seguros "
          f"({total_descartado} descartados: {motivos})")
    return catalog


def load_catalog() -> pd.DataFrame:
    """Backend 'mongo': lê as três coleções ao vivo (somente leitura, nunca escreve)
    e normaliza+cura para o esquema que FeatureSpace já espera -- substituto direto
    de load_dataset() no ponto de entrada do sistema."""
    return _build_catalog(
        video_docs=get_read_only_client("video").find({}),
        audio_docs=get_read_only_client("audio").find({}),
        image_docs=get_read_only_client("image").find({}),
    )


def load_from_json_export() -> pd.DataFrame:
    """Backend 'json_export': lê os arquivos locais gerados por
    `mongoexport --jsonArray`, um por modalidade, do diretório de referência
    sysrec.JSON_EXPORT_DIR (pasta dbs/ na raiz do projeto -- ver _resolve_json_path).
    Modalidade cujo arquivo não existir é pulada com aviso (não bloqueia as demais)
    -- permite testar hoje só com videos.json, antes de audios.json/images.json
    existirem."""
    docs_by_modality: dict = {}
    for modality, filename in sysrec.JSON_EXPORT_PATHS.items():
        try:
            docs_by_modality[modality] = _load_json_export(filename)
        except FileNotFoundError:
            print(f"[aviso] Export de '{modality}' não encontrado em "
                  f"'{os.path.join(sysrec.JSON_EXPORT_DIR, filename)}' -- catálogo "
                  f"ficará sem itens desta modalidade.")
            docs_by_modality[modality] = []

    return _build_catalog(
        video_docs=docs_by_modality["video"],
        audio_docs=docs_by_modality["audio"],
        image_docs=docs_by_modality["image"],
    )


def iter_raw_docs(modality: str = None, dataset_filter: str = None):
    """
    Itera documentos BRUTOS (antes de normalize_*_doc), do backend ativo em
    sysrec.DATA_BACKEND ('mongo' ou 'json_export'). Ponto único de acesso a dados
    brutos: normalization.py e safety.py usam esta função, nunca acessam Mongo ou
    arquivo diretamente -- assim funcionam sem alteração contra qualquer um dos
    dois backends.

    modality: 'video' | 'audio' | 'image' | None (as três, em sequência).
    dataset_filter: nome do dataset (ex. 'DEAM'), comparado sem diferenciar
        maiúsculas/minúsculas (a grafia varia entre datasets: 'DEAM' vs 'MuVi').

    IMPORTANTE: esta função contorna a curadoria de segurança de propósito -- é uma
    ferramenta de diagnóstico, usada justamente para DECIDIR o que a curadoria deve
    aprovar (ver safety.explore_taxonomy). O caminho de produção (load_catalog /
    load_from_json_export) sempre passa por safety.apply_safety_filter; este não.
    """
    modalities = [modality] if modality else ["video", "audio", "image"]

    for mod in modalities:
        if sysrec.DATA_BACKEND == "mongo":
            docs = get_read_only_client(mod).find({})
        elif sysrec.DATA_BACKEND == "json_export":
            try:
                docs = _load_json_export(sysrec.JSON_EXPORT_PATHS[mod])
            except FileNotFoundError:
                continue  # modalidade sem export ainda -- pular silenciosamente aqui
                          # (quem quiser saber disso usa load_from_json_export, que avisa)
        else:
            raise ValueError(
                f"iter_raw_docs não suporta DATA_BACKEND={sysrec.DATA_BACKEND!r} "
                "(use 'mongo' ou 'json_export')."
            )

        for doc in docs:
            if dataset_filter:
                doc_dataset = ((doc.get("sourceMeta") or {}).get("dataset")
                                or (doc.get("source") or {}).get("dataset") or "")
                if doc_dataset.upper() != dataset_filter.upper():
                    continue
            yield doc


def build_raw_catalog() -> pd.DataFrame:
    """
    Constrói o catálogo completo (todas as modalidades, todos os datasets) já
    normalizado para UNIFIED_SCHEMA_FIELDS, SEM aplicar a curadoria de segurança.

    Usado por normalization.run_consistency_audit(), que precisa ver os itens ANTES
    do filtro de segurança: com a allowlist vazia, itens de dataset ainda não
    revisado (ex.: EMOPIA) nunca apareceriam no catálogo final para serem
    sinalizados. Backend-agnóstico por construção: itera via iter_raw_docs(), o
    ponto único de acesso a documentos brutos (mongo ao vivo ou json_export,
    conforme sysrec.DATA_BACKEND).
    """
    rows: list = []
    for modality, normalize in _NORMALIZERS_BY_MODALITY.items():
        for doc in iter_raw_docs(modality=modality):
            if not doc.get("active", True):
                continue
            row = normalize(doc)
            if row["valencia_norm"] is None or row["arousal_norm"] is None:
                continue
            rows.append(row)
    return pd.DataFrame(rows, columns=UNIFIED_SCHEMA_FIELDS)
