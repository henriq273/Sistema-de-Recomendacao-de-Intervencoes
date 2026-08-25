"""
Testes de aceitação da auditoria de normalização e da curadoria de segurança, sobre
os backends "mongo" (coleções separadas por modalidade, ver MONGO_COLLECTIONS) e
"json_export" (arquivos locais em dbs/, ver JSON_EXPORT_DIR). Roda sem Mongo real:
usa uma FakeCollection em memória que implementa só find(), o suficiente para
exercitar data_source.py, safety.py e normalization.py de ponta a ponta.

Não é integrado ao --eval do sistema_de_recomendacao_v3.py (que é a bancada do
protótipo, sempre sobre o CSV sintético) -- este arquivo testa especificamente os
módulos novos desta migração para Mongo/json_export. Roda como script, sem framework
de teste (mesmo estilo de sanity_checks() em sistema_de_recomendacao_v3.py: funções
que imprimem PASS/FAIL, sem dependência de pytest).

Uso:
    python test_data_pipeline.py
"""
import io
import json
import os
import re
import sys
import tempfile
import types
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import data_source
import safety
import sistema_de_recomendacao_v3 as sysrec

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

_FAILURES = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


class FakeCollection:
    """Substitui pymongo.Collection nos testes: find() sobre uma lista de dicts em
    memória. Não implementa update/insert/delete de propósito -- se algum código sob
    teste tentasse chamar um desses métodos, o teste quebraria com AttributeError, o
    que é o comportamento correto a se ter."""

    def __init__(self, docs: list):
        self._docs = docs

    def find(self, query: dict | None = None):
        query = query or {}
        return [doc for doc in self._docs if self._matches(doc, query)]

    @staticmethod
    def _matches(doc: dict, query: dict) -> bool:
        for key, cond in query.items():
            if key == "$or":
                if not any(FakeCollection._matches(doc, sub) for sub in cond):
                    return False
                continue
            value = data_source._get(doc, key)
            if isinstance(cond, dict):
                for op, opval in cond.items():
                    if op == "$lt" and not (value is not None and value < opval):
                        return False
                    if op == "$gt" and not (value is not None and value > opval):
                        return False
            elif value != cond:
                return False
        return True


def _fake_mongo_backend(video: list = None, audio: list = None, image: list = None) -> None:
    """Monkeypatcha data_source.get_read_only_client para devolver uma FakeCollection
    por modalidade (espelhando MONGO_COLLECTIONS: uma coleção por modalidade, não uma
    coleção única com campo mediaType) e força sysrec.DATA_BACKEND = 'mongo', que é o
    que iter_raw_docs() consulta para decidir por onde iterar."""
    by_modality = {"video": video or [], "audio": audio or [], "image": image or []}
    sysrec.DATA_BACKEND = "mongo"
    data_source.get_read_only_client = lambda modality: FakeCollection(by_modality[modality])


# ---------- Documentos fake, um grupo por modalidade (uma coleção cada, sem campo
# mediaType -- a modalidade é dada por qual coleção/lista o documento veio) ----------

DOCS_VIDEO = [
    # 1) vídeo OASIS válido, categoria aprovada -> deve passar.
    {
        "_id": "v1", "title": "Praia ao pôr do sol",
        "ratings": {"valenceMean": 0.6, "arousalMean": -0.3},
        "durationSeconds": 120, "tags": ["nature", "calm"], "category": "nature",
        "sourceMeta": {"dataset": "OASIS"}, "videoOctant": 8, "videoUrl": "http://x/v1",
    },
    # 7) vídeo com categoria aprovada, mas item_id vai para BLOCKED_ITEM_IDS -> excluído (Camada 4).
    {
        "_id": "v2", "title": "Bloqueado manualmente",
        "ratings": {"valenceMean": 0.5, "arousalMean": -0.2},
        "durationSeconds": 60, "tags": ["nature"], "category": "nature",
        "sourceMeta": {"dataset": "OASIS"}, "videoOctant": 8,
    },
    # 9) vídeo sem V/A (ratings ausente) -> descartado por va_ausente.
    {
        "_id": "v3", "title": "Sem avaliação",
        "durationSeconds": 30, "category": "nature", "sourceMeta": {"dataset": "OASIS"},
    },
]

DOCS_AUDIO = [
    # 2) áudio DEAM válido, categoria aprovada -> deve passar.
    {
        "_id": "a1", "title": "Chuva suave",
        "staticAnnotations": {"valenceNormalized": 0.1, "arousalNormalized": -0.6},
        "source": {"dataset": "DEAM"}, "durationSeconds": 180,
        "tags": ["rain"], "category": "ambient", "soundOctant": 7, "audioUrl": "http://x/a1",
    },
    # 3) áudio EMOPIA com quadrante reconhecido via tag -> resolve via centroide,
    #    categoria aprovada. Quadrante vem de tags/subcategories (formato
    #    "quadrant_qN"), nunca de staticAnnotations -- ver correção da seção 5.
    {
        "_id": "a2", "title": "Trilha alegre",
        "source": {"dataset": "EMOPIA"}, "durationSeconds": 90,
        "tags": ["quadrant_q1", "energetic"], "category": "music_energetic",
    },
    # 4) áudio EMOPIA SEM quadrante reconhecido nas tags -> descartado (va_ausente), não incluído com nulo.
    {
        "_id": "a3", "title": "Sem quadrante",
        "source": {"dataset": "EMOPIA"}, "tags": ["ambient"],
        "category": "music_energetic",
    },
]

DOCS_IMAGE = [
    # 5) imagem com keyword de denylist na tag, categoria aprovada -> deve ser excluída (Camada 2).
    {
        "_id": "i1", "title": "Cena de guerra",
        "ratings": {"valenceNormalized": -0.5, "arousalNormalized": 0.7},
        "tags": ["war"], "category": "nature", "sourceMeta": {"dataset": "OASIS"},
    },
    # 6) imagem com valência muito negativa, categoria NÃO aprovada -> excluída (Camadas 1+3).
    {
        "_id": "i2", "title": "Estímulo aversivo",
        "ratings": {"valenceNormalized": -0.9, "arousalNormalized": 0.5},
        "tags": [], "category": "aversive_research", "sourceMeta": {"dataset": "GAPED"},
    },
]


def test_write_methods_absent():
    """Teste 1: nenhuma chamada de escrita aparece em nenhum arquivo do projeto."""
    forbidden_calls = re.compile(
        r"\.(update_one|update_many|insert_one|insert_many|delete_one|delete_many|"
        r"replace_one|find_one_and_update|find_one_and_delete|find_one_and_replace|"
        r"bulk_write)\s*\("
    )
    forbidden_stages = re.compile(r"""["'](\$out|\$merge)["']""")

    offenders = []
    for filename in os.listdir(REPO_DIR):
        if not filename.endswith(".py"):
            continue
        content = open(os.path.join(REPO_DIR, filename), encoding="utf-8").read()
        if forbidden_calls.search(content) or forbidden_stages.search(content):
            offenders.append(filename)

    _check("1. nenhuma chamada de escrita no código-fonte", not offenders, f"encontrados em: {offenders}")


def test_pymongo_import_is_local_not_module_level():
    """Import de pymongo deve estar dentro de get_read_only_client, não no topo do
    arquivo -- senão `import data_source` exigiria pymongo instalado mesmo para quem
    só usa os backends 'csv'/'json_export'."""
    content = open(os.path.join(REPO_DIR, "data_source.py"), encoding="utf-8").read()
    top_level = content.split("def get_read_only_client")[0]
    _check(
        "pymongo importado apenas dentro de get_read_only_client, não no topo do módulo",
        "import pymongo" not in top_level and "from pymongo" not in top_level,
        top_level,
    )


def test_empty_allowlist_returns_empty_catalog():
    """Teste 2: APPROVED_CATEGORIES vazio -> DataFrame vazio, sem exceção, sem travar."""
    sysrec.APPROVED_CATEGORIES.clear()
    sysrec.BLOCKED_ITEM_IDS.clear()
    _fake_mongo_backend(video=DOCS_VIDEO, audio=DOCS_AUDIO, image=DOCS_IMAGE)

    try:
        catalog = data_source.load_catalog()
        ok = len(catalog) == 0
    except Exception as exc:  # não deveria lançar
        ok = False
        print(f"      exceção inesperada: {exc!r}")
    _check("2. allowlist vazia -> catálogo vazio, sem exceção", ok)


def test_approved_category_filters_correctly():
    """Testes 3, 4 e 5: allowlist populada, denylist de keyword, revisão geométrica e
    bloqueio por ID -- todos verificados sobre o mesmo carregamento."""
    sysrec.APPROVED_CATEGORIES.clear()
    sysrec.APPROVED_CATEGORIES.update({
        ("OASIS", "nature"), ("DEAM", "ambient"), ("EMOPIA", "music_energetic"),
    })
    sysrec.BLOCKED_ITEM_IDS.clear()
    sysrec.BLOCKED_ITEM_IDS.add("v2")
    _fake_mongo_backend(video=DOCS_VIDEO, audio=DOCS_AUDIO, image=DOCS_IMAGE)

    buf = io.StringIO()
    with redirect_stdout(buf):
        catalog = data_source.load_catalog()
    output = buf.getvalue()
    print(output, end="")

    ids = set(catalog["item_id"])

    # Teste 3: só itens de (dataset, category) aprovados, nenhum com keyword de denylist.
    _check(
        "3. só categorias aprovadas entram, nenhuma keyword de denylist passa",
        "v1" in ids and "a1" in ids and "a2" in ids and "i1" not in ids,
        f"ids presentes: {sorted(ids)}",
    )

    # Teste 4: valência muito negativa + categoria não aprovada -> excluído mesmo sem keyword.
    _check("4. valência muito negativa + categoria não aprovada é excluída", "i2" not in ids)

    # Teste 5: item_id em BLOCKED_ITEM_IDS é excluído mesmo com categoria aprovada.
    _check("5. item bloqueado por ID é excluído mesmo com categoria aprovada", "v2" not in ids)

    # EMOPIA com quadrante reconhecido resolve V/A corretamente (base do teste 8).
    if "a2" in ids:
        row = catalog[catalog["item_id"] == "a2"].iloc[0]
        v_ok = abs(row["Valencia"] - 0.5) < 1e-6
        a_ok = abs(row["Arousal"] - 0.5) < 1e-6
        _check("EMOPIA Q1 resolve para o centroide (0.5, 0.5)", v_ok and a_ok, f"got ({row['Valencia']}, {row['Arousal']})")

    # Descartes: V/A ausente (vídeo v3 e EMOPIA a3 sem quadrante). Não há mais a
    # categoria "modalidade_desconhecida" -- cada coleção já é uma modalidade fixa.
    _check(
        "descartes reportados por motivo (va_ausente)",
        "va_ausente=2" in output,
        f"saída: {output.strip().splitlines()[-1] if output else '(vazia)'}",
    )


def test_feature_space_schema_retains_diagnostic_columns():
    """Recaracterização estatística (characterize.py) precisa de colunas do esquema
    unificado (dataset, tipo_modalidade, category, tags, octant_raw, confidence_tier)
    além de EXPECTED_COLUMNS -- ver data_source._to_feature_space_schema. Confirma
    também que FeatureSpace(catalog) não lança exceção sobre o catálogo já adaptado
    (pré-requisito bloqueante do plano de recaracterização). Allowlist é sempre
    estrita nesta branch (sem flag) -- popula APPROVED_CATEGORIES para o catálogo
    não sair vazio, senão a checagem de colunas não teria o que checar."""
    sysrec.APPROVED_CATEGORIES.clear()
    sysrec.APPROVED_CATEGORIES.update({
        ("OASIS", "nature"), ("DEAM", "ambient"), ("EMOPIA", "music_energetic"),
    })
    sysrec.BLOCKED_ITEM_IDS.clear()
    _fake_mongo_backend(video=DOCS_VIDEO, audio=DOCS_AUDIO, image=DOCS_IMAGE)

    catalog = data_source.load_catalog()

    expected_extra = {
        "dataset", "tipo_modalidade", "category", "tags", "octant_raw", "confidence_tier",
    }
    _check(
        "catálogo adaptado preserva colunas de diagnóstico além de EXPECTED_COLUMNS",
        expected_extra.issubset(set(catalog.columns)),
        f"colunas: {sorted(catalog.columns)}",
    )

    try:
        sysrec.FeatureSpace(catalog)
        ok = True
    except Exception as exc:  # não deveria lançar
        ok = False
        print(f"      exceção inesperada: {exc!r}")
    _check("FeatureSpace(catalog) não lança exceção sobre o catálogo real adaptado", ok)


def _make_safety_filter_df(arousals, valencias) -> pd.DataFrame:
    n = len(arousals)
    return pd.DataFrame({
        "Nome": [f"item{i}" for i in range(n)],
        "Tipo": ["Áudio"] * n,
        "Valencia": valencias,
        "Arousal": arousals,
        "Duracao": [5.0] * n,
        "Indoor": [0] * n,
        "Tag": ["x"] * n,
        "Oitante": [1] * n,
    })


def test_safety_filter_r2_blocks_high_energy_octant():
    """R2 (nova regra, expandindo o guardrail): item de alta ativação é bloqueado
    quando curr_oct está em HIGH_ENERGY_OCTANTS -- a versão anterior só cobria
    LOW_ENERGY_OCTANTS (R1), deixando hiperativação (3/4) sem proteção equivalente.
    dest_oct=1 (Excitado/Eufórico, ta=0.8>=0 e tv=0.8>0) é escolhido para não
    acionar R3 (precisa ta<0) nem R4 (precisa Valencia abaixo do limiar aversivo)."""
    df = _make_safety_filter_df(
        arousals=[sysrec.SAFETY_AROUSAL_THRESHOLD - 0.1, sysrec.SAFETY_AROUSAL_THRESHOLD + 0.1],
        valencias=[0.1, 0.1],
    )
    fake_self = types.SimpleNamespace(df=df, safety_checked=0, safety_blocked=0)
    eligible = df.index.to_numpy()

    safe = sysrec.Recommender._apply_safety_filter(fake_self, eligible, curr_oct=3, dest_oct=1)

    _check(
        "R2: item de alta ativação bloqueado quando curr_oct=3 (HIGH_ENERGY_OCTANTS)",
        0 in safe and 1 not in safe,
        f"eligible={list(eligible)} safe={list(safe)}",
    )


def test_safety_filter_never_empties_eligible():
    """Garantia mantida do guardrail original: se o filtro bloquearia TODOS os itens
    elegíveis, reverte para o conjunto anterior em vez de esvaziar."""
    df = _make_safety_filter_df(
        arousals=[sysrec.SAFETY_AROUSAL_THRESHOLD + 0.1] * 2, valencias=[0.1, 0.1],
    )
    fake_self = types.SimpleNamespace(df=df, safety_checked=0, safety_blocked=0)
    eligible = df.index.to_numpy()

    # curr_oct=5 (LOW_ENERGY_OCTANTS) + dest_oct=1 -> R1 bloquearia os dois únicos itens.
    safe = sysrec.Recommender._apply_safety_filter(fake_self, eligible, curr_oct=5, dest_oct=1)

    _check(
        "guardrail nunca esvazia o conjunto elegível mesmo quando tudo seria bloqueado",
        len(safe) == len(eligible),
        f"safe={list(safe)}",
    )


def test_extract_emopia_quadrant():
    """Teste 8: quadrante extraído de tags/subcategories (formato 'quadrant_qN'),
    nunca de um campo 'quadrant' solto ou de staticAnnotations (ver correção da
    seção 5 do plano de revisões -- staticAnnotations é espúrio para EMOPIA)."""
    _check(
        "8a. EMOPIA com tag quadrant_q3 -> 'Q3'",
        data_source._extract_emopia_quadrant({"tags": ["quadrant_q3", "calm"]}) == "Q3",
    )
    _check(
        "8b. EMOPIA com subcategories quadrant_q2 (maiúsculas) -> 'Q2'",
        data_source._extract_emopia_quadrant({"tags": [], "subcategories": ["QUADRANT_Q2"]}) == "Q2",
    )
    _check(
        "8c. EMOPIA sem tag de quadrante -> None",
        data_source._extract_emopia_quadrant({"tags": ["ambient"]}) is None,
    )
    _check(
        "8d. EMOPIA sem tags/subcategories -> None",
        data_source._extract_emopia_quadrant({}) is None,
    )


def test_scale_none_is_skipped_generically():
    """
    Teste 7 (adaptado): datasets com scale=None são pulados com aviso, sem erro nem
    fórmula assumida. audit_dataset checa scale=None ANTES de tocar em
    iter_raw_docs/backend, então este teste não precisa de nenhum backend fake.
    """
    import normalization

    sysrec.NORMALIZATION_REFERENCE["_FAKE_UNCONFIRMED"] = {
        "raw_field": "ratings.valenceMean", "scale": None,
    }
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            discrepancias = normalization.audit_dataset("_FAKE_UNCONFIRMED")
        _check(
            "7. scale=None é pulado sem erro (mecanismo genérico)",
            discrepancias == [] and "PULAR" in buf.getvalue(),
        )
    finally:
        del sysrec.NORMALIZATION_REFERENCE["_FAKE_UNCONFIRMED"]


def test_emomadrid_scale_confirmed_by_example():
    """
    Correção seção 2: fórmula (-2,2) confirmada com o par real
    valenceMean=1.13 -> valenceNormalized=0.565 (1.13/2 = 0.565). A escala 1-9
    tradicional (assumida antes desta correção) dava (1.13-5)/4 = -0.9675, que NÃO
    bate -- é exatamente o cálculo que motivou a correção.
    """
    import normalization

    _fake_mongo_backend(video=[
        {"_id": "em1", "sourceMeta": {"dataset": "EmoMadrid"},
         "ratings": {"valenceMean": 1.13, "valenceNormalized": 0.565}},
    ])
    discrepancias = normalization.audit_dataset("EmoMadrid")
    _check("EmoMadrid (-2,2): exemplo real bate sem discrepância", discrepancias == [], f"{discrepancias}")


def test_meditation_local_scale_confirmed_by_example():
    """
    Correção seção 4: escala (1,9) confirmada pela própria description do dado
    ("x' = (x - 5) / 4"). Exemplo real: valenceMean=6.8 -> (6.8-5)/4 = 0.45.
    """
    import normalization

    _fake_mongo_backend(video=[
        {"_id": "ml1", "source": {"dataset": "MEDITATION_LOCAL"},
         "staticAnnotations": {"valenceMean": 6.8, "valenceNormalized": 0.45}},
    ])
    discrepancias = normalization.audit_dataset("MEDITATION_LOCAL")
    _check("MEDITATION_LOCAL (1,9): exemplo real bate sem discrepância", discrepancias == [], f"{discrepancias}")


def test_muvi_identity_range_check():
    """
    Correção seção 3: MuVi não tem campo *Normalized separado (scale="IDENTITY") --
    a auditoria vira checagem de faixa sobre o valor bruto, usado diretamente. Exemplo
    real (dentro da faixa) + um caso injetado fora de [-1,1] para confirmar detecção.
    """
    import normalization

    _fake_mongo_backend(video=[
        {"_id": "mv1", "sourceMeta": {"dataset": "MuVi"},
         "ratings": {"valenceMean": -0.0017458, "arousalMean": 0.479212}},  # exemplo real, dentro da faixa
        {"_id": "mv2", "sourceMeta": {"dataset": "MuVi"},
         "ratings": {"valenceMean": 1.5, "arousalMean": 0.2}},  # fora de [-1,1], injetado
    ])
    discrepancias = normalization.audit_dataset("MuVi")
    _check(
        "MuVi (IDENTITY): detecta valor fora de [-1,1], exemplo real não é sinalizado",
        len(discrepancias) == 1 and discrepancias[0]["id"] == "mv2",
        f"{discrepancias}",
    )


# Item de exemplo real do EMOPIA usado na correção crítica da seção 5: quadrante,
# tags e oitante concordam entre si (alta valência, alto arousal); só o valor
# armazenado em staticAnnotations diverge -- é exatamente esse valor que deve ser
# ignorado por normalize_audio_doc.
_EMOPIA_SPURIOUS_EXAMPLE = {
    "_id": "quadrant_q1_example", "title": "Exemplo real EMOPIA",
    "source": {"dataset": "EMOPIA"},
    "staticAnnotations": {
        "valenceMean": -0.0427, "valenceNormalized": -0.0427,
        "arousalMean": -0.3747, "arousalNormalized": -0.3747,
    },
    "annotationType": "russell_4q_inferred_va_from_reference_octant_centroid",
    "description": "Valence/arousal inferred from audio_deam_ octant centroid",
    "tags": ["quadrant_q1", "positive_valence", "high_arousal"],
    "soundOctant": 7,
}


def test_emopia_spurious_value_ignored_uses_quadrant():
    """Correção crítica (seção 5): normalize_audio_doc deve ignorar o valor espúrio
    armazenado e usar sempre o centroide do quadrante extraído das tags."""
    row = data_source.normalize_audio_doc(_EMOPIA_SPURIOUS_EXAMPLE)
    v_ok = abs(row["valencia_norm"] - 0.5) < 1e-9
    a_ok = abs(row["arousal_norm"] - 0.5) < 1e-9
    nao_espurio = row["valencia_norm"] != -0.0427 and row["arousal_norm"] != -0.3747
    _check(
        "EMOPIA espúrio -> normalize_audio_doc usa centroide de Q1 (0.5, 0.5), não o valor armazenado",
        v_ok and a_ok and nao_espurio,
        f"got ({row['valencia_norm']}, {row['arousal_norm']})",
    )


def test_internal_consistency_flags_emopia_before_and_after():
    """
    Seções 5 e 6: check_internal_consistency, aplicado ao valor espúrio antigo do
    item de exemplo, encontra 3+ contradições simultâneas (tags de valência e de
    arousal, quadrante) -- é essa multiplicidade que classifica como severo, não
    ambiguidade de fronteira. Aplicado ao valor já corrigido (centroide de Q1), as
    contradições de tag/quadrante desaparecem (pode restar divergência de oitante
    geométrico, que é esperada e vai para revisão de baixa prioridade, não para
    correção de código -- ver contraste com meditation_local na seção 5.3 do plano).
    """
    import normalization

    row_antes = {
        "item_id": "quadrant_q1_example", "dataset": "EMOPIA",
        "valencia_norm": -0.0427, "arousal_norm": -0.3747,  # valor espúrio, pré-correção
        "tags": ["quadrant_q1", "positive_valence", "high_arousal"], "octant_raw": 7,
    }
    problems_antes = normalization.check_internal_consistency(row_antes)
    _check(
        "consistência ANTES da correção: 3+ problemas para o item espúrio do EMOPIA",
        len(problems_antes) >= 3,
        f"{problems_antes}",
    )

    row_depois = dict(row_antes)
    row_depois["valencia_norm"], row_depois["arousal_norm"] = sysrec.EMOPIA_QUADRANT_CENTROIDS["Q1"]
    problems_depois = normalization.check_internal_consistency(row_depois)
    problemas_tag_ou_quadrante = [p for p in problems_depois if "tag " in p or "contradiz sinal" in p]
    _check(
        "consistência DEPOIS da correção: sem contradição de tag/quadrante (só pode restar oitante)",
        problemas_tag_ou_quadrante == [],
        f"{problems_depois}",
    )


def test_consistency_audit_severity_drops_after_correction():
    """
    Item 7 da seção 9 (adaptado -- sem Mongo real para rodar sobre o catálogo
    completo): demonstra sobre o item de exemplo que run_consistency_audit classifica
    como severo o valor espúrio antigo e deixa de classificar como severo o valor já
    corrigido.
    """
    import normalization
    import pandas as pd

    catalogo_antes = pd.DataFrame([{
        "item_id": "quadrant_q1_example", "dataset": "EMOPIA",
        "valencia_norm": -0.0427, "arousal_norm": -0.3747,
        "tags": ["quadrant_q1", "positive_valence", "high_arousal"], "octant_raw": 7,
    }])
    severos_antes, _ = normalization.run_consistency_audit(catalogo_antes)

    catalogo_depois = pd.DataFrame([{
        "item_id": "quadrant_q1_example", "dataset": "EMOPIA",
        "valencia_norm": 0.5, "arousal_norm": 0.5,
        "tags": ["quadrant_q1", "positive_valence", "high_arousal"], "octant_raw": 7,
    }])
    severos_depois, _ = normalization.run_consistency_audit(catalogo_depois)

    _check(
        "run_consistency_audit: contagem de severos cai a 0 após a correção do EMOPIA",
        len(severos_antes) == 1 and len(severos_depois) == 0,
        f"antes={severos_antes} depois={severos_depois}",
    )


def test_dataset_field_path_por_modalidade():
    """
    Seção 7: regressão explícita -- áudio usa source.dataset, vídeo e imagem usam
    sourceMeta.dataset. Protege contra uma refatoração futura que "simplifique"
    isso incorretamente para um caminho único.
    """
    audio_doc = {
        "_id": "reg_a", "source": {"dataset": "DEAM"},
        "staticAnnotations": {"valenceNormalized": 0.0, "arousalNormalized": 0.0},
    }
    video_doc = {"_id": "reg_v", "sourceMeta": {"dataset": "MuVi"}}
    image_doc = {"_id": "reg_i", "sourceMeta": {"dataset": "OASIS"}}
    _check(
        "regressão: áudio lê dataset de source.dataset",
        data_source.normalize_audio_doc(audio_doc)["dataset"] == "DEAM",
    )
    _check(
        "regressão: vídeo lê dataset de sourceMeta.dataset",
        data_source.normalize_video_doc(video_doc)["dataset"] == "MuVi",
    )
    _check(
        "regressão: imagem lê dataset de sourceMeta.dataset",
        data_source.normalize_image_doc(image_doc)["dataset"] == "OASIS",
    )


def test_audit_detects_injected_discrepancy():
    """Teste 6: normalization detecta uma discrepância injetada de propósito."""
    import normalization

    # OASIS: escala (1, 7). raw=7 (máximo) deveria normalizar para +1.0; armazenamos
    # errado de propósito (-5.0) para confirmar que a auditoria detecta a discrepância.
    _fake_mongo_backend(video=[
        {"_id": "d1", "sourceMeta": {"dataset": "OASIS"},
         "ratings": {"valenceMean": 7, "valenceNormalized": -5.0}},
        {"_id": "d2", "sourceMeta": {"dataset": "OASIS"},
         "ratings": {"valenceMean": 4, "valenceNormalized": 0.0}},  # correto, sem discrepância
    ])
    discrepancias = normalization.audit_dataset("OASIS")
    _check(
        "6. discrepância injetada é detectada, e só ela",
        len(discrepancias) == 1 and discrepancias[0]["id"] == "d1",
        f"discrepâncias encontradas: {discrepancias}",
    )


def test_determinism():
    """Teste 9: duas cargas seguidas produzem o mesmo DataFrame."""
    sysrec.APPROVED_CATEGORIES.clear()
    sysrec.APPROVED_CATEGORIES.update({("OASIS", "nature"), ("DEAM", "ambient")})
    sysrec.BLOCKED_ITEM_IDS.clear()
    _fake_mongo_backend(video=DOCS_VIDEO, audio=DOCS_AUDIO, image=DOCS_IMAGE)

    buf = io.StringIO()
    with redirect_stdout(buf):
        cat1 = data_source.load_catalog()
        cat2 = data_source.load_catalog()
    _check("9. duas cargas seguidas produzem o mesmo DataFrame", cat1.equals(cat2))


def test_statistical_checks_run_without_error():
    """Checagens estatísticas complementares (seção 6): rodam sem erro sobre dados fake,
    inclusive detectando um valor fora de [-1, 1] injetado de propósito."""
    import normalization

    _fake_mongo_backend(video=[
        {"_id": "s1", "sourceMeta": {"dataset": "OASIS"}, "videoOctant": 1,
         "ratings": {"valenceNormalized": 1.4, "arousalNormalized": 0.2}},  # fora de [-1,1] de propósito
        {"_id": "s2", "sourceMeta": {"dataset": "OASIS"}, "videoOctant": 1,
         "ratings": {"valenceNormalized": 0.3, "arousalNormalized": 0.1}},
    ])
    buf = io.StringIO()
    ok = True
    try:
        with redirect_stdout(buf):
            normalization.check_out_of_range()
            normalization.check_zero_variance()
            normalization.check_octant_region_agreement()
    except Exception as exc:
        ok = False
        print(f"      exceção inesperada: {exc!r}")
    output = buf.getvalue()
    _check(
        "checagens estatísticas rodam sem erro e detectam valor fora de [-1,1]",
        ok and "1 documento" in output,
        output,
    )


# ---------- Backend json_export (arquivos locais em dbs/) ----------

def test_unwrap_extended_json():
    result = data_source._unwrap_extended_json({
        "_id": {"$oid": "abc"}, "createdAt": {"$date": "2026-01-01T00:00:00Z"}, "x": 1,
    })
    _check(
        "_unwrap_extended_json: _id/datas viram tipos nativos, não dicts aninhados",
        result == {"_id": "abc", "createdAt": "2026-01-01T00:00:00Z", "x": 1},
        f"{result}",
    )


def test_load_json_export_reads_only_from_dbs_dir():
    """dbs/ é o diretório de referência único (instrução explícita do usuário) --
    _load_json_export/_resolve_json_path só encontram arquivos ali, nunca em outro
    diretório mesmo que o nome bata (diferente da busca em múltiplos candidatos usada
    para o CSV em resolve_dataset_path)."""
    with tempfile.TemporaryDirectory() as tmp_dbs:
        original_dir = sysrec.JSON_EXPORT_DIR
        sysrec.JSON_EXPORT_DIR = tmp_dbs
        try:
            docs = [{"_id": {"$oid": "x1"}, "title": "a"}, {"_id": {"$oid": "x2"}, "title": "b"}]
            with open(os.path.join(tmp_dbs, "videos.json"), "w", encoding="utf-8") as fh:
                json.dump(docs, fh)

            loaded = data_source._load_json_export("videos.json")
            _check(
                "_load_json_export lê de dbs/ e desembrulha _id para string",
                len(loaded) == 2 and loaded[0]["_id"] == "x1" and isinstance(loaded[0]["_id"], str),
                f"{loaded}",
            )

            try:
                data_source._resolve_json_path("nao_existe.json")
                achou_fora = False
            except FileNotFoundError:
                achou_fora = True
            _check("_resolve_json_path lança FileNotFoundError para arquivo ausente em dbs/", achou_fora)
        finally:
            sysrec.JSON_EXPORT_DIR = original_dir


def test_load_from_json_export_missing_modality_no_exception():
    """Só videos.json presente -> load_from_json_export não lança exceção, avisa
    sobre audios.json/images.json ausentes, e devolve DataFrame normalmente."""
    with tempfile.TemporaryDirectory() as tmp_dbs:
        original_dir = sysrec.JSON_EXPORT_DIR
        sysrec.JSON_EXPORT_DIR = tmp_dbs
        sysrec.APPROVED_CATEGORIES.clear()
        sysrec.BLOCKED_ITEM_IDS.clear()
        try:
            docs = [{
                "_id": {"$oid": "v1"}, "title": "Praia", "category": "nature",
                "sourceMeta": {"dataset": "OASIS"}, "tags": ["nature"],
                "ratings": {"valenceMean": 0.6, "arousalMean": -0.3},
                "durationSeconds": 120,
            }]
            with open(os.path.join(tmp_dbs, "videos.json"), "w", encoding="utf-8") as fh:
                json.dump(docs, fh)

            buf = io.StringIO()
            ok = True
            catalog = None
            try:
                with redirect_stdout(buf):
                    catalog = data_source.load_from_json_export()
            except Exception as exc:
                ok = False
                print(f"      exceção inesperada: {exc!r}")
            output = buf.getvalue()
            _check(
                "load_from_json_export com só videos.json não lança exceção e avisa das modalidades ausentes",
                ok and catalog is not None and "audio" in output and "image" in output,
                output,
            )
        finally:
            sysrec.JSON_EXPORT_DIR = original_dir


def test_iter_raw_docs_json_export_dataset_filter_case_insensitive():
    with tempfile.TemporaryDirectory() as tmp_dbs:
        original_dir = sysrec.JSON_EXPORT_DIR
        original_backend = sysrec.DATA_BACKEND
        sysrec.JSON_EXPORT_DIR = tmp_dbs
        sysrec.DATA_BACKEND = "json_export"
        try:
            docs = [
                {"_id": "mv1", "sourceMeta": {"dataset": "MuVi"}, "title": "a"},
                {"_id": "mv2", "sourceMeta": {"dataset": "OtherDataset"}, "title": "b"},
            ]
            with open(os.path.join(tmp_dbs, "videos.json"), "w", encoding="utf-8") as fh:
                json.dump(docs, fh)
            # audios.json/images.json ausentes de propósito -- iter_raw_docs deve
            # pular essas modalidades silenciosamente, sem lançar exceção.

            found_lower = list(data_source.iter_raw_docs(modality="video", dataset_filter="muvi"))
            found_exact = list(data_source.iter_raw_docs(modality="video", dataset_filter="MuVi"))
            found_all_modalities = list(data_source.iter_raw_docs(dataset_filter="MuVi"))
            _check(
                "iter_raw_docs(json_export): dataset_filter insensível a maiúsculas, "
                "ignora modalidade sem export sem lançar exceção",
                len(found_lower) == 1 and len(found_exact) == 1 and found_lower[0]["_id"] == "mv1"
                and len(found_all_modalities) == 1,
                f"lower={found_lower} exact={found_exact} all={found_all_modalities}",
            )
        finally:
            sysrec.JSON_EXPORT_DIR = original_dir
            sysrec.DATA_BACKEND = original_backend


def _build_tiny_recommender(n_items: int = 4, valencia: list = None, arousal: list = None):
    """Catálogo sintético mínimo (EXPECTED_COLUMNS) + FeatureSpace/Agent/Recommender
    reais -- usado pelos testes de fadiga/filtro de tempo/determinismo do slot 1,
    que precisam exercitar Recommender.recommend() de ponta a ponta, não só funções
    isoladas. `valencia`/`arousal` são opcionais -- por padrão usam a progressão
    linear original; passe valores explícitos quando o teste precisar de um
    ranking de score conhecido e sem empates."""
    df = pd.DataFrame({
        "Nome": [f"item{i}" for i in range(n_items)],
        "Tipo": ["Áudio"] * n_items,
        "Valencia": valencia if valencia is not None else [0.1 * i for i in range(n_items)],
        "Arousal": arousal if arousal is not None else [-0.1 * i for i in range(n_items)],
        "Duracao": [5.0] * n_items,
        "Indoor": [0] * n_items,
        "Tag": ["x"] * n_items,
        "Oitante": [1] * n_items,
    })
    feature_space = sysrec.FeatureSpace(df)
    agent = sysrec.Agent(df, feature_space)
    recommender = sysrec.Recommender(df, feature_space, agent)
    return recommender, df


def test_fatigue_blocked_mask_correctness():
    """FatigueTracker.blocked_mask bloqueia exatamente os itens com
    delta < FATIGUE_MIN_GAP desde a última EXECUÇÃO, e libera os demais.
    mark_executed()/advance_round() substituem o antigo register() único -- só o
    item explicitamente marcado como executado entra em last_seen."""
    tracker = sysrec.FatigueTracker()
    tracker.mark_executed(10)   # item 10 executado no counter=0
    tracker.advance_round()     # counter vira 1
    tracker.mark_executed(20)   # item 20 executado no counter=1
    tracker.advance_round()     # counter vira 2

    # counter agora é 2: delta(10)=2-0=2, delta(20)=2-1=1, delta(30)=nunca executado.
    mask = tracker.blocked_mask(np.array([10, 20, 30]))
    expected = np.array([2 < sysrec.FATIGUE_MIN_GAP, 1 < sysrec.FATIGUE_MIN_GAP, False])
    _check(
        "FatigueTracker.blocked_mask bloqueia deltas < FATIGUE_MIN_GAP e libera o resto",
        np.array_equal(mask, expected),
        f"mask={mask} expected={expected}",
    )


def test_fatigue_only_tracks_executed_items():
    """Requisito central desta mudança: itens apenas EXIBIDOS (não escolhidos, ou
    exibidos numa rodada onde o usuário não executou nada) não entram em cooldown
    -- só mark_executed() registra um item. recommend() sozinho (sem
    mark_executed() depois) nunca bloqueia nada."""
    recommender, df = _build_tiny_recommender(n_items=4)

    results = recommender.recommend(curr_oct=1, dest_oct=1, time_avail=60, k=2)
    shown_ids = {r["item_idx"] for r in results}
    _check(
        "recommend() sozinho não marca nenhum item como executado (last_seen vazio)",
        len(recommender.fatigue.last_seen) == 0,
        f"last_seen={recommender.fatigue.last_seen} shown_ids={shown_ids}",
    )

    # Simula execução de apenas UM dos itens mostrados.
    executed_id = results[0]["item_idx"]
    recommender.fatigue.mark_executed(executed_id)
    mask = recommender.fatigue.blocked_mask(df.index.to_numpy())
    blocked_ids = set(df.index.to_numpy()[mask])
    _check(
        "só o item executado entra em cooldown -- os demais itens mostrados "
        "(não escolhidos) continuam livres",
        blocked_ids == {executed_id},
        f"blocked_ids={blocked_ids} executed_id={executed_id} shown_ids={shown_ids}",
    )


def test_fatigue_never_empties_pool():
    """Garantia da Parte 2: se TODOS os itens do pool foram executados há menos de
    FATIGUE_MIN_GAP interações, o bloqueio rígido não pode esvaziar o pool -- o
    Recommender reverte para o pool anterior (mesma regra do guardrail/curadoria)."""
    recommender, df = _build_tiny_recommender(n_items=4)

    # Força TODOS os itens do catálogo como "recém-executados" (delta=1 < FATIGUE_MIN_GAP).
    for item_idx in df.index.to_numpy():
        recommender.fatigue.mark_executed(item_idx)
    recommender.fatigue.advance_round()

    results = recommender.recommend(curr_oct=1, dest_oct=1, time_avail=60, k=2)
    _check(
        "fadiga nunca esvazia o pool: recommend() ainda devolve itens mesmo com "
        "todo o catálogo 'recém-executado'",
        len(results) > 0,
        f"results={results}",
    )


def test_time_filter_toggle():
    """Testes 5 e 6: com TIME_FILTER_ENABLED=False (padrão), recommend() ignora
    Duracao (mesmo pedindo bem menos tempo do que qualquer item exige); com True,
    volta a filtrar -- regressão zero quando reativado."""
    recommender, df = _build_tiny_recommender(n_items=3)
    # Todos os itens têm Duracao=5.0 -- time_avail=1 exclui todos SE o filtro estiver ativo.

    original = sysrec.TIME_FILTER_ENABLED
    try:
        sysrec.TIME_FILTER_ENABLED = False
        results_disabled = recommender.recommend(curr_oct=1, dest_oct=1, time_avail=1, k=2)
        _check(
            "TIME_FILTER_ENABLED=False: recommend() ignora Duracao "
            "(devolve itens mesmo com time_avail menor que qualquer duração do catálogo)",
            len(results_disabled) > 0,
            f"results={results_disabled}",
        )

        sysrec.TIME_FILTER_ENABLED = True
        results_enabled = recommender.recommend(curr_oct=1, dest_oct=1, time_avail=1, k=2)
        _check(
            "TIME_FILTER_ENABLED=True: recommend() volta a filtrar por Duracao "
            "(nenhum item cabe em time_avail=1 -- lista vazia, comportamento pré-flag)",
            results_enabled == [],
            f"results={results_enabled}",
        )
    finally:
        sysrec.TIME_FILTER_ENABLED = original


def test_select_slots_slot1_always_greedy_argmax():
    """Requisito central: o slot 1 deve ser SEMPRE o item de maior `adjusted`
    score, de forma determinística -- nunca sorteado, mesmo quando o slot
    exploratório entra em jogo (P_EXPLORE_SLOT) ou quando a cauda é embaralhada.
    _select_slots não lê nada de `self` -- testável isoladamente como função pura,
    sem Agent/FeatureSpace reais (mesmo padrão de self falso já usado para
    _apply_safety_filter)."""
    fake_self = types.SimpleNamespace()
    adjusted = np.array([0.1, 0.9, 0.3, 0.5], dtype=np.float32)  # posição 1 = argmax, sem empate
    pool_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    true_best = int(np.argmax(adjusted))

    violations = []
    for _ in range(200):
        chosen, slot_types, propensities = sysrec.Recommender._select_slots(
            fake_self, adjusted, pool_vectors, k=3
        )
        if chosen[0] != true_best or slot_types[0] != "greedy" or propensities[0] != 1.0:
            violations.append((chosen, slot_types, propensities))

    _check(
        "slot 1 é sempre o argmax determinístico de `adjusted` em 200 chamadas "
        "(cobrindo o sorteio do slot exploratório e o embaralhamento da cauda) -- "
        "nunca sorteado, sempre slot_type='greedy' e propensão 1.0",
        violations == [],
        f"{len(violations)}/200 violações; exemplos: {violations[:3]}",
    )


def test_slot1_deterministic_and_respects_fatigue_spacing():
    """Prova de ponta a ponta (recommend() completo) das duas metades do
    requisito juntas:
      (a) slot 1 nunca tem score menor que os demais slots retornados na mesma
          chamada (condição necessária de ser o item de maior score do pool);
      (b) o espaçamento (FATIGUE_MIN_GAP) continua valendo PARA ITENS EXECUTADOS:
          simulando que o usuário sempre executa a recomendação do slot 1 (via
          fatigue.mark_executed, o mesmo caminho que _run_interaction usa), nenhum
          item reaparece no slot 1 com intervalo menor que FATIGUE_MIN_GAP chamadas
          consecutivas no MESMO contexto. recommend() sozinho não marca mais nada
          como executado -- é por isso que o teste simula a execução explicitamente
          a cada rodada, em vez de depender do registro automático que existia
          antes desta mudança.

    Catálogo dimensionado com folga -- FATIGUE_MIN_GAP + TOP_K itens -- para que o
    fallback de "nunca esvaziar o pool" do FatigueTracker nunca precise disparar:
    com poucos itens, esse fallback (que reverte a filtragem quando ela zeraria o
    pool) pode reintroduzir um item ainda em cooldown, invalidando a garantia de
    espaçamento por escassez de catálogo, não por regressão real. Como só o item
    executado (1 por rodada, não mais os TOP_K exibidos) entra em cooldown, no pior
    caso até FATIGUE_MIN_GAP-1 itens distintos ficam em cooldown simultaneamente;
    FATIGUE_MIN_GAP+TOP_K garante sempre pelo menos TOP_K itens livres. (Valencia,
    Arousal) em progressão linear até (0.8, 0.8) = centroide do octante 1
    (ISO_ALPHA=1.0) -- distâncias únicas, sem empate, então o ranking geométrico é
    conhecido: item0 (distância 0) é o argmax inequívoco na 1ª chamada."""
    n_items = sysrec.FATIGUE_MIN_GAP + sysrec.TOP_K
    valencia = [0.8 - 0.05 * i for i in range(n_items)]
    arousal = [0.8 - 0.05 * i for i in range(n_items)]
    recommender, df = _build_tiny_recommender(n_items=n_items, valencia=valencia, arousal=arousal)

    n_calls = sysrec.FATIGUE_MIN_GAP + 3
    slot1_history = []
    necessary_condition_ok = True
    for _ in range(n_calls):
        results = recommender.recommend(curr_oct=1, dest_oct=1, time_avail=60, k=sysrec.TOP_K)
        if not results:
            necessary_condition_ok = False
            break
        slot1_history.append(results[0]["item_idx"])
        if len(results) > 1 and results[0]["score"] < max(r["score"] for r in results[1:]):
            necessary_condition_ok = False
        # Simula o usuário sempre executando a recomendação do slot 1 -- mesmo
        # caminho que _run_interaction usa de verdade.
        recommender.fatigue.mark_executed(results[0]["item_idx"])

    _check(
        "primeira chamada: slot 1 é o item geometricamente mais próximo do alvo "
        "(distância 0 a (0.8,0.8)) -- argmax inequívoco do catálogo sintético",
        bool(slot1_history) and slot1_history[0] == 0,
        f"slot1_history={slot1_history}",
    )
    _check(
        "slot 1 nunca tem score menor que os demais slots retornados na mesma "
        "chamada (condição necessária de ser sempre o item de maior score do pool)",
        necessary_condition_ok,
        f"slot1_history={slot1_history}",
    )

    last_seen_at = {}
    min_gap_seen = None
    for call_idx, item_idx in enumerate(slot1_history):
        if item_idx in last_seen_at:
            gap = call_idx - last_seen_at[item_idx]
            min_gap_seen = gap if min_gap_seen is None else min(min_gap_seen, gap)
        last_seen_at[item_idx] = call_idx

    _check(
        f"espaçamento preservado: nenhum item reaparece no slot 1 com intervalo "
        f"menor que FATIGUE_MIN_GAP={sysrec.FATIGUE_MIN_GAP} chamadas consecutivas "
        f"com o MESMO contexto",
        min_gap_seen is None or min_gap_seen >= sysrec.FATIGUE_MIN_GAP,
        f"slot1_history={slot1_history} min_gap_seen={min_gap_seen}",
    )


def main() -> None:
    test_write_methods_absent()
    test_pymongo_import_is_local_not_module_level()
    test_empty_allowlist_returns_empty_catalog()
    test_approved_category_filters_correctly()
    test_feature_space_schema_retains_diagnostic_columns()
    test_safety_filter_r2_blocks_high_energy_octant()
    test_safety_filter_never_empties_eligible()
    test_fatigue_blocked_mask_correctness()
    test_fatigue_only_tracks_executed_items()
    test_fatigue_never_empties_pool()
    test_time_filter_toggle()
    test_select_slots_slot1_always_greedy_argmax()
    test_slot1_deterministic_and_respects_fatigue_spacing()
    test_extract_emopia_quadrant()
    test_scale_none_is_skipped_generically()
    test_emomadrid_scale_confirmed_by_example()
    test_meditation_local_scale_confirmed_by_example()
    test_muvi_identity_range_check()
    test_emopia_spurious_value_ignored_uses_quadrant()
    test_internal_consistency_flags_emopia_before_and_after()
    test_consistency_audit_severity_drops_after_correction()
    test_dataset_field_path_por_modalidade()
    test_audit_detects_injected_discrepancy()
    test_determinism()
    test_statistical_checks_run_without_error()
    test_unwrap_extended_json()
    test_load_json_export_reads_only_from_dbs_dir()
    test_load_from_json_export_missing_modality_no_exception()
    test_iter_raw_docs_json_export_dataset_filter_case_insensitive()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} teste(s) falharam: {_FAILURES}")
        sys.exit(1)
    print("Todos os testes passaram.")


if __name__ == "__main__":
    main()
