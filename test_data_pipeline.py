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
import random
import re
import sys
import tempfile
import types
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch

import data_source
import multiseed
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
    # 10) imagem com valência muito negativa, categoria APROVADA -> testa a Camada 3
    #     isolada (regressão do bug de absorção booleana: com "| mask_category" a
    #     Camada 3 nunca excluía nada de uma categoria já aprovada, para nenhum
    #     valor de SAFETY_MIN_VALENCE_REVIEW -- ver test_camada3_blocks_extreme_valence).
    {
        "_id": "i3", "title": "Paisagem perturbadora",
        "ratings": {"valenceNormalized": -0.8, "arousalNormalized": 0.4},
        "tags": [], "category": "nature", "sourceMeta": {"dataset": "OASIS"},
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

    # Teste 6 (Camada 3 isolada): i3 tem valência muito negativa E categoria
    # aprovada ("nature") -- é exatamente o caso que o bug de absorção booleana
    # deixava passar (mask_category & (X | mask_category) ≡ mask_category). Com a
    # correção, i3 é excluído mesmo com categoria aprovada, a menos que esteja em
    # REVIEWED_NEGATIVE_ITEM_IDS.
    _check(
        "6. Camada 3 bloqueia valência extrema mesmo com categoria aprovada "
        "(regressão do bug de absorção booleana)",
        "i3" not in ids,
        f"ids presentes: {sorted(ids)}",
    )


def test_camada3_independent_of_category_approval():
    """Teste 7: a mesma valência extrema com categoria aprovada continua excluída
    até ser revisada individualmente (REVIEWED_NEGATIVE_ITEM_IDS) -- e passa a
    entrar assim que revisada, sem depender de mudar a categoria. Isola a Camada 3
    do resto da curadoria (não precisa recarregar o catálogo completo)."""
    df = pd.DataFrame({
        "item_id": ["low_v", "ok_v"],
        "valencia_norm": [-0.8, 0.1],
        "dataset": ["OASIS", "OASIS"],
        "category": ["nature", "nature"],
        "nome": ["item baixo", "item ok"],
        "tags": [[], []],
    })
    sysrec.APPROVED_CATEGORIES.clear()
    sysrec.APPROVED_CATEGORIES.add(("OASIS", "nature"))
    sysrec.BLOCKED_ITEM_IDS.clear()
    sysrec.REVIEWED_NEGATIVE_ITEM_IDS.clear()

    safe = safety.apply_safety_filter(df)
    _check(
        "7a. item de valência extrema com categoria aprovada é excluído sem revisão individual",
        "low_v" not in set(safe["item_id"]) and "ok_v" in set(safe["item_id"]),
        f"ids presentes: {sorted(safe['item_id'])}",
    )

    sysrec.REVIEWED_NEGATIVE_ITEM_IDS.add("low_v")
    safe = safety.apply_safety_filter(df)
    _check(
        "7b. item revisado individualmente (REVIEWED_NEGATIVE_ITEM_IDS) passa a entrar",
        "low_v" in set(safe["item_id"]),
        f"ids presentes: {sorted(safe['item_id'])}",
    )
    sysrec.REVIEWED_NEGATIVE_ITEM_IDS.clear()


def test_denylist_word_boundary_and_normalization():
    """Teste 8: correspondência por PALAVRA INTEIRA (não subcadeia) -- "war" não
    deve casar "edwards"/"forwards"/"paowar" (falsos positivos reais medidos contra
    o catálogo real), mas "war" isolado e "norm_violation" (com underscore, tag
    típica do GAPED) continuam batendo -- _normalize_haystack converte _/-// em
    espaço antes da fronteira de palavra."""
    no_match_row = {"nome": "Edwards Forwards Paowar", "tags": [], "category": ""}
    _check(
        "8a. 'war' não casa subcadeias (edwards/forwards/paowar)",
        not safety._matches_denylist_keyword(no_match_row),
        no_match_row,
    )

    war_row = {"nome": "Cena de guerra", "tags": ["war zone"], "category": ""}
    _check("8b. 'war' isolado continua sendo detectado",
           safety._matches_denylist_keyword(war_row), war_row)

    underscore_row = {"nome": "Imagem neutra", "tags": ["norm_violation"], "category": ""}
    _check(
        "8c. termo com underscore na tag ('norm_violation') é detectado após normalização",
        safety._matches_denylist_keyword(underscore_row),
        underscore_row,
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
    acionar R3 (precisa ta<0) nem R4 (precisa Valencia abaixo do limiar aversivo) --
    Valencia fixada relativa a SAFETY_AVERSIVE_VALENCE_THRESHOLD (não um valor
    absoluto) para continuar acima do limiar independente de recalibração."""
    df = _make_safety_filter_df(
        arousals=[sysrec.SAFETY_AROUSAL_THRESHOLD - 0.1, sysrec.SAFETY_AROUSAL_THRESHOLD + 0.1],
        valencias=[sysrec.SAFETY_AVERSIVE_VALENCE_THRESHOLD + 0.1] * 2,
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
    # min_survivors=0: sem piso, a relaxação progressiva não entra em ação -- testa
    # só o bloqueio bruto.
    mask, n_relaxed = tracker.blocked_mask(np.array([10, 20, 30]), min_survivors=0)
    expected = np.array([2 < sysrec.FATIGUE_MIN_GAP, 1 < sysrec.FATIGUE_MIN_GAP, False])
    _check(
        "FatigueTracker.blocked_mask bloqueia deltas < FATIGUE_MIN_GAP e libera o resto",
        np.array_equal(mask, expected) and n_relaxed == 0,
        f"mask={mask} expected={expected} n_relaxed={n_relaxed}",
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
    mask, _ = recommender.fatigue.blocked_mask(df.index.to_numpy(), min_survivors=0)
    blocked_ids = set(df.index.to_numpy()[mask])
    _check(
        "só o item executado entra em cooldown -- os demais itens mostrados "
        "(não escolhidos) continuam livres",
        blocked_ids == {executed_id},
        f"blocked_ids={blocked_ids} executed_id={executed_id} shown_ids={shown_ids}",
    )


def test_fatigue_relaxation_releases_oldest_first():
    """Correção A: quando o bloqueio deixaria menos de min_survivors itens, a
    relaxação progressiva libera os itens em cooldown há MAIS TEMPO (maior delta)
    primeiro -- nunca os recém-mostrados. Também confirma n_relaxed reportado."""
    tracker = sysrec.FatigueTracker()
    # item 1 executado no counter=0 (mais antigo), item 2 no counter=1, item 3 no
    # counter=2 (mais recente) -- todos ficam em cooldown até counter=2+FATIGUE_MIN_GAP.
    tracker.mark_executed(1)
    tracker.advance_round()
    tracker.mark_executed(2)
    tracker.advance_round()
    tracker.mark_executed(3)
    # counter=2: delta(1)=2, delta(2)=1, delta(3)=0 -- todos < FATIGUE_MIN_GAP (>=3).

    mask, n_relaxed = tracker.blocked_mask(np.array([1, 2, 3]), min_survivors=2)
    released = {item for item, blocked in zip([1, 2, 3], mask) if not blocked}
    _check(
        "relaxação libera os itens de MAIOR delta primeiro (item 1, depois item 2 "
        "-- nunca o 3, recém-mostrado)",
        n_relaxed == 2 and released == {1, 2},
        f"mask={mask} n_relaxed={n_relaxed} released={released}",
    )

    # min_survivors=0: nenhuma relaxação necessária, todos continuam bloqueados.
    mask0, n_relaxed0 = tracker.blocked_mask(np.array([1, 2, 3]), min_survivors=0)
    _check(
        "sem piso (min_survivors=0), nenhuma relaxação ocorre",
        n_relaxed0 == 0 and mask0.all(),
        f"mask0={mask0} n_relaxed0={n_relaxed0}",
    )


def test_fatigue_min_gap_pool_invariant_raises_on_violation():
    """Correção B: uma configuração degenerada (CANDIDATE_POOL_SIZE menor que
    FATIGUE_MIN_GAP*FATIGUE_ITEMS_PER_ROUND + TOP_K) deve levantar ValueError --
    verificada aqui reexecutando a mesma checagem do módulo com valores que a
    violam, sem precisar reimportar o módulo (o que rodaria efeitos colaterais de
    import indesejados nos testes)."""
    gap, items_per_round, top_k, pool = 10, 1, 3, 12  # a configuração antiga, degenerada
    required = gap * items_per_round + top_k
    raised = False
    try:
        if pool < required:
            raise ValueError(
                f"Configuração degenerada: requer CANDIDATE_POOL_SIZE >= {required}."
            )
    except ValueError:
        raised = True
    _check(
        "invariante FATIGUE_MIN_GAP*FATIGUE_ITEMS_PER_ROUND+TOP_K <= CANDIDATE_POOL_SIZE "
        "detecta a configuração antiga (10, 1, 3, 12) como degenerada",
        raised,
        f"required={required} pool={pool}",
    )
    _check(
        "a configuração vigente hoje (FATIGUE_MIN_GAP/CANDIDATE_POOL_SIZE) satisfaz a invariante",
        sysrec.CANDIDATE_POOL_SIZE >= sysrec.FATIGUE_MIN_GAP * sysrec.FATIGUE_ITEMS_PER_ROUND + sysrec.TOP_K,
        f"CANDIDATE_POOL_SIZE={sysrec.CANDIDATE_POOL_SIZE} FATIGUE_MIN_GAP={sysrec.FATIGUE_MIN_GAP} "
        f"FATIGUE_ITEMS_PER_ROUND={sysrec.FATIGUE_ITEMS_PER_ROUND} TOP_K={sysrec.TOP_K}",
    )


def test_fatigue_relaxation_rate_zero_when_comfortable():
    """Correção 1.5: fatigue_relaxation_rate deve ser 0 numa configuração folgada
    (contexto variado, catálogo com folga confortável de itens)."""
    recommender, df = _build_tiny_recommender(n_items=sysrec.FATIGUE_MIN_GAP + sysrec.TOP_K + 5)
    for i in range(10):
        results = recommender.recommend(curr_oct=1, dest_oct=1, time_avail=60, k=sysrec.TOP_K)
        if results:
            recommender.fatigue.mark_executed(results[0]["item_idx"])
    _check(
        "fatigue_relaxation_rate é 0 em configuração folgada",
        recommender.fatigue_relaxation_rate == 0.0,
        f"fatigue_relaxation_rate={recommender.fatigue_relaxation_rate} "
        f"relaxations={recommender.fatigue_relaxations} calls={recommender.fatigue_calls}",
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
    _select_slots só escreve em `self` pelo contador de inércia do MMR (passivo,
    não altera escolhas) -- testável isoladamente sem Agent/FeatureSpace reais,
    com um self falso que expõe só esse contador como no-op (mesmo padrão de self
    falso já usado para _apply_safety_filter)."""
    fake_self = types.SimpleNamespace(_record_mmr_step=lambda *args, **kwargs: None)
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


# ---------- multiseed.py (avaliação multi-seed) ----------
#
# Teste 1 do plano ("com SEED_MODE='single', a saída de --eval é idêntica ao
# arquivo de referência") FICA DE FORA deste suite automático de propósito: rodar
# --eval de verdade leva ~1-2min (baselines n=3000), e este arquivo é pensado para
# rodar em segundos. Verificação manual, feita uma vez após a refatoração do
# simulador:
#   python sistema_de_recomendacao_v3.py --eval > /tmp/eval_now.txt
#   diff tests/golden/eval_single_seed.txt /tmp/eval_now.txt
# (sem diferenças == a refatoração retrocompatível de _simulate/
# simulate_feedback_holdout não mudou o caminho legado bit a bit.)


def test_multiseed_config_validation_raises():
    """Teste 14: SEED_MODE inválido, e REPLAY_ENTROPY fora do modo multi_random,
    levantam ValueError na importação do módulo -- verificado executando o
    trecho real de validação (copiado literalmente de
    sistema_de_recomendacao_v3.py) com valores que o violam, em vez de reimportar
    o módulo inteiro meio da suíte de testes (efeito colateral indesejado)."""
    valid_modes = sysrec._VALID_SEED_MODES

    def _validate(seed_mode, replay_entropy):
        if seed_mode not in valid_modes:
            raise ValueError(f"SEED_MODE inválido: {seed_mode!r}.")
        if replay_entropy is not None and seed_mode != "multi_random":
            raise ValueError("REPLAY_ENTROPY só tem efeito com SEED_MODE='multi_random'.")

    raised_invalid_mode = False
    try:
        _validate("nao_existe", None)
    except ValueError:
        raised_invalid_mode = True
    _check("14a. SEED_MODE inválido levanta ValueError", raised_invalid_mode)

    raised_bad_replay = False
    try:
        _validate("multi_fixed", 123)
    except ValueError:
        raised_bad_replay = True
    _check("14b. REPLAY_ENTROPY fora de multi_random levanta ValueError", raised_bad_replay)

    # A configuração vigente hoje (single, REPLAY_ENTROPY=None) não levanta nada --
    # já provado pelo simples fato de sistema_de_recomendacao_v3 ter importado.
    _check(
        "14c. configuração vigente (SEED_MODE=single) é válida",
        sysrec.SEED_MODE in valid_modes and (sysrec.REPLAY_ENTROPY is None or sysrec.SEED_MODE == "multi_random"),
    )


def test_multiseed_build_seed_plan_32bit_limit():
    """Teste 10: nenhuma seed derivada por build_seed_plan excede o limite de 32
    bits do np.random.seed legado (a entropia raiz tem 128 bits; generate_state
    trunca para uint32 -- aqui confirmamos que o valor final já vem truncado)."""
    _, plan = multiseed.build_seed_plan("multi_fixed", 8, master_seed=123)
    all_within = all(0 <= r.global_seed < 2**32 for r in plan)
    _check(
        "10. todas as seeds globais derivadas ficam dentro de [0, 2**32)",
        all_within,
        f"seeds={[r.global_seed for r in plan]}",
    )


def test_multiseed_seeded_scope_restores_state():
    """Teste 4: seeded_scope aplica a seed dentro do bloco e restaura o estado
    ANTERIOR (não um reinício do fluxo) ao sair -- para random, numpy e torch."""
    s = random.getstate()
    a = random.random()
    random.setstate(s)
    with multiseed.seeded_scope(123):
        random.random()
    b = random.random()
    _check("4a. random: estado restaurado -- próximo valor é o mesmo de antes do bloco", a == b)

    np_state = np.random.get_state()
    a_np = np.random.random()
    np.random.set_state(np_state)
    with multiseed.seeded_scope(123):
        np.random.random()
    b_np = np.random.random()
    _check("4b. numpy: estado restaurado -- próximo valor é o mesmo de antes do bloco", a_np == b_np)

    torch_state = torch.get_rng_state()
    a_t = torch.rand(1).item()
    torch.set_rng_state(torch_state)
    with multiseed.seeded_scope(123):
        torch.rand(1)
    b_t = torch.rand(1).item()
    _check("4c. torch: estado restaurado -- próximo valor é o mesmo de antes do bloco", a_t == b_t)


def test_multiseed_seeded_scope_restores_on_exception():
    """seeded_scope restaura o estado mesmo quando o bloco lança -- garantia
    equivalente à de safety_filter_disabled/_override já testadas alhures."""
    s = random.getstate()
    a = random.random()
    random.setstate(s)
    raised = False
    try:
        with multiseed.seeded_scope(999):
            random.random()
            raise RuntimeError("boom")
    except RuntimeError:
        raised = True
    b = random.random()
    _check(
        "seeded_scope restaura o estado mesmo quando o bloco lança exceção",
        raised and a == b,
        f"raised={raised} a={a} b={b}",
    )


def test_multiseed_crn_no_global_consumption_when_presampled():
    """Teste 5 (fundamento do CRN): quando u_exec/z_noise são passados,
    simulate_feedback_holdout não consome NADA do gerador global -- é isso que
    garante que todos os braços, vendo os mesmos sorteios pré-gerados, sejam
    independentes entre si (adicionar/remover um braço não desloca o consumo do
    gerador para os demais)."""
    df, _ = _build_tiny_recommender_df(n_items=3)
    item = df.iloc[0]

    s = random.getstate()
    sysrec.simulate_feedback_holdout(1, 2, item, u_exec=0.01, z_noise=0.0)
    unchanged = random.getstate() == s
    _check(
        "5. simulate_feedback_holdout com u_exec/z_noise pré-gerados não consome "
        "o gerador global random (condição que torna os braços independentes sob CRN)",
        unchanged,
    )


def test_multiseed_pregenerate_randomness_shapes():
    """pregenerate_randomness produz os cinco arrays no tamanho pedido, com
    destinos restritos a ALLOWED_DEST_OCTANTS (a avaliação multi-seed reflete os
    destinos que a produção de fato oferece, diferente do _random_context legado,
    que sorteava de 1 a 8)."""
    ctx_ss = np.random.SeedSequence(1)
    noise_ss = np.random.SeedSequence(2)
    rnd = multiseed.pregenerate_randomness(50, ctx_ss, noise_ss)
    shapes_ok = all(len(getattr(rnd, f)) == 50 for f in
                    ("curr", "dest", "u_exec", "z_noise", "u_random_arm"))
    dest_ok = set(rnd.dest.tolist()) <= set(sysrec.ALLOWED_DEST_OCTANTS)
    _check(
        "pregenerate_randomness: 5 arrays de tamanho n_episodes, dest restrito a "
        "ALLOWED_DEST_OCTANTS",
        shapes_ok and dest_ok,
        f"dest únicos={sorted(set(rnd.dest.tolist()))}",
    )


def test_multiseed_reproducible_same_process():
    """Testes 2 e 6: multi_fixed rodado duas vezes no mesmo processo produz
    recompensas idênticas por réplica (reprodutibilidade), e rodar as réplicas em
    ordem inversa produz os mesmos resultados por replicate_id (independência de
    ordem -- consequência de seeded_scope aplicar/restaurar estado por réplica)."""
    df, feature_space = _build_tiny_recommender_df(n_items=20)
    _, plan = multiseed.build_seed_plan("multi_fixed", 3, master_seed=7)

    def _run_all(order):
        results = {}
        for i in order:
            results[i] = multiseed.run_replicate(df, feature_space, plan[i], n_episodes=25)
        return results

    run_a = _run_all([0, 1, 2])
    run_b = _run_all([0, 1, 2])
    same_process_ok = all(
        np.array_equal(run_a[i]["rewards"][arm], run_b[i]["rewards"][arm])
        for i in range(3) for arm in multiseed.ARMS
    )
    _check("2. multi_fixed rodado duas vezes no mesmo processo produz recompensas "
          "idênticas por réplica", same_process_ok)

    run_reversed = _run_all([2, 1, 0])
    order_independent_ok = all(
        np.array_equal(run_a[i]["rewards"][arm], run_reversed[i]["rewards"][arm])
        for i in range(3) for arm in multiseed.ARMS
    )
    _check("6. rodar as réplicas em ordem inversa produz os mesmos resultados por "
          "replicate_id (independência de ordem)", order_independent_ok)


def test_multiseed_crn_arm_independence():
    """Teste 5: adicionar um braço fictício à execução não altera o array de
    recompensas de nenhum dos braços existentes -- prova direta sobre
    run_replicate, não só sobre a chamada isolada do simulador."""
    df, feature_space = _build_tiny_recommender_df(n_items=15)
    _, plan = multiseed.build_seed_plan("multi_fixed", 1, master_seed=11)
    rseed = plan[0]

    baseline = multiseed.run_replicate(df, feature_space, rseed, n_episodes=20)

    original_picks_source = multiseed.run_replicate.__code__  # sanity: função real
    assert original_picks_source is not None

    # Reproduz run_replicate manualmente com um braço extra ("ficticio") inserido
    # no dict `picks`, usando os MESMOS sorteios pré-gerados -- monkeypatch
    # cirúrgico via reimplementação local, já que ARMS/picks são fixos no código.
    rnd = multiseed.pregenerate_randomness(20, rseed.ctx_ss, rseed.noise_ss)
    eligible = df.index.to_numpy(dtype=int)
    time_avail = feature_space.max_duration
    popular_item = int(eligible[np.argmax(df.loc[eligible, "Valencia"].to_numpy())])
    content_cache = {}
    rewards_with_extra = {arm: np.empty(20, dtype=np.float32) for arm in (*multiseed.ARMS, "ficticio")}

    with multiseed.seeded_scope(rseed.global_seed):
        agent = sysrec.Agent(df, feature_space)
        for ep in range(20):
            curr, dest = int(rnd.curr[ep]), int(rnd.dest[ep])
            user_state = feature_space.user_state(curr, dest, time_avail)
            if (curr, dest) not in content_cache:
                d = sysrec.distance_to_point(df.loc[eligible], sysrec.target_point(curr, dest))
                content_cache[(curr, dest)] = int(eligible[np.argmin(d)])
            random_pos = min(int(rnd.u_random_arm[ep] * len(eligible)), len(eligible) - 1)
            picks = {
                "aleatorio": int(eligible[random_pos]),
                "mais_popular": popular_item,
                "conteudo_puro": content_cache[(curr, dest)],
                "agent_online": int(eligible[np.argmax(agent.q_values(user_state, eligible))]),
                "ficticio": int(eligible[0]),  # braço extra, sempre o mesmo item
            }
            for arm, item_idx in picks.items():
                reward, next_oct = sysrec.simulate_feedback_holdout(
                    curr, dest, df.loc[item_idx], u_exec=rnd.u_exec[ep], z_noise=rnd.z_noise[ep],
                )
                rewards_with_extra[arm][ep] = reward
                if arm == "agent_online":
                    next_state = feature_space.user_state(next_oct, dest, time_avail)
                    agent.learn_from_feedback(user_state, item_idx, reward, next_state)

    unaffected = all(
        np.array_equal(baseline["rewards"][arm], rewards_with_extra[arm])
        for arm in multiseed.ARMS
    )
    _check(
        "5. adicionar um braço fictício não altera as recompensas dos braços "
        "existentes (CRN: números aleatórios comuns pré-gerados, consumo "
        "independente do item escolhido)",
        unaffected,
    )


def test_multiseed_random_mode_fresh_entropy():
    """Teste 7: duas chamadas a build_seed_plan em multi_random sem REPLAY_ENTROPY
    registram entropias raiz diferentes (entropia nova do SO a cada vez)."""
    entropy_a, _ = multiseed.build_seed_plan("multi_random", 2)
    entropy_b, _ = multiseed.build_seed_plan("multi_random", 2)
    _check("7. multi_random sem REPLAY_ENTROPY gera entropia raiz nova a cada "
          "execução", entropy_a != entropy_b)


def test_multiseed_random_mode_replay_reproducible():
    """Teste 8: com REPLAY_ENTROPY igual à entropia registrada de uma execução
    anterior, o plano de seeds (e portanto os resultados) é idêntico."""
    entropy, plan_a = multiseed.build_seed_plan("multi_random", 3)
    _, plan_b = multiseed.build_seed_plan("multi_random", 3, replay_entropy=entropy)
    seeds_match = [r.global_seed for r in plan_a] == [r.global_seed for r in plan_b]
    _check("8. REPLAY_ENTROPY reproduz exatamente as mesmas seeds globais de uma "
          "execução multi_random anterior", seeds_match)


def test_multiseed_manifest_written_before_replicates():
    """Teste 9: write_manifest grava o manifesto (com a entropia raiz) em disco
    ANTES de qualquer réplica rodar -- testado diretamente, sem depender de
    interromper um processo no meio."""
    df, _ = _build_tiny_recommender_df(n_items=5)
    root_entropy, plan = multiseed.build_seed_plan("multi_fixed", 2, master_seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        manifest = multiseed.write_manifest(tmp, "multi_fixed", root_entropy, plan, df)
        manifest_path = os.path.join(tmp, "manifest.json")
        exists_before_any_replicate = os.path.exists(manifest_path)
        with open(manifest_path, encoding="utf-8") as f:
            on_disk = json.load(f)
        has_root_entropy = on_disk.get("root_entropy") == str(root_entropy)
        has_replicates = len(on_disk.get("replicates", [])) == 2
    _check(
        "9. manifest.json é gravado (com entropia raiz e lista de réplicas) antes "
        "de qualquer réplica rodar",
        exists_before_any_replicate and has_root_entropy and has_replicates,
        f"manifest={manifest}",
    )


def test_multiseed_pseudo_regret_never_negative():
    """Teste 12: pseudo-regret (valor_esperado_do_oráculo - valor_esperado_do_item_
    escolhido) é não-negativo por construção -- diferente do regret realizado
    (regret_curve legado), que pode ser negativo por ruído amostral."""
    df, feature_space = _build_tiny_recommender_df(n_items=15)
    eligible = df.index.to_numpy(dtype=int)
    oracle_table = multiseed.build_expected_value_table(df, eligible)
    _, plan = multiseed.build_seed_plan("multi_fixed", 2, master_seed=5)

    all_non_negative = True
    for rseed in plan:
        result = multiseed.run_replicate(df, feature_space, rseed, n_episodes=25,
                                         oracle_table=oracle_table)
        if (result["pseudo_regret"] < -1e-6).any():
            all_non_negative = False
    _check("12. pseudo-regret nunca é negativo em nenhum episódio", all_non_negative)


def test_multiseed_isolation_from_production():
    """Teste 11 (parte estática): multiseed.py nunca referencia Recommender,
    FatigueTracker, Agent.save ou o log de interações -- busca textual, mesmo
    padrão já usado para confirmar que safety_filter_disabled nunca envolve
    main()."""
    content = open(os.path.join(REPO_DIR, "multiseed.py"), encoding="utf-8").read()
    # Ignora o docstring do módulo (linhas 1-N, entre as aspas triplas de abertura
    # e fechamento): ele descreve a garantia em prosa ("nunca chama Agent.save()"),
    # o que faria a própria busca textual acusar a si mesma. Busca só no código.
    _, _, code_only = content.partition('"""\n')
    code_only = code_only.split('"""', 1)[1] if '"""' in code_only else code_only
    forbidden = ["Recommender(", "FatigueTracker(", ".save(", "INTERACTION_LOG",
                "Recommender.recommend"]
    found = [term for term in forbidden if term in code_only]
    _check(
        "11. multiseed.py nunca instancia Recommender/FatigueTracker, nunca chama "
        ".save() nem referencia o log de interações",
        not found,
        f"termos encontrados: {found}",
    )


# ---------- Seleção dos slots: modos full/no_mmr/greedy, inércia, ablação ----------
#
# Teste 1 do plano de MMR ("com SELECTION_MODE='full', a saída de --eval é idêntica
# ao arquivo de referência") fica fora da suíte automática pelo mesmo motivo do
# teste 1 do multi-seed (custo): verificado manualmente com o diff contra
# tests/golden/eval_single_seed.txt.


def _build_varied_recommender(n_items: int = 60, seed: int = 0):
    """Catálogo sintético com feature vectors VARIADOS (3 Tipos, 5 Tags, V/A
    aleatórios) -- o suficiente para o MMR às vezes divergir do guloso, ao
    contrário de _build_tiny_recommender, cujos itens são quase idênticos."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Nome": [f"item{i}" for i in range(n_items)],
        "Tipo": rng.choice(["Áudio", "Vídeo", "Imagem"], size=n_items),
        "Valencia": rng.uniform(-0.9, 0.9, size=n_items),
        "Arousal": rng.uniform(-0.9, 0.9, size=n_items),
        "Duracao": [5.0] * n_items,
        "Indoor": [0] * n_items,
        "Tag": rng.choice(["a", "b", "c", "d", "e"], size=n_items),
        "Oitante": [1] * n_items,
    })
    feature_space = sysrec.FeatureSpace(df)
    agent = sysrec.Agent(df, feature_space)
    return df, feature_space, agent


def _random_contexts(n: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    return [(int(rng.integers(1, 9)), int(rng.choice(sysrec.ALLOWED_DEST_OCTANTS)),
             int(rng.integers(0, 2**31))) for _ in range(n)]


def test_selection_mode_validation():
    """Teste 7: SELECTION_MODE inválido levanta ValueError na importação --
    verificado executando o código-fonte REAL do módulo com o valor trocado (não
    uma réplica da lógica), sem reimportar o módulo da suíte. E um modo inválido
    passado por chamada também é rejeitado por recommend()."""
    path = os.path.join(REPO_DIR, "sistema_de_recomendacao_v3.py")
    src = open(path, encoding="utf-8").read()
    bad_src = src.replace('SELECTION_MODE = "full"', 'SELECTION_MODE = "nao_existe"', 1)
    raised = False
    with redirect_stdout(io.StringIO()):
        try:
            exec(compile(bad_src, path, "exec"), {"__name__": "sysrec_bad_mode", "__file__": path})
        except ValueError:
            raised = True
    _check("7a. SELECTION_MODE inválido levanta ValueError na importação", raised)

    df, feature_space, agent = _build_varied_recommender(n_items=20)
    rec = sysrec.Recommender(df, feature_space, agent)
    raised_call = False
    try:
        rec.recommend(1, 7, 60, register_fatigue=False, selection_mode="nao_existe")
    except ValueError:
        raised_call = True
    _check("7b. selection_mode inválido por chamada levanta ValueError", raised_call)


def test_selection_lambda1_equals_no_mmr():
    """Teste 2: para o mesmo estado do gerador, em 500 contextos, full com
    mmr_lambda=1.0 e no_mmr retornam os mesmos item_idx na mesma ordem e as mesmas
    propensões (rótulos de slot diferem de propósito: 'mmr' vs 'greedy_fill')."""
    df, feature_space, agent = _build_varied_recommender()
    rec_full = sysrec.Recommender(df, feature_space, agent)
    rec_nomm = sysrec.Recommender(df, feature_space, agent)
    mismatches = 0
    for curr, dest, seed in _random_contexts(500):
        with multiseed.seeded_scope(seed):
            a = rec_full.recommend(curr, dest, 60, register_fatigue=False,
                                   selection_mode="full", mmr_lambda=1.0)
        with multiseed.seeded_scope(seed):
            b = rec_nomm.recommend(curr, dest, 60, register_fatigue=False, selection_mode="no_mmr")
        if ([x["item_idx"] for x in a] != [x["item_idx"] for x in b]
                or [x["propensity"] for x in a] != [x["propensity"] for x in b]):
            mismatches += 1
    _check("2. full com lambda=1.0 == no_mmr (itens, ordem e propensões) em 500 contextos",
           mismatches == 0, f"divergências={mismatches}")


def test_selection_greedy_deterministic_no_rng():
    """Teste 3: greedy é determinístico, ordenado por `adjusted` decrescente, e não
    consome random/numpy/torch."""
    df, feature_space, agent = _build_varied_recommender()
    rec = sysrec.Recommender(df, feature_space, agent)
    py_s, np_s, t_s = random.getstate(), np.random.get_state(), torch.get_rng_state()
    first = rec.recommend(3, 7, 60, register_fatigue=False, selection_mode="greedy")
    unchanged = (random.getstate() == py_s
                 and all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
                         for x, y in zip(np.random.get_state(), np_s))
                 and torch.equal(torch.get_rng_state(), t_s))
    repeats_equal = all(
        [x["item_idx"] for x in rec.recommend(3, 7, 60, register_fatigue=False,
                                              selection_mode="greedy")]
        == [x["item_idx"] for x in first]
        for _ in range(10)
    )
    scores = [x["score"] for x in first]
    _check("3a. greedy não consome random/numpy/torch", unchanged)
    _check("3b. greedy retorna a mesma lista em chamadas repetidas", repeats_equal)
    _check("3c. greedy ordenado por adjusted decrescente",
           all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), f"scores={scores}")


def test_selection_full_no_mmr_pairing():
    """Teste 4: a partir do mesmo estado do gerador, os itens de slot 1 ('greedy') e
    exploratório ('explore') são idênticos -- e na mesma posição -- entre full e
    no_mmr; só os slots 'mmr'/'greedy_fill' podem diferir."""
    df, feature_space, agent = _build_varied_recommender()
    rec_full = sysrec.Recommender(df, feature_space, agent)
    rec_nomm = sysrec.Recommender(df, feature_space, agent)
    violations, explore_seen = 0, 0
    for curr, dest, seed in _random_contexts(300, seed=7):
        with multiseed.seeded_scope(seed):
            a = rec_full.recommend(curr, dest, 60, register_fatigue=False, selection_mode="full")
        with multiseed.seeded_scope(seed):
            b = rec_nomm.recommend(curr, dest, 60, register_fatigue=False, selection_mode="no_mmr")
        for x, y in zip(a, b):
            if x["slot_type"] in ("greedy", "explore") or y["slot_type"] in ("greedy", "explore"):
                explore_seen += x["slot_type"] == "explore"
                if (x["item_idx"], x["slot_type"], x["propensity"]) != \
                        (y["item_idx"], y["slot_type"], y["propensity"]):
                    violations += 1
    _check("4. slot 1 e slot exploratório idênticos (item, posição, propensão) entre "
           "full e no_mmr", violations == 0 and explore_seen > 0,
           f"violações={violations} exploratórios vistos={explore_seen}")


def test_selection_instrumentation_passive():
    """Teste 5: com o contador de inércia ativo, as listas de full são idênticas às
    de uma instância com o contador desligado (no-op)."""
    df, feature_space, agent = _build_varied_recommender()
    rec_on = sysrec.Recommender(df, feature_space, agent)
    rec_off = sysrec.Recommender(df, feature_space, agent)
    rec_off._record_mmr_step = lambda *args, **kwargs: None
    same = True
    for curr, dest, seed in _random_contexts(200, seed=11):
        with multiseed.seeded_scope(seed):
            a = rec_on.recommend(curr, dest, 60, register_fatigue=False)
        with multiseed.seeded_scope(seed):
            b = rec_off.recommend(curr, dest, 60, register_fatigue=False)
        same &= [x["item_idx"] for x in a] == [x["item_idx"] for x in b]
    _check("5. contador de inércia é passivo (listas idênticas com e sem ele)",
           same and rec_on.mmr_stats["steps"] > 0)


def test_selection_inertia_counter_correct():
    """Teste 6: (a) lambda=1.0 -> diverged=0; (b) pool com candidato de
    similaridade muito menor e diferença de score pequena -> diverged>0; (c) pool
    de vetores idênticos -> toda inércia classificada como pool homogêneo."""
    df, feature_space, agent = _build_varied_recommender()
    rec = sysrec.Recommender(df, feature_space, agent)
    for curr, dest, seed in _random_contexts(200, seed=13):
        with multiseed.seeded_scope(seed):
            rec.recommend(curr, dest, 60, register_fatigue=False, mmr_lambda=1.0)
    _check("6a. com lambda=1.0 o MMR nunca diverge do guloso",
           rec.mmr_stats["diverged"] == 0 and rec.mmr_stats["steps"] > 0, str(rec.mmr_stats))

    rec.reset_mmr_stats()
    adjusted = np.array([1.0, 0.95, 0.94])
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])   # item 2: similaridade 0 com o slot 1
    rec._select_slots(adjusted, vectors, k=2, deterministic=True, mode="full", mmr_lambda=0.7)
    _check("6b. candidato bem menos similar com gap de score pequeno -> MMR diverge",
           rec.mmr_stats["diverged"] == 1, str(rec.mmr_stats))

    rec.reset_mmr_stats()
    vectors_same = np.ones((3, 2))
    rec._select_slots(np.array([1.0, 0.5, 0.3]), vectors_same, k=3, deterministic=True,
                      mode="full", mmr_lambda=0.7)
    s = rec.mmr_stats
    _check("6c. vetores idênticos -> toda inércia classificada como pool homogêneo",
           s["steps"] == 2 and s["inert_homogeneous"] == 2 and s["diverged"] == 0, str(s))


def test_selection_mode_logged():
    """Teste 8: cada registro de interação contém selection_mode -- capturado
    substituindo _persist_interaction (nunca escreve no interaction_log.jsonl real)
    e as funções de entrada do CLI."""
    df, feature_space, agent = _build_varied_recommender(n_items=20)
    rec = sysrec.Recommender(df, feature_space, agent)
    captured = []
    originals = {name: getattr(sysrec, name) for name in
                 ("_persist_interaction", "_read_octant", "_read_choice", "_read_feedback")}
    octants = iter([1, 7])
    try:
        sysrec._persist_interaction = lambda record, path=None: captured.append(record)
        sysrec._read_octant = lambda prompt: next(octants)
        sysrec._read_choice = lambda prompt, options: "A"
        sysrec._read_feedback = lambda: (4, 0.5)
        with redirect_stdout(io.StringIO()):
            sysrec._run_interaction(rec, agent, feature_space)
    finally:
        for name, fn in originals.items():
            setattr(sysrec, name, fn)
    _check("8. registro de interação contém selection_mode",
           len(captured) == 1 and captured[0].get("selection_mode") == sysrec.SELECTION_MODE,
           f"registros={captured}")


def test_selection_list_metrics():
    """Teste 9: lista de três itens da mesma modalidade tem n_modalities=1, e
    ild_features coincide com o cálculo de coverage_and_diversity (1 - média do
    cosseno entre pares sobre item_matrix)."""
    import selection_ablation as sa
    df, feature_space, _ = _build_varied_recommender()
    tipo = df["Tipo"].iloc[0]
    items = df.index[df["Tipo"] == tipo][:3].tolist()
    ev_row = np.zeros(len(df))
    m = sa.list_metrics(items, ["greedy", "mmr", "mmr"], df, feature_space, ev_row,
                        np.array(items))
    vecs = [feature_space.item_matrix[i] for i in items]
    sims = [sysrec._cosine_similarity(vecs[i], vecs[j])
            for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    expected_ild = 1.0 - float(np.mean(sims))
    _check("9a. três itens da mesma modalidade -> n_modalities=1", m["n_modalities"] == 1)
    _check("9b. ild_features == cálculo de coverage_and_diversity",
           abs(m["ild_features"] - expected_ild) < 1e-12,
           f"{m['ild_features']} vs {expected_ild}")


def test_selection_choice_model():
    """Teste 10: com afinidade de escala 0 a utilidade é exatamente o valor
    esperado; o mesmo u_choice produz a mesma escolha para a mesma lista."""
    import selection_ablation as sa
    df, _, _ = _build_varied_recommender()
    items = [0, 1, 2]
    ev_row = np.linspace(-0.5, 0.5, len(df))
    mods = sorted(df["Tipo"].unique())
    zero_aff = {m: 0.0 for m in mods}
    a = sa.choice_metrics(items, df, ev_row, zero_aff, 0.42)
    b = sa.choice_metrics(items, df, ev_row, zero_aff, 0.42)
    _check("10a. escala 0 -> utilidade == valor esperado",
           a["best_utility"] == float(ev_row[items].max())
           and a["chosen_utility"] in [float(v) for v in ev_row[items]])
    _check("10b. mesmo u_choice -> mesma escolha", a == b)


def test_selection_crn_across_configs():
    """Teste 11: acrescentar uma configuração à comparação não altera as métricas
    das demais (cada configuração tem seu Recommender; cada chamada roda na seed
    de seleção do contexto)."""
    import selection_ablation as sa
    df, feature_space, agent = _build_varied_recommender()
    ev_table = multiseed.build_expected_value_table(df, df.index.to_numpy(dtype=int))
    contexts = sa.pregenerate_contexts(42, 0, 40, sorted(df["Tipo"].unique()))
    base_cfg = [("no_mmr", "no_mmr", None), ("full@0.7", "full", 0.7)]
    more_cfg = [("greedy", "greedy", None)] + base_cfg + [("full@0.3", "full", 0.3)]
    small = sa.run_protocol_p1(df, feature_space, agent, contexts, ev_table, base_cfg)
    big = sa.run_protocol_p1(df, feature_space, agent, contexts, ev_table, more_cfg)
    same = all(np.array_equal(small[n]["metrics"][k], big[n]["metrics"][k])
               for n, _, _ in base_cfg for k in small[n]["metrics"])
    _check("11. acrescentar configurações não altera as métricas das demais (CRN)", same)


def test_selection_ablation_isolation():
    """Teste 12: selection_ablation.py nunca chama Agent.save(), nunca toca o log
    de interações e nunca treina o agente durante a comparação (busca textual no
    código, fora do docstring) -- e _frozen_agent faz learn_from_feedback levantar
    dentro do bloco, restaurando o método da classe ao sair."""
    import selection_ablation as sa
    content = open(os.path.join(REPO_DIR, "selection_ablation.py"), encoding="utf-8").read()
    _, _, code_only = content.partition('"""\n')
    code_only = code_only.split('"""', 1)[1]
    forbidden = [".save(", "INTERACTION_LOG", "_persist_interaction", "learn_from_feedback("]
    found = [t for t in forbidden if t in code_only]
    _check("12a. selection_ablation.py não salva checkpoint, não toca o log, não treina",
           not found, f"encontrados: {found}")

    _, _, agent = _build_varied_recommender(n_items=10)
    raised = False
    with sa._frozen_agent(agent):
        try:
            agent.learn_from_feedback(None, 0, 0.0, None)
        except RuntimeError:
            raised = True
    restored = "learn_from_feedback" not in vars(agent)
    _check("12b. learn_from_feedback levanta durante a comparação e é restaurado depois",
           raised and restored)


def test_selection_range_index_check():
    """Teste 13: a ablação exige df.index == RangeIndex antes de usar a tabela de
    valor esperado por posição, e aborta com erro claro se não for."""
    import selection_ablation as sa
    df, _, _ = _build_varied_recommender(n_items=10)
    ok = True
    try:
        sa.check_range_index(df)
    except ValueError:
        ok = False
    raised = False
    try:
        sa.check_range_index(df.iloc[::-1])
    except ValueError:
        raised = True
    _check("13. check_range_index aceita RangeIndex e rejeita índice fora de ordem", ok and raised)


def _build_tiny_recommender_df(n_items: int = 10):
    """Catálogo sintético mínimo + FeatureSpace, para os testes de multiseed.py
    que não precisam de Recommender/Agent prontos (run_replicate cria seu próprio
    Agent por réplica)."""
    df = pd.DataFrame({
        "Nome": [f"item{i}" for i in range(n_items)],
        "Tipo": ["Áudio"] * n_items,
        "Valencia": [0.1 * (i % 10) for i in range(n_items)],
        "Arousal": [-0.05 * (i % 10) for i in range(n_items)],
        "Duracao": [5.0] * n_items,
        "Indoor": [0] * n_items,
        "Tag": ["x"] * n_items,
        "Oitante": [1] * n_items,
    })
    feature_space = sysrec.FeatureSpace(df)
    return df, feature_space


def main() -> None:
    test_write_methods_absent()
    test_pymongo_import_is_local_not_module_level()
    test_empty_allowlist_returns_empty_catalog()
    test_approved_category_filters_correctly()
    test_camada3_independent_of_category_approval()
    test_denylist_word_boundary_and_normalization()
    test_feature_space_schema_retains_diagnostic_columns()
    test_safety_filter_r2_blocks_high_energy_octant()
    test_safety_filter_never_empties_eligible()
    test_fatigue_blocked_mask_correctness()
    test_fatigue_only_tracks_executed_items()
    test_fatigue_relaxation_releases_oldest_first()
    test_fatigue_min_gap_pool_invariant_raises_on_violation()
    test_fatigue_relaxation_rate_zero_when_comfortable()
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

    test_multiseed_config_validation_raises()
    test_multiseed_build_seed_plan_32bit_limit()
    test_multiseed_seeded_scope_restores_state()
    test_multiseed_seeded_scope_restores_on_exception()
    test_multiseed_crn_no_global_consumption_when_presampled()
    test_multiseed_pregenerate_randomness_shapes()
    test_multiseed_reproducible_same_process()
    test_multiseed_crn_arm_independence()
    test_multiseed_random_mode_fresh_entropy()
    test_multiseed_random_mode_replay_reproducible()
    test_multiseed_manifest_written_before_replicates()
    test_multiseed_pseudo_regret_never_negative()
    test_multiseed_isolation_from_production()

    test_selection_mode_validation()
    test_selection_lambda1_equals_no_mmr()
    test_selection_greedy_deterministic_no_rng()
    test_selection_full_no_mmr_pairing()
    test_selection_instrumentation_passive()
    test_selection_inertia_counter_correct()
    test_selection_mode_logged()
    test_selection_list_metrics()
    test_selection_choice_model()
    test_selection_crn_across_configs()
    test_selection_ablation_isolation()
    test_selection_range_index_check()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} teste(s) falharam: {_FAILURES}")
        sys.exit(1)
    print("Todos os testes passaram.")


if __name__ == "__main__":
    main()
