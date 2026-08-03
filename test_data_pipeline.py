"""
Testes de aceitação da auditoria de normalização e da curadoria de segurança
(ver seção 7 do plano). Roda sem Mongo real: usa uma FakeCollection em memória que
implementa só find()/aggregate(), o suficiente para exercitar data_source.py,
safety.py e normalization.py de ponta a ponta.

Não é integrado ao --eval do sistema_de_recomendacao_v3.py (que é a bancada do
protótipo, sempre sobre o CSV sintético) -- este arquivo testa especificamente os três
módulos novos desta migração para Mongo. Roda como script, sem framework de teste
(mesmo estilo de sanity_checks() em sistema_de_recomendacao_v3.py: funções que
imprimem PASS/FAIL, sem dependência de pytest).

Uso:
    python test_data_pipeline.py
"""
import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np

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
    """Substitui pymongo.Collection nos testes: find()/aggregate() sobre uma lista de
    dicts em memória. Não implementa update/insert/delete de propósito -- se algum
    código sob teste tentasse chamar um desses métodos, o teste quebraria com
    AttributeError, o que é o comportamento correto a se ter."""

    def __init__(self, docs: list):
        self._docs = docs

    def find(self, query: dict | None = None):
        query = query or {}
        return [doc for doc in self._docs if self._matches(doc, query)]

    def aggregate(self, pipeline: list):
        # Suficiente para o pipeline usado em safety.explore_taxonomy
        # ($group por dataset/category + $addToSet + $sum, depois $sort).
        docs = list(self._docs)
        result = docs
        for stage in pipeline:
            if "$group" in stage:
                result = self._group(result, stage["$group"])
            elif "$sort" in stage:
                result = self._sort(result, stage["$sort"])
        return result

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

    @staticmethod
    def _group(docs: list, spec: dict) -> list:
        buckets = {}
        for doc in docs:
            key = tuple(data_source._get(doc, path.lstrip("$")) for path in spec["_id"].values())
            bucket = buckets.setdefault(key, {"_id": dict(zip(spec["_id"].keys(), key)), "count": 0, "subcats": set()})
            bucket["count"] += 1
        return [{**b, "subcats": list(b["subcats"])} for b in buckets.values()]

    @staticmethod
    def _sort(docs: list, spec: dict) -> list:
        for field, direction in reversed(list(spec.items())):
            path = field.split(".")
            docs = sorted(docs, key=lambda d: _dig(d, path), reverse=(direction < 0))
        return docs


def _dig(d: dict, path: list):
    for key in path:
        d = d.get(key, {}) if isinstance(d, dict) else {}
    return d if not isinstance(d, dict) else ""


# ---------- Documentos fake, cobrindo os casos da seção 7 ----------

DOCS = [
    # 1) vídeo OASIS válido, categoria aprovada -> deve passar.
    {
        "_id": "v1", "mediaType": "video", "title": "Praia ao pôr do sol",
        "ratings": {"valenceMean": 0.6, "arousalMean": -0.3},
        "durationSeconds": 120, "tags": ["nature", "calm"], "category": "nature",
        "sourceMeta": {"dataset": "OASIS"}, "videoOctant": 8, "videoUrl": "http://x/v1",
    },
    # 2) áudio DEAM válido, categoria aprovada -> deve passar.
    {
        "_id": "a1", "mediaType": "audio", "title": "Chuva suave",
        "staticAnnotations": {"valenceNormalized": 0.1, "arousalNormalized": -0.6},
        "source": {"dataset": "DEAM"}, "durationSeconds": 180,
        "tags": ["rain"], "category": "ambient", "soundOctant": 7, "audioUrl": "http://x/a1",
    },
    # 3) áudio EMOPIA com quadrante reconhecido via tag -> resolve via centroide,
    #    categoria aprovada. Quadrante vem de tags/subcategories (formato
    #    "quadrant_qN"), nunca de staticAnnotations -- ver correção da seção 5.
    {
        "_id": "a2", "mediaType": "audio", "title": "Trilha alegre",
        "source": {"dataset": "EMOPIA"}, "durationSeconds": 90,
        "tags": ["quadrant_q1", "energetic"], "category": "music_energetic",
    },
    # 4) áudio EMOPIA SEM quadrante reconhecido nas tags -> descartado (va_ausente), não incluído com nulo.
    {
        "_id": "a3", "mediaType": "audio", "title": "Sem quadrante",
        "source": {"dataset": "EMOPIA"}, "tags": ["ambient"],
        "category": "music_energetic",
    },
    # 5) imagem com keyword de denylist na tag, categoria aprovada -> deve ser excluída (Camada 2).
    {
        "_id": "i1", "mediaType": "image", "title": "Cena de guerra",
        "ratings": {"valenceNormalized": -0.5, "arousalNormalized": 0.7},
        "tags": ["war"], "category": "nature", "sourceMeta": {"dataset": "OASIS"},
    },
    # 6) imagem com valência muito negativa, categoria NÃO aprovada -> excluída (Camadas 1+3).
    {
        "_id": "i2", "mediaType": "image", "title": "Estímulo aversivo",
        "ratings": {"valenceNormalized": -0.9, "arousalNormalized": 0.5},
        "tags": [], "category": "aversive_research", "sourceMeta": {"dataset": "GAPED"},
    },
    # 7) vídeo com categoria aprovada, mas item_id vai para BLOCKED_ITEM_IDS -> excluído (Camada 4).
    {
        "_id": "v2", "mediaType": "video", "title": "Bloqueado manualmente",
        "ratings": {"valenceMean": 0.5, "arousalMean": -0.2},
        "durationSeconds": 60, "tags": ["nature"], "category": "nature",
        "sourceMeta": {"dataset": "OASIS"}, "videoOctant": 8,
    },
    # 8) modalidade desconhecida -> descartada antes mesmo da curadoria.
    {"_id": "x1", "mediaType": "haptic", "title": "Modalidade nova"},
    # 9) vídeo sem V/A (ratings ausente) -> descartado por va_ausente.
    {
        "_id": "v3", "mediaType": "video", "title": "Sem avaliação",
        "durationSeconds": 30, "category": "nature", "sourceMeta": {"dataset": "OASIS"},
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


def test_empty_allowlist_returns_empty_catalog():
    """Teste 2: APPROVED_CATEGORIES vazio -> DataFrame vazio, sem exceção, sem travar."""
    sysrec.APPROVED_CATEGORIES.clear()
    sysrec.BLOCKED_ITEM_IDS.clear()
    data_source.get_read_only_client = lambda: FakeCollection(DOCS)

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
    data_source.get_read_only_client = lambda: FakeCollection(DOCS)

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

    # Descartes: modalidade desconhecida, V/A ausente (vídeo v3 e EMOPIA sem quadrante).
    _check(
        "descartes reportados por motivo (modalidade_desconhecida e va_ausente)",
        "modalidade_desconhecida=1" in output and "va_ausente=2" in output,
        f"saída: {output.strip().splitlines()[-1] if output else '(vazia)'}",
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
    fórmula assumida. EmoMadrid e MuVi -- os dois scale=None originais -- foram
    resolvidos nesta rodada de correções (seções 2 e 3), então o mecanismo é testado
    aqui com uma entrada temporária injetada em NORMALIZATION_REFERENCE, não com um
    dataset real (que não existe mais no estado 'não confirmado').
    """
    import normalization

    sysrec.NORMALIZATION_REFERENCE["_FAKE_UNCONFIRMED"] = {
        "raw_field": "ratings.valenceMean", "scale": None,
    }
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            discrepancias = normalization.audit_dataset(FakeCollection([]), "_FAKE_UNCONFIRMED")
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

    fake = FakeCollection([
        {"_id": "em1", "sourceMeta": {"dataset": "EmoMadrid"},
         "ratings": {"valenceMean": 1.13, "valenceNormalized": 0.565}},
    ])
    discrepancias = normalization.audit_dataset(fake, "EmoMadrid")
    _check("EmoMadrid (-2,2): exemplo real bate sem discrepância", discrepancias == [], f"{discrepancias}")


def test_meditation_local_scale_confirmed_by_example():
    """
    Correção seção 4: escala (1,9) confirmada pela própria description do dado
    ("x' = (x - 5) / 4"). Exemplo real: valenceMean=6.8 -> (6.8-5)/4 = 0.45.
    """
    import normalization

    fake = FakeCollection([
        {"_id": "ml1", "source": {"dataset": "MEDITATION_LOCAL"},
         "staticAnnotations": {"valenceMean": 6.8, "valenceNormalized": 0.45}},
    ])
    discrepancias = normalization.audit_dataset(fake, "MEDITATION_LOCAL")
    _check("MEDITATION_LOCAL (1,9): exemplo real bate sem discrepância", discrepancias == [], f"{discrepancias}")


def test_muvi_identity_range_check():
    """
    Correção seção 3: MuVi não tem campo *Normalized separado (scale="IDENTITY") --
    a auditoria vira checagem de faixa sobre o valor bruto, usado diretamente. Exemplo
    real (dentro da faixa) + um caso injetado fora de [-1,1] para confirmar detecção.
    """
    import normalization

    fake = FakeCollection([
        {"_id": "mv1", "sourceMeta": {"dataset": "MuVi"},
         "ratings": {"valenceMean": -0.0017458, "arousalMean": 0.479212}},  # exemplo real, dentro da faixa
        {"_id": "mv2", "sourceMeta": {"dataset": "MuVi"},
         "ratings": {"valenceMean": 1.5, "arousalMean": 0.2}},  # fora de [-1,1], injetado
    ])
    discrepancias = normalization.audit_dataset(fake, "MuVi")
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
    "_id": "quadrant_q1_example", "mediaType": "audio", "title": "Exemplo real EMOPIA",
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
    fake = FakeCollection([
        {"_id": "d1", "sourceMeta": {"dataset": "OASIS"},
         "ratings": {"valenceMean": 7, "valenceNormalized": -5.0}},
        {"_id": "d2", "sourceMeta": {"dataset": "OASIS"},
         "ratings": {"valenceMean": 4, "valenceNormalized": 0.0}},  # correto, sem discrepância
    ])
    discrepancias = normalization.audit_dataset(fake, "OASIS")
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
    data_source.get_read_only_client = lambda: FakeCollection(DOCS)

    buf = io.StringIO()
    with redirect_stdout(buf):
        cat1 = data_source.load_catalog()
        cat2 = data_source.load_catalog()
    _check("9. duas cargas seguidas produzem o mesmo DataFrame", cat1.equals(cat2))


def test_statistical_checks_run_without_error():
    """Checagens estatísticas complementares (seção 6): rodam sem erro sobre dados fake,
    inclusive detectando um valor fora de [-1, 1] injetado de propósito."""
    import normalization

    fake = FakeCollection([
        {"_id": "s1", "sourceMeta": {"dataset": "OASIS"}, "videoOctant": 1,
         "ratings": {"valenceNormalized": 1.4, "arousalNormalized": 0.2}},  # fora de [-1,1] de propósito
        {"_id": "s2", "sourceMeta": {"dataset": "OASIS"}, "videoOctant": 1,
         "ratings": {"valenceNormalized": 0.3, "arousalNormalized": 0.1}},
    ])
    buf = io.StringIO()
    ok = True
    try:
        with redirect_stdout(buf):
            normalization.check_out_of_range(fake)
            normalization.check_zero_variance(fake)
            normalization.check_octant_region_agreement(fake)
    except Exception as exc:
        ok = False
        print(f"      exceção inesperada: {exc!r}")
    output = buf.getvalue()
    _check(
        "checagens estatísticas rodam sem erro e detectam valor fora de [-1,1]",
        ok and "1 documento" in output,
        output,
    )


def main() -> None:
    test_write_methods_absent()
    test_empty_allowlist_returns_empty_catalog()
    test_approved_category_filters_correctly()
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

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} teste(s) falharam: {_FAILURES}")
        sys.exit(1)
    print("Todos os testes passaram.")


if __name__ == "__main__":
    main()
