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
    # 3) áudio EMOPIA com quadrante reconhecido -> resolve via centroide, categoria aprovada.
    {
        "_id": "a2", "mediaType": "audio", "title": "Trilha alegre",
        "source": {"dataset": "EMOPIA"}, "quadrant": "Q1", "durationSeconds": 90,
        "tags": ["energetic"], "category": "music_energetic",
    },
    # 4) áudio EMOPIA SEM quadrante reconhecido -> descartado (va_ausente), não incluído com nulo.
    {
        "_id": "a3", "mediaType": "audio", "title": "Sem quadrante",
        "source": {"dataset": "EMOPIA"}, "quadrant": "desconhecido",
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


def test_resolve_non_continuous_va():
    """Teste 8: centroide correto para quadrante válido, (None, None) para inválido/ausente."""
    v, a = data_source._resolve_non_continuous_va({"quadrant": "Q3"}, "EMOPIA")
    _check("8a. EMOPIA Q3 -> centroide (-0.5, -0.5)", (v, a) == (-0.5, -0.5))

    v, a = data_source._resolve_non_continuous_va({"quadrant": "Q9"}, "EMOPIA")
    _check("8b. EMOPIA quadrante desconhecido -> (None, None)", (v, a) == (None, None))

    v, a = data_source._resolve_non_continuous_va({}, "EMOPIA")
    _check("8c. EMOPIA sem campo quadrant -> (None, None)", (v, a) == (None, None))


def test_normalization_reference_scale_none_is_skipped():
    """Teste 7: datasets com scale=None são pulados com aviso, sem erro nem fórmula assumida."""
    fake = FakeCollection([])
    import normalization

    for dataset_name in ("MuVi", "EmoMadrid"):
        buf = io.StringIO()
        with redirect_stdout(buf):
            discrepancias = normalization.audit_dataset(fake, dataset_name)
        _check(
            f"7. {dataset_name} (scale=None) é pulado sem erro",
            discrepancias == [] and "PULAR" in buf.getvalue(),
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
    test_resolve_non_continuous_va()
    test_normalization_reference_scale_none_is_skipped()
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
