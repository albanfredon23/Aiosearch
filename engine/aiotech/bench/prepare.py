"""
Construit les échantillons figés du benchmark à partir des fichiers publiés (exécuté une fois, résultat versionné).

    python -m aiotech.bench.prepare --hotpot hotpot_dev_distractor_v1.json \
        --fever-dir FeverSymmetric/symmetric_v0.2 --out aiotech/bench/data

HotpotQA : 300 questions tirées sans remise (random.Random(20261006).sample) dans les 7 405 questions
du dev « distractor » officiel, chacune avec ses 10 paragraphes (2 utiles, 8 leurres).

FEVER : les 355 paires originales (affirmation, phrase de preuve Wikipédia, SUPPORTS/REFUTES) du
dev FEVER publiées dans FeverSymmetric v0.2 (dev + test). Les phrases modifiées de FeverSymmetric
(dédupliquées) servent uniquement de leurres difficiles dans le corpus commun de preuves.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any

HOTPOT_SHA256 = "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea"
SEED = 20261006
HOTPOT_SAMPLE = 300
FEVER_FILES = ("fever_symmetric_dev.jsonl", "fever_symmetric_test.jsonl")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_generated(fever_id: str) -> bool:
    return len(fever_id) > 7 and fever_id[-7:-1] == "000000"


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.GzipFile(path, "wb", mtime=0) as raw:
        for row in rows:
            raw.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())


def hotpot_sample(source: Path) -> list[dict[str, Any]]:
    digest = sha256(source)
    if digest != HOTPOT_SHA256:
        raise SystemExit(f"{source} : empreinte {digest} différente du fichier officiel attendu")
    data = json.loads(source.read_text("utf-8"))
    chosen = random.Random(SEED).sample(range(len(data)), HOTPOT_SAMPLE)
    rows = []
    for index in sorted(chosen):
        item = data[index]
        rows.append({
            "id": item["_id"], "question": item["question"], "answer": item["answer"], "type": item["type"],
            "supporting_titles": sorted({title for title, _ in item["supporting_facts"]}),
            "supporting_facts": item["supporting_facts"],
            "context": [{"title": title, "sentences": sentences} for title, sentences in item["context"]],
        })
    return rows


def fever_pairs(directory: Path) -> list[dict[str, Any]]:
    claims: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}

    def add_evidence(text: str, synthetic: bool) -> str:
        key = "e" + hashlib.blake2b(text.encode(), digest_size=6).hexdigest()
        current = evidence.get(key)
        if current is None or (current["synthetic"] and not synthetic):
            evidence[key] = {"kind": "evidence", "id": key, "text": text, "synthetic": synthetic}
        return key

    for name in FEVER_FILES:
        for line in (directory / name).read_text("utf-8").splitlines():
            row = json.loads(line)
            text = row.get("evidence_sentence") or row["evidence"]
            label = row.get("gold_label") or row["label"]
            if is_generated(row["id"]):
                add_evidence(text, synthetic=True)
                continue
            evidence_id = add_evidence(text, synthetic=False)
            claims[row["id"]] = {"kind": "claim", "id": row["id"], "claim": row["claim"], "label": label,
                                 "evidence_id": evidence_id}
    return [*sorted(claims.values(), key=lambda r: int(r["id"])), *sorted(evidence.values(), key=lambda r: r["id"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hotpot", type=Path, required=True)
    parser.add_argument("--fever-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hotpot = hotpot_sample(args.hotpot)
    write_jsonl_gz(args.out / "hotpotqa_dev_distractor_s300.jsonl.gz", hotpot)
    fever = fever_pairs(args.fever_dir)
    write_jsonl_gz(args.out / "fever_dev_pairs.jsonl.gz", fever)
    n_claims = sum(1 for r in fever if r["kind"] == "claim")
    print(f"HotpotQA : {len(hotpot)} questions ; FEVER : {n_claims} affirmations, {len(fever) - n_claims} phrases de preuve")


if __name__ == "__main__":
    main()
