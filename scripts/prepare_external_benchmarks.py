from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


DATASETS = (
    "scifact_open",
    "evidence_inference",
    "nli4ct",
    "limitgen",
    "claimdecomp",
    "litsearch",
)

URLS = {
    "scifact_open": "https://scifact.s3.us-west-2.amazonaws.com/scifact-open/latest/scifact_open.tar.gz",
    "evidence_annotations": "https://raw.githubusercontent.com/jayded/evidence-inference/master/annotations/annotations_merged.csv",
    "evidence_prompts": "https://raw.githubusercontent.com/jayded/evidence-inference/master/annotations/prompts_merged.csv",
    "nli4ct": "https://raw.githubusercontent.com/ai-systems/nli4ct/main/Complete_dataset.zip",
    "limitgen": "https://huggingface.co/datasets/yale-nlp/LimitGen/resolve/main/human/classified_limitations.json?download=true",
    "claimdecomp": "https://docs.google.com/spreadsheets/d/1g6bmuc1D5jbLE0U9Y68MZWH_VCr6ZXfeKA2muA1uJD0/export?format=csv&gid=0",
    "litsearch_query": "https://huggingface.co/datasets/princeton-nlp/LitSearch/resolve/main/query/full-00000-of-00001.parquet?download=true",
}

LITSEARCH_CORPUS_URLS = tuple(
    "https://huggingface.co/datasets/princeton-nlp/LitSearch/resolve/main/"
    f"corpus_clean/full-{index:05d}-of-00006.parquet"
    for index in range(6)
)

DATASET_URL_KEYS = {
    "scifact_open": ("scifact_open",),
    "evidence_inference": ("evidence_annotations", "evidence_prompts"),
    "nli4ct": ("nli4ct",),
    "limitgen": ("limitgen",),
    "claimdecomp": ("claimdecomp",),
    "litsearch": ("litsearch_query",),
}

DATASET_FILES = {
    "scifact_open": (
        "scifact_open.tar.gz",
        "scifact-open/data/claims.jsonl",
        "scifact-open/data/corpus_candidates.jsonl",
    ),
    "evidence_inference": ("evidence_annotations.csv", "evidence_prompts.csv"),
    "nli4ct": ("nli4ct.zip", "nli4ct/Complete_dataset/Gold_test.json"),
    "limitgen": ("limitgen_human.json",),
    "claimdecomp": ("claimdecomp_annotations.csv",),
    "litsearch": (
        "litsearch_query.parquet",
        "litsearch_queries.jsonl",
        "litsearch_gold_corpus.jsonl",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download official external benchmark artifacts for ClaimScope."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/external-benchmarks/data"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    selected = list(dict.fromkeys(args.datasets))
    for dataset in selected:
        print(f"Preparing {dataset}...")
        PREPARERS[dataset](args.data_dir, args.force)
    _write_manifest(args.data_dir, selected)


def _prepare_scifact_open(root: Path, force: bool) -> None:
    archive = root / "scifact_open.tar.gz"
    _download(URLS["scifact_open"], archive, force)
    output = root / "scifact-open"
    wanted = {
        "data/claims.jsonl",
        "data/corpus_candidates.jsonl",
    }
    if force or not all((output / name).exists() for name in wanted):
        with tarfile.open(archive, "r:gz") as handle:
            members = [member for member in handle.getmembers() if member.name in wanted]
            if {member.name for member in members} != wanted:
                raise RuntimeError("SciFact-Open archive is missing required files.")
            handle.extractall(output, members=members)


def _prepare_evidence_inference(root: Path, force: bool) -> None:
    _download(
        URLS["evidence_annotations"], root / "evidence_annotations.csv", force
    )
    _download(URLS["evidence_prompts"], root / "evidence_prompts.csv", force)


def _prepare_nli4ct(root: Path, force: bool) -> None:
    archive = root / "nli4ct.zip"
    _download(URLS["nli4ct"], archive, force)
    output = root / "nli4ct"
    expected = output / "Complete_dataset" / "Gold_test.json"
    if expected.exists() and not force:
        return
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            target = (output / info.filename).resolve()
            if output.resolve() not in target.parents and target != output.resolve():
                raise RuntimeError("NLI4CT archive contains an unsafe path.")
        handle.extractall(output)


def _prepare_limitgen(root: Path, force: bool) -> None:
    _download(URLS["limitgen"], root / "limitgen_human.json", force)


def _prepare_claimdecomp(root: Path, force: bool) -> None:
    _download(
        URLS["claimdecomp"], root / "claimdecomp_annotations.csv", force
    )


def _prepare_litsearch(root: Path, force: bool) -> None:
    query_parquet = root / "litsearch_query.parquet"
    queries_jsonl = root / "litsearch_queries.jsonl"
    corpus_jsonl = root / "litsearch_gold_corpus.jsonl"
    _download(URLS["litsearch_query"], query_parquet, force)
    if not force and queries_jsonl.exists() and corpus_jsonl.exists():
        return
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "LitSearch preparation requires DuckDB. Install with "
            "`pip install -e .[benchmark]`."
        ) from exc
    connection = duckdb.connect()
    try:
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")
        query_rows = connection.execute(
            "SELECT query_set, query, specificity, quality, corpusids "
            "FROM read_parquet(?)",
            [str(query_parquet.resolve())],
        ).fetchall()
        connection.execute(
            "CREATE OR REPLACE TEMP TABLE gold_ids AS "
            "SELECT DISTINCT UNNEST(corpusids) AS corpusid FROM read_parquet(?)",
            [str(query_parquet.resolve())],
        )
        corpus_rows = connection.execute(
            "SELECT corpusid, title, abstract FROM read_parquet(?) "
            "WHERE corpusid IN (SELECT corpusid FROM gold_ids)",
            [list(LITSEARCH_CORPUS_URLS)],
        ).fetchall()
    finally:
        connection.close()
    _write_jsonl(
        queries_jsonl,
        (
            {
                "query_set": row[0],
                "query": row[1],
                "specificity": row[2],
                "quality": row[3],
                "corpusids": row[4],
            }
            for row in query_rows
        ),
    )
    _write_jsonl(
        corpus_jsonl,
        (
            {"corpusid": row[0], "title": row[1], "abstract": row[2]}
            for row in corpus_rows
        ),
    )


def _download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ClaimScope-eval/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_manifest(root: Path, datasets: list[str]) -> None:
    entries: dict[str, object] = {}
    for dataset in datasets:
        files = []
        for relative_path in DATASET_FILES[dataset]:
            path = root / relative_path
            if not path.exists():
                continue
            files.append(
                {
                    "path": relative_path.replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        entries[dataset] = {
            "source_urls": [
                *[URLS[key] for key in DATASET_URL_KEYS[dataset]],
                *(LITSEARCH_CORPUS_URLS if dataset == "litsearch" else ()),
            ],
            "files": files,
        }
    payload = {
        "schema_version": 1,
        "datasets": entries,
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PREPARERS = {
    "scifact_open": _prepare_scifact_open,
    "evidence_inference": _prepare_evidence_inference,
    "nli4ct": _prepare_nli4ct,
    "limitgen": _prepare_limitgen,
    "claimdecomp": _prepare_claimdecomp,
    "litsearch": _prepare_litsearch,
}


if __name__ == "__main__":
    main()
