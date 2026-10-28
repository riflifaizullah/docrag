"""Core document reconciliation (new/updated/unchanged/deleted via content
hashing). Shared by the scheduled job, the /sync endpoint, the UI button,
and direct manual runs (`python -m app.sync_documents`) -- one
implementation, four trigger paths."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from app import config, database, ingestion, retrieval

_scheduler: BackgroundScheduler | None = None


def sync_once(corpus_dir: Path = config.CORPUS_DIR, verbose: bool = False) -> dict:
    """verbose=True prints per-file progress as it happens -- off by default
    so the scheduled background job and the /sync API endpoint (which call
    this every run) don't spam server logs; the manual `python -m
    app.sync_documents` CLI entrypoint below turns it on, since that's the
    one place a human is actually watching the terminal live."""
    collection = retrieval.get_collection()
    embedder = retrieval.get_embedder()

    disk_files = {p.name: p for p in corpus_dir.glob("*.pdf")}
    known_sources = set(database.list_document_sources())

    new_files, updated_files, unchanged_files, deleted_files = [], [], [], []
    to_ingest: dict[str, Path] = {}

    for name, path in disk_files.items():
        existing_hash = database.get_document_hash(name)
        if existing_hash is None:
            to_ingest[name] = path
            new_files.append(name)
            continue

        # Cheap mtime+size check first -- at thousands-of-files scale,
        # reading every file's full bytes to hash it on every sync tick
        # just to confirm "unchanged" is real, avoidable I/O. A missing
        # fingerprint (a file recorded before this existed) always falls
        # through to a real hash rather than being trusted blindly.
        existing_fp = database.get_document_fingerprint(name)
        current_fp = ingestion.file_fingerprint(path)
        if existing_fp is not None and existing_fp == current_fp:
            unchanged_files.append(name)
            continue

        current_hash = ingestion.content_hash(path)
        if existing_hash != current_hash:
            to_ingest[name] = path
            updated_files.append(name)
        else:
            unchanged_files.append(name)

    to_delete = known_sources - set(disk_files.keys())
    if verbose:
        print(
            f"Found {len(new_files)} new, {len(updated_files)} updated, "
            f"{len(unchanged_files)} unchanged, {len(to_delete)} to delete.",
            flush=True,
        )

    # PDF parsing/OCR (extract_ingest_data) is CPU-bound and per-file
    # independent, so it's fanned out across worker processes. The actual
    # writes (write_ingest_data) touch Chroma/the embedder/sqlite, which
    # must stay on one consistent thread/process (see retrieval.py's
    # _chroma_thread comment on the cross-thread ChromaDB bug) -- so results
    # are written back sequentially here in the main process as each
    # extraction finishes, never in parallel.
    total = len(to_ingest)
    done = 0
    if len(to_ingest) == 1:
        ((name, path),) = to_ingest.items()
        if verbose:
            print(f"[1/1] Ingesting {name} ...", flush=True)
        data = ingestion.extract_ingest_data(path)
        n_chunks = ingestion.write_ingest_data(path, data, collection, embedder)
        if verbose:
            print(f"[1/1] Done: {name} ({n_chunks} chunks)", flush=True)
    elif to_ingest:
        workers = min(config.INGEST_PARALLEL_WORKERS, len(to_ingest))
        if verbose:
            print(f"Extracting {total} files across {workers} worker processes ...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(ingestion.extract_ingest_data, path): (name, path)
                for name, path in to_ingest.items()
            }
            for future in as_completed(futures):
                name, path = futures[future]
                data = future.result()
                n_chunks = ingestion.write_ingest_data(path, data, collection, embedder)
                done += 1
                if verbose:
                    print(f"[{done}/{total}] Ingested: {name} ({n_chunks} chunks)", flush=True)

    for source in to_delete:
        if verbose:
            print(f"Removing (no longer on disk): {source}", flush=True)
        ingestion.remove_file(source, collection)
        deleted_files.append(source)

    retrieval.invalidate_sources_cache()

    if verbose:
        print("Sync complete.", flush=True)

    return {
        "new": new_files,
        "updated": updated_files,
        "unchanged": unchanged_files,
        "deleted": deleted_files,
    }


def _scheduled_sync_once() -> None:
    # APScheduler's BackgroundScheduler runs jobs on its own dedicated
    # thread, separate from both FastAPI's request threadpool and
    # retrieval.py's single Chroma-IO thread -- funnel through the same
    # one Chroma thread everything else uses, or a periodic scheduled sync
    # would hit the exact same cross-thread ChromaDB bug the /ask and
    # /sync endpoints were fixed for.
    retrieval.run_on_chroma_thread(sync_once)


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _scheduled_sync_once, "interval", minutes=config.SYNC_INTERVAL_MINUTES, id="doc_sync"
    )
    _scheduler.start()
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


if __name__ == "__main__":
    database.init_db()
    result = sync_once(verbose=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
