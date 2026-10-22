"""
File-system watcher — auto-ingests documents dropped into ``my_data/``.

On startup it processes every file already present, then watches for
new / modified / moved files and re-ingests them in real time.
"""

import logging
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.services.ingestion.embedding import run_file
from app.services.ingestion.chunking import FORMAT_HANDLER

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent / "my_data"


# ── Helpers ───────────────────────────────────────────────────────

def wait_until_ready(path: Path, attempts: int = 10) -> bool:
    """Wait until a copied file stops changing in size."""
    previous_size = -1
    stable_checks = 0

    for _ in range(attempts):
        try:
            current_size = path.stat().st_size
        except OSError:
            time.sleep(1)
            continue

        if current_size == previous_size and current_size > 0:
            stable_checks += 1
            if stable_checks >= 2:
                return True
        else:
            stable_checks = 0

        previous_size = current_size
        time.sleep(1)

    return False


def process_file(file_path):
    """Validate and ingest a single file."""
    path = Path(file_path)

    if path.suffix.lower() not in FORMAT_HANDLER:
        return

    if not path.is_file():
        return

    if not wait_until_ready(path):
        logger.warning("File not ready; skipping: %s", path.name)
        return

    try:
        run_file(path)
    except Exception as e:
        logger.exception("Ingestion failed for %s: %s", path.name, e)


# ── Watchdog handler ─────────────────────────────────────────────

class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            process_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            process_file(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            process_file(event.dest_path)


# ── Entry points ─────────────────────────────────────────────────

def start_watcher() -> Observer:
    """
    Ingest existing files, start the watchdog observer, and return it.

    Called from ``main.py``'s lifespan — runs in a daemon thread so it
    doesn't block the event loop.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Ingest documents already present at startup
    for path in DATA_DIR.rglob("*"):
        if path.is_file():
            process_file(path)

    observer = Observer()
    observer.schedule(DocumentHandler(), str(DATA_DIR), recursive=True)
    observer.start()

    logger.info("👁  Watching for documents in: %s", DATA_DIR)
    return observer


def main():
    """Standalone entry-point (``python -m app.watcher``)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for path in DATA_DIR.rglob("*"):
        if path.is_file():
            process_file(path)

    observer = Observer()
    observer.schedule(DocumentHandler(), str(DATA_DIR), recursive=True)
    observer.start()

    print(f"Watching for documents in: {DATA_DIR}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        observer.join()


if __name__ == "__main__":
    main()
