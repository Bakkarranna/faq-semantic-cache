#!/usr/bin/env python
"""Record the HTTP status the published report answers, at generation time.

The report says whether GitHub Pages is serving the site or answers 404, and
that claim is measured rather than assumed: this script requests the URL and
stores the status it answers with the measured numbers, so the page, the
README and ``results/summary.json`` quote one artifact.

    python scripts/check_pages.py

Run it when the report is regenerated, so the status stored beside the numbers
is the one the site answered at generation time.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES_URL = "https://bakkarrarna.github.io/faq-semantic-cache/"
TIMEOUT_SECONDS = 20.0


def probe_pages(url: str = PAGES_URL, timeout: float = TIMEOUT_SECONDS) -> dict:
    """One GET against the published report, with the status it answered.

    A 404 is an answer and is recorded as one; only a transport failure leaves
    the status unset, and that is recorded in ``error``.
    """

    entry = {
        "url": url,
        "http_status": None,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "error": None,
    }
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            entry["http_status"] = response.status
    except urllib.error.HTTPError as exc:
        entry["http_status"] = exc.code
    except OSError as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def refresh(path: Path, url: str = PAGES_URL) -> dict:
    """Store a fresh probe in the record, leaving every measured number as is.

    The file is rewritten in the shape the evaluation writer emits -- two-space
    indent, no trailing newline -- so a refresh changes this field and nothing
    else.
    """

    record = json.loads(path.read_text(encoding="utf-8"))
    record["pages_status"] = probe_pages(url)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    return record["pages_status"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", default=str(ROOT / "results" / "summary.json"))
    args = parser.parse_args()

    status = refresh(Path(args.record))
    url = status["url"]
    if status["error"] is not None:
        print(f"could not request {url}: {status['error']}")
        return 2
    print(
        f"{url} answered HTTP {status['http_status']} at {status['checked_at']}\n"
        f"recorded in {args.record}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
