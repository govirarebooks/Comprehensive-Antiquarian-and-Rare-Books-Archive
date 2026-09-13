#!/usr/bin/env python3
"""Build an external scholar knowledge cache from Govi entities.

The Govi JSON remains authoritative for catalogue facts. This script only adds
context from external public knowledge sources, with source URLs and retrieval
metadata, and never writes back into the source dataset.

Primary external sources:
- Wikidata entity search + entity claims
- Wikimedia/Wikipedia page summaries
- Optional DBpedia lookup as a secondary identity/context source

The cache is incremental: unchanged entity names are reused from the previous
external_knowledge.json file unless --refresh is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "govi-rare-books-academic-dataset.json"
OUTPUT_DIR = ROOT / "rag" / "external_knowledge"
OUTPUT_PATH = OUTPUT_DIR / "external_knowledge.json"

USER_AGENT = "GoviRareBooksScholar/1.0 (+https://www.govirarebooks.com/)"
TIMEOUT = 8
PROBE_TIMEOUT = 4
SLEEP_SECONDS = 0.15
RETRY_HOURS = 24

WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/w/api.php"
WIKI_API_BASES = {
    "en": "https://en.wikipedia.org/w/rest.php/v1",
    "it": "https://it.wikipedia.org/w/rest.php/v1",
    "fr": "https://fr.wikipedia.org/w/rest.php/v1",
    "de": "https://de.wikipedia.org/w/rest.php/v1",
}
DBPEDIA_LOOKUP = "https://lookup.dbpedia.org/api/search"

# Properties chosen because they help build scholar context without turning the
# external layer into a noisy mirror of every Wikidata claim.
CLAIM_PROPERTIES = {
    "P31": "instance_of",
    "P106": "occupation",
    "P19": "place_of_birth",
    "P20": "place_of_death",
    "P69": "educated_at",
    "P551": "residence",
    "P800": "notable_work",
    "P737": "influenced_by",
    "P144": "influenced",
    "P463": "member_of",
    "P1416": "affiliation",
    "P108": "employer",
    "P17": "country",
    "P135": "movement",
    "P2348": "time_period",
    "P361": "part_of",
}


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def http_json(url: str, params: dict[str, Any] | None = None, timeout: int | None = None) -> dict[str, Any] | None:
    final_url = url
    if params:
        final_url = f"{url}?{urlencode(params)}"
    request = Request(final_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout or TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"WARN external request failed: {final_url} :: {exc}", file=sys.stderr)
        return None




def network_probe() -> tuple[bool, str | None]:
    url = "https://www.wikidata.org/wiki/Special:EntityData/Q42.json"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=PROBE_TIMEOUT) as response:
            response.read(32)
        return True, None
    except Exception as exc:
        return False, f"network_probe_failed: {exc}"


def retry_due(prior: dict[str, Any]) -> bool:
    next_retry = prior.get("next_retry_at")
    if not next_retry:
        return True
    try:
        return datetime.now(timezone.utc) >= datetime.fromisoformat(next_retry)
    except ValueError:
        return True


def mark_pending(name: str, roles: list[str], prior: dict[str, Any] | None, error: str) -> dict[str, Any]:
    prior = prior or {}
    attempts = int(prior.get("attempts", 0)) + 1
    next_retry = datetime.now(timezone.utc) + timedelta(hours=RETRY_HOURS)
    errors = list(prior.get("errors", []))[-4:]
    errors.append({"at": utc_now(), "error": error})
    return {
        "entity": name,
        "roles": roles,
        "updated_at": utc_now(),
        "status": "pending",
        "attempts": attempts,
        "last_attempt_at": utc_now(),
        "next_retry_at": next_retry.isoformat(),
        "errors": errors,
        "sources": {
            "wikidata": None, "wikipedia": None, "dbpedia": None
        },
        "scholar_context": prior.get("scholar_context", {}),
    }


def collect_entities(records: list[dict[str, Any]], scopes: set[str]) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for record in records:
        if "authors" in scopes:
            for item in record.get("authors", []) or []:
                name = norm(item.get("name") if isinstance(item, dict) else item)
                if name:
                    entities.setdefault(name, {"name": name, "roles": set()})["roles"].add("author")
        if "publishers" in scopes:
            for item in record.get("publishers", []) or []:
                name = norm(item.get("name") if isinstance(item, dict) else item)
                if name:
                    entities.setdefault(name, {"name": name, "roles": set()})["roles"].add("publisher")
        if "related_names" in scopes:
            for item in record.get("related_names", []) or []:
                name = norm(item.get("name") if isinstance(item, dict) else item)
                if name:
                    entities.setdefault(name, {"name": name, "roles": set()})["roles"].add("related_name")
    for value in entities.values():
        value["roles"] = sorted(value["roles"])
    return entities


def wikidata_search(name: str, language: str) -> dict[str, Any] | None:
    data = http_json(
        WIKIDATA_SEARCH,
        {
            "action": "wbsearchentities",
            "format": "json",
            "search": name,
            "language": language,
            "uselang": language,
            "type": "item",
            "limit": 5,
        },
    )
    if not data:
        return None
    results = data.get("search") or []
    if not results:
        return None
    # Prefer exact label, then the strongest scored result returned by Wikidata.
    lname = name.casefold()
    ranked = sorted(
        results,
        key=lambda x: (
            1 if norm(x.get("label")).casefold() == lname else 0,
            float(x.get("match", {}).get("score", 0) or 0),
        ),
        reverse=True,
    )
    best = ranked[0]
    return {
        "id": best.get("id"),
        "label": best.get("label"),
        "description": best.get("description"),
        "match_score": best.get("match", {}).get("score"),
        "language": language,
    }


def entity_ids_for_claims(entity: dict[str, Any], property_id: str) -> list[str]:
    values = []
    for claim in entity.get("claims", {}).get(property_id, []) or []:
        snak = claim.get("mainsnak", {})
        datavalue = snak.get("datavalue", {})
        if datavalue.get("type") == "wikibase-entityid":
            qid = datavalue.get("value", {}).get("id")
            if qid:
                values.append(qid)
    return list(dict.fromkeys(values))


def fetch_wikidata_entity(qid: str, language: str) -> dict[str, Any] | None:
    data = http_json(
        WIKIDATA_ENTITY,
        {
            "action": "wbgetentities",
            "format": "json",
            "ids": qid,
            "props": "labels|descriptions|aliases|claims|sitelinks",
            "languages": f"{language}|en|it|fr|de",
        },
    )
    if not data:
        return None
    return (data.get("entities") or {}).get(qid)


def labels_for_qids(qids: list[str], language: str) -> dict[str, str]:
    if not qids:
        return {}
    data = http_json(
        WIKIDATA_ENTITY,
        {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(qids[:50]),
            "props": "labels",
            "languages": f"{language}|en|it|fr|de",
        },
    )
    if not data:
        return {}
    out = {}
    for qid, item in (data.get("entities") or {}).items():
        labels = item.get("labels") or {}
        label = None
        for lang in (language, "en", "it", "fr", "de"):
            if lang in labels:
                label = labels[lang].get("value")
                if label:
                    break
        if label:
            out[qid] = label
    return out


def wikipedia_search(name: str, language: str) -> dict[str, Any] | None:
    base = WIKI_API_BASES[language]
    data = http_json(f"{base}/search/page", {"q": name, "limit": 5})
    if not data:
        return None
    pages = data.get("pages") or []
    if not pages:
        return None
    lname = name.casefold()
    ranked = sorted(
        pages,
        key=lambda p: (
            1 if norm(p.get("title")).casefold() == lname else 0,
            len(norm(p.get("description"))),
        ),
        reverse=True,
    )
    best = ranked[0]
    title = best.get("title")
    if not title:
        return None
    summary = http_json(f"{base}/page/summary/{quote(title, safe='')}")
    if not summary:
        return None
    return {
        "language": language,
        "title": summary.get("title") or title,
        "description": summary.get("description"),
        "extract": summary.get("extract"),
        "url": ((summary.get("content_urls") or {}).get("desktop") or {}).get("page"),
    }


def dbpedia_lookup(name: str) -> dict[str, Any] | None:
    data = http_json(
        DBPEDIA_LOOKUP,
        {"format": "JSON", "query": name, "maxResults": 3},
    )
    if not data:
        return None
    docs = (data.get("docs") or [])
    if not docs:
        return None
    lname = name.casefold()
    ranked = sorted(
        docs,
        key=lambda d: (
            1 if any(norm(x).casefold() == lname for x in (d.get("label") or [])) else 0,
            float((d.get("refCount") or [0])[0] if d.get("refCount") else 0),
        ),
        reverse=True,
    )
    best = ranked[0]
    return {
        "uri": (best.get("resource") or [None])[0] if best.get("resource") else None,
        "label": (best.get("label") or [None])[0] if best.get("label") else None,
        "description": (best.get("description") or [None])[0] if best.get("description") else None,
    }


def build_entity(name: str, roles: list[str], refresh: bool) -> dict[str, Any]:
    print(f"[{name}]")
    wiki = None
    wikidata = None
    wikidata_search_result = None
    for lang in ("en", "it", "fr", "de"):
        wikidata_search_result = wikidata_search(name, lang)
        if wikidata_search_result:
            break
        time.sleep(SLEEP_SECONDS)
    if wikidata_search_result:
        qid = wikidata_search_result.get("id")
        time.sleep(SLEEP_SECONDS)
        wikidata = fetch_wikidata_entity(qid, wikidata_search_result.get("language", "en")) if qid else None

    claims = {}
    labels = {}
    sitelinks = {}
    if wikidata:
        qid = wikidata.get("id")
        sitelinks = wikidata.get("sitelinks") or {}
        qids = []
        for prop in CLAIM_PROPERTIES:
            qids.extend(entity_ids_for_claims(wikidata, prop))
        labels = labels_for_qids(list(dict.fromkeys(qids)), "en")
        for prop, key in CLAIM_PROPERTIES.items():
            ids = entity_ids_for_claims(wikidata, prop)
            claims[key] = [
                {"id": x, "label": labels.get(x, x)} for x in ids[:15]
            ]
        wikidata_payload = {
            "id": qid,
            "label": wikidata_search_result.get("label"),
            "description": wikidata_search_result.get("description"),
            "match_score": wikidata_search_result.get("match_score"),
            "claims": claims,
            "sitelinks": {
                lang: link.get("title")
                for lang, link in sitelinks.items()
                if lang in {"enwiki", "itwiki", "frwiki", "dewiki"}
            },
            "url": f"https://www.wikidata.org/wiki/{qid}" if qid else None,
        }
    else:
        wikidata_payload = None

    for lang in ("en", "it", "fr", "de"):
        time.sleep(SLEEP_SECONDS)
        wiki = wikipedia_search(name, lang)
        if wiki:
            break

    time.sleep(SLEEP_SECONDS)
    dbpedia = dbpedia_lookup(name)

    return {
        "entity": name,
        "roles": roles,
        "updated_at": utc_now(),
        "status": "resolved" if (wikidata_payload or wiki or dbpedia) else "partial",
        "attempts": 1,
        "last_attempt_at": utc_now(),
        "next_retry_at": None,
        "errors": [],
        "sources": {
            "wikidata": wikidata_payload,
            "wikipedia": wiki,
            "dbpedia": dbpedia,
        },
        "scholar_context": {
            "periods": [c["label"] for c in claims.get("time_period", [])],
            "movements": [c["label"] for c in claims.get("movement", [])],
            "occupations": [c["label"] for c in claims.get("occupation", [])],
            "influenced_by": [c["label"] for c in claims.get("influenced_by", [])],
            "influenced": [c["label"] for c in claims.get("influenced", [])],
            "memberships": [c["label"] for c in claims.get("member_of", [])],
            "affiliations": [c["label"] for c in claims.get("affiliation", [])],
            "educated_at": [c["label"] for c in claims.get("educated_at", [])],
            "notable_works": [c["label"] for c in claims.get("notable_work", [])],
            "residences": [c["label"] for c in claims.get("residence", [])],
            "places_of_birth": [c["label"] for c in claims.get("place_of_birth", [])],
            "places_of_death": [c["label"] for c in claims.get("place_of_death", [])],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="authors,related_names", help="authors,publishers,related_names")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names", default="", help="Only these catalogue entity names, separated by ||")
    parser.add_argument("--record", default="", help="Only entities attached to this Govi record ID")
    parser.add_argument("--offline", action="store_true", help="Never use the network; mark selected entities pending")
    parser.add_argument("--retry-pending", action="store_true", help="Retry pending entries even before next_retry_at")
    args = parser.parse_args()

    scopes = {x.strip() for x in args.scope.split(",") if x.strip()}
    records = load_json(DATASET_PATH, [])
    existing = load_json(OUTPUT_PATH, {"version": 1, "sources": {}, "entities": {}})
    entities = collect_entities(records, scopes)
    names = sorted(entities)
    if args.record:
        selected_record = next((r for r in records if str(r.get("id")) == str(args.record)), None)
        if selected_record:
            selected = set()
            for field in ("authors", "publishers", "related_names"):
                if field not in scopes:
                    continue
                for item in selected_record.get(field, []) or []:
                    selected.add(norm(item.get("name") if isinstance(item, dict) else item))
            names = [n for n in names if norm(n) in selected]
    if args.names:
        wanted = {norm(x) for x in args.names.split("||") if norm(x)}
        names = [n for n in names if norm(n) in wanted]
    if args.limit:
        names = names[: args.limit]

    output = {
        "version": 1,
        "generated_at": existing.get("generated_at") or utc_now(),
        "last_run_at": utc_now(),
        "source_policy": {
            "catalogue_truth": "govi-rare-books-academic-dataset.json",
            "external_context": ["Wikidata", "Wikipedia", "DBpedia"],
            "rule": "External knowledge enriches context and never overwrites catalogue facts.",
        },
        "entities": dict(existing.get("entities") or {}),
    }

    changed = 0
    online = False
    probe_error = None
    if not args.offline and names:
        online, probe_error = network_probe()
        if not online:
            print(f"WARN: external network unavailable; queueing {len(names)} entity/ies without blocking.", file=sys.stderr)
            print(f"WARN: {probe_error}", file=sys.stderr)

    for name in names:
        prior = output["entities"].get(name)
        roles = entities[name]["roles"]
        if prior and not args.refresh and sorted(prior.get("roles", [])) == roles:
            if prior.get("status") == "pending" and (args.retry_pending or retry_due(prior)):
                pass
            else:
                continue
        if args.offline or not online:
            output["entities"][name] = mark_pending(name, roles, prior, probe_error or "offline_mode")
        else:
            fresh = build_entity(name, roles, args.refresh)
            if prior and prior.get("attempts"):
                fresh["attempts"] = int(prior.get("attempts", 0)) + 1
            output["entities"][name] = fresh
        changed += 1
        save_json(OUTPUT_PATH, output)

    # Keep counts explicit for auditability.
    statuses = [x.get("status", "unknown") for x in output["entities"].values()]
    output["counts"] = {
        "catalogue_records": len(records),
        "entities_cached": len(output["entities"]),
        "entities_changed_this_run": changed,
        "resolved": statuses.count("resolved"),
        "partial": statuses.count("partial"),
        "pending": statuses.count("pending"),
    }
    save_json(OUTPUT_PATH, output)

    print("=" * 80)
    print("GOVI EXTERNAL SCHOLAR KNOWLEDGE")
    print("=" * 80)
    print(f"Catalogue records: {len(records)}")
    print(f"Entities cached: {len(output['entities'])}")
    print(f"Changed this run: {changed}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
