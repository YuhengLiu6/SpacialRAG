import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from spatial_rag.graph_builder import (
    create_neo4j_driver,
    query_direction_neighbors,
    query_direction_objects,
    query_place_objects,
    query_places_for_object,
    query_same_node,
)


CARDINAL_DIRECTIONS = ("north", "east", "south", "west")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    details: Dict[str, Any]


def _run_scalar(driver, cypher: str, database: Optional[str] = None, **params) -> List[Dict[str, Any]]:
    with driver.session(database=database) as session:
        return [record.data() for record in session.run(cypher, **params)]


def _graph_counts(driver, database: Optional[str] = None) -> Dict[str, int]:
    counts = {}
    for key, cypher in (
        ("places", "MATCH (p:Place) RETURN count(p) AS value"),
        ("views", "MATCH (v:View) RETURN count(v) AS value"),
        ("objects", "MATCH (o:ObjectObservation) RETURN count(o) AS value"),
        ("object_classes", "MATCH (c:ObjectClass) RETURN count(c) AS value"),
    ):
        rows = _run_scalar(driver, cypher, database=database)
        counts[key] = int(rows[0]["value"]) if rows else 0
    return counts


def _top_object_labels(driver, limit: int = 10, database: Optional[str] = None) -> List[Dict[str, Any]]:
    return _run_scalar(
        driver,
        """
        MATCH (o:ObjectObservation)
        RETURN o.label AS label, count(o) AS seen_count
        ORDER BY seen_count DESC, label ASC
        LIMIT $limit
        """,
        database=database,
        limit=int(limit),
    )


def _all_place_ids(driver, database: Optional[str] = None) -> List[str]:
    rows = _run_scalar(
        driver,
        "MATCH (p:Place) RETURN p.place_id AS place_id ORDER BY p.place_id ASC",
        database=database,
    )
    return [str(row["place_id"]) for row in rows]


def _choose_anchor_place(driver, preferred_place_id: Optional[str], database: Optional[str] = None) -> Optional[str]:
    if preferred_place_id:
        rows = _run_scalar(
            driver,
            "MATCH (p:Place {place_id: $place_id}) RETURN p.place_id AS place_id",
            database=database,
            place_id=preferred_place_id,
        )
        if rows:
            return str(rows[0]["place_id"])
    place_ids = _all_place_ids(driver, database=database)
    for place_id in place_ids:
        same_node = query_same_node(driver, place_id=place_id, database=database)
        if same_node.get("views"):
            return place_id
    return place_ids[0] if place_ids else None


def _choose_object_label(driver, preferred_object_label: Optional[str], database: Optional[str] = None) -> Optional[str]:
    if preferred_object_label:
        rows = query_places_for_object(driver, preferred_object_label, database=database)
        if rows:
            return str(preferred_object_label)
    top_labels = _top_object_labels(driver, limit=10, database=database)
    return str(top_labels[0]["label"]) if top_labels else None


def _find_directional_object_case(
    driver,
    place_id: str,
    database: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    for direction in CARDINAL_DIRECTIONS:
        neighbors = query_direction_neighbors(driver, place_id=place_id, direction=direction, database=database)
        for neighbor in neighbors:
            objects = query_place_objects(driver, place_id=str(neighbor["place_id"]), database=database)
            if not objects:
                continue
            label = str(objects[0]["label"])
            return {
                "direction": direction,
                "neighbor_place_id": str(neighbor["place_id"]),
                "object_label": label,
            }
    return None


def _collect_file_names(value: Any) -> List[str]:
    found: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "file_name" and isinstance(item, str) and item.strip():
                found.append(item)
            else:
                found.extend(_collect_file_names(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_file_names(item))
    return found


def export_graph_query_artifacts(
    report: Dict[str, Any],
    db_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    views_root = root / "views"
    views_root.mkdir(parents=True, exist_ok=True)

    copied: List[Dict[str, str]] = []
    missing: List[str] = []
    seen = set()
    for file_name in _collect_file_names(report):
        if file_name in seen:
            continue
        seen.add(file_name)
        source_path = Path(file_name)
        if not source_path.is_absolute():
            source_path = Path(db_dir) / source_path
        if not source_path.exists():
            missing.append(str(file_name))
            continue
        try:
            rel_path = source_path.relative_to(Path(db_dir))
        except ValueError:
            rel_path = Path(source_path.name)
        destination = views_root / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied.append({"source": str(source_path), "copied_to": str(destination.relative_to(root))})

    report_with_export = dict(report)
    report_with_export["artifacts"] = {
        "root_dir": str(root),
        "report_json": str(root / "report.json"),
        "views_dir": str(views_root),
        "copied_images": copied,
        "missing_images": missing,
    }
    (root / "report.json").write_text(json.dumps(report_with_export, indent=2, ensure_ascii=True), encoding="utf-8")
    return report_with_export


def run_graph_query_test_pipeline(
    driver,
    preferred_place_id: Optional[str] = None,
    preferred_object_label: Optional[str] = None,
    database: Optional[str] = None,
) -> Dict[str, Any]:
    counts = _graph_counts(driver, database=database)
    checks: List[CheckResult] = []

    graph_non_empty = all(int(counts.get(key, 0)) > 0 for key in ("places", "views", "objects"))
    checks.append(
        CheckResult(
            name="graph_non_empty",
            status="passed" if graph_non_empty else "failed",
            details=counts,
        )
    )

    anchor_place_id = _choose_anchor_place(driver, preferred_place_id=preferred_place_id, database=database)
    if anchor_place_id:
        same_node = query_same_node(driver, place_id=anchor_place_id, database=database)
        views = list(same_node.get("views", []))
        place_object_rows = query_place_objects(driver, place_id=anchor_place_id, database=database)
        checks.append(
            CheckResult(
                name="same_node_views",
                status="passed" if views else "failed",
                details={
                    "place_id": anchor_place_id,
                    "num_views": len(views),
                    "orientations": [row.get("orientation_deg") for row in views],
                    "views": views[:8],
                },
            )
        )
        checks.append(
            CheckResult(
                name="same_node_objects",
                status="passed" if place_object_rows else "failed",
                details={
                    "place_id": anchor_place_id,
                    "num_objects": len(place_object_rows),
                    "sample_rows": place_object_rows[:8],
                },
            )
        )
    else:
        checks.append(CheckResult(name="same_node_views", status="failed", details={"reason": "no_place_found"}))
        checks.append(CheckResult(name="same_node_objects", status="failed", details={"reason": "no_place_found"}))

    object_label = _choose_object_label(driver, preferred_object_label=preferred_object_label, database=database)
    if object_label:
        object_hits = query_places_for_object(driver, object_label=object_label, database=database)
        checks.append(
            CheckResult(
                name="global_object_lookup",
                status="passed" if object_hits else "failed",
                details={
                    "object_label": object_label,
                    "num_hits": len(object_hits),
                    "sample_place_ids": [row.get("place_id") for row in object_hits[:5]],
                    "sample_hits": object_hits[:8],
                },
            )
        )
    else:
        checks.append(CheckResult(name="global_object_lookup", status="failed", details={"reason": "no_object_label_found"}))

    if anchor_place_id:
        for direction in ("north", "east"):
            neighbors = query_direction_neighbors(driver, place_id=anchor_place_id, direction=direction, database=database)
            checks.append(
                CheckResult(
                    name=f"{direction}_neighbors",
                    status="passed" if neighbors else "skipped",
                    details={
                        "place_id": anchor_place_id,
                        "direction": direction,
                        "num_neighbors": len(neighbors),
                        "sample_place_ids": [row.get("place_id") for row in neighbors[:5]],
                    },
                )
            )

    if anchor_place_id:
        directional_case = _find_directional_object_case(driver, place_id=anchor_place_id, database=database)
        if directional_case:
            direction_object_rows = query_direction_objects(
                driver,
                place_id=anchor_place_id,
                direction=str(directional_case["direction"]),
                object_label=str(directional_case["object_label"]),
                database=database,
            )
            checks.append(
                CheckResult(
                    name="directional_object_lookup",
                    status="passed" if direction_object_rows else "failed",
                    details={
                        "place_id": anchor_place_id,
                        "direction": directional_case["direction"],
                        "object_label": directional_case["object_label"],
                        "neighbor_place_id": directional_case["neighbor_place_id"],
                        "num_hits": len(direction_object_rows),
                        "sample_hits": direction_object_rows[:8],
                    },
                )
            )
        else:
            checks.append(
                CheckResult(
                    name="directional_object_lookup",
                    status="skipped",
                    details={"place_id": anchor_place_id, "reason": "no_directional_case_found"},
                )
            )

    status_counts: Dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}
    for check in checks:
        status_counts[check.status] = status_counts.get(check.status, 0) + 1

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "anchor_place_id": anchor_place_id,
        "object_label": object_label,
        "counts": counts,
        "summary": {
            "total_checks": len(checks),
            "passed": status_counts.get("passed", 0),
            "failed": status_counts.get("failed", 0),
            "skipped": status_counts.get("skipped", 0),
            "ok": status_counts.get("failed", 0) == 0,
        },
        "checks": [asdict(check) for check in checks],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reusable Neo4j graph query smoke-test pipeline.")
    parser.add_argument("--neo4j_uri", type=str, required=True, help="Neo4j bolt URI, e.g. bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, required=True, help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, required=True, help="Neo4j password")
    parser.add_argument("--neo4j_database", type=str, default=None, help="Optional Neo4j database name")
    parser.add_argument("--place_id", type=str, default=None, help="Optional preferred anchor place_id")
    parser.add_argument("--object_label", type=str, default=None, help="Optional preferred object label for global lookup")
    parser.add_argument("--db_dir", type=str, default=None, help="Optional spatial DB dir used to resolve image paths for artifact export")
    parser.add_argument("--report_out", type=str, default=None, help="Optional JSON report output path")
    parser.add_argument("--artifact_dir", type=str, default=None, help="Optional output directory containing report.json plus copied view images")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    driver = create_neo4j_driver(uri=args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        report = run_graph_query_test_pipeline(
            driver,
            preferred_place_id=args.place_id,
            preferred_object_label=args.object_label,
            database=args.neo4j_database,
        )
    finally:
        driver.close()

    if args.artifact_dir:
        if not args.db_dir:
            raise ValueError("--db_dir is required when --artifact_dir is set")
        report = export_graph_query_artifacts(report=report, db_dir=args.db_dir, output_dir=args.artifact_dir)
    elif args.report_out:
        report_path = Path(args.report_out)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
