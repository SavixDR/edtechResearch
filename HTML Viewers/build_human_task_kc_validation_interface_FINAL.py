#!/usr/bin/env python3
"""
Build the final standalone Human Task + KC Validation HTML.

What this script does
---------------------
1. Loads the 24 baseline student-LLM conversations.
2. Loads one or more model result JSONL files.
3. Takes the UNION of model-selected assignment Tasks and KCs per conversation.
4. Loads the six KC CSV tables.
5. Loads the six official KC-model JSON files and uses their exact Task descriptions,
   required outputs, and assessed behaviours.
6. Marks model-union KCs as GREEN.
7. Detects nearby/related KCs and places them in YELLOW cards.
8. Keeps all other assignment KCs in the searchable scroll list.
9. Produces a standalone evaluator HTML:
      - Present / Not present only
      - optional KC presence category
      - optional 1-10 presence-strength rating
      - optional missed Task/KC additions
      - flexible case review: reviewers do NOT have to decide every item
      - local browser autosave
      - CSV and JSON export
10. Writes a researcher-only model-union map. Do NOT send that file to evaluators.

No external Python packages are required.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------
# Default input/output filenames
# ---------------------------------------------------------------------

DEFAULT_CONVERSATIONS = "balanced_conversations_24.jsonl"

DEFAULT_RESULTS = [
    "results_claude.jsonl",
    "results.jsonl",
    "results2.jsonl",
    "results3.jsonl",
]

DEFAULT_KC_FILES = [
    "f24_a2_kc_table.csv",
    "f24_a3_corrected_kc_table.csv",
    "f24_a4_corrected_kc_table.csv",
    "f24_a5_corrected_kc_table.csv",
    "f24_a6_corrected_kc_table.csv",
    "f24_a7_corrected_kc_table.csv",
]

DEFAULT_TASK_MODEL_FILES = [
    "f24_a2_kc_model.json",
    "f24_a3_corrected_kc_model.json",
    "f24_a4_corrected_kc_model.json",
    "f24_a5_corrected_kc_model.json",
    "f24_a6_corrected_complete_kc_model.json",
    "f24_a7_corrected_kc_model.json",
]

DEFAULT_OUTPUT = "human_task_kc_validation_interface_FLEXIBLE_REVIEW.html"
DEFAULT_INTERNAL_MAP = "human_task_kc_union_model_map.json"
DEFAULT_COMBINED_KC = "combined_kc_catalogue.csv"


# ---------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------

def read_jsonl(path):
    rows = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_no}: {e}"
                ) from e
    return rows


def assignment_key(value):
    """
    f24_a2, s25_a2, [f24_a2] -> a2
    """
    text = str(value or "").strip()
    m = re.search(r"a([2-7])", text, re.I)
    return f"a{int(m.group(1))}" if m else text.lower()


def assignment_parts(assignment_id):
    """
    f24_a2 -> ('f24', 'A2')
    """
    text = str(assignment_id or "")
    m = re.match(r"(.+?)_a(\d+)$", text, re.I)
    if m:
        return m.group(1), f"A{int(m.group(2))}"
    return "", text


def kc_key(value):
    """
    Normalises both:
      f24_a4_KC_001
      [f24_a4]_KC_001
    to:
      a4_KC_001
    """
    text = str(value or "").strip()
    m = re.search(r"a([2-7])\]?_KC_(\d+)$", text, re.I)
    if m:
        return f"a{int(m.group(1))}_KC_{int(m.group(2)):03d}"
    return text.lower()


def task_key(value):
    """
    Normalises both:
      f24_a4_TASK_001
      [f24_a4]_TASK_001
    to:
      a4_TASK_001
    """
    text = str(value or "").strip()
    m = re.search(r"a([2-7])\]?_TASK_(\d+)$", text, re.I)
    if m:
        return f"a{int(m.group(1))}_TASK_{int(m.group(2)):03d}"
    return text.lower()


def clean_bracketed_id(value):
    """
    [f24_a4]_TASK_001 -> f24_a4_TASK_001
    [f24_a4]_KC_001   -> f24_a4_KC_001
    """
    return re.sub(r"^\[([^\]]+)\]", r"\1", str(value or "").strip())


def split_ids(text):
    if not text:
        return []
    return [x.strip() for x in str(text).split(";") if x.strip()]


def natural_id_key(value):
    text = str(value or "")
    m = re.search(r"_(KC|TASK)_(\d+)$", text.replace("]", ""), re.I)
    return (int(m.group(2)) if m else 10**9, text)


# ---------------------------------------------------------------------
# Load KC CSV catalogue + Task-KC relationships
# ---------------------------------------------------------------------

def load_kc_catalogue(kc_files, combined_output=None):
    catalogue = {}
    by_assignment = defaultdict(list)

    task_to_kcs = defaultdict(list)
    csv_task_ids = {}
    combined_rows = []

    for path_str in kc_files:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"KC table not found: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames or "KC_ID" not in reader.fieldnames:
                raise ValueError(f"{path}: expected a KC_ID column")

            for row in reader:
                raw_kc = str(row.get("KC_ID") or "").strip()
                if not raw_kc:
                    continue

                kkey = kc_key(raw_kc)
                akey = assignment_key(raw_kc)

                raw_related_tasks = split_ids(row.get("Related_Task_IDs"))
                related_task_keys = [task_key(x) for x in raw_related_tasks]

                parent_keys = [
                    kc_key(x)
                    for x in split_ids(row.get("Parent_KC_ID"))
                ]
                prereq_keys = [
                    kc_key(x)
                    for x in split_ids(row.get("Prerequisite_KC_IDs"))
                ]
                alternative_keys = [
                    kc_key(x)
                    for x in split_ids(row.get("Alternative_KC_IDs"))
                ]

                info = {
                    "kc_id": clean_bracketed_id(raw_kc),
                    "match_key": kkey,
                    "assignment_key": akey,
                    "name": str(row.get("Canonical_Name") or "").strip(),
                    "description": str(row.get("Definition") or "").strip(),
                    "kc_type": str(row.get("KC_Type") or "").strip(),
                    "granularity": str(row.get("Granularity") or "").strip(),
                    "requirement_level": str(
                        row.get("Requirement_Level") or ""
                    ).strip(),
                    "successful_performance": str(
                        row.get("Successful_Performance") or ""
                    ).strip(),
                    "difficulty_signals": str(
                        row.get("Difficulty_Signals") or ""
                    ).strip(),
                    "related_task_ids": [
                        clean_bracketed_id(x) for x in raw_related_tasks
                    ],
                    "related_task_keys": related_task_keys,
                    "parent_keys": parent_keys,
                    "prerequisite_keys": prereq_keys,
                    "alternative_keys": alternative_keys,
                    "source_file": path.name,
                }

                if kkey not in catalogue:
                    catalogue[kkey] = info
                    by_assignment[akey].append(info)

                for raw_task, tkey in zip(
                    raw_related_tasks, related_task_keys
                ):
                    csv_task_ids.setdefault(
                        tkey, clean_bracketed_id(raw_task)
                    )
                    if kkey not in task_to_kcs[tkey]:
                        task_to_kcs[tkey].append(kkey)

                combined = dict(row)
                combined["_Assignment_Key"] = akey
                combined["_Normalized_KC_Key"] = kkey
                combined["_Source_Table"] = path.name
                combined_rows.append(combined)

    for akey in by_assignment:
        by_assignment[akey].sort(
            key=lambda x: natural_id_key(x["kc_id"])
        )

    for tkey in task_to_kcs:
        task_to_kcs[tkey].sort(
            key=lambda k: natural_id_key(
                catalogue.get(k, {}).get("kc_id", k)
            )
        )

    # Explicit structural KC graph:
    # parent, prerequisite, and alternative links are treated as undirected
    # one-hop neighborhood relations for yellow suggestions.
    kc_graph = defaultdict(set)

    for kkey, info in catalogue.items():
        structural = (
            info["parent_keys"]
            + info["prerequisite_keys"]
            + info["alternative_keys"]
        )
        for other in structural:
            if other and other in catalogue:
                kc_graph[kkey].add(other)
                kc_graph[other].add(kkey)

    if combined_output:
        fields = []
        seen = set()

        for row in combined_rows:
            for field in row:
                if field not in seen:
                    fields.append(field)
                    seen.add(field)

        with Path(combined_output).open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(combined_rows)

    return {
        "catalogue": catalogue,
        "by_assignment": by_assignment,
        "task_to_kcs": task_to_kcs,
        "csv_task_ids": csv_task_ids,
        "kc_graph": kc_graph,
        "combined_rows": combined_rows,
    }


# ---------------------------------------------------------------------
# Load exact official Task descriptions from KC-model JSON files
# ---------------------------------------------------------------------

def official_assignment_id(assignment_obj):
    if assignment_obj.get("assignment_id"):
        return str(assignment_obj["assignment_id"])

    # Some files store f24_aN as a key inside assignment.
    for key in assignment_obj:
        if re.fullmatch(r"f24_a[2-7]", str(key), re.I):
            return str(key)

    raise ValueError("Could not determine assignment ID in task JSON.")


def load_official_tasks(task_model_files):
    tasks = {}
    tasks_by_assignment = defaultdict(list)
    assignment_meta = {}

    for path_str in task_model_files:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(
                f"Official task/KC model JSON not found: {path}"
            )

        data = json.loads(path.read_text(encoding="utf-8"))
        assignment = data.get("assignment") or {}

        aid = official_assignment_id(assignment)
        akey = assignment_key(aid)

        assignment_meta[akey] = {
            "assignment_id": aid,
            "assignment_title": str(
                assignment.get("assignment_title") or ""
            ),
            "objective": str(assignment.get("objective") or ""),
        }

        for raw_task in assignment.get("tasks", []) or []:
            raw_id = raw_task.get("task_id")
            if not raw_id:
                continue

            tkey = task_key(raw_id)

            info = {
                "task_id": clean_bracketed_id(raw_id),
                "official_task_id": raw_id,
                "match_key": tkey,
                "assignment_key": akey,
                "description": str(
                    raw_task.get("description") or ""
                ).strip(),
                "required_output": str(
                    raw_task.get("required_output") or ""
                ).strip(),
                "assessed_behaviour": str(
                    raw_task.get("assessed_behaviour") or ""
                ).strip(),
                "support_basis": str(
                    raw_task.get("support_basis") or ""
                ).strip(),
            }

            if tkey in tasks:
                raise ValueError(
                    f"Duplicate official Task after normalisation: {tkey}"
                )

            tasks[tkey] = info
            tasks_by_assignment[akey].append(info)

    for akey in tasks_by_assignment:
        tasks_by_assignment[akey].sort(
            key=lambda x: natural_id_key(x["task_id"])
        )

    return {
        "tasks": tasks,
        "tasks_by_assignment": tasks_by_assignment,
        "assignment_meta": assignment_meta,
    }


# ---------------------------------------------------------------------
# Build model unions + reviewer dataset
# ---------------------------------------------------------------------

def mapped_kcs_for_task(
    tkey,
    task_to_kcs,
    catalogue,
    candidate_kc_keys,
):
    mapped = []

    for kkey in task_to_kcs.get(tkey, []):
        kinfo = catalogue.get(kkey, {})

        mapped.append({
            "kc_id": kinfo.get("kc_id", kkey),
            "name": kinfo.get("name", ""),
            "candidate": kkey in candidate_kc_keys,
        })

    return mapped


def build_dataset(
    conversation_file,
    result_files,
    kc_files,
    task_model_files,
    combined_kc_output,
):
    kc_data = load_kc_catalogue(
        kc_files,
        combined_output=combined_kc_output,
    )
    catalogue = kc_data["catalogue"]
    by_assignment = kc_data["by_assignment"]
    task_to_kcs = kc_data["task_to_kcs"]
    csv_task_ids = kc_data["csv_task_ids"]
    kc_graph = kc_data["kc_graph"]

    official = load_official_tasks(task_model_files)
    official_tasks = official["tasks"]
    official_tasks_by_assignment = official["tasks_by_assignment"]
    assignment_meta = official["assignment_meta"]

    conversations = read_jsonl(conversation_file)

    case_order = []
    conv_by_case = {}
    conv_by_id = {}

    for idx, rec0 in enumerate(conversations, start=1):
        rec = dict(rec0)

        case_id = (
            rec.get("case_id")
            or rec.get("conversation_id")
            or f"case-{idx}"
        )

        rec["_sample_id"] = f"H{idx:02d}"
        rec["_case_id"] = case_id

        case_order.append(case_id)
        conv_by_case[case_id] = rec

        if rec.get("conversation_id"):
            conv_by_id[rec["conversation_id"]] = rec

    kc_union = defaultdict(
        lambda: defaultdict(
            lambda: {
                "models": set(),
                "statuses": set(),
                "evidence": [],
                "notes": [],
                "raw_ids": set(),
                "linked_task_keys": set(),
            }
        )
    )

    task_union = defaultdict(
        lambda: defaultdict(
            lambda: {
                "models": set(),
                "evidence": [],
                "raw_ids": set(),
            }
        )
    )

    model_coverage = defaultdict(set)
    model_names = set()

    for result_file in result_files:
        for rec in read_jsonl(result_file):
            result_case = rec.get("case", {}) or {}

            target = conv_by_case.get(
                result_case.get("case_id")
            )

            if not target:
                cid = (
                    result_case.get("conversation_id")
                    or result_case.get("base_conversation_id")
                )
                target = conv_by_id.get(cid)

            if not target:
                continue

            target_case_id = target["_case_id"]

            judge = rec.get("judge", {}) or {}
            model = (
                judge.get("display_name")
                or judge.get("model_key")
                or Path(result_file).stem
            )

            model_names.add(model)
            model_coverage[target_case_id].add(model)

            parsed = (
                (rec.get("response", {}) or {}).get("parsed_json")
                or {}
            )

            if not isinstance(parsed, dict):
                continue

            # Model-selected Tasks
            for item in parsed.get("tasks", []) or []:
                raw_id = item.get("task_id")
                if not raw_id:
                    continue

                tkey = task_key(raw_id)
                bucket = task_union[target_case_id][tkey]

                bucket["models"].add(model)
                bucket["raw_ids"].add(raw_id)

                for ev in item.get("evidence", []) or []:
                    if ev and ev not in bucket["evidence"]:
                        bucket["evidence"].append(ev)

            # Model-selected KCs
            for item in parsed.get("labels", []) or []:
                raw_id = item.get("kc_id")
                if not raw_id:
                    continue

                kkey = kc_key(raw_id)
                bucket = kc_union[target_case_id][kkey]

                bucket["models"].add(model)
                bucket["raw_ids"].add(raw_id)

                if item.get("status"):
                    bucket["statuses"].add(item["status"])

                for ev in item.get("evidence", []) or []:
                    if ev and ev not in bucket["evidence"]:
                        bucket["evidence"].append(ev)

                note = item.get("note")
                if note and note not in bucket["notes"]:
                    bucket["notes"].append(note)

                for raw_task in item.get(
                    "linked_task_ids", []
                ) or []:
                    bucket["linked_task_keys"].add(
                        task_key(raw_task)
                    )

    cases = []
    internal_map = []

    for case_id in case_order:
        rec = conv_by_case[case_id]

        assignment_id = rec.get("assignment_id", "")
        akey = assignment_key(assignment_id)
        semester, assignment = assignment_parts(
            assignment_id
        )

        turns = rec.get("turns", []) or []

        candidate_task_keys = set(
            task_union.get(case_id, {}).keys()
        )
        candidate_kc_keys = set(
            kc_union.get(case_id, {}).keys()
        )

        # Model-union Tasks + Tasks linked to model-union KCs
        relevant_task_keys = set(candidate_task_keys)

        for kkey in candidate_kc_keys:
            if kkey in catalogue:
                relevant_task_keys.update(
                    catalogue[kkey]["related_task_keys"]
                )

            relevant_task_keys.update(
                kc_union[case_id][kkey][
                    "linked_task_keys"
                ]
            )

        # Direct structural one-hop KC neighbors
        graph_neighbors = set()

        for kkey in candidate_kc_keys:
            graph_neighbors.update(
                kc_graph.get(kkey, set())
            )

        # -------------------------------------------------------------
        # GREEN Task cards = union of model-selected Tasks
        # -------------------------------------------------------------
        candidate_tasks = []

        for tkey in sorted(
            candidate_task_keys,
            key=lambda x: natural_id_key(
                official_tasks.get(
                    x, {}
                ).get(
                    "task_id",
                    csv_task_ids.get(x, x),
                )
            ),
        ):
            bucket = task_union[case_id][tkey]
            official_info = official_tasks.get(
                tkey, {}
            )

            raw_ids = sorted(bucket["raw_ids"])

            display_id = (
                official_info.get("task_id")
                or csv_task_ids.get(tkey)
                or (
                    clean_bracketed_id(raw_ids[0])
                    if raw_ids else tkey
                )
            )

            candidate_tasks.append({
                "task_id": display_id,
                "official_task_id": official_info.get(
                    "official_task_id", display_id
                ),
                "match_key": tkey,
                "description": official_info.get(
                    "description", ""
                ),
                "required_output": official_info.get(
                    "required_output", ""
                ),
                "assessed_behaviour": official_info.get(
                    "assessed_behaviour", ""
                ),
                "support_basis": official_info.get(
                    "support_basis", ""
                ),
                "mapped_kcs": mapped_kcs_for_task(
                    tkey,
                    task_to_kcs,
                    catalogue,
                    candidate_kc_keys,
                ),
            })

        # -------------------------------------------------------------
        # GREEN KCs = union of model-selected KCs
        # -------------------------------------------------------------
        candidate_kcs = []

        for kkey in sorted(
            candidate_kc_keys,
            key=lambda x: natural_id_key(
                catalogue.get(
                    x, {}
                ).get("kc_id", x)
            ),
        ):
            info = catalogue.get(kkey, {})
            raw_ids = sorted(
                kc_union[case_id][kkey]["raw_ids"]
            )

            display_id = (
                info.get("kc_id")
                or (
                    clean_bracketed_id(raw_ids[0])
                    if raw_ids else kkey
                )
            )

            candidate_kcs.append({
                "kc_id": display_id,
                "match_key": kkey,
                "name": info.get("name", ""),
                "description": info.get(
                    "description", ""
                ),
                "kc_type": info.get("kc_type", ""),
                "granularity": info.get(
                    "granularity", ""
                ),
                "requirement_level": info.get(
                    "requirement_level", ""
                ),
                "successful_performance": info.get(
                    "successful_performance", ""
                ),
                "difficulty_signals": info.get(
                    "difficulty_signals", ""
                ),
                "related_task_ids": [
                    official_tasks.get(
                        t, {}
                    ).get(
                        "task_id",
                        csv_task_ids.get(t, t),
                    )
                    for t in info.get(
                        "related_task_keys", []
                    )
                ],
            })

        # -------------------------------------------------------------
        # YELLOW KC cards = nearby / related KCs
        # All remaining KCs stay in the scrollable "Other KCs" list.
        # -------------------------------------------------------------
        assignment_kcs = []

        for info in by_assignment.get(akey, []):
            kkey = info["match_key"]

            if kkey in candidate_kc_keys:
                continue

            reasons = []

            shared_tasks = (
                set(info["related_task_keys"])
                & relevant_task_keys
            )

            if shared_tasks:
                task_labels = [
                    official_tasks.get(
                        t, {}
                    ).get(
                        "task_id",
                        csv_task_ids.get(t, t),
                    )
                    for t in sorted(shared_tasks)
                ]

                reasons.append(
                    "same/relevant task: "
                    + ", ".join(task_labels)
                )

            if kkey in graph_neighbors:
                reasons.append(
                    "direct KC graph neighbor"
                )

            assignment_kcs.append({
                "kc_id": info["kc_id"],
                "match_key": kkey,
                "name": info["name"],
                "description": info[
                    "description"
                ],
                "related_task_ids": [
                    official_tasks.get(
                        t, {}
                    ).get(
                        "task_id",
                        csv_task_ids.get(t, t),
                    )
                    for t in info[
                        "related_task_keys"
                    ]
                ],
                "successful_performance": info.get(
                    "successful_performance", ""
                ),
                "difficulty_signals": info.get(
                    "difficulty_signals", ""
                ),
                "near_neighbor": bool(reasons),
                "neighbor_reason": "; ".join(
                    reasons
                ),
            })

        assignment_kcs.sort(
            key=lambda x: (
                not x["near_neighbor"],
                natural_id_key(x["kc_id"]),
            )
        )

        # -------------------------------------------------------------
        # COMPLETE official Task list for "Other assignment tasks"
        # -------------------------------------------------------------
        assignment_tasks = []

        for tinfo in official_tasks_by_assignment.get(
            akey, []
        ):
            tkey = tinfo["match_key"]

            mapped = mapped_kcs_for_task(
                tkey,
                task_to_kcs,
                catalogue,
                candidate_kc_keys,
            )

            contains_candidate_kc = any(
                k["candidate"] for k in mapped
            )

            assignment_tasks.append({
                "task_id": tinfo["task_id"],
                "official_task_id": tinfo[
                    "official_task_id"
                ],
                "match_key": tkey,
                "description": tinfo[
                    "description"
                ],
                "required_output": tinfo[
                    "required_output"
                ],
                "assessed_behaviour": tinfo[
                    "assessed_behaviour"
                ],
                "support_basis": tinfo[
                    "support_basis"
                ],
                "mapped_kcs": mapped,
                "near_neighbor": (
                    contains_candidate_kc
                ),
                "candidate": (
                    tkey in candidate_task_keys
                ),
            })

        assignment_tasks.sort(
            key=lambda x: natural_id_key(
                x["task_id"]
            )
        )

        meta = assignment_meta.get(akey, {})

        cases.append({
            "sample_id": rec["_sample_id"],
            "case_id": case_id,
            "base_conversation_id": rec.get(
                "base_conversation_id", ""
            ),
            "conversation_id": rec.get(
                "conversation_id", ""
            ),
            "chunk_id": rec.get("chunk_id", ""),
            "assignment_id": assignment_id,
            "assignment": assignment,
            "semester": semester,
            "student_id": rec.get(
                "student_id", ""
            ),
            "test_id": rec.get("test_id", ""),
            "variant_id": rec.get(
                "variant_id", ""
            ),
            "variant_type": rec.get(
                "variant_type", ""
            ),
            "turns": turns,
            "num_interactions": sum(
                1
                for t in turns
                if str(
                    t.get("speaker", "")
                ).lower() == "student"
            ),
            "candidate_tasks": candidate_tasks,
            "candidate_kcs": candidate_kcs,
            "assignment_tasks": assignment_tasks,
            "assignment_kcs": assignment_kcs,
            "model_count": len(
                model_coverage.get(
                    case_id, set()
                )
            ),
            "assignment_title": meta.get(
                "assignment_title", assignment
            ),
            "assignment_objective": meta.get(
                "objective", ""
            ),
        })

        # Researcher-only provenance map.
        for tkey, bucket in task_union.get(
            case_id, {}
        ).items():
            tinfo = official_tasks.get(
                tkey, {}
            )
            raw_ids = sorted(
                bucket["raw_ids"]
            )

            internal_map.append({
                "sample_id": rec["_sample_id"],
                "case_id": case_id,
                "conversation_id": rec.get(
                    "conversation_id", ""
                ),
                "assignment_id": assignment_id,
                "item_type": "task",
                "item_id": (
                    tinfo.get("task_id")
                    or csv_task_ids.get(tkey)
                    or (
                        clean_bracketed_id(
                            raw_ids[0]
                        )
                        if raw_ids else tkey
                    )
                ),
                "normalized_key": tkey,
                "proposed_by_models": sorted(
                    bucket["models"]
                ),
                "model_evidence_union": bucket[
                    "evidence"
                ],
                "raw_model_ids": raw_ids,
            })

        for kkey, bucket in kc_union.get(
            case_id, {}
        ).items():
            kinfo = catalogue.get(
                kkey, {}
            )
            raw_ids = sorted(
                bucket["raw_ids"]
            )

            internal_map.append({
                "sample_id": rec["_sample_id"],
                "case_id": case_id,
                "conversation_id": rec.get(
                    "conversation_id", ""
                ),
                "assignment_id": assignment_id,
                "item_type": "kc",
                "item_id": (
                    kinfo.get("kc_id")
                    or (
                        clean_bracketed_id(
                            raw_ids[0]
                        )
                        if raw_ids else kkey
                    )
                ),
                "normalized_key": kkey,
                "proposed_by_models": sorted(
                    bucket["models"]
                ),
                "model_proposed_statuses": sorted(
                    bucket["statuses"]
                ),
                "model_evidence_union": bucket[
                    "evidence"
                ],
                "model_notes_union": bucket[
                    "notes"
                ],
                "raw_model_ids": raw_ids,
            })

    return {
        "cases": cases,
        "model_count": len(model_names),
        "result_files": [
            Path(p).name for p in result_files
        ],
        "codebook_kcs_loaded": len(catalogue),
        "official_tasks_loaded": len(
            official_tasks
        ),
        "_internal_model_map": internal_map,
    }


# ---------------------------------------------------------------------
# Final standalone HTML template
# ---------------------------------------------------------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human Task + KC Validation</title>
<style>
:root{
  --bg:#f5f7fb;--surface:#fff;--surface2:#f8fafc;--ink:#172033;--muted:#667085;
  --line:#dce2eb;--accent:#3157d5;--accentSoft:#edf2ff;
  --green:#176b43;--greenBg:#eaf7ef;--greenLine:#a8d7ba;
  --yellow:#7d5b00;--yellowBg:#fff5c9;--yellowLine:#e5cd68;
  --bad:#b42318;--badBg:#fff0ee;--shadow:0 10px 30px rgba(30,45,70,.08)
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
button,input,textarea,select{font:inherit;color:inherit}
.app{display:grid;grid-template-columns:310px minmax(0,1fr);min-height:100vh}
.sidebar{position:sticky;top:0;height:100vh;background:var(--surface);border-right:1px solid var(--line);display:flex;flex-direction:column;overflow:hidden}
.brand{padding:17px;border-bottom:1px solid var(--line)}
.brand h1{font-size:16px;margin:0}
.brand p{margin:4px 0 0;font-size:11px;color:var(--muted);line-height:1.45}
.evalbox,.progress,.side-controls{padding:11px 14px;border-bottom:1px solid var(--line)}
.evalbox label{font-size:10px;color:var(--muted);font-weight:800;display:block;margin-bottom:5px}
.evalbox input,.side-controls input{width:100%;padding:8px;border:1px solid var(--line);border-radius:8px;background:#fff}
.progress{font-size:11px;color:var(--muted)}
.progressbar{height:7px;background:#edf0f5;border-radius:99px;overflow:hidden;margin-top:7px}
.progressfill{height:100%;background:var(--accent);width:0}
.case-list{padding:8px;overflow-y:auto}
.case-card{width:100%;text-align:left;border:1px solid transparent;background:#fff;border-radius:9px;padding:10px;margin-bottom:6px;cursor:pointer}
.case-card:hover{background:#f8faff;border-color:#d7def3}
.case-card.active{background:var(--accentSoft);border-color:#9fb0eb}
.case-top{display:flex;justify-content:space-between;gap:7px}
.case-id{font-weight:800;font-size:12px}
.case-assignment{font-size:10px;background:#eef1f5;padding:3px 6px;border-radius:99px}
.case-preview{font-size:11px;font-weight:650;line-height:1.35;margin-top:6px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.case-meta{font-size:10px;color:var(--muted);margin-top:5px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#c7ced9;margin-right:5px}
.dot.done{background:var(--green)}

.main{padding:18px;min-width:0}
.topbar{max-width:1540px;margin:0 auto 12px;display:flex;justify-content:space-between;gap:12px;align-items:center}
.topbar h2{margin:0;font-size:19px}
.topbar p{margin:3px 0 0;color:var(--muted);font-size:11px}
.actions{display:flex;gap:7px;flex-wrap:wrap}
.btn{border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 11px;cursor:pointer;font-weight:750;font-size:11px}
.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.workspace{max-width:1540px;margin:auto;display:grid;grid-template-columns:minmax(0,1.25fr) minmax(440px,.75fr);gap:14px;align-items:start}
.panel{background:#fff;border:1px solid var(--line);border-radius:13px;box-shadow:var(--shadow);overflow:hidden}
.panel-head{padding:12px 14px;border-bottom:1px solid var(--line);background:#fbfcfe}
.panel-head h3{margin:0;font-size:14px}
.panel-head p{margin:3px 0 0;color:var(--muted);font-size:10px}
.panel-body{padding:13px}
.meta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-bottom:12px}
.meta-item{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:8px;min-width:0}
.meta-label{font-size:8px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:800}
.meta-value{font-size:9px;margin-top:3px;overflow-wrap:anywhere}
.message{border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:9px;background:#fff}
.message.student{border-left:4px solid #5879dc}
.message.assistant{border-left:4px solid #9aa4b2}
.msg-head{display:flex;justify-content:space-between;gap:8px;padding:7px 10px;border-bottom:1px solid var(--line);background:#fafbfc;font-size:9px}
.speaker{font-weight:850}.turn{color:var(--muted)}
.msg-text{white-space:pre-wrap;overflow-wrap:anywhere;padding:10px 11px;font:10.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace}

.tabs{display:grid;grid-template-columns:1fr 1fr;gap:5px;padding:7px;border-bottom:1px solid var(--line);background:#f8fafc}
.tab{border:1px solid transparent;background:transparent;border-radius:7px;padding:8px;cursor:pointer;font-size:10px;font-weight:850}
.tab.active{background:#fff;border-color:var(--line);box-shadow:0 2px 5px rgba(0,0,0,.04)}
.section{display:none;padding:11px}.section.active{display:block}
.notice{font-size:9.5px;line-height:1.45;color:var(--muted);padding:8px 9px;border:1px solid var(--line);border-radius:8px;background:#fafbfc;margin-bottom:9px}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:9px;color:var(--muted);margin:7px 0 10px}
.legend span{display:inline-flex;align-items:center;gap:5px}
.swatch{width:13px;height:13px;border-radius:4px;border:1px solid var(--line);background:#fff}
.swatch.green{background:var(--greenBg);border-color:var(--greenLine)}
.swatch.yellow{background:var(--yellowBg);border-color:var(--yellowLine)}
.group-title{font-size:9px;text-transform:uppercase;letter-spacing:.08em;font-weight:900;color:var(--muted);margin:11px 0 6px}

.item-card{border:1px solid var(--line);border-radius:10px;padding:10px;margin-bottom:8px;background:#fff}
.item-card.union{background:var(--greenBg);border-color:var(--greenLine)}
.item-card.neighbor{background:var(--yellowBg);border-color:var(--yellowLine)}
.item-card.human-added{background:#fff;border-style:dashed}
.source-row{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:4px}
.source-badge{font-size:8px;font-weight:900;text-transform:uppercase;letter-spacing:.06em;padding:3px 6px;border-radius:99px;border:1px solid var(--line);background:#fff}
.source-badge.union{color:var(--green);border-color:var(--greenLine)}
.source-badge.neighbor{color:var(--yellow);border-color:var(--yellowLine)}
.source-badge.human-added{color:var(--muted)}
.item-id{font:700 10px ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}
.item-name{font-size:11px;font-weight:850;margin-top:3px}
.task-official-title{font-size:12px;line-height:1.4;margin-top:5px}
.task-detail-block{margin-top:7px;padding:7px 8px;background:rgba(255,255,255,.68);border:1px solid rgba(120,130,145,.20);border-radius:7px}
.task-detail-label{font-size:8px;text-transform:uppercase;letter-spacing:.06em;font-weight:900;color:var(--muted)}
.task-detail-text{font-size:9.5px;line-height:1.45;margin-top:3px;color:#465164}

.item-desc{font-size:9.5px;color:#535f70;line-height:1.45;margin-top:4px}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:7px}
.chip{font-size:8.5px;padding:3px 5px;border-radius:99px;background:#fff;border:1px solid rgba(120,130,145,.28)}
.chip.candidate{background:#eef2ff;border-color:#cbd6ff;color:#2446a3}
.neighbor-why{font-size:8.5px;color:var(--yellow);font-weight:700;margin-top:6px}

.decision{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:9px}
.decision button{border:1px solid #cfd5df;background:#fff;border-radius:7px;padding:7px 4px;cursor:pointer;font-size:10px;font-weight:850}
.decision button.selected.present{border-color:var(--green);background:#dff3e8;color:var(--green)}
.decision button.selected.not_present{border-color:var(--bad);background:var(--badBg);color:var(--bad)}

.present-details{margin-top:8px;padding:8px;border:1px solid rgba(90,100,120,.18);background:rgba(255,255,255,.72);border-radius:8px}
.field-label{display:block;font-size:8.5px;color:var(--muted);font-weight:850;margin-bottom:4px}
.presence-select{width:100%;border:1px solid var(--line);border-radius:7px;background:#fff;padding:7px;font-size:9px}
.rating-wrap{margin-top:9px}
.rating-head{display:flex;justify-content:space-between;gap:8px;font-size:8.5px;font-weight:800;color:var(--muted)}
.rating-value{color:var(--ink);font-weight:900}
.rating-slider{width:100%;margin:6px 0 0;accent-color:var(--accent)}
.rating-scale{display:flex;justify-content:space-between;color:var(--muted);font-size:8px}
.required-note{font-size:8px;color:var(--bad);margin-top:4px}

details.more{margin-top:7px;font-size:9px;color:var(--muted)}
details.more summary{cursor:pointer;font-weight:800}
.more-body{padding-top:5px;line-height:1.45}

.addbox{border-top:1px solid var(--line);margin-top:12px;padding-top:11px}
.addbox h4{margin:0 0 4px;font-size:11px}
.addbox p{margin:0 0 7px;font-size:9px;color:var(--muted);line-height:1.4}
.search-add{width:100%;padding:7px 8px;border:1px solid var(--line);border-radius:7px;font-size:10px}
.add-results{margin-top:6px;max-height:270px;overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
.add-row{padding:7px 8px;border-bottom:1px solid var(--line);display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:start;background:#fff}
.add-row:last-child{border-bottom:0}
.add-row .rid{font:700 9px ui-monospace,SFMono-Regular,Menlo,monospace}
.add-row .rname{font-size:9px;font-weight:700;margin-top:2px}
.addbtn{border:1px solid var(--line);background:#fff;border-radius:6px;padding:4px 7px;font-size:9px;font-weight:850;cursor:pointer}
.remove-card{border:0;background:transparent;color:var(--bad);cursor:pointer;font-size:8.5px;font-weight:850;padding:2px}
.case-note{width:100%;border:1px solid var(--line);border-radius:7px;padding:7px;font-size:10px;min-height:52px;resize:vertical}
.review-row{margin-top:9px;padding:9px;border-radius:8px;background:#fafbfc;border:1px solid var(--line);font-size:10px}
.review-row .incomplete{color:var(--bad);font-weight:850}.review-row .complete{color:var(--green);font-weight:850}
.empty{padding:12px;border:1px dashed var(--line);border-radius:8px;color:var(--muted);font-size:10px;text-align:center}
.modal{position:fixed;inset:0;background:rgba(17,24,39,.48);display:none;align-items:center;justify-content:center;padding:20px;z-index:50}
.modal.open{display:flex}.modal-card{background:#fff;border-radius:13px;max-width:670px;width:100%;max-height:88vh;overflow:auto;padding:20px;box-shadow:0 24px 70px rgba(0,0,0,.2)}
.modal-card h2{margin:0 0 10px;font-size:18px}.modal-card ol{font-size:11px;line-height:1.55;padding-left:20px}
.navrow{display:flex;justify-content:space-between;gap:8px;margin-top:11px}
@media(max-width:1050px){.app{grid-template-columns:1fr}.sidebar{position:static;height:auto;max-height:430px;border-right:0;border-bottom:1px solid var(--line)}.workspace{grid-template-columns:1fr}.meta{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="app">
<aside class="sidebar">
  <div class="brand"><h1>Human Task + KC Validation</h1><p>Validate model-union items and nearby KCs without seeing model identities.</p></div>
  <div class="evalbox"><label for="evaluatorId">Evaluator ID / initials</label><input id="evaluatorId" placeholder="e.g. E1"></div>
  <div class="progress"><div id="progressText">0 of 24 cases reviewed</div><div class="progressbar"><div class="progressfill" id="progressFill"></div></div></div>
  <div class="side-controls"><input id="searchCases" type="search" placeholder="Search sample or assignment"></div>
  <div class="case-list" id="caseList"></div>
</aside>

<main class="main">
  <div class="topbar">
    <div><h2 id="pageTitle"></h2><p id="pageSub"></p></div>
    <div class="actions">
      <button class="btn" id="instructionsBtn">Instructions</button>
      <button class="btn" id="backupBtn">Download backup JSON</button>
      <button class="btn primary" id="csvBtn">Download report CSV</button>
    </div>
  </div>

  <div class="workspace">
    <section class="panel">
      <div class="panel-head"><h3>Conversation</h3><p>Read the full student–assistant conversation before making decisions.</p></div>
      <div class="panel-body"><div class="meta" id="meta"></div><div id="conversation"></div></div>
    </section>

    <section class="panel">
      <div class="panel-head"><h3>Human validation</h3><p>Green = model union. Yellow = nearby KC to check. White = manually added other item.</p></div>
      <div class="tabs">
        <button class="tab active" data-tab="tasks">Tasks <span id="taskCount"></span></button>
        <button class="tab" data-tab="kcs">Knowledge Components <span id="kcCount"></span></button>
      </div>

      <div class="section active" id="section-tasks">
        <div class="notice">Review the model-union tasks that you can judge confidently. You do not need to select every task. If you mark a task <b>Present</b>, you may also give a 1–10 presence-strength rating.</div>
        <div class="legend"><span><i class="swatch green"></i>model-union task</span><span><i class="swatch"></i>manually added task</span></div>
        <div id="taskList"></div>

        <div class="addbox">
          <h4>Other assignment tasks</h4>
          <p>If a task was missed by every model, find it here and add it to the decision panel.</p>
          <input class="search-add" id="taskSearch" type="search" placeholder="Search task ID or mapped KC">
          <div class="add-results" id="taskAddResults"></div>
        </div>
      </div>

      <div class="section" id="section-kcs">
        <div class="notice"><b>Green KCs</b> are in the model union. <b>Yellow KCs</b> are nearby/related suggestions. You do not need to select every KC. Review only the items you can judge confidently; all other assignment KCs remain in the scrollable search list below.</div>
        <div class="legend"><span><i class="swatch green"></i>model-union KC</span><span><i class="swatch yellow"></i>nearby / related KC</span><span><i class="swatch"></i>manually added other KC</span></div>
        <div id="kcList"></div>

        <div class="addbox">
          <h4>Other KCs in this assignment</h4>
          <p>These are neither in the model union nor in the yellow nearby set. Add one only if you want it moved into the decision panel.</p>
          <input class="search-add" id="kcSearch" type="search" placeholder="Search KC ID, name, definition, or task">
          <div class="add-results" id="kcAddResults"></div>
        </div>
      </div>

      <div class="section" style="display:block;padding-top:0">
        <div class="addbox">
          <label style="font-size:10px;font-weight:850">Case notes (optional)</label>
          <textarea class="case-note" id="caseNotes" placeholder="Any ambiguity or comment about this conversation"></textarea>
          <div class="review-row">
            <div id="completionText"></div>
            <label style="display:block;margin-top:7px"><input type="checkbox" id="caseReviewed"> Mark case reviewed</label>
          </div>
          <div class="navrow"><button class="btn" id="prevBtn">← Previous</button><button class="btn" id="nextBtn">Next →</button></div>
        </div>
      </div>
    </section>
  </div>
</main>
</div>

<div class="modal" id="instructionsModal">
  <div class="modal-card">
    <h2>Evaluator instructions</h2>
    <ol>
      <li>Read the complete conversation.</li>
      <li>Validate each <b>green model-union Task</b> as Present or Not present.</li>
      <li>Validate each <b>green model-union KC</b> and each <b>yellow nearby KC</b> as Present or Not present.</li>
      <li>If a Task or KC is <b>Present</b>, you may use the 1–10 slider to rate how strongly it is evidenced.</li>
      <li>For a <b>Present KC</b>, you may also select the evidence/presence category.</li>
      <li>If another KC not shown in green/yellow is relevant, find it in the scrollable Other KCs list and add it. It will appear as a white decision card.</li>
      <li>Green/yellow colors are search aids only. They do not mean the item is correct.</li>
      <li>You do not have to make a decision for every listed Task or KC. Mark only the items you are comfortable evaluating, and you may mark the case reviewed at any time.</li>
      <li>When all cases are complete, download the CSV report and return it to the researcher.</li>
    </ol>
    <button class="btn primary" id="closeInstructions">Close</button>
  </div>
</div>

<script id="dataset" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById("dataset").textContent);
const KEY="human-task-kc-validation-v3";
const PRESENCE_STATES=[
  ["","Select presence category…"],
  ["difficulty_evidence","Difficulty evidence"],
  ["misconception","Misconception"],
  ["information_seeking","Information seeking"],
  ["resolved_understanding","Resolved understanding"],
  ["neutral_mention","Neutral mention"],
  ["insufficient_evidence","Insufficient evidence"]
];

let state={evaluator_id:"",current:0,cases:{}};

function esc(s){
  return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));
}
function load(){
  try{
    const x=JSON.parse(localStorage.getItem(KEY));
    if(x&&x.cases)state=x;
  }catch(e){}
}
function save(){localStorage.setItem(KEY,JSON.stringify(state))}
function currentCase(){return DATA.cases[state.current]}
function caseState(id){
  if(!state.cases[id])state.cases[id]={
    task_decisions:{},kc_decisions:{},
    task_ratings:{},kc_ratings:{},kc_presence_categories:{},
    added_tasks:[],added_kcs:[],notes:"",reviewed:false
  };
  const st=state.cases[id];
  st.task_decisions=st.task_decisions||{};
  st.kc_decisions=st.kc_decisions||{};
  st.task_ratings=st.task_ratings||{};
  st.kc_ratings=st.kc_ratings||{};
  st.kc_presence_categories=st.kc_presence_categories||{};
  st.added_tasks=st.added_tasks||[];
  st.added_kcs=st.added_kcs||[];
  return st;
}
function preview(c){
  const t=(c.turns||[]).find(x=>String(x.speaker).toLowerCase()==="student");
  return t?String(t.text).replace(/\s+/g," ").slice(0,100):"No student text";
}
function nearbyKcs(c){return (c.assignment_kcs||[]).filter(k=>k.near_neighbor)}
function otherKcs(c){return (c.assignment_kcs||[]).filter(k=>!k.near_neighbor)}
function allMainKcs(c){
  const st=caseState(c.case_id);
  return []
    .concat((c.candidate_kcs||[]).map(k=>({...k,_source:"model_union"})))
    .concat(nearbyKcs(c).map(k=>({...k,_source:"near_neighbor"})))
    .concat((st.added_kcs||[]).map(k=>({...k,_source:"human_added_other"})));
}
function allMainTasks(c){
  const st=caseState(c.case_id);
  return []
    .concat((c.candidate_tasks||[]).map(t=>({...t,_source:"model_union"})))
    .concat((st.added_tasks||[]).map(t=>({...t,_source:"human_added_other"})));
}

function renderSidebar(){
  const q=document.getElementById("searchCases").value.trim().toLowerCase();
  const box=document.getElementById("caseList");
  box.innerHTML=DATA.cases.map((c,i)=>{
    const hay=(c.sample_id+" "+c.assignment_id+" "+c.conversation_id+" "+preview(c)).toLowerCase();
    if(q&&!hay.includes(q))return"";
    const st=caseState(c.case_id);
    const yellow=nearbyKcs(c).length;
    return `<button class="case-card ${i===state.current?"active":""}" data-i="${i}">
      <div class="case-top"><span class="case-id"><i class="dot ${st.reviewed?"done":""}"></i>${esc(c.sample_id)}</span><span class="case-assignment">${esc(c.assignment)}</span></div>
      <div class="case-preview">${esc(preview(c))}</div>
      <div class="case-meta">${c.candidate_tasks.length} task union · ${c.candidate_kcs.length} green KCs · ${yellow} yellow KCs · ${c.num_interactions} interactions</div>
    </button>`;
  }).join("");
  box.querySelectorAll(".case-card").forEach(b=>b.onclick=()=>{
    commit();state.current=Number(b.dataset.i);save();renderAll();window.scrollTo({top:0,behavior:"smooth"});
  });
}
function renderConversation(c){
  const yellow=nearbyKcs(c).length;
  document.getElementById("pageTitle").textContent=`${c.sample_id} · ${c.assignment_title||c.assignment}`;
  document.getElementById("pageSub").textContent=`${c.assignment_id} · ${c.num_interactions} interactions · ${c.candidate_tasks.length} union tasks · ${c.candidate_kcs.length} green KCs · ${yellow} yellow KCs`;

  const meta=[
    ["Sample",c.sample_id],
    ["Assignment",c.assignment_id],
    ["Assignment title",c.assignment_title||""],
    ["Assignment objective",c.assignment_objective||""],
    ["Case ID",c.case_id],
    ["Conversation ID",c.conversation_id],
    ["Student ID",c.student_id],
    ["Variant",c.variant_type||""]
  ];
  document.getElementById("meta").innerHTML=meta.map(x=>`<div class="meta-item"><div class="meta-label">${esc(x[0])}</div><div class="meta-value">${esc(x[1])}</div></div>`).join("");
  document.getElementById("conversation").innerHTML=(c.turns||[]).map(t=>{
    const sp=String(t.speaker||"").toLowerCase();
    return `<div class="message ${sp}"><div class="msg-head"><span class="speaker">${esc(sp==="student"?"Student":sp==="assistant"?"Assistant":sp)}</span><span class="turn">Turn ${esc(t.turn_index)}</span></div><div class="msg-text">${esc(t.text)}</div></div>`;
  }).join("");
}
function decisionButtons(type,id,value){
  return `<div class="decision">
    <button class="${value==="present"?"selected present":""}" data-kind="${type}" data-id="${esc(id)}" data-decision="present">✓ Present</button>
    <button class="${value==="not_present"?"selected not_present":""}" data-kind="${type}" data-id="${esc(id)}" data-decision="not_present">✕ Not present</button>
  </div>`;
}
function ratingControl(type,id,rating){
  const shown=(rating!==undefined&&rating!==null&&rating!=="")?rating:"—";
  const sliderValue=(rating!==undefined&&rating!==null&&rating!=="")?rating:5;
  return `<div class="rating-wrap">
    <div class="rating-head"><span>Presence strength (1–10, optional)</span><span class="rating-value" id="rating-value-${type}-${esc(id)}">${shown}</span></div>
    <input class="rating-slider" type="range" min="1" max="10" step="1" value="${sliderValue}" data-rating-kind="${type}" data-rating-id="${esc(id)}">
    <div class="rating-scale"><span>1 weak</span><span>10 very strong</span></div>

  </div>`;
}
function presenceSelect(id,value){
  return `<div style="margin-top:2px">
    <label class="field-label">Evidence / presence category</label>
    <select class="presence-select" data-presence-id="${esc(id)}">
      ${PRESENCE_STATES.map(x=>`<option value="${esc(x[0])}" ${x[0]===value?"selected":""}>${esc(x[1])}</option>`).join("")}
    </select>

  </div>`;
}
function taskCard(c,t){
  const st=caseState(c.case_id),d=st.task_decisions[t.task_id]||"",rating=st.task_ratings[t.task_id];
  const chips=(t.mapped_kcs||[]).map(k=>`<span class="chip ${k.candidate?"candidate":""}" title="${esc(k.name)}">${esc(k.kc_id)}</span>`).join("");
  const source=t._source||"model_union";
  const sourceClass=source==="model_union"?"union":"human-added";
  const sourceText=source==="model_union"?"Model union":"Manually added";
  const remove=source==="human_added_other"?`<button class="remove-card" data-remove-task="${esc(t.task_id)}">Remove</button>`:"";

  return `<div class="item-card ${sourceClass}">
    <div class="source-row"><span class="source-badge ${sourceClass}">${sourceText}</span>${remove}</div>

    <div class="item-id">${esc(t.task_id)}</div>
    <div class="item-name task-official-title">${esc(t.description||"Task description unavailable")}</div>

    <div class="task-detail-block">
      <div class="task-detail-label">Required output</div>
      <div class="task-detail-text">${esc(t.required_output||"Not specified")}</div>
    </div>

    <div class="task-detail-block">
      <div class="task-detail-label">What the task assesses</div>
      <div class="task-detail-text">${esc(t.assessed_behaviour||"Not specified")}</div>
    </div>

    <div class="item-desc" style="margin-top:8px">KCs mapped to this assignment task:</div>
    <div class="chips">${chips||'<span class="item-desc">No KC mapping found in supplied tables.</span>'}</div>

    ${decisionButtons("task",t.task_id,d)}
    ${d==="present"?`<div class="present-details">${ratingControl("task",t.task_id,rating)}</div>`:""}
  </div>`;
}
function kcCard(c,k){
  const st=caseState(c.case_id),d=st.kc_decisions[k.kc_id]||"",rating=st.kc_ratings[k.kc_id],category=st.kc_presence_categories[k.kc_id]||"";
  const tasks=(k.related_task_ids||[]).map(t=>`<span class="chip">${esc(t)}</span>`).join("");
  const more=(k.successful_performance||k.difficulty_signals)?`<details class="more"><summary>More KC guidance</summary><div class="more-body">${k.successful_performance?'<b>Successful performance:</b> '+esc(k.successful_performance)+'<br><br>':''}${k.difficulty_signals?'<b>Difficulty signals:</b> '+esc(k.difficulty_signals):''}</div></details>`:"";
  const source=k._source||"model_union";
  const sourceClass=source==="model_union"?"union":source==="near_neighbor"?"neighbor":"human-added";
  const sourceText=source==="model_union"?"Model union":source==="near_neighbor"?"Nearby / related":"Manually added";
  const remove=source==="human_added_other"?`<button class="remove-card" data-remove-kc="${esc(k.kc_id)}">Remove</button>`:"";
  return `<div class="item-card ${sourceClass}">
    <div class="source-row"><span class="source-badge ${sourceClass}">${sourceText}</span>${remove}</div>
    <div class="item-id">${esc(k.kc_id)}</div>
    <div class="item-name">${esc(k.name||"")}</div>
    <div class="item-desc">${esc(k.description||"")}</div>
    ${tasks?`<div class="chips">${tasks}</div>`:""}
    ${source==="near_neighbor"&&k.neighbor_reason?`<div class="neighbor-why">${esc(k.neighbor_reason)}</div>`:""}
    ${more}
    ${decisionButtons("kc",k.kc_id,d)}
    ${d==="present"?`<div class="present-details">${presenceSelect(k.kc_id,category)}${ratingControl("kc",k.kc_id,rating)}</div>`:""}
  </div>`;
}
function wireControls(c){
  const st=caseState(c.case_id);

  document.querySelectorAll("[data-decision]").forEach(b=>b.onclick=()=>{
    const kind=b.dataset.kind,id=b.dataset.id,decision=b.dataset.decision;
    if(kind==="task"){
      st.task_decisions[id]=decision;
      if(decision!=="present")delete st.task_ratings[id];
    }else{
      st.kc_decisions[id]=decision;
      if(decision!=="present"){
        delete st.kc_ratings[id];
        delete st.kc_presence_categories[id];
      }
    }
    if(st.reviewed && !completion(c).complete) st.reviewed=false;
    save();renderValidation(c);renderSidebar();updateProgress();
  });

  document.querySelectorAll("[data-rating-kind]").forEach(slider=>slider.oninput=()=>{
    const kind=slider.dataset.ratingKind,id=slider.dataset.ratingId,v=Number(slider.value);
    if(kind==="task")st.task_ratings[id]=v;else st.kc_ratings[id]=v;
    const display=document.getElementById(`rating-value-${kind}-${id}`);
    if(display)display.textContent=String(v);
    save();updateCompletion(c);
  });

  document.querySelectorAll("[data-presence-id]").forEach(sel=>sel.onchange=()=>{
    st.kc_presence_categories[sel.dataset.presenceId]=sel.value;
    if(st.reviewed && !completion(c).complete) st.reviewed=false;
    save();updateCompletion(c);updateProgress();renderSidebar();
  });

  document.querySelectorAll("[data-remove-task]").forEach(b=>b.onclick=()=>{
    const id=b.dataset.removeTask;
    st.added_tasks=st.added_tasks.filter(x=>x.task_id!==id);
    delete st.task_decisions[id];delete st.task_ratings[id];
    st.reviewed=false;save();renderValidation(c);renderSidebar();updateProgress();
  });

  document.querySelectorAll("[data-remove-kc]").forEach(b=>b.onclick=()=>{
    const id=b.dataset.removeKc;
    st.added_kcs=st.added_kcs.filter(x=>x.kc_id!==id);
    delete st.kc_decisions[id];delete st.kc_ratings[id];delete st.kc_presence_categories[id];
    st.reviewed=false;save();renderValidation(c);renderSidebar();updateProgress();
  });
}
function filterAdd(items,q,kind){
  q=q.trim().toLowerCase();
  return items.filter(x=>{
    if(!q)return true;
    const text=kind==="kc"
      ?[x.kc_id,x.name,x.description,(x.related_task_ids||[]).join(" ")].join(" ")
      :[x.task_id,x.description,x.required_output,x.assessed_behaviour,(x.mapped_kcs||[]).map(k=>k.kc_id+" "+k.name).join(" ")].join(" ");
    return text.toLowerCase().includes(q);
  }).slice(0,100);
}
function renderTaskAdd(c){
  const st=caseState(c.case_id);
  const blocked=new Set((c.candidate_tasks||[]).map(x=>x.task_id).concat(st.added_tasks.map(x=>x.task_id)));
  const q=document.getElementById("taskSearch").value;
  const items=filterAdd((c.assignment_tasks||[]).filter(x=>!blocked.has(x.task_id)),q,"task");
  document.getElementById("taskAddResults").innerHTML=items.length?items.map(t=>`
    <div class="add-row"><div>
      <div class="rid">${esc(t.task_id)}</div>
      <div class="rname">${esc(t.description||"Task description unavailable")}</div>
      <div class="item-desc" style="margin-top:3px"><b>Required output:</b> ${esc(t.required_output||"Not specified")}</div>
    </div>
    <button class="addbtn" data-add-task="${esc(t.task_id)}">Add to panel</button></div>`).join(""):'<div class="empty">No matching remaining tasks.</div>';
  document.querySelectorAll("[data-add-task]").forEach(b=>b.onclick=()=>{
    const t=(c.assignment_tasks||[]).find(x=>x.task_id===b.dataset.addTask);
    if(t){
      st.added_tasks.push({
        task_id:t.task_id,
        official_task_id:t.official_task_id||t.task_id,
        description:t.description||"",
        required_output:t.required_output||"",
        assessed_behaviour:t.assessed_behaviour||"",
        support_basis:t.support_basis||"",
        mapped_kcs:t.mapped_kcs||[]
      });
      st.reviewed=false;save();renderValidation(c);
    }
  });
}
function renderKcAdd(c){
  const st=caseState(c.case_id);
  const blocked=new Set((c.candidate_kcs||[]).map(x=>x.kc_id).concat(nearbyKcs(c).map(x=>x.kc_id)).concat(st.added_kcs.map(x=>x.kc_id)));
  const q=document.getElementById("kcSearch").value;
  const items=filterAdd(otherKcs(c).filter(x=>!blocked.has(x.kc_id)),q,"kc");
  document.getElementById("kcAddResults").innerHTML=items.length?items.map(k=>`
    <div class="add-row"><div><div class="rid">${esc(k.kc_id)}</div><div class="rname">${esc(k.name)}</div></div>
    <button class="addbtn" data-add-kc="${esc(k.kc_id)}">Add to panel</button></div>`).join(""):'<div class="empty">No matching remaining KCs.</div>';
  document.querySelectorAll("[data-add-kc]").forEach(b=>b.onclick=()=>{
    const k=otherKcs(c).find(x=>x.kc_id===b.dataset.addKc);
    if(k){
      st.added_kcs.push({
        kc_id:k.kc_id,name:k.name,description:k.description,
        related_task_ids:k.related_task_ids||[],successful_performance:k.successful_performance||"",
        difficulty_signals:k.difficulty_signals||"",neighbor_reason:""
      });
      st.reviewed=false;save();renderValidation(c);
    }
  });
}
function itemComplete(decision,rating,categoryRequired,category){
  return decision==="present" || decision==="not_present";
}
function completion(c){
  const st=caseState(c.case_id),tasks=allMainTasks(c),kcs=allMainKcs(c);
  const taskDone=tasks.filter(t=>itemComplete(st.task_decisions[t.task_id],st.task_ratings[t.task_id],false,"")).length;
  const kcDone=kcs.filter(k=>itemComplete(st.kc_decisions[k.kc_id],st.kc_ratings[k.kc_id],true,st.kc_presence_categories[k.kc_id])).length;
  return {taskDone,kcDone,totalTasks:tasks.length,totalKcs:kcs.length,complete:taskDone===tasks.length&&kcDone===kcs.length};
}
function updateCompletion(c){
  const cm=completion(c);
  document.getElementById("completionText").innerHTML=
    `Decisions recorded: <b>${cm.taskDone}/${cm.totalTasks} tasks</b> · <b>${cm.kcDone}/${cm.totalKcs} KCs</b> · <span class="muted">You may mark this case reviewed at any time.</span>`;
}
function renderValidation(c){
  const st=caseState(c.case_id);
  const tasks=allMainTasks(c),green=(c.candidate_kcs||[]).map(k=>({...k,_source:"model_union"})),yellow=nearbyKcs(c).map(k=>({...k,_source:"near_neighbor"})),added=st.added_kcs.map(k=>({...k,_source:"human_added_other"}));
  document.getElementById("taskCount").textContent=`(${tasks.length})`;
  document.getElementById("kcCount").textContent=`(${green.length+yellow.length+added.length})`;

  let taskHtml="";
  if(c.candidate_tasks.length){
    taskHtml+=`<div class="group-title">Green · model union</div>`+(c.candidate_tasks||[]).map(t=>taskCard(c,{...t,_source:"model_union"})).join("");
  }
  if(st.added_tasks.length){
    taskHtml+=`<div class="group-title">White · manually added</div>`+st.added_tasks.map(t=>taskCard(c,{...t,_source:"human_added_other"})).join("");
  }
  if(!taskHtml)taskHtml='<div class="empty">No model-union task is shown for this conversation. Use the search below if a task is clearly present.</div>';
  document.getElementById("taskList").innerHTML=taskHtml;

  let kcHtml="";
  if(green.length)kcHtml+=`<div class="group-title">Green · model union</div>`+green.map(k=>kcCard(c,k)).join("");
  if(yellow.length)kcHtml+=`<div class="group-title">Yellow · nearby / related KCs</div>`+yellow.map(k=>kcCard(c,k)).join("");
  if(added.length)kcHtml+=`<div class="group-title">White · manually added other KCs</div>`+added.map(k=>kcCard(c,k)).join("");
  if(!kcHtml)kcHtml='<div class="empty">No green or yellow KC is shown for this conversation. Use the Other KCs list below if needed.</div>';
  document.getElementById("kcList").innerHTML=kcHtml;

  document.getElementById("caseNotes").value=st.notes||"";
  document.getElementById("caseReviewed").checked=!!st.reviewed;
  updateCompletion(c);
  wireControls(c);
  renderTaskAdd(c);
  renderKcAdd(c);
  document.getElementById("taskSearch").oninput=()=>renderTaskAdd(c);
  document.getElementById("kcSearch").oninput=()=>renderKcAdd(c);
}
function commit(){
  const c=currentCase();if(!c)return;
  const st=caseState(c.case_id);
  st.notes=document.getElementById("caseNotes")?.value||st.notes;
  state.evaluator_id=document.getElementById("evaluatorId").value.trim();
  save();
}
function updateProgress(){
  const done=DATA.cases.filter(c=>caseState(c.case_id).reviewed).length,total=DATA.cases.length;
  document.getElementById("progressText").textContent=`${done} of ${total} cases reviewed`;
  document.getElementById("progressFill").style.width=(total?done/total*100:0)+"%";
}
function renderAll(){const c=currentCase();renderSidebar();renderConversation(c);renderValidation(c);updateProgress()}
function changeCase(d){commit();state.current=Math.max(0,Math.min(DATA.cases.length-1,state.current+d));save();renderAll();window.scrollTo({top:0,behavior:"smooth"})}
function download(name,text,type){const blob=new Blob([text],{type});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function csvCell(v){const s=String(v??"");return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
function rowFor(base,itemType,item,source,decision,rating,category){
  return {
    ...base,
    item_type:itemType,
    item_id:itemType==="task"?item.task_id:item.kc_id,
    item_name:itemType==="task"?"":(item.name||""),
    item_source:source,
    candidate_from_model_union:source==="model_union",
    near_neighbor:source==="near_neighbor",
    human_decision:decision||"",
    human_presence_category:decision==="present"&&itemType==="kc"?(category||""):"",
    presence_strength_1_10:decision==="present"?(rating??""):"",
    related_ids:itemType==="task"?(item.mapped_kcs||[]).map(k=>k.kc_id).join("; "):(item.related_task_ids||[]).join("; "),
    neighbor_reason:source==="near_neighbor"?(item.neighbor_reason||""):""
  };
}
function buildRows(){
  commit();const rows=[];
  DATA.cases.forEach(c=>{
    const st=caseState(c.case_id);
    const base={
      evaluator_id:state.evaluator_id,sample_id:c.sample_id,case_id:c.case_id,
      conversation_id:c.conversation_id,assignment_id:c.assignment_id,
      case_notes:st.notes||"",case_reviewed:!!st.reviewed
    };
    (c.candidate_tasks||[]).forEach(t=>rows.push(rowFor(base,"task",t,"model_union",st.task_decisions[t.task_id],st.task_ratings[t.task_id],"")));
    (st.added_tasks||[]).forEach(t=>rows.push(rowFor(base,"task",t,"human_added_other",st.task_decisions[t.task_id],st.task_ratings[t.task_id],"")));
    (c.candidate_kcs||[]).forEach(k=>rows.push(rowFor(base,"kc",k,"model_union",st.kc_decisions[k.kc_id],st.kc_ratings[k.kc_id],st.kc_presence_categories[k.kc_id])));
    nearbyKcs(c).forEach(k=>rows.push(rowFor(base,"kc",k,"near_neighbor",st.kc_decisions[k.kc_id],st.kc_ratings[k.kc_id],st.kc_presence_categories[k.kc_id])));
    (st.added_kcs||[]).forEach(k=>rows.push(rowFor(base,"kc",k,"human_added_other",st.kc_decisions[k.kc_id],st.kc_ratings[k.kc_id],st.kc_presence_categories[k.kc_id])));
    if(!rows.length)rows.push({...base,item_type:"none",item_id:"",item_name:"",item_source:"",candidate_from_model_union:false,near_neighbor:false,human_decision:"",human_presence_category:"",presence_strength_1_10:"",related_ids:"",neighbor_reason:""});
  });
  return rows;
}
function exportCsv(){
  const rows=buildRows(),headers=Object.keys(rows[0]||{}),text=[headers.join(",")].concat(rows.map(r=>headers.map(h=>csvCell(r[h])).join(","))).join("\n");
  download(`human_task_kc_validation_${state.evaluator_id||"evaluator"}.csv`,text,"text/csv;charset=utf-8");
}
function exportBackup(){
  commit();
  download(`human_task_kc_validation_backup_${state.evaluator_id||"evaluator"}.json`,JSON.stringify({dataset:{cases:DATA.cases.length,result_files:DATA.result_files},evaluation:state},null,2),"application/json");
}

document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active",x===b));
  document.querySelectorAll(".section[id^='section-']").forEach(x=>x.classList.toggle("active",x.id===`section-${b.dataset.tab}`));
});
document.getElementById("evaluatorId").oninput=e=>{state.evaluator_id=e.target.value;save()};
document.getElementById("searchCases").oninput=renderSidebar;
document.getElementById("prevBtn").onclick=()=>changeCase(-1);
document.getElementById("nextBtn").onclick=()=>changeCase(1);
document.getElementById("caseNotes").oninput=e=>{caseState(currentCase().case_id).notes=e.target.value;save()};
document.getElementById("caseReviewed").onchange=e=>{
  const c=currentCase(),st=caseState(c.case_id);
  st.reviewed=e.target.checked;
  save();
  updateProgress();
  renderSidebar();
};
document.getElementById("csvBtn").onclick=exportCsv;
document.getElementById("backupBtn").onclick=exportBackup;
document.getElementById("instructionsBtn").onclick=()=>document.getElementById("instructionsModal").classList.add("open");
document.getElementById("closeInstructions").onclick=()=>document.getElementById("instructionsModal").classList.remove("open");
document.getElementById("instructionsModal").onclick=e=>{if(e.target.id==="instructionsModal")e.currentTarget.classList.remove("open")};

load();
document.getElementById("evaluatorId").value=state.evaluator_id||"";
if(state.current>=DATA.cases.length)state.current=0;
renderAll();
</script>
</body>
</html>"""


def build_html(data, output_path):
    payload = json.dumps(
        data, ensure_ascii=False
    ).replace("</", "<\\/")

    final_html = HTML_TEMPLATE.replace(
        "__DATA__", payload
    )

    Path(output_path).write_text(
        final_html,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the final standalone Human "
            "Task + KC Validation interface."
        )
    )

    parser.add_argument(
        "--conversations",
        default=DEFAULT_CONVERSATIONS,
    )

    parser.add_argument(
        "--results",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--kc-files",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--task-model-files",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--combined-kc-output",
        default=DEFAULT_COMBINED_KC,
    )

    parser.add_argument(
        "--internal-map-output",
        default=DEFAULT_INTERNAL_MAP,
    )

    args = parser.parse_args()

    if not Path(args.conversations).exists():
        raise SystemExit(
            "Conversation file not found: "
            + args.conversations
        )

    result_files = args.results or [
        p for p in DEFAULT_RESULTS
        if Path(p).exists()
    ]

    if not result_files:
        result_files = sorted(
            str(p)
            for p in Path(".").glob(
                "results*.jsonl"
            )
        )

    if not result_files:
        raise SystemExit(
            "No model result JSONL files found."
        )

    kc_files = args.kc_files or [
        p for p in DEFAULT_KC_FILES
        if Path(p).exists()
    ]

    if len(kc_files) != 6:
        missing = [
            p for p in DEFAULT_KC_FILES
            if not Path(p).exists()
        ]
        raise SystemExit(
            "Expected all six KC CSV tables. "
            "Missing: "
            + ", ".join(missing)
        )

    task_model_files = (
        args.task_model_files
        or [
            p
            for p in DEFAULT_TASK_MODEL_FILES
            if Path(p).exists()
        ]
    )

    if len(task_model_files) != 6:
        missing = [
            p
            for p in DEFAULT_TASK_MODEL_FILES
            if not Path(p).exists()
        ]
        raise SystemExit(
            "Expected all six official task/KC "
            "model JSON files. Missing: "
            + ", ".join(missing)
        )

    data = build_dataset(
        conversation_file=args.conversations,
        result_files=result_files,
        kc_files=kc_files,
        task_model_files=task_model_files,
        combined_kc_output=(
            args.combined_kc_output
        ),
    )

    internal_map = data.pop(
        "_internal_model_map", []
    )

    Path(
        args.internal_map_output
    ).write_text(
        json.dumps(
            internal_map,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    build_html(
        data=data,
        output_path=args.output,
    )

    green_tasks = sum(
        len(c["candidate_tasks"])
        for c in data["cases"]
    )

    green_kcs = sum(
        len(c["candidate_kcs"])
        for c in data["cases"]
    )

    yellow_kcs = sum(
        sum(
            1
            for k in c["assignment_kcs"]
            if k["near_neighbor"]
        )
        for c in data["cases"]
    )

    print("=" * 72)
    print(
        "FINAL FLEXIBLE HUMAN TASK + KC "
        "VALIDATION INTERFACE CREATED"
    )
    print("=" * 72)
    print(
        f"Base conversations: "
        f"{len(data['cases'])}"
    )
    print(
        f"Model result files: "
        f"{len(result_files)}"
    )
    print(
        f"Distinct models: "
        f"{data['model_count']}"
    )
    print(
        f"Official Tasks loaded: "
        f"{data['official_tasks_loaded']}"
    )
    print(
        f"KC definitions loaded: "
        f"{data['codebook_kcs_loaded']}"
    )
    print(
        f"Green model-union Task cards: "
        f"{green_tasks}"
    )
    print(
        f"Green model-union KC cards: "
        f"{green_kcs}"
    )
    print(
        f"Yellow nearby/related KC cards: "
        f"{yellow_kcs}"
    )
    print(
        f"Researcher-only model map: "
        f"{args.internal_map_output}"
    )
    print(
        f"Human evaluator HTML: "
        f"{args.output}"
    )
    print()
    print(
        "SEND ONLY the generated HTML "
        "file to human evaluators."
    )


if __name__ == "__main__":
    main()
