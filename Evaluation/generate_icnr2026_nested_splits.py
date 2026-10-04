"""Generate deterministic subject-wise nested CV splits for ICNR 2026.

Design constraints
------------------
* Six outer folds with sizes 4, 4, 4, 4, 4, 3.
* Subject-level stratification uses max UPDRS_GAIT because eight subjects
  change UPDRS across medication states.
* Every outer-test and inner-validation set should cover U0/U1/U2 and both
  medication states whenever feasible.
* Clinical coverage has priority. Duration and walk-count balance are only
  tie-breakers because validation objectives are aggregated subject-first.
* Each outer fold has exactly four inner-validation subjects. The same split
  artifact must be reused by every sensor/loss/capacity condition.

The script writes JSON/CSV/Markdown audit artifacts and does not train a model.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "TransPose/data/dataset_work/CAREPD_BMCLab/manifest.json"
OUT_DIR = PROJECT_ROOT / "ModelComparison/data/splits/CAREPD_BMCLab_ICNR2026_nested_cv"

N_OUTER_FOLDS = 6
N_INNER_VAL_SUBJECTS = 4
BASE_SEED = 20260802
N_OUTER_CANDIDATES = 10_000
SEVERITY_LEVELS = (0, 1, 2)
MEDICATION_STATES = ("off", "on")


def load_subject_summary() -> pd.DataFrame:
    manifest = pd.DataFrame(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    required = {"subject_id", "walk_id", "n_frames", "target_fps", "UPDRS_GAIT", "medication"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    manifest["duration_s"] = manifest["n_frames"] / manifest["target_fps"]
    rows = []
    for subject, sub in manifest.groupby("subject_id", sort=True):
        states = sorted(set(sub["medication"].dropna().astype(str)))
        rows.append(
            {
                "subject": str(subject),
                "n_walks": int(len(sub)),
                "duration_s": float(sub["duration_s"].sum()),
                "min_updrs": int(sub["UPDRS_GAIT"].min()),
                "max_updrs": int(sub["UPDRS_GAIT"].max()),
                "medication_states": ",".join(states),
                "has_off": "off" in states,
                "has_on": "on" in states,
                "is_paired_medication": set(states) == set(MEDICATION_STATES),
            }
        )
    summary = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)

    observed = set(summary["max_updrs"].unique())
    if observed != set(SEVERITY_LEVELS):
        raise ValueError(f"Unexpected max-UPDRS levels: {sorted(observed)}")
    if len(summary) != 23:
        raise ValueError(f"Expected 23 BMCLab subjects, found {len(summary)}")
    return summary


def severity_counts(sub: pd.DataFrame) -> dict[int, int]:
    counts = sub["max_updrs"].value_counts().to_dict()
    return {level: int(counts.get(level, 0)) for level in SEVERITY_LEVELS}


def medication_union(sub: pd.DataFrame) -> set[str]:
    states: set[str] = set()
    for value in sub["medication_states"]:
        states.update(str(value).split(","))
    states.discard("")
    return states


def outer_score(summary: pd.DataFrame, test_folds: list[list[str]]) -> tuple:
    global_severity = severity_counts(summary)
    global_paired_rate = float(summary["is_paired_medication"].mean())
    total_duration = float(summary["duration_s"].sum())
    total_walks = int(summary["n_walks"].sum())

    missing_severity = 0
    severity_deviation = 0.0
    missing_medication = 0
    paired_deviation = 0.0
    duration_deviation = 0.0
    walk_deviation = 0.0

    for subjects in test_folds:
        sub = summary[summary["subject"].isin(subjects)]
        size = len(sub)
        counts = severity_counts(sub)
        missing_severity += sum(counts[level] == 0 for level in SEVERITY_LEVELS)
        for level in SEVERITY_LEVELS:
            expected = size * global_severity[level] / len(summary)
            severity_deviation += abs(counts[level] - expected)

        states = medication_union(sub)
        missing_medication += sum(state not in states for state in MEDICATION_STATES)
        paired_deviation += abs(int(sub["is_paired_medication"].sum()) - size * global_paired_rate)

        expected_duration = total_duration * size / len(summary)
        expected_walks = total_walks * size / len(summary)
        duration_deviation += abs(float(sub["duration_s"].sum()) - expected_duration) / total_duration
        walk_deviation += abs(int(sub["n_walks"].sum()) - expected_walks) / total_walks

    # Lexicographic priority: clinical coverage, medication coverage, then size proxies.
    return (
        int(missing_severity),
        round(severity_deviation, 12),
        int(missing_medication),
        round(paired_deviation, 12),
        round(duration_deviation, 12),
        round(walk_deviation, 12),
    )


def choose_outer_folds(summary: pd.DataFrame) -> tuple[list[list[str]], int, tuple]:
    subjects = summary["subject"].to_numpy()
    strata = summary["max_updrs"].to_numpy()
    best = None
    for offset in range(N_OUTER_CANDIDATES):
        seed = BASE_SEED + offset
        splitter = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=seed)
        folds = [sorted(subjects[test_idx].tolist()) for _, test_idx in splitter.split(subjects, strata)]
        score = outer_score(summary, folds)
        candidate = (score, seed, folds)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("Failed to generate outer folds")
    score, seed, folds = best
    return folds, seed, score


def inner_score(train_pool: pd.DataFrame, val_subjects: tuple[str, ...]) -> tuple:
    val = train_pool[train_pool["subject"].isin(val_subjects)]
    counts = severity_counts(val)
    pool_counts = severity_counts(train_pool)
    missing_severity = sum(counts[level] == 0 for level in SEVERITY_LEVELS)
    severity_deviation = sum(
        abs(counts[level] - len(val) * pool_counts[level] / len(train_pool)) for level in SEVERITY_LEVELS
    )

    states = medication_union(val)
    missing_medication = sum(state not in states for state in MEDICATION_STATES)
    expected_paired = len(val) * float(train_pool["is_paired_medication"].mean())
    paired_deviation = abs(int(val["is_paired_medication"].sum()) - expected_paired)

    target_fraction = len(val) / len(train_pool)
    duration_deviation = abs(
        float(val["duration_s"].sum()) / float(train_pool["duration_s"].sum()) - target_fraction
    )
    walk_deviation = abs(int(val["n_walks"].sum()) / int(train_pool["n_walks"].sum()) - target_fraction)

    # Subject tuple is the final deterministic tie-breaker.
    return (
        int(missing_severity),
        round(severity_deviation, 12),
        int(missing_medication),
        round(paired_deviation, 12),
        round(duration_deviation, 12),
        round(walk_deviation, 12),
        tuple(val_subjects),
    )


def choose_inner_validation(summary: pd.DataFrame, outer_test: list[str]) -> tuple[list[str], tuple]:
    train_pool = summary[~summary["subject"].isin(outer_test)].copy()
    best = None
    for subjects in itertools.combinations(train_pool["subject"].tolist(), N_INNER_VAL_SUBJECTS):
        score = inner_score(train_pool, subjects)
        if best is None or score < best[0]:
            best = (score, sorted(subjects))
    if best is None:
        raise RuntimeError("Failed to choose inner-validation subjects")
    score, subjects = best
    return subjects, score[:-1]


def audit_subset(summary: pd.DataFrame, subjects: list[str]) -> dict:
    sub = summary[summary["subject"].isin(subjects)]
    counts = severity_counts(sub)
    states = medication_union(sub)
    return {
        "n_subjects": int(len(sub)),
        "n_walks": int(sub["n_walks"].sum()),
        "duration_s": round(float(sub["duration_s"].sum()), 6),
        "duration_min": round(float(sub["duration_s"].sum()) / 60.0, 6),
        "max_updrs_counts": {str(k): v for k, v in counts.items()},
        "medication_states_present": sorted(states),
        "n_paired_medication_subjects": int(sub["is_paired_medication"].sum()),
        "subjects": sorted(subjects),
    }


def validate_splits(summary: pd.DataFrame, folds: list[dict]) -> None:
    all_subjects = set(summary["subject"])
    test_counter: Counter[str] = Counter()
    for fold in folds:
        outer_test = set(fold["outer_test_subjects"])
        inner_val = set(fold["inner_val_subjects"])
        inner_train = set(fold["inner_train_subjects"])
        if outer_test & inner_val or outer_test & inner_train or inner_val & inner_train:
            raise AssertionError(f"Fold {fold['outer_fold']} has overlapping roles")
        if outer_test | inner_val | inner_train != all_subjects:
            raise AssertionError(f"Fold {fold['outer_fold']} does not cover every subject")
        if len(inner_val) != N_INNER_VAL_SUBJECTS:
            raise AssertionError(f"Fold {fold['outer_fold']} has wrong inner-validation size")

        for name, subjects in [("outer_test", outer_test), ("inner_val", inner_val)]:
            sub = summary[summary["subject"].isin(subjects)]
            if set(sub["max_updrs"]) != set(SEVERITY_LEVELS):
                raise AssertionError(f"Fold {fold['outer_fold']} {name} lacks a max-UPDRS level")
            if medication_union(sub) != set(MEDICATION_STATES):
                raise AssertionError(f"Fold {fold['outer_fold']} {name} lacks a medication state")
        test_counter.update(outer_test)

    if set(test_counter) != all_subjects or any(count != 1 for count in test_counter.values()):
        raise AssertionError("Each subject must appear in exactly one outer-test fold")


def write_outputs(summary: pd.DataFrame, folds: list[dict], outer_seed: int, outer_objective: tuple) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_to_fold = {
        subject: fold["outer_fold"] for fold in folds for subject in fold["outer_test_subjects"]
    }
    subject_table = summary.copy()
    subject_table["outer_test_fold"] = subject_table["subject"].map(subject_to_fold)
    subject_table.to_csv(OUT_DIR / "subject_summary_with_outer_fold.csv", index=False)

    inner_val_counter: Counter[str] = Counter(
        subject for fold in folds for subject in fold["inner_val_subjects"]
    )
    reuse_table = subject_table[
        ["subject", "n_walks", "duration_s", "max_updrs", "medication_states"]
    ].copy()
    reuse_table["inner_val_appearances"] = reuse_table["subject"].map(inner_val_counter).fillna(0).astype(int)
    reuse_table = reuse_table.sort_values(
        ["inner_val_appearances", "subject"], ascending=[False, True]
    )
    reuse_table.to_csv(OUT_DIR / "inner_validation_reuse.csv", index=False)

    assignment_rows = []
    audit_rows = []
    for fold in folds:
        roles = {
            "inner_train": fold["inner_train_subjects"],
            "inner_val": fold["inner_val_subjects"],
            "outer_test": fold["outer_test_subjects"],
        }
        for role, subjects in roles.items():
            for subject in subjects:
                assignment_rows.append(
                    {"outer_fold": fold["outer_fold"], "subject": subject, "role": role}
                )
            audit = fold["audit"][role]
            audit_rows.append(
                {
                    "outer_fold": fold["outer_fold"],
                    "role": role,
                    "n_subjects": audit["n_subjects"],
                    "n_walks": audit["n_walks"],
                    "duration_s": audit["duration_s"],
                    "duration_min": audit["duration_min"],
                    "n_max_updrs_0": audit["max_updrs_counts"]["0"],
                    "n_max_updrs_1": audit["max_updrs_counts"]["1"],
                    "n_max_updrs_2": audit["max_updrs_counts"]["2"],
                    "medication_states_present": ",".join(audit["medication_states_present"]),
                    "n_paired_medication_subjects": audit["n_paired_medication_subjects"],
                    "subjects": ",".join(audit["subjects"]),
                }
            )
    pd.DataFrame(assignment_rows).sort_values(["outer_fold", "role", "subject"]).to_csv(
        OUT_DIR / "nested_split_assignments.csv", index=False
    )
    audit_df = pd.DataFrame(audit_rows).sort_values(["outer_fold", "role"])
    audit_df.to_csv(OUT_DIR / "fold_audit.csv", index=False)

    payload = {
        "purpose": "ICNR2026 PD-aware TransPose fine-tuning",
        "source_manifest": str(MANIFEST_PATH),
        "subject_stratification_key": "max UPDRS_GAIT across medication states",
        "base_seed": BASE_SEED,
        "outer_candidate_count": N_OUTER_CANDIDATES,
        "selected_outer_seed": outer_seed,
        "outer_objective": list(outer_objective),
        "constraint_priority": [
            "max-UPDRS level coverage and balance",
            "medication-state coverage and paired-subject balance",
            "duration balance (tie-breaker)",
            "walk-count balance (tie-breaker)",
        ],
        "validation_aggregation": "subject-first, equal weight across inner-validation subjects",
        "inner_validation_reuse": {
            row.subject: int(row.inner_val_appearances) for row in reuse_table.itertuples()
        },
        "capacity_count_caveat": (
            "Inner-validation subjects recur across outer folds; S1/S1.5/S2 selection counts "
            "are correlated diagnostic outcomes, not six independent observations."
        ),
        "folds": folds,
    }
    (OUT_DIR / "nested_splits.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# ICNR 2026 BMCLab nested subject-wise CV splits",
        "",
        f"Source: `{MANIFEST_PATH}`",
        "",
        "Subject-level stratification uses each subject's maximum UPDRS_GAIT across medication states. ",
        "Clinical severity and medication coverage have priority; duration and walk count are tie-breakers.",
        "Inner-validation metrics must be aggregated per subject before the four subjects are averaged.",
        "",
        f"Selected outer seed: `{outer_seed}` from {N_OUTER_CANDIDATES} deterministic candidates.",
        "",
        "| Fold | Role | N subj | N walks | Minutes | U0/U1/U2 | Paired med | Subjects |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    role_order = {"outer_test": 0, "inner_val": 1, "inner_train": 2}
    for _, row in audit_df.sort_values(["outer_fold", "role"], key=lambda s: s.map(role_order) if s.name == "role" else s).iterrows():
        levels = f"{row.n_max_updrs_0}/{row.n_max_updrs_1}/{row.n_max_updrs_2}"
        lines.append(
            f"| {row.outer_fold} | {row.role} | {row.n_subjects} | {row.n_walks} | "
            f"{row.duration_min:.2f} | {levels} | {row.n_paired_medication_subjects} | {row.subjects} |"
        )
    lines.extend(
        [
            "",
            "## Inner-validation reuse diagnostic",
            "",
            "Validation duration is closely balanced, but the same subjects can recur across outer folds. "
            "Consequently, the reported number of folds selecting S1/S1.5/S2 is a correlated diagnostic "
            "summary rather than six independent observations. The assignments remain frozen.",
            "",
            "| Subject | Inner-val appearances | N walks | Duration (s) | Max UPDRS | Medication states |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in reuse_table.itertuples():
        lines.append(
            f"| {row.subject} | {row.inner_val_appearances} | {row.n_walks} | "
            f"{row.duration_s:.1f} | {row.max_updrs} | {row.medication_states} |"
        )
    lines.extend(
        [
            "",
            "## Reuse rule",
            "",
            "The exact same subject assignments must be reused for P6/C6/P5/C5, all capacities, masking ",
            "sensitivity analyses, seeds, and noise conditions. Outer-test subjects must never influence ",
            "capacity selection, early stopping, imputation statistics, or loss weights.",
        ]
    )
    (OUT_DIR / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = load_subject_summary()
    outer_folds, outer_seed, outer_objective = choose_outer_folds(summary)
    all_subjects = set(summary["subject"])
    folds = []
    for fold_idx, outer_test in enumerate(outer_folds):
        inner_val, inner_objective = choose_inner_validation(summary, outer_test)
        inner_train = sorted(all_subjects - set(outer_test) - set(inner_val))
        fold = {
            "outer_fold": fold_idx,
            "outer_test_subjects": sorted(outer_test),
            "inner_val_subjects": sorted(inner_val),
            "inner_train_subjects": inner_train,
            "inner_objective": list(inner_objective),
        }
        fold["audit"] = {
            "outer_test": audit_subset(summary, fold["outer_test_subjects"]),
            "inner_val": audit_subset(summary, fold["inner_val_subjects"]),
            "inner_train": audit_subset(summary, fold["inner_train_subjects"]),
        }
        folds.append(fold)

    validate_splits(summary, folds)
    write_outputs(summary, folds, outer_seed, outer_objective)
    print(f"Saved nested CV splits to {OUT_DIR}")
    print(f"Selected outer seed: {outer_seed}; objective={outer_objective}")
    for fold in folds:
        print(
            f"fold {fold['outer_fold']}: test={fold['outer_test_subjects']} "
            f"val={fold['inner_val_subjects']}"
        )


if __name__ == "__main__":
    main()
