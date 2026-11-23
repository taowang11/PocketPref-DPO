"""Evaluate saved multi-pocket generation results with a common scoring protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import statistics
import tempfile
import time
from collections import defaultdict
from importlib import import_module, metadata
from pathlib import Path

VINA_METRICS = ("vina_score", "vina_min", "vina_dock")
CHEMISTRY_METRICS = (
    "qed", "sa", "lipinski",
    "logp", "heavy_atoms", "mw", "tpsa", "rings", "aromatic_rings", "rotatable_bonds",
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-pockets", type=positive_int, default=100)
    parser.add_argument("--expected-attempts", type=positive_int, default=100,
                        help="Attempts per model per pocket, including failures.")
    parser.add_argument("--workers", type=positive_int, default=min(32, len(os.sched_getaffinity(0))))
    parser.add_argument("--analysis", choices=("vina", "structure", "all"), default="vina",
                        help="Evaluate docking, saved structures, or both.")
    parser.add_argument("--dock", action="store_true", help="Also run full Vina docking.")
    parser.add_argument("--vina-seed", type=int, default=0)
    parser.add_argument("--exhaustiveness", type=positive_int, default=16)
    parser.add_argument("--min-docks-for-high-affinity", type=positive_int, default=50)
    parser.add_argument("--timeout", type=positive_int, default=1800,
                        help="Timeout in seconds for each isolated molecule evaluation.")
    args = parser.parse_args()
    if args.dock and args.analysis == "structure":
        parser.error("--dock requires --analysis vina or all.")
    return args


def digest(path):
    with Path(path).open("rb") as handle:
        sha = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024**2), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_manifest(path, expected_pockets):
    data = json.loads(path.read_text())
    pockets = data["pockets"]
    if len(pockets) != expected_pockets:
        raise ValueError(f"Expected {expected_pockets} pockets, found {len(pockets)}.")
    identifiers = set()
    models = set(pockets[0]["runs"])
    if not models or any(not name.strip() for name in models):
        raise ValueError("Provide at least one named model in each pocket's runs.")

    def existing(value):
        resolved = (path.parent / value).resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved

    for pocket in pockets:
        pocket_id = str(pocket["id"])
        if not pocket_id or pocket_id in identifiers:
            raise ValueError(f"Expected distinct pocket IDs, got {pocket_id!r}.")
        identifiers.add(pocket_id)
        if set(pocket["runs"]) != models:
            raise ValueError(f"Model names differ at pocket {pocket_id!r}.")
        pocket["id"] = pocket_id
        pocket["runs"] = {name: existing(value) for name, value in pocket["runs"].items()}
        for field in ("pocket", "receptor", "receptor_pdbqt", "structure_receptor", "prolif_receptor", "reference_sdf"):
            if field in pocket:
                pocket[field] = existing(pocket[field])
        if not any(field in pocket for field in ("pocket", "receptor", "receptor_pdbqt", "structure_receptor")):
            raise ValueError(f"Pocket {pocket_id!r} has no receptor input.")
        if "box" in pocket:
            for key in ("center", "size"):
                numbers = pocket["box"][key]
                if len(numbers) != 3 or any(not math.isfinite(n) for n in numbers):
                    raise ValueError(f"Invalid box {key} at pocket {pocket_id!r}.")
            if min(pocket["box"]["size"]) <= 0:
                raise ValueError(f"Invalid box size at pocket {pocket_id!r}.")
        ref = pocket.get("reference_vina_dock")
        if ref is not None and not math.isfinite(ref):
            raise ValueError(f"Invalid reference docking score at pocket {pocket_id!r}.")
    return pockets


def read_generation(path, expected, input_pocket):
    import numpy as np
    from rdkit import Chem

    generation = None
    if path.is_dir():
        report = path / "run.json"
        generation = json.loads(report.read_text())
        if generation["attempted"] != expected:
            raise ValueError(f"{report} declares {generation['attempted']} attempts, expected {expected}.")
        if input_pocket and generation.get("pocket_sha256") != digest(input_pocket):
            raise ValueError(f"Generation used a different input pocket according to {report}.")
        sdf = path / "molecules.sdf"
        attempts = {row["attempt"]: row for row in generation["attempts"]}
        if set(attempts) != set(range(expected)) or len(generation["attempts"]) != expected:
            raise ValueError(f"{report} must account for each attempt exactly once.")
        valid_ids = [index for index in range(expected) if attempts[index]["valid"]]
        if generation["valid"] != len(valid_ids):
            raise ValueError(f"{report} has inconsistent validity counts.")
    else:
        if path.suffix.lower() != ".sdf":
            raise ValueError(f"Expected a generation directory or an SDF, got {path}.")
        sdf = path
        attempts = {}
        valid_ids = list(range(expected))

    if not sdf.is_file():
        raise FileNotFoundError(sdf)
    molecules = {}
    record_ids = set()
    if sdf.stat().st_size:
        for index, mol in enumerate(Chem.SDMolSupplier(str(sdf), removeHs=True)):
            if index >= len(valid_ids):
                raise ValueError(f"{sdf} contains more records than the declared budget.")
            label = int(mol.GetProp("attempt")) if mol is not None and mol.HasProp("attempt") else valid_ids[index]
            if label not in valid_ids or label in record_ids:
                raise ValueError(f"Unexpected or repeated attempt {label} in {sdf}.")
            record_ids.add(label)
            if (mol is not None and mol.GetNumAtoms() and len(Chem.GetMolFrags(mol)) == 1
                    and mol.GetNumConformers() == 1 and np.isfinite(mol.GetConformer().GetPositions()).all()):
                molecules[label] = mol
        if generation is not None and record_ids != set(valid_ids):
            raise ValueError(f"{sdf} does not account for all successful generation records.")
    elif generation is not None and valid_ids:
        raise ValueError(f"{sdf} is empty but {len(valid_ids)} valid molecules were declared.")

    provenance = {"source": str(path), "sdf_sha256": digest(sdf)}
    if generation:
        provenance.update({key: generation.get(key) for key in ("checkpoint_sha256", "seed", "reverse_steps")})
    return molecules, attempts, provenance


def finite_score(value):
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("Nonfinite Vina score.")
    return score


def evaluate_molecule(job, args, work, result):
    import numpy as np
    from rdkit import Chem

    from utils.evaluation import chemistry

    row, mol, receptor, box, structure, context = job
    row["errors"] = {}
    try:
        row.update(chemistry.descriptors(mol, row["smiles"]))
    except (ValueError, RuntimeError, OSError) as error:
        row["errors"]["chemistry"] = str(error)
    write_json(result, row)
    if structure is not None:
        structure.measure(mol, context, row)
        write_json(result, row)
    if receptor is None:
        return
    from utils.evaluation import fixed_vina

    try:
        ligand = work / f"{result.stem}.pdbqt"
        fixed_vina.prepare_ligand(mol, ligand)
        if box is None:
            positions = Chem.AddHs(mol, addCoords=True).GetConformer().GetPositions()
            low, high = positions.min(0), positions.max(0)
            box = {"center": ((low + high) / 2).tolist(), "size": (high - low + 5).tolist()}
        else:
            positions = mol.GetConformer().GetPositions()
            if np.any(np.abs(positions - box["center"]) >= np.asarray(box["size"]) / 2):
                raise ValueError("Original ligand lies outside the supplied Vina box.")
        vina = fixed_vina.engine(receptor, box, args.vina_seed, ligand=ligand)
        row["vina_box"] = box
        row["vina_score"] = finite_score(vina.score()[0])
        write_json(result, row)
        try:
            row["vina_min"] = finite_score(vina.optimize()[0])
        except (RuntimeError, ValueError) as error:
            row["errors"]["vina_min"] = str(error)
        write_json(result, row)
        if args.dock:
            try:
                # Start full docking from the original prepared pose.
                dock = fixed_vina.engine(receptor, box, args.vina_seed, ligand=ligand)
                row["vina_dock_seed"] = int(dock.info()["seed"])
                dock.dock(exhaustiveness=args.exhaustiveness, n_poses=1)
                row["vina_dock"] = finite_score(dock.energies(n_poses=1)[0][0])
            except (RuntimeError, ValueError, IndexError) as error:
                row["errors"]["vina_dock"] = str(error)
    except (KeyError, RuntimeError, ValueError, OSError) as error:
        row["errors"]["vina_preparation_or_score"] = str(error)
    write_json(result, row)


def collect_scoring(jobs, args, temporary):
    """Isolate native exits and preserve completed metrics from each attempt."""
    pending = iter(enumerate(jobs))
    active = {}
    exhausted = False
    total = len(jobs)
    completed = 0
    try:
        while active or not exhausted:
            while len(active) < args.workers and not exhausted:
                assignment = next(pending, None)
                if assignment is None:
                    exhausted = True
                    break
                index, job = assignment
                dest = temporary / f"{index}.json"
                pid = os.fork()
                if pid == 0:
                    log = os.open(temporary / f"{index}.log", os.O_CREAT | os.O_WRONLY, 0o600)
                    os.dup2(log, 1)
                    os.dup2(log, 2)
                    os.close(log)
                    try:
                        evaluate_molecule(job, args, temporary, dest)
                    except BaseException:
                        import traceback
                        traceback.print_exc()
                        os._exit(1)
                    os._exit(0)
                active[pid] = (job[0], dest, time.monotonic())
            for pid, (row, dest, start) in list(active.items()):
                done, status = os.waitpid(pid, os.WNOHANG)
                timed_out = not done and time.monotonic() - start > args.timeout
                if timed_out:
                    os.kill(pid, signal.SIGKILL)
                    done, status = os.waitpid(pid, 0)
                if not done:
                    continue
                if dest.exists():
                    row.update(json.loads(dest.read_text()))
                if timed_out or status or not dest.exists():
                    errors = row.setdefault("errors", {})
                    log = dest.with_suffix(".log")
                    reason = "timed out" if timed_out else f"worker exit status {status}"
                    errors["worker"] = reason + (": " + log.read_text(errors="replace")[-500:] if log.exists() else "")
                del active[pid]
                completed += 1
                if completed == total or completed % 100 == 0:
                    print(f"Evaluated {completed}/{total} valid input molecules", flush=True)
            if active:
                time.sleep(0.025)
    finally:
        for pid in active:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


def statistics_row(rows, metrics):
    result = {}
    for metric in metrics:
        valid = [row[metric] for row in rows if row.get(metric) is not None]
        result[f"{metric}_n"] = len(valid)
        result[f"{metric}_avg"] = statistics.fmean(valid) if valid else None
        result[f"{metric}_med"] = statistics.median(valid) if valid else None
    return result


def summarise(rows, pocket_data, args, metrics, structure_metrics):
    groups = defaultdict(list)
    models = defaultdict(list)
    for row in rows:
        groups[row["model"], row["pocket"]].append(row)
        models[row["model"]].append(row)
    per_pocket = []
    for (model, pocket), records in groups.items():
        info = pocket_data[model, pocket]
        docked = [row["vina_dock"] for row in records if "vina_dock" in row]
        reference = info["reference"]
        valid = sum(row["valid"] for row in records)
        per_pocket.append(dict(
            model=model, pocket=pocket, attempts=len(records), valid=valid, valid_rate=valid / len(records),
            diversity=info["diversity"], reference_vina_dock=reference,
            high_affinity_pct=(100 * sum(score <= reference for score in docked) / len(docked)
                               if reference is not None and len(docked) >= args.min_docks_for_high_affinity else None),
            **statistics_row(records, metrics),
        ))
    summary = []
    for model, records in models.items():
        pockets = [row for row in per_pocket if row["model"] == model]
        valid = sum(row["valid"] for row in records)
        record = dict(
            model=model, pockets=len(pockets), attempts=len(records), valid=valid, valid_rate=valid / len(records),
            **statistics_row(records, metrics), **statistics_row(pockets, ("diversity", "high_affinity_pct")),
        )
        for metric in structure_metrics:
            values = [pocket[metric + "_avg"] for pocket in pockets if pocket[metric + "_avg"] is not None]
            record[metric + "_pocket_n"] = len(values)
            record[metric + "_pocket_equal_avg"] = statistics.fmean(values) if values else None
        summary.append(record)
    return summary, per_pocket


def write_csv(path, rows, empty_columns=()):
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else empty_columns
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def prepare_structure(pocket, structure, chemical_space, record, work):
    receptor = pocket.get("structure_receptor", pocket.get("receptor", pocket.get("pocket")))
    if receptor is None or receptor.suffix.lower() != ".pdb" or not receptor.is_file():
        raise ValueError(f"Provide a receptor PDB for structure evaluation at {pocket['id']!r}.")
    clash_receptor = pocket.get("receptor_pdbqt", receptor)
    prolif_receptor = pocket.get("prolif_receptor", receptor)
    if not prolif_receptor.is_file() or prolif_receptor.suffix.lower() != ".pdb":
        raise ValueError(f"Provide a protein PDB for ProLIF at {pocket['id']!r}.")
    reference_path = pocket.get("reference_sdf")
    reference = None
    if reference_path is not None:
        references, _, info = read_generation(reference_path, 1, None)
        if len(references) != 1:
            raise ValueError(f"Provide one valid reference ligand with its saved pose in {reference_path}.")
        reference = references[0]
        chemical_space.add(dict(model="Reference", pocket=pocket["id"], attempt=None), reference, role="reference")
        record["reference_sdf"] = info
    work.mkdir()
    context = structure.prepare(receptor, clash_receptor, prolif_receptor, reference, work)
    record["structure_inputs"] = {
        role: {"source": str(path), "sha256": digest(path)}
        for role, path in (("posebusters", receptor), ("steric", clash_receptor), ("prolif", prolif_receptor))
    }
    if context["reference"] is not None:
        record["reference_interaction_count"] = len(context["reference"])
        record["reference_interactions"] = sorted(context["reference"])
    if context["errors"]:
        record["structure_errors"] = context["errors"]
    return context


def write_structure_outputs(output_dir, chemical_space):
    import numpy as np

    per_pocket, summary = chemical_space.ring_tables()
    write_csv(output_dir / "ring_sizes_by_pocket.csv", per_pocket,
              ("model", "pocket", "role", "ring_size", "molecules", "containing_ring", "occurrence"))
    write_csv(output_dir / "ring_size_summary.csv", summary,
              ("model", "role", "ring_size", "pockets", "pocket_equal_occurrence"))
    projections, pca_info, state = chemical_space.pca()
    write_csv(output_dir / "morgan_pca.csv", projections, ("model", "pocket", "attempt", "role", "pc1", "pc2"))
    write_json(output_dir / "morgan_pca.json", pca_info)
    if state is not None:
        np.savez_compressed(output_dir / "morgan_pca_model.npz", **state)


def evaluate(args):
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    import numpy as np
    from rdkit import Chem

    from utils.evaluation import chemistry

    pockets = load_manifest(args.manifest, args.expected_pockets)
    versions = {library: metadata.version(library) for library in ("rdkit", "numpy")}
    use_vina = args.analysis in ("vina", "all")
    if use_vina:
        from utils.evaluation import fixed_vina

        # Historical Open Babel-backed Meeko uses this NumPy alias.
        if "int" not in np.__dict__:
            np.int = int
        for library in ("vina", "meeko", "openbabel.pybel"):
            import_module(library)
        versions.update({library: metadata.version(library) for library in ("vina", "meeko")})
    structure, chemical_space, structure_metrics = None, None, ()
    if args.analysis in ("structure", "all"):
        from utils.evaluation import structure as structure_checks

        structure = structure_checks.StructureEvaluation()
        structure_metrics = structure_checks.STRUCTURE_METRICS
        capacity = len(pockets) * (len(pockets[0]["runs"]) * args.expected_attempts + 1)
        chemical_space = structure_checks.ChemicalSpace(capacity)
        versions.update({library: metadata.version(library) for library in ("posebusters", "prolif", "scipy")})
    chemistry.rdkit_sa_scorer().readFragmentScores()
    args.output_dir.mkdir(parents=True)
    rows, pocket_data, jobs = [], {}, []
    provenance = []
    with tempfile.TemporaryDirectory(prefix="pocketpref-eval-") as scratch:
        temporary = Path(scratch)
        for index, pocket in enumerate(pockets):
            name = pocket["id"]
            record = dict(pocket=name, runs={})
            receptor, context = None, None
            if use_vina:
                receptor = pocket.get("receptor_pdbqt")
                if receptor is None:
                    raw_receptor = pocket.get("receptor", pocket.get("pocket"))
                    if raw_receptor is None:
                        raise ValueError(f"Provide a Vina receptor input for pocket {name!r}.")
                    receptor = fixed_vina.prepare_receptor(raw_receptor, temporary / f"receptor_{index}")
                record.update(receptor_sha256=digest(receptor), box=pocket.get("box"))
            if structure is not None:
                context = prepare_structure(pocket, structure, chemical_space, record, temporary / f"structure_{index}")
            provenance.append(record)
            for model, source in pocket["runs"].items():
                molecules, attempts, info = read_generation(source, args.expected_attempts, pocket.get("pocket"))
                record["runs"][model] = info
                pocket_data[model, name] = dict(diversity=chemistry.internal_diversity(molecules.values()),
                                                reference=pocket.get("reference_vina_dock"))
                for attempt in range(args.expected_attempts):
                    mol = molecules.get(attempt)
                    row = dict(model=model, pocket=name, attempt=attempt, valid=mol is not None)
                    rows.append(row)
                    if mol is None:
                        reason = attempts.get(attempt, {}).get("error") or "No valid reconstructed molecule for this attempt"
                        row["errors"] = {"generation_or_sdf": reason}
                        continue
                    row["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if chemical_space is not None:
                        row["ring_sizes"] = chemical_space.add(row, mol)
                    jobs.append((row, mol, receptor, pocket.get("box"), structure, context))
        protocol = dict(
            versions=versions, manifest_sha256=digest(args.manifest), pockets=provenance,
            analysis=args.analysis, expected_attempts=args.expected_attempts, workers=args.workers, dock=args.dock,
        )
        if use_vina:
            protocol.update(
                vina_seed=args.vina_seed, exhaustiveness=args.exhaustiveness,
                box_default="Per-ligand hydrogen-added pose bbox midpoint and span plus 5 Å",
                high_affinity_minimum_docked=args.min_docks_for_high_affinity,
            )
        if structure is not None:
            protocol["structure"] = dict(
                posebusters_config="dock", clash_raw_vdw_overlap_cutoff_A=structure_checks.CLASH_OVERLAP,
                prolif_interactions=structure_checks.INTERACTIONS, prolif_implicit_hydrogens=True,
                ring_bins=structure_checks.RING_BINS,
            )
        write_json(args.output_dir / "protocol.json", protocol)
        collect_scoring(jobs, args, temporary)

    metrics = (VINA_METRICS if use_vina else ()) + CHEMISTRY_METRICS + structure_metrics
    summary, per_pocket = summarise(rows, pocket_data, args, metrics, structure_metrics)
    write_csv(args.output_dir / "molecules.csv", rows)
    write_csv(args.output_dir / "pockets.csv", per_pocket)
    write_csv(args.output_dir / "summary.csv", summary)
    write_json(args.output_dir / "summary.json", summary)
    if chemical_space is not None:
        write_structure_outputs(args.output_dir, chemical_space)
    for model in summary:
        message = f"{model['model']}: {model['valid']}/{model['attempts']} valid"
        if use_vina:
            message += f", Vina score/min/dock counts {model['vina_score_n']}/{model['vina_min_n']}/{model['vina_dock_n']}"
        if structure is not None:
            message += (f", PoseBusters/ProLIF/clash counts {model['posebusters_pass_n']}/"
                        f"{model['prolif_interaction_count_n']}/{model['clash_pairs_n']}")
        print(message)


if __name__ == "__main__":
    try:
        evaluate(parse_arguments())
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
        raise SystemExit(f"Evaluation failed: {error}") from None
