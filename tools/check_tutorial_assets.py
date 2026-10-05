"""Statically validate public tutorial downloads; never import or run the solver.

Run from any working directory with Python 3.10+:
    python tools/check_tutorial_assets.py
Optionally compare against the authorized original delivery:
    python tools/check_tutorial_assets.py --originals /path/to/originals
"""

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "examples" / "tutorials"
GENERIC_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
ALLOWED_IMPORT_ROOTS = {
    "pathlib", "datetime", "hashlib", "importlib", "inspect", "json", "math",
    "platform", "random", "warnings", "zipfile", "numpy", "torch", "matplotlib", "starwave",
}
PRIVATE_PATH = re.compile(r"(?:/(?:home|Users|root|workspace|mnt|scratch)/|\b[A-Za-z]:[\\/]|file://|ssh://)")
SHA256 = re.compile(r"[a-f0-9]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def canonical_digest(data):
    return digest(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def source(cell):
    return "".join(cell["source"])


def code_source(notebook):
    return "\n\n".join(source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")


def export_script(notebook):
    """A deterministic percent-format export with code-cell bytes unchanged."""
    pieces = ["# -*- coding: utf-8 -*-\n# Paired executable script exported from the public notebook.\n\n"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            pieces.append("# %% [markdown]\n")
            pieces.append("\n".join("# " + line for line in source(cell).splitlines()) + "\n\n")
        else:
            pieces.append("# %%\n" + source(cell))
            if not source(cell).endswith("\n"):
                pieces.append("\n")
            pieces.append("\n")
    return "".join(pieces)


def literal_config(notebook):
    """Read only literal CONFIG declarations/updates; no eval or execution."""
    config = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse(source(cell)).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id == "CONFIG":
                    require(config is None, "Duplicate CONFIG declaration")
                    config = ast.literal_eval(node.value)
                elif (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                      and target.value.id == "CONFIG"):
                    require(config is not None, "CONFIG update before declaration")
                    config[ast.literal_eval(target.slice)] = ast.literal_eval(node.value)
            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute)
                  and isinstance(node.value.func.value, ast.Name)
                  and node.value.func.value.id == "CONFIG" and node.value.func.attr == "update"):
                require(config is not None, "CONFIG update before declaration")
                require(len(node.value.args) == 1 and not node.value.keywords, "Unexpected CONFIG update")
                config.update(ast.literal_eval(node.value.args[0]))
    require(isinstance(config, dict), "Missing CONFIG dictionary")
    return config


def numerical_config(config):
    return {key: value for key, value in config.items() if key not in {"gpu_id", "output_root"}}


def helper_without_reference_fields(text):
    """Return an AST with only the two declared provenance dictionary entries removed."""
    tree = copy.deepcopy(ast.parse(text))
    removed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            kept = []
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in {
                    "reference_dev_commit", "reference_is_not_installed_commit_claim",
                }:
                    removed.append(key.value)
                else:
                    kept.append((key, value))
            node.keys = [key for key, _ in kept]
            node.values = [value for _, value in kept]
    require(sorted(removed) == ["reference_dev_commit", "reference_is_not_installed_commit_claim"],
            "Original helper does not contain exactly the two expected provenance fields")
    return tree


def check_code(text, label):
    tree = ast.parse(text, filename=label)
    require(not PRIVATE_PATH.search(text), f"{label}: private absolute path or remote URI")
    require("reference_dev_commit" not in text, f"{label}: unnecessary development commit")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            require(all(module.split(".")[0] in ALLOWED_IMPORT_ROOTS for module in modules),
                    f"{label}: unexpected/network/process import: {modules}")
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            require(name not in {"eval", "exec", "compile", "__import__", "get_ipython"},
                    f"{label}: dynamic execution is not permitted")
            require(not any(part in name.lower() for part in ("urlopen", "urlretrieve", "download", "subprocess", "socket")),
                    f"{label}: network/download/process call: {name}")
            if name == "importlib.import_module":
                require(len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == "starwave.scalar", f"{label}: unexpected dynamic import")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            require(not re.search(r"(?:https?|ftp)://", node.value), f"{label}: executable network URL")
            # A lone slash is a fragment of the relative ZIP-entry f-string.
            require(node.value == "/" or not node.value.startswith(("/", "~", "\\\\")),
                    f"{label}: absolute/private path literal")
    return tree


def check_finite(value, label="results"):
    if isinstance(value, float):
        require(math.isfinite(value), f"{label}: nonfinite numeric value")
    elif isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            check_finite(item, f"{label}[{i}]")


def check(originals=None):
    manifest = json.loads((ASSETS / "source_manifest.json").read_text(encoding="utf-8"))
    summary_path = ASSETS / "results_summary.json"
    results = json.loads(summary_path.read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1, "Unsupported provenance schema")
    require(manifest["public_edition_executed_byte_for_byte"] is False, "Execution provenance must be explicit")
    require(digest(summary_path.read_bytes()) == manifest["results_summary_sha256"], "Results summary hash changed")
    require("figures" not in results, "Use the separately curated figure manifest")
    check_finite(results)
    require(results["recorded_environment"] == {
        "gpu": "NVIDIA A30", "python": "3.10.18", "torch": "2.5.1",
        "torch_cuda": "11.8", "starwave_distribution": "0.1.0.dev9",
    }, "Measured environment changed; these results are not a public 2.0.0 wheel run")
    require(results["metrics"]["02_forward_gradient"]["directional_check"]["status"] == "not_requested",
            "Do not imply the directional check ran")
    require(results["metrics"]["02_forward_gradient"]["full_adjoint_validation"] is False,
            "Do not imply full adjoint validation")
    fwi = results["metrics"]["03_simple_fwi"]
    require(fwi["updates"] == 25 and fwi["final"]["iteration"] == 25, "Expected 25 updates and last iterate")
    require(fwi["synthetic_same_solver_data"] and fwi["final_is_last_iterate"], "Missing teaching/iterate caveats")
    require(fwi["gpu_full_acceptance"] is False, "Do not imply full GPU acceptance")
    require(fwi["status"] == "optimization_completed_not_convergence_certified", "Do not certify convergence")
    limits = " ".join(results["limits"]).lower()
    for caveat in ("same-solver", "convergence is not certified", "lens", "full adjoint", "multi-gpu", "0.1.0.dev9", "2.0.0", "not calibrated pa"):
        require(caveat in limits, f"Missing measured-results caveat: {caveat}")

    cells_checked = 0
    require(len(manifest["tutorials"]) == 3, "Expected three standalone tutorials")
    expected_notebooks = {item["notebook"] for item in manifest["tutorials"]}
    require({p.name for p in ASSETS.glob("*.ipynb")} == expected_notebooks, "Unexpected/missing public notebook")
    require({p.name for p in ASSETS.glob("*.py")} == {Path(name).with_suffix(".py").name for name in expected_notebooks},
            "Unexpected/missing public script")
    for item in manifest["tutorials"]:
        name = item["notebook"]
        require(Path(name).name == name and name.endswith(".ipynb"), "Unsafe notebook filename")
        notebook_path = ASSETS / name
        script_path = notebook_path.with_suffix(".py")
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        require(set(nb) == {"cells", "metadata", "nbformat", "nbformat_minor"}, f"{name}: unexpected top-level data")
        require(nb["nbformat"] == 4, f"{name}: unsupported format")
        require(nb["metadata"] == GENERIC_METADATA, f"{name}: non-generic notebook metadata")
        require(digest(notebook_path.read_bytes()) == item["public_notebook_sha256"], f"{name}: notebook hash changed")
        require(digest(script_path.read_bytes()) == item["public_script_sha256"], f"{name}: script hash changed")
        require(digest(code_source(nb)) == item["public_code_sha256"], f"{name}: source hash changed")
        for field in ("original_notebook_sha256", "original_script_sha256", "original_code_sha256"):
            require(SHA256.fullmatch(item[field]) is not None, f"{name}: invalid provenance hash")
        require(len(nb["cells"]) == item["cell_count"], f"{name}: cell count changed")
        code_cells = {index: cell for index, cell in enumerate(nb["cells"]) if cell["cell_type"] == "code"}
        require({row["cell_index"] for row in item["code_cells"]} == set(code_cells), f"{name}: incomplete code snapshot")
        for index, cell in enumerate(nb["cells"]):
            expected_keys = {"cell_type", "metadata", "source"}
            if cell["cell_type"] == "code":
                expected_keys |= {"outputs", "execution_count"}
            require(set(cell) == expected_keys, f"{name} cell {index}: unexpected cell data")
            require(cell["metadata"] == {}, f"{name} cell {index}: nonempty cell metadata")
            require(not cell.get("attachments"), f"{name} cell {index}: embedded attachment")
            require(not PRIVATE_PATH.search(source(cell)), f"{name} cell {index}: private path")
            if cell["cell_type"] == "code":
                require(cell["outputs"] == [] and cell["execution_count"] is None, f"{name} cell {index}: saved execution output")
                check_code(source(cell), f"{name}:{index}")
                cells_checked += 1
            else:
                require(cell["cell_type"] == "markdown", f"{name}: unexpected cell type")
        for snapshot in item["code_cells"]:
            current = digest(source(code_cells[snapshot["cell_index"]]))
            require(current == snapshot["public_sha256"], f"{name}: cell source drift")
            if snapshot["role"] != "helpers":
                require(current == snapshot["original_sha256"], f"{name}: numerical/plot/export cell changed from original")
            else:
                require(current == snapshot["original_without_reference_fields_sha256"], f"{name}: helper edit exceeds declared scope")
        script = script_path.read_text(encoding="utf-8")
        require(script == export_script(nb), f"{name}: paired script is not the exact deterministic cell export")
        require(ast.dump(ast.parse(script)) == ast.dump(ast.parse(code_source(nb))), f"{name}: notebook/script AST mismatch")
        cfg = literal_config(nb)
        require(canonical_digest(cfg) == item["original_config_sha256"], f"{name}: original config drift")
        require(numerical_config(cfg) == results["configs"][item["experiment"]], f"{name}: measured config mismatch")
        require(canonical_digest(numerical_config(cfg)) == item["result_numerical_config_sha256"], f"{name}: numerical config hash mismatch")
        intro = source(nb["cells"][0])
        for caveat in ("0.1.0.dev9", "2.0.0", "not been run byte-for-byte", "same original Chinese-language"):
            require(caveat in intro, f"{name}: missing public edition caveat: {caveat}")
        if originals:
            original_path = Path(originals) / name
            original = json.loads(original_path.read_text(encoding="utf-8"))
            require(digest(original_path.read_bytes()) == item["original_notebook_sha256"], f"{name}: original notebook mismatch")
            require(digest(original_path.with_suffix(".py").read_bytes()) == item["original_script_sha256"], f"{name}: original script mismatch")
            require(digest(code_source(original)) == item["original_code_sha256"], f"{name}: original code mismatch")
            for snapshot in item["code_cells"]:
                original_source = source(original["cells"][snapshot["cell_index"]])
                require(digest(original_source) == snapshot["original_sha256"],
                        f"{name}: original cell snapshot mismatch")
                if snapshot["role"] == "helpers":
                    require(ast.dump(helper_without_reference_fields(original_source)) ==
                            ast.dump(ast.parse(source(code_cells[snapshot["cell_index"]]))),
                            f"{name}: helper AST changed beyond the two declared provenance fields")
    print(f"PASS: 3 clean notebooks, 3 exact paired scripts, {cells_checked} parsed code cells; unchanged original numerical cells/configs.")
    print("PASS: finite curated metrics, dev9 environment, provenance hashes and teaching/validation caveats. No solver was run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", type=Path, help="Optional original delivery directory, read only")
    check(parser.parse_args().originals)
