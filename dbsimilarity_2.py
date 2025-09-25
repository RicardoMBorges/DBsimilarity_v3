"""
Utility toolkit for:
- MassQL query generation (MS1 and MS2) from metadata tables
- Morgan fingerprint similarity matrices and network export
- Building MZmine custom MS1 database CSVs
- Mordred descriptor computation
- Hierarchical clustering (dendrogram) + Agglomerative clustering mirror
- t-SNE projection with Plotly export

Notes
-----
• Designed to be imported from notebooks or scripts.
• Optional dependencies (handle gracefully where possible):
    - RDKit
    - Mordred (often requires numpy<2 with some RDKit builds)
    - chembl_webresource_client (optional; only used if you call it)

Author: Ricardo M. Borges + ChatGPT (refactor)
"""
from __future__ import annotations

# --- Standard library
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --- Third-party (optional in some blocks)
import numpy as np
import pandas as pd

# Plotting (only used where needed)
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Network
import networkx as nx

# Optional external clients
try:  # chembl client is optional; code will still import without it
    from chembl_webresource_client.new_client import new_client  # noqa: F401
except Exception:  # pragma: no cover
    new_client = None

# =============================
# Canonical shared definitions
# =============================
ADDUCT_MASS: Dict[str, float] = {
    "[M+H]+":   1.007276466,
    "[M+Na]+":  22.989218,     # Na+
    "[M+K]+":   38.963157,     # K+
    "[M+NH4]+": 18.033823,     # NH4+
    "[M+2H]+2": 2 * 1.007276466,  # doubly protonated
}

ADDUCT_CHARGE: Dict[str, int] = {
    "[M+H]+": 1,
    "[M+Na]+": 1,
    "[M+K]+": 1,
    "[M+NH4]+": 1,
    "[M+2H]+2": 2,
}

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _resolve_column(df: pd.DataFrame, colname: str) -> str:
    """Case-insensitive column resolver with friendly error."""
    if colname in df.columns:
        return colname
    lowmap = {str(c).lower(): c for c in df.columns}
    key = str(colname).lower()
    if key in lowmap:
        return lowmap[key]
    raise KeyError(f"Column '{colname}' not found. Available: {', '.join(map(str, df.columns))}")


def _to_float(val) -> Optional[float]:
    """Robust numeric parser for strings with commas, blanks, etc."""
    if val is None:
        return None
    s = str(val).strip().replace(",", ".")
    m = _num_re.search(s)
    return float(m.group(0)) if m else None


def _fmt(x: float, decimals: int = 5) -> str:
    return f"{x:.{decimals}f}"


def _prec_mzs(mono_mass: float, adducts: Sequence[str], decimals: int) -> List[str]:
    return [_fmt(mono_mass + ADDUCT_MASS[a], decimals) for a in adducts]


# =============================
# Dataframe utilities
# =============================

# Robust loader for CSV/TSV/TXT + SMILES column normalization + InchiKey
import os, re
from pathlib import Path
import pandas as pd
from rdkit import Chem

def _detect_file(stem_or_path, candidates=(".csv", ".tsv", ".txt")) -> Path:
    p = Path(stem_or_path)
    if p.suffix:  # user provided extension
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p
    for ext in candidates:
        cand = Path(stem_or_path + ext)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No file found for '{stem_or_path}' with extensions {candidates}")

def _guess_sep(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".csv":
        return ","
    if ext == ".tsv":
        return "\t"
    # .txt → sniff
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
    # simple heuristic: prefer tab, then semicolon, then comma
    if "\t" in head:
        return "\t"
    if head.count(";") >= head.count(","):
        return ";"
    return ","

def _read_table(path: Path, sep: str) -> tuple[pd.DataFrame, str]:
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc)
            return df, enc
        except UnicodeDecodeError:
            continue
    # last try with engine='python' for odd separators
    df = pd.read_csv(path, sep=sep, encoding="utf-8", engine="python")
    return df, "utf-8 (python-engine)"

def _normalize_smiles_column(df: pd.DataFrame) -> str:
    """Find any 'smiles'-like column (case/punct-insensitive) and rename to 'SMILES'."""
    def norm(s): return re.sub(r"[^a-z]", "", str(s).lower())
    candidates = [c for c in df.columns if "smiles" in norm(c)]
    if not candidates:
        raise KeyError(
            "No SMILES-like column found. Looked for any header containing 'smiles' "
            f"(case/format-insensitive). Columns: {list(df.columns)}"
        )
    # If multiple candidates, pick the one with the most non-null entries
    best = max(candidates, key=lambda c: df[c].notna().sum())
    if best != "SMILES":
        df.rename(columns={best: "SMILES"}, inplace=True)
    return "SMILES"

def _add_inchikey(df: pd.DataFrame, smiles_col: str = "SMILES", out_col: str = "InchiKey") -> None:
    def smi_to_inchikey(s):
        if pd.isna(s): return None
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return None
        try:
            return Chem.MolToInchiKey(m)
        except Exception:
            return None
    df[out_col] = df[smiles_col].apply(smi_to_inchikey)

from typing import List, Sequence, Optional
import pandas as pd

def merge_dataframes(
    df_list: List[pd.DataFrame],
    key_column: str,
    name_columns: Optional[Sequence[str]] = None,
    unified_name_col: str = "Compound name (unified)",
) -> pd.DataFrame:
    """
    Merge multiple DataFrames on `key_column`, add presence flags per source,
    and create a unified compound-name column by coalescing across `name_columns`.
    Original name columns are preserved.

    Parameters
    ----------
    df_list : list[pd.DataFrame]
    key_column : str
        Column used as the merge key across frames.
    name_columns : sequence[str], optional
        Candidate columns that may contain the compound name (e.g., ["Compound name","Name"]).
        The first non-empty value among these (left-to-right) is used.
    unified_name_col : str
        Output column name for the unified compound name.

    Returns
    -------
    pd.DataFrame
    """
    # 1) Row/column union without dropping anything
    merged = pd.concat(df_list, axis=0, ignore_index=True, sort=False)

    # 2) Build unified name column by coalescing across possible headers
    if name_columns:
        present = [c for c in name_columns if c in merged.columns]
        if present:
            # start with all-NA string series
            out = pd.Series(pd.NA, index=merged.index, dtype="string")
            for c in present:
                s = merged[c].astype("string", errors="ignore")
                s = s.astype("string") if s.dtype != "string" else s
                s = s.str.strip()
                s = s.mask(s.eq("") | s.isna(), pd.NA)
                out = out.fillna(s)
            merged[unified_name_col] = out
        else:
            # none of the suggested columns exist; create empty unified column
            merged[unified_name_col] = pd.Series(pd.NA, index=merged.index, dtype="string")

    # 3) Collapse duplicates by key (keep first occurrence; preserves non-numeric cols)
    if key_column in merged.columns:
        merged = (merged
                  .sort_values(key_column, kind="stable")
                  .drop_duplicates(subset=[key_column], keep="first"))
    else:
        raise KeyError(f"`key_column` '{key_column}' not found in merged columns.")

    # 4) Presence flags per original DF
    for i, df in enumerate(df_list, start=1):
        col = f"additional_column_{i}"
        keys = df[key_column].dropna().unique() if key_column in df.columns else []
        merged[col] = merged[key_column].isin(keys).astype("Int8")  # 0/1, nullable int

    return merged


import re
import pandas as pd

def rename_by_aliases(df: pd.DataFrame,
                      aliases: dict[str, list[str]],
                      *,
                      inplace: bool = False,
                      prefer: str = "max_notna") -> pd.DataFrame:
    """
    Rename columns by your chosen targets using alias lists (case/format agnostic).

    aliases: dict where keys are your desired final names,
             and values are lists of alias headers you want to match.
             Matching is done on a normalized form: lowercase and non-letters stripped.

    prefer: how to pick if multiple columns match the same target
            - "max_notna": pick the column with most non-null values
            - "first": pick the first match encountered

    Example:
    aliases = {
        "SMILES":   ["SMILES", "smiles", "***smiles***", "Smiles"],
        "InChIKey": ["InChIKey", "INCHIKEY", "inchikey", "InChlKey", "InChI Key"]
    }
    """
    def norm(s: str) -> str:
        # remove anything not a letter/number; lowercase
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    work = df if inplace else df.copy()

    # Precompute normalized forms of existing columns
    cols = list(work.columns)
    norm_cols = {c: norm(c) for c in cols}

    # Build mapping: desired_name -> list of matching existing columns
    matches: dict[str, list[str]] = {}
    for desired, alist in aliases.items():
        desired_norm = norm(desired)
        alias_norms = {norm(a) for a in (alist or [])} | {desired_norm}  # include desired itself
        # any column whose normalized form equals any alias normalized form
        found = [c for c in cols if norm_cols[c] in alias_norms]
        if found:
            matches[desired] = found

    # Decide winners per desired name, then assemble rename mapping
    rename_map = {}
    for desired, candidates in matches.items():
        if len(candidates) == 1 or prefer == "first":
            winner = candidates[0]
        else:  # prefer == "max_notna"
            winner = max(candidates, key=lambda c: work[c].notna().sum())
        if winner != desired:
            # Avoid creating duplicate column names by accidental multi-mapping
            if desired in work.columns and winner != desired:
                # If target already exists, skip renaming to avoid duplicate header
                # You can change this policy as needed.
                continue
            rename_map[winner] = desired

    if rename_map:
        work.rename(columns=rename_map, inplace=True)

    return work


# ========================================
# SMILES → properties (InChI, InChIKey, formula, exact mass, adduct m/z, logP)
# ========================================

def annotate_smiles_table(
    df: pd.DataFrame,
    *,
    smiles_col: str = "SMILES",
    # Which properties to compute
    compute_inchi: bool = True,
    compute_inchikey: bool = True,
    compute_formula: bool = True,
    compute_exact_mass: bool = True,
    compute_adduct_columns: bool = True,
    compute_logp: bool = True,
    # How to compute logP
    logp_source: str = "mordred",          # "mordred" | "rdkit"
    # Which adduct columns to emit (name → adduct key)
    adduct_columns: Optional[Dict[str, str]] = None,  # default set below
    # Mass constants
    use_proton_mass: bool = True,            # True → 1.007276466 (recommended for MS)
                                             # False → 1.007825032 (hydrogen atom mass)
    decimals: int = 6,                       # rounding for numeric outputs
    drop_invalid_smiles: bool = True,
    inplace: bool = False,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Annotate a SMILES table with InChI, InChIKey, formula, exact mass, adduct m/z and SLogP.

    Parameters
    ----------
    df : DataFrame containing a SMILES column.
    smiles_col : str
        Column with SMILES strings.
    compute_* : bool
        Toggle which properties to compute.
    logp_source : {"mordred", "rdkit"}
        Where to compute logP from. Falls back to RDKit if Mordred is unavailable.
    adduct_columns : dict[str,str]
        Mapping of output column name → adduct key. If None, uses
        {
           "MolWeight-1H": "[M-H]-",
           "MolWeight+1H": "[M+H]+",
           "MolWeight+Na": "[M+Na]+",
           "MolWeight+K":  "[M+K]+",
        }
    use_proton_mass : bool
        If True, use ±1.007276466 for H+/H−; if False, use 1.007825032 for H atom.
    decimals : int
        Round numeric outputs to this many decimals.
    drop_invalid_smiles : bool
        Drop rows where SMILES could not be parsed.
    inplace : bool
        Modify the original dataframe.
    out_csv : str | None
        If provided, save the annotated table to CSV.

    Returns
    -------
    DataFrame (annotated). If inplace=True, returns the same object reference.
    """
    try:
        from rdkit.Chem import AllChem as Chem
        from rdkit.Chem import Descriptors as RDDesc
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from rdkit.Chem import Crippen
    except Exception as e:  # pragma: no cover
        raise ImportError("RDKit is required for annotate_smiles_table().") from e

    work = df if inplace else df.copy()
    if smiles_col not in work.columns:
        raise KeyError(f"Column '{smiles_col}' not found in df.")

    # Parse SMILES once
    smiles_series = work[smiles_col].astype(str)
    mols = [Chem.MolFromSmiles(s) if pd.notna(s) else None for s in smiles_series]
    valid_mask = [m is not None for m in mols]

    if drop_invalid_smiles:
        work = work.loc[valid_mask].copy()
        mols = [m for m, ok in zip(mols, valid_mask) if ok]

    # Exact (neutral) mass & formula
    if compute_exact_mass:
        neutral_mass = [RDDesc.ExactMolWt(m) if m is not None else np.nan for m in mols]
        work["MolWeight"] = np.round(neutral_mass, decimals)

    if compute_formula:
        work["MolFormula"] = [CalcMolFormula(m) if m is not None else None for m in mols]

    # InChI / InChIKey
    if compute_inchi:
        def _mol_to_inchi(m):
            try:
                return Chem.MolToInchi(m) if m is not None else None
            except Exception:
                return None
        work["Inchi"] = [_mol_to_inchi(m) for m in mols]

    if compute_inchikey:
        def _mol_to_inchikey(m):
            try:
                return Chem.MolToInchiKey(m) if m is not None else None
            except Exception:
                return None
        work["InchiKey"] = [_mol_to_inchikey(m) for m in mols]

    # Adduct m/z columns
    if compute_adduct_columns:
        # Extend ADDUCT_MASS with negative H if needed
        proton = 1.007276466
        hydrogen_atom = 1.007825032
        h_mass = proton if use_proton_mass else hydrogen_atom
        local_adducts = dict(ADDUCT_MASS)
        local_adducts["[M-H]-"] = -h_mass

        if adduct_columns is None:
            adduct_columns = {
                "MolWeight-1H": "[M-H]-",
                "MolWeight+1H": "[M+H]+",
                "MolWeight+Na": "[M+Na]+",
                "MolWeight+K":  "[M+K]+",
            }
        # Ensure we have MolWeight available for neutral mass
        if "MolWeight" not in work.columns:
            neutral_mass = [RDDesc.ExactMolWt(m) if m is not None else np.nan for m in mols]
            work["MolWeight"] = np.round(neutral_mass, decimals)
        base_mass = work["MolWeight"].to_numpy(dtype=float)

        for out_col, adduct_key in adduct_columns.items():
            if adduct_key not in local_adducts:
                raise KeyError(f"Unknown adduct key '{adduct_key}'. Known: {list(local_adducts.keys())}")
            delta = local_adducts[adduct_key]
            work[out_col] = np.round(base_mass + float(delta), decimals)

    # logP
    if compute_logp:
        values = None
        if logp_source.lower() == "mordred":
            try:
                from mordred import Calculator, descriptors
                calc = Calculator(descriptors.SLogP.SLogP)
                df_slogp = calc.pandas(mols)
                # Mordred returns a DataFrame; pick the first column (SLogP)
                values = df_slogp.iloc[:, 0].to_list()
            except Exception:
                # Fallback to RDKit
                values = [Crippen.MolLogP(m) if m is not None else np.nan for m in mols]
        else:
            values = [Crippen.MolLogP(m) if m is not None else np.nan for m in mols]
        work["SLogP"] = np.round(values, decimals)

    # Save
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        work.to_csv(out_csv, index=False)

    return work

# --- Convenience: minimal wrapper that mirrors user's original quick path ---

def annotate_and_save_quick(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    out_csv: str = "annotated_structures.csv",
    *,
    use_proton_mass: bool = True,
    decimals: int = 6,
) -> pd.DataFrame:
    """One-liner to reproduce the user's original behavior with improved accuracy.

    Computes formula, InChI, InChIKey, MolWeight, [M−H]−, [M+H]+, [M+Na]+, [M+K]+ and SLogP,
    and writes to CSV.
    """
    annotated = annotate_smiles_table(
        df,
        smiles_col=smiles_col,
        compute_inchi=True,
        compute_inchikey=True,
        compute_formula=True,
        compute_exact_mass=True,
        compute_adduct_columns=True,
        compute_logp=True,
        logp_source="mordred",
        adduct_columns=None,  # default set inside
        use_proton_mass=use_proton_mass,
        decimals=decimals,
        drop_invalid_smiles=True,
        inplace=False,
        out_csv=out_csv,
    )
    return annotated



# =============================
# MassQL (MS1)
# =============================

def compute_adduct_mzs(mono_mass: float, adducts: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Return dict {adduct: m/z} as floats (format later when emitting query)."""
    if adducts is None:
        adducts = list(ADDUCT_MASS.keys())
    mzs: Dict[str, float] = {}
    for a in adducts:
        if a not in ADDUCT_MASS:
            raise KeyError(f"Unknown adduct '{a}'. Known: {list(ADDUCT_MASS.keys())}")
        z = ADDUCT_CHARGE.get(a, 1)
        mzs[a] = (mono_mass + ADDUCT_MASS[a]) / z
    return mzs

def massql_query_for_compound(
    name: str,
    mono_mass: float,
    adducts: Optional[Sequence[str]] = None,
    ppm: int = 10,
    intensity_percent: int = 1,
    decimals: int = 5,
    separate_adducts: bool = False,
) -> str | Dict[str, str]:
    """Build MassQL query(ies) for one compound (MS1)."""
    if adducts is None:
        adducts = ("[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+2H]+2")

    mzs = compute_adduct_mzs(mono_mass, adducts=adducts)

    if not separate_adducts:
        mz_list = " OR ".join(_fmt(mzs[a], decimals) for a in adducts)
        q = (
            f"# {name}\n"
            "QUERY scaninfo(MS1DATA) WHERE\n"
            f"MS1MZ=(\n\t{mz_list}\n"
            f"):TOLERANCEPPM={ppm}:INTENSITYPERCENT={intensity_percent}"
        )
        return q

    out: Dict[str, str] = {}
    for a in adducts:
        q = (
            f"# {name} {a}\n"
            "QUERY scaninfo(MS1DATA) WHERE\n"
            f"MS1MZ=(\n\t{_fmt(mzs[a], decimals)}\n"
            f"):TOLERANCEPPM={ppm}:INTENSITYPERCENT={intensity_percent}"
        )
        out[a] = q
    return out
    
def massql_query_for_compound2(
    name: str,
    mono_mass: float,
    adducts: Optional[Sequence[str]] = None,
    ppm: int = 10,
    intensity_percent: int = 1,
    decimals: int = 5,
    separate_adducts: bool = False,
) -> str | Dict[str, str]:
    """Build MassQL query(ies) for one compound (MS1)."""
    if adducts is None:
        adducts = ("[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+2H]+2")

    mzs = compute_adduct_mzs(mono_mass, adducts=adducts)

    if not separate_adducts:
        mz_list = " OR ".join(_fmt(mzs[a], decimals) for a in adducts)
        q = (
            f"# {name}\n"
            "QUERY scaninfo(MS2DATA) WHERE\n"
            f"MS2PREC=(\n\t{mz_list}\n"
            f"):TOLERANCEPPM={ppm}:INTENSITYPERCENT={intensity_percent}"
        )
        return q

    out: Dict[str, str] = {}
    for a in adducts:
        q = (
            f"# {name} {a}\n"
            "QUERY scaninfo(MS2DATA) WHERE\n"
            f"MS2PREC=(\n\t{_fmt(mzs[a], decimals)}\n"
            f"):TOLERANCEPPM={ppm}:INTENSITYPERCENT={intensity_percent}"
        )
        out[a] = q
    return out    


def generate_massql_queries(
    df_metadata: pd.DataFrame,
    ppm: int = 10,
    intensity_percent: int = 1,
    decimals: int = 5,
    separate_adducts: bool = False,
    adducts: Optional[Sequence[str]] = None,
    name_col: str = "Compound name",
    mass_col: str = "MolWeight",
) -> Dict[str, str] | Dict[str, Dict[str, str]]:
    """Build MassQL MS1 queries from a metadata dataframe."""
    if adducts is None:
        adducts = ("[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+2H]+2")

    name_col = _resolve_column(df_metadata, name_col)
    mass_col = _resolve_column(df_metadata, mass_col)

    results: Dict[str, str] | Dict[str, Dict[str, str]] = {}
    for _, row in df_metadata.iterrows():
        name = str(row[name_col])
        mono = _to_float(row[mass_col])
        if mono is None:
            continue
        results[name] = massql_query_for_compound(
            name=name,
            mono_mass=mono,
            adducts=adducts,
            ppm=ppm,
            intensity_percent=intensity_percent,
            decimals=decimals,
            separate_adducts=separate_adducts,
        )
    return results

def generate_massqlPrec_queries(
    df_metadata: pd.DataFrame,
    ppm: int = 10,
    intensity_percent: int = 1,
    decimals: int = 5,
    separate_adducts: bool = False,
    adducts: Optional[Sequence[str]] = None,
    name_col: str = "Compound name",
    mass_col: str = "MolWeight",
) -> Dict[str, str] | Dict[str, Dict[str, str]]:
    """Build MassQL MS2 (precursors) queries from a metadata dataframe."""
    if adducts is None:
        adducts = ("[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+2H]+2")

    name_col = _resolve_column(df_metadata, name_col)
    mass_col = _resolve_column(df_metadata, mass_col)

    results: Dict[str, str] | Dict[str, Dict[str, str]] = {}
    for _, row in df_metadata.iterrows():
        name = str(row[name_col])
        mono = _to_float(row[mass_col])
        if mono is None:
            continue
        results[name] = massql_query_for_compound2(
            name=name,
            mono_mass=mono,
            adducts=adducts,
            ppm=ppm,
            intensity_percent=intensity_percent,
            decimals=decimals,
            separate_adducts=separate_adducts,
        )
    return results

# =============================
# MassQL (MS2)
# =============================

def _prec_mzs(mono_mass: float, adducts: Sequence[str], decimals: int) -> List[str]:
    """Return formatted precursor m/z strings for the given adducts, charge-aware."""
    mz_list: List[str] = []
    for a in adducts:
        a_norm = a.strip()
        if a_norm not in ADDUCT_MASS:
            # silently skip unknown adducts
            continue
        z = ADDUCT_CHARGE.get(a_norm, 1)
        mz = (mono_mass + ADDUCT_MASS[a_norm]) / z
        mz_list.append(_fmt(mz, decimals))
    return mz_list

def generate_massql_ms2_queries(
    df_metadata: pd.DataFrame,
    name_col: str = "Compound name",
    mass_col: str = "Monoisotopic mass",  # or "MolWeight"
    fragments_col: str = "Fragments",
    adducts: Optional[Sequence[str]] = None,
    ppm_prec: int = 10,
    ppm_prod: int = 10,
    intensity_percent: int = 5,
    decimals: int = 5,
    cardinality_min: int = 1,
    cardinality_max: int = 5,
    clamp_cardinality: bool = True,
    ignore_zero_fragments: bool = True,
) -> Dict[str, str]:
    """
    Build per-row MassQL MS2 queries.

    If a row has fragment m/z values (semicolon/space/comma separated), include MS2PROD with
    CARDINALITY and INTENSITYPERCENT. Otherwise, restrict only by MS2PREC (precursor list).
    """
    if adducts is None:
        adducts = ("[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+2H]+2")

    name_col = _resolve_column(df_metadata, name_col)
    mass_col = _resolve_column(df_metadata, mass_col)
    fragments_col = _resolve_column(df_metadata, fragments_col)

    results: Dict[str, str] = {}

    for _, row in df_metadata.iterrows():
        name = str(row[name_col]).strip()
        mono = _to_float(row[mass_col])
        if mono is None:
            continue

        precs = _prec_mzs(mono, adducts, decimals)
        if not precs:
            # no valid adducts -> skip row
            continue
        prec_list = " OR ".join(precs)

        # parse fragments
        frags: List[float] = []
        raw = row[fragments_col]
        if pd.notna(raw) and str(raw).strip():
            tokens = re.split(r"[;,\s]+", str(raw))
            seen: set[float] = set()
            for t in tokens:
                v = _to_float(t)
                if v is None:
                    continue
                if ignore_zero_fragments and v <= 0:
                    continue
                v_round = float(_fmt(v, decimals))
                if v_round not in seen:
                    frags.append(v_round)
                    seen.add(v_round)

        if frags:
            n = len(frags)
            cmin = max(1, min(cardinality_min, n) if clamp_cardinality else cardinality_min)
            cmax = max(cmin, min(cardinality_max, n) if clamp_cardinality else cardinality_max)
            prod_list = " OR ".join(_fmt(v, decimals) for v in frags)
            q = (
                f"# {name}\n"
                "QUERY scaninfo(MS2DATA) WHERE\n"
                f"MS2PREC=({prec_list}):TOLERANCEPPM={ppm_prec} AND\n"
                f"MS2PROD=({prod_list}):CARDINALITY=range(min={cmin},max={cmax}):"
                f"TOLERANCEPPM={ppm_prod}:INTENSITYPERCENT={intensity_percent}"
            )
        else:
            q = (
                f"# {name}\n"
                "QUERY scaninfo(MS2DATA) WHERE\n"
                f"MS2PREC=({prec_list}):TOLERANCEPPM={ppm_prec}"
            )

        results[name] = q

    return results


# ========================================
# Morgan similarity + network export
# ========================================

def morgan_similarity_matrix(
    df: pd.DataFrame,
    mol_col: str = "Molecule",
    id_col: str = "InchiKey",
    radius: int = 2,
    nbits: int = 2048,
    metric: str = "dice",  # "dice" or "tanimoto"
    return_fps: bool = False,
):
    """Compute a square Morgan similarity matrix among molecules in df.

    Returns (sim_df, fps?) where sim_df index/columns = kept ids.
    """
    try:
        from rdkit.Chem import AllChem as Chem
        from rdkit import DataStructs
    except Exception as e:  # pragma: no cover
        raise ImportError("RDKit is required for morgan_similarity_matrix().") from e

    mols = df[mol_col].tolist()
    ids = df[id_col].tolist()
    keep_idx: List[int] = []
    fps = []
    for i, m in enumerate(mols):
        if m is None:
            continue
        try:
            fp = Chem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        except Exception:
            continue
        fps.append(fp)
        keep_idx.append(i)

    kept_ids = [ids[i] for i in keep_idx]
    n = len(kept_ids)
    if n == 0:
        raise ValueError("No valid molecules/fingerprints to compare.")

    sim = np.eye(n, dtype=float)
    use_tanimoto = (metric.lower() == "tanimoto")
    for i in range(n):
        if use_tanimoto:
            sims_i = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i:])
        else:
            sims_i = DataStructs.BulkDiceSimilarity(fps[i], fps[i:])
        sim[i, i:] = sims_i
        sim[i:, i] = sims_i

    sim_df = pd.DataFrame(sim, index=kept_ids, columns=kept_ids)
    return (sim_df, fps) if return_fps else sim_df


def build_similarity_network(
    sim_df: pd.DataFrame,
    threshold: float = 0.85,
    project: Optional[str] = None,
    save_csv: bool = True,
    csv_sep: str = ";",
    out_name: Optional[str] = None,
    *,
    metadata_df: Optional[pd.DataFrame] = None,
    id_col: str = "InchiKey",
    identity_col: Optional[str] = "file_path",
    save_isolated_csv: bool = True,
    isolated_out_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, nx.Graph, pd.DataFrame]:
    """Create an edge list from a similarity matrix and return the network and isolated nodes."""
    if not (set(sim_df.index) == set(sim_df.columns) and len(sim_df.index) == len(sim_df.columns)):
        raise ValueError("sim_df must be square with identical row/column labels.")

    sim_df = sim_df.loc[sim_df.index, sim_df.index]

    long_df = sim_df.stack().reset_index()
    long_df.columns = ["SOURCE", "TARGET", "CORRELATION"]

    mask_dups = (
        long_df[["SOURCE", "TARGET"]].apply(frozenset, axis=1).duplicated()
        | (long_df["SOURCE"] == long_df["TARGET"]) 
    )
    links = long_df[~mask_dups]

    links_filtered = links.loc[
        (links["CORRELATION"] > float(threshold)) & (links["SOURCE"] != links["TARGET"])
    ].reset_index(drop=True)

    G = nx.from_pandas_edgelist(links_filtered, "SOURCE", "TARGET")

    universe_ids = list(sim_df.index)
    used_ids = set(links_filtered["SOURCE"]).union(set(links_filtered["TARGET"]))
    isolated_ids = [i for i in universe_ids if i not in used_ids]

    if metadata_df is not None and id_col in metadata_df.columns:
        isolated_df = metadata_df[metadata_df[id_col].isin(isolated_ids)].copy()
        keep_cols = [id_col]
        if identity_col and identity_col in isolated_df.columns:
            keep_cols.append(identity_col)
        isolated_df = isolated_df[keep_cols]
    else:
        isolated_df = pd.DataFrame({id_col: isolated_ids})

    if save_csv:
        if out_name is None:
            project = project or "Project"
            out_name = f"{project}_DB_compounds_Similarity_{threshold}.csv"
        Path(out_name).parent.mkdir(parents=True, exist_ok=True)
        links_filtered.to_csv(out_name, sep=csv_sep, index=False)

    if save_isolated_csv:
        if isolated_out_name is None:
            project = project or "Project"
            isolated_out_name = f"{project}_isolated_nodes.csv"
        Path(isolated_out_name).parent.mkdir(parents=True, exist_ok=True)
        isolated_df.to_csv(isolated_out_name, index=False)

    return links_filtered, G, isolated_df

def add_massdiff_from_metadata(
    links_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    id_col: str = "InchiKey",
    weight_col: str = "MolWeight",
    source_col: str = "SOURCE",
    target_col: str = "TARGET",
    drop_missing_weights: bool = False,
    add_ppm: bool = False
) -> pd.DataFrame:
    """
    Add SOURCE_MW, TARGET_MW, and MassDiff_Da to links_df using weights from metadata_df[weight_col]
    keyed by metadata_df[id_col]. Optionally add MassDiff_ppm.
    """
    # Basic column checks
    for col in (source_col, target_col):
        if col not in links_df.columns:
            raise ValueError(f"links_df must contain '{col}' column.")
    for col in (id_col, weight_col):
        if col not in metadata_df.columns:
            raise ValueError(f"metadata_df must contain '{col}' column.")

    # Build a mapping InchiKey -> MolWeight (first occurrence if duplicates)
    weight_map = (
        metadata_df[[id_col, weight_col]]
        .drop_duplicates(subset=[id_col])
        .set_index(id_col)[weight_col]
    )
    weight_map = pd.to_numeric(weight_map, errors="coerce")  # ensure numeric

    out = links_df.copy()
    out["SOURCE_MW"] = out[source_col].map(weight_map)
    out["TARGET_MW"] = out[target_col].map(weight_map)
    out["MassDiff_Da"] = (out["SOURCE_MW"] - out["TARGET_MW"]).abs()

    if drop_missing_weights:
        out = out.dropna(subset=["SOURCE_MW", "TARGET_MW"]).reset_index(drop=True)

    if add_ppm:
        denom = out[["SOURCE_MW", "TARGET_MW"]].mean(axis=1)
        out["MassDiff_ppm"] = (out["MassDiff_Da"] / denom) * 1e6

    return out

# ========================================
# MZmine custom MS1 database CSV
# ========================================

def make_mzmine_custom_db(
    df: pd.DataFrame,
    *,
    mz_col: str = "MolWeight+1H",     # use "MolWeight-1H" for NEG mode
    formula_col: str = "MolFormula",
    identity_col: str = "file_path",
    default_rt: float = 0.0,
    project: str = "Project",
    mode: str = "POS",                # "POS" or "NEG" (filename suffix only)
    start_id: int = 1,
    save_csv: bool = True,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build an MZmine custom MS1 database table with columns:
       ID, m/z, Retention Time, identity, Formula
    """
    needed = [identity_col, mz_col, formula_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    base = df[[identity_col, mz_col, formula_col]].copy()
    base["Retention Time"] = float(default_rt)
    base["ID"] = range(start_id, start_id + len(base))

    out = base[["ID", mz_col, "Retention Time", identity_col, formula_col]].rename(
        columns={mz_col: "m/z", identity_col: "identity", formula_col: "Formula"}
    )

    if save_csv:
        if out_path is None:
            out_path = f"{project}_MZMine_CustomDB_{mode}.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)

    return out


# ========================================
# Mordred descriptors
# ========================================

def compute_mordred_descriptors(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    ignore_3D: bool = False,
    project: str = "Project",
    save_csv: bool = True,
    out_sep: str = ";",
    out_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Compute Mordred descriptors for a DataFrame of SMILES.

    Returns
    -------
    df_descriptors : pd.DataFrame
    info : dict -> {"n_2D": int, "n_3D": int, "n_total": int}
    """
    try:
        # Lazy imports to avoid hard dependency if user doesn't call this function
        from rdkit.Chem import AllChem as Chem
        from mordred import Calculator, descriptors
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Mordred (and RDKit) are required for compute_mordred_descriptors().\n"
            "Tip: some builds work best with numpy<2."
        ) from e

    # Version hint for numpy
    try:
        if np.__version__.startswith("2."):
            print("Warning: Mordred + NumPy 2.x may fail with some RDKit wheels; consider numpy<2 if issues occur.")
    except Exception:  # pragma: no cover
        pass

    # Count available descriptors (2D/3D/total)
    n_all = len(Calculator(descriptors, ignore_3D=False).descriptors)
    n_2D = len(Calculator(descriptors, ignore_3D=True).descriptors)
    n_3D = n_all - n_2D
    info = {"n_2D": n_2D, "n_3D": n_3D, "n_total": n_all}
    print(f"2D: {n_2D:5}\n3D: {n_3D:5}\n------------\ntotal: {n_all:5}")

    # RDKit mols
    mols = [Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None for smi in df[smiles_col]]

    calc = Calculator(descriptors, ignore_3D=ignore_3D)
    df_descriptors = calc.pandas(mols)

    if save_csv:
        if out_path is None:
            out_path = f"{project}_DB_compounds_MordredDescriptors.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_descriptors.to_csv(out_path, sep=out_sep, index=False)
        print(
            f"Done! Descriptors table saved with {len(df.index)} rows → {out_path}"
        )

    return df_descriptors, info


# ========================================
# Dendrogram + clustering + t-SNE
# ========================================

def dendrogram_and_cluster_descriptors(
    df_descriptors: pd.DataFrame,
    *,
    labels: Optional[Sequence[str]] = None,
    color_threshold: float = 250000.0,
    palette: Optional[Sequence[str]] = (
        '#109c2c', '#d41e29', '#a65628', '#5c5c5c', '#984ea3', '#ff7f00', '#ffff33'
    ),
    figsize: Tuple[int, int] = (10, 40),
    xlim: Tuple[float, float] | None = (0, 700000),
    orientation: str = "right",
    distance_metric: str = "euclidean",
    linkage_method: str = "ward",
    save_dir: str = "images",
    filename: str = "dendrogram.png",
    do_save_png: bool = True,
    cut_distance: Optional[float] = None,
    run_agglomerative: bool = True,
    metadata_df: Optional[pd.DataFrame] = None,
    id_col: str = "InchiKey",
    identity_col: str = "file_path",
    cluster_col: str = "cluster",
    project: Optional[str] = None,
    save_cluster_csv: bool = True,
    cluster_csv_name: Optional[str] = None,
) -> Tuple[pd.Series, np.ndarray, plt.Figure, plt.Axes, pd.DataFrame]:
    """Ward dendrogram from numeric descriptor columns + AgglomerativeClustering mirror."""
    X = df_descriptors.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric columns found in df_descriptors.")

    row_index = X.index
    dists = pdist(X.values, metric=distance_metric)
    Z = linkage(dists, method=linkage_method)

    if palette:
        set_link_color_palette(list(palette))
    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(
        Z,
        orientation=orientation,
        color_threshold=color_threshold,
        distance_sort="ascending",
        show_leaf_counts=True,
        ax=ax,
        labels=(labels if labels is not None else row_index.astype(str).tolist()),
        leaf_font_size=10,
    )
    ax.set_title("Dendrogram")
    ax.set_xlabel("Observations")
    ax.set_ylabel("Distance")
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.tight_layout()

    if do_save_png:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / filename, dpi=300)

    if cut_distance is None:
        cut_distance = float(color_threshold)
    clusters = pd.Series(
        fcluster(Z, t=float(cut_distance), criterion="distance"),
        index=row_index,
        name=cluster_col,
    )

    if run_agglomerative:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(cut_distance),
            metric="euclidean",
            linkage="ward",
        )
        model.fit(X.values)
        aggl_labels = pd.Series(model.labels_, index=row_index, name="Clust")
    else:
        aggl_labels = pd.Series(np.nan, index=row_index, name="Clust")

    if metadata_df is not None:
        meta = metadata_df.copy()
        if identity_col in meta.columns and meta.index.equals(row_index):
            df_clustered = pd.DataFrame({identity_col: meta[identity_col], "Clust": aggl_labels})
        elif (id_col in meta.columns) and (id_col in df_descriptors.columns):
            df_clustered = pd.DataFrame({id_col: df_descriptors[id_col], "Clust": aggl_labels.values})
            if identity_col in meta.columns:
                df_clustered = df_clustered.merge(meta[[id_col, identity_col]], on=id_col, how="left")
        else:
            df_clustered = pd.DataFrame({"Clust": aggl_labels})
    else:
        df_clustered = pd.DataFrame({"Clust": aggl_labels})

    if save_cluster_csv:
        thresh = int(cut_distance) if float(cut_distance).is_integer() else cut_distance
        if cluster_csv_name is None:
            cluster_csv_name = f"df_descriptors_values_dist_threshold_{thresh}.csv"
            if project:
                cluster_csv_name = f"{project}_" + cluster_csv_name
        Path(cluster_csv_name).parent.mkdir(parents=True, exist_ok=True)
        df_clustered.to_csv(cluster_csv_name, index=False)

    # Optional write-back to metadata_df (in-place-like best effort)
    if (metadata_df is not None) and (cluster_col not in metadata_df.columns):
        try:
            if metadata_df.index.equals(clusters.index):
                metadata_df[cluster_col] = clusters.values
            elif (id_col in metadata_df.columns) and (id_col in df_descriptors.columns):
                tmp = pd.DataFrame({id_col: df_descriptors[id_col], cluster_col: clusters.values})
                metadata_df.merge(tmp, on=id_col, how="left")
        except Exception:
            pass

    return clusters, Z, fig, ax, df_clustered


from typing import Optional, Sequence, Tuple, Union, List
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def tsne_projection_plot(
    df_descriptors: pd.DataFrame,
    *,
    metadata_df: Optional[pd.DataFrame] = None,
    cluster_series: Optional[pd.Series] = None,
    hover_cols: Sequence[str] = ("file_path", "cluster"),
    standardize: bool = False,
    n_components: int = 2,
    init: str = "random",
    learning_rate: str | float = "auto",
    perplexity: float = 20.0,
    random_state: int = 0,
    title: str = "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
    color_palette: Sequence[str] = None,  # Plotly picks defaults if None
    save_html: bool = True,
    out_html: str = "images/t-sne.html",
    # --- New highlight options ---
    activities_df: Optional[pd.DataFrame] = None,   # e.g., merged_df2 (index must align or be reindexable)
    activities_col: str = "activities",
    highlight_activity: Optional[Union[str, List[str]]] = None,  # e.g., "antimicrobial", ["cytotoxic","analgesic"], or "any"
    highlight_mode: str = "overlay",  # "overlay" (triangles added as a new trace). ("symbol" available if you prefer single-trace)
    base_marker_size: int = 7,
    highlight_size: int = 11,
) -> Tuple[pd.DataFrame, "plotly.graph_objs.Figure"]:
    """
    Compute a 2D t-SNE projection from descriptor table and make an interactive Plotly scatter.

    Highlighting:
      - Provide activities_df with a column `activities_col` (default "activities").
      - Set highlight_activity to a string (case-insensitive, matches tokens split by ';'),
        a list of strings, or "any" to highlight any non-empty activities.
      - Highlighted points appear as triangle-up markers (overlay trace) with larger size.
    """
    try:
        import plotly.express as px
        import plotly.io as pio
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover
        raise ImportError("Plotly is required for tsne_projection_plot().") from e

    # --- Prepare data ---
    X = df_descriptors.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric columns found in df_descriptors.")
    n_samples = len(X)

    X_use = X.values
    if standardize:
        X_use = StandardScaler().fit_transform(X_use)

    if n_samples <= 3:
        raise ValueError("t-SNE requires more than 3 samples.")
    perp = min(float(perplexity), max(5.0, (n_samples - 1) / 3.0))

    tsne = TSNE(
        n_components=n_components,
        learning_rate=learning_rate,
        random_state=random_state,
        init=init,
        perplexity=perp,
    )
    Y = tsne.fit_transform(X_use)
    proj = pd.DataFrame(Y, index=X.index, columns=[f"TSNE{i+1}" for i in range(n_components)])

    # --- Color & hover data ---
    color_data = None
    if cluster_series is not None:
        color_data = cluster_series.reindex(proj.index) if not cluster_series.index.equals(proj.index) else cluster_series
    elif (metadata_df is not None) and ("cluster" in metadata_df.columns):
        try:
            color_data = metadata_df.loc[proj.index, "cluster"]
        except Exception:
            color_data = None

    hover_data = {}
    if metadata_df is not None:
        for col in hover_cols:
            if col in metadata_df.columns:
                s = metadata_df[col]
                hover_data[col] = s.reindex(proj.index) if not s.index.equals(proj.index) else s

    # Include activities in hover, if present
    activities_series = None
    if activities_df is not None and activities_col in activities_df.columns:
        s = activities_df[activities_col]
        activities_series = s.reindex(proj.index) if not s.index.equals(proj.index) else s
        hover_data[activities_col] = activities_series

    # --- Base scatter (circles) ---
    fig = px.scatter(
        proj,
        x="TSNE1",
        y="TSNE2",
        color=color_data if color_data is not None else None,
        color_discrete_sequence=(color_palette if color_palette is not None else px.colors.sequential.Turbo),
        hover_data=hover_data if hover_data else None,
        title=title,
    )
    # Normalize base marker size
    fig.update_traces(marker=dict(size=base_marker_size), selector=dict(mode="markers"))

    # --- Highlight logic (overlay triangles) ---
    def _tokenize(val: str) -> list[str]:
        # split on ';' and commas, trim whitespace
        return [t.strip() for t in str(val).replace(",", ";").split(";") if t.strip()]

    if activities_series is not None and highlight_activity is not None:
        # Build boolean mask for highlighting
        if isinstance(highlight_activity, str) and highlight_activity.lower() == "any":
            mask = activities_series.fillna("").astype(str).str.strip().ne("")
        else:
            # allow a single string or list of strings; case-insensitive token match
            targets = [highlight_activity] if isinstance(highlight_activity, str) else list(highlight_activity)
            targets = [str(t).lower().strip() for t in targets if str(t).strip()]
            def match_tokens(cell):
                if pd.isna(cell) or str(cell).strip() == "":
                    return False
                toks = [t.lower() for t in _tokenize(cell)]
                # match if any target is contained as full token or substring in any token
                return any(any(targ in tok for tok in toks) for targ in targets)
            mask = activities_series.apply(match_tokens)

        if mask.any():
            hi_idx = proj.index[mask]
            # Build overlay trace only for highlighted points
            fig.add_trace(
                go.Scattergl(
                    x=proj.loc[hi_idx, "TSNE1"],
                    y=proj.loc[hi_idx, "TSNE2"],
                    mode="markers",
                    name=f"Highlighted: {highlight_activity}",
                    marker=dict(symbol="triangle-up", size=highlight_size, line=dict(width=1)),
                    hovertemplate=(
                        "TSNE1=%{x:.3f}<br>"
                        "TSNE2=%{y:.3f}<br>"
                        + "<br>".join([f"{k}=%{{customdata[{i}]}}" for i, k in enumerate(hover_data.keys())])
                        if hover_data else "TSNE1=%{x:.3f}<br>TSNE2=%{y:.3f}"
                    ),
                    customdata=np.column_stack([v.reindex(hi_idx).values for v in hover_data.values()]) if hover_data else None,
                    showlegend=True,
                )
            )
        else:
            print("ℹ️ No points matched the requested activity filter; no highlight overlay added.")

    if save_html:
        Path(os.path.dirname(out_html) or ".").mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, out_html)

    return proj, fig



# ========================================
# WordCloud of activity column
# ========================================


from collections import Counter
from typing import Iterable, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def wordcloud_from_semicolon_column(
    df: pd.DataFrame,
    column: str = "activities",
    *,
    delimiter: str = ";",
    normalize_case: bool = True,
    stopwords: Optional[Iterable[str]] = None,
    min_count: int = 1,
    max_words: int = 500,
    width: int = 1400,
    height: int = 900,
    background_color: str = "white",
    colormap: Optional[str] = None,           # e.g. "viridis", "plasma", etc.
    collocations: bool = False,               # keep False to avoid auto bigrams
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[WordCloud, Dict[str, int]]:
    """
    Create a word cloud from a DataFrame column whose cells may contain multiple
    strings separated by a delimiter (default: ';').

    Returns (wordcloud_object, frequency_dict).
    """

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    # Collect tokens
    series = df[column].dropna().astype(str)
    tokens = []
    for cell in series:
        for raw in cell.split(delimiter):
            token = raw.strip()
            if not token:
                continue
            if normalize_case:
                token = token.lower()
            tokens.append(token)

    # Apply stopwords (if any)
    if stopwords is not None:
        stop_set = {s.lower() if normalize_case else s for s in stopwords}
        tokens = [t for t in tokens if t not in stop_set]

    # Count frequencies and enforce min_count
    freqs = Counter(tokens)
    if min_count > 1:
        freqs = {k: v for k, v in freqs.items() if v >= min_count}
    else:
        freqs = dict(freqs)

    if not freqs:
        raise ValueError(
            "No tokens found after processing. "
            "Check the column name, delimiter, stopwords/min_count, or data."
        )

    # Build the word cloud from frequencies (stable and exact)
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        collocations=collocations,
        prefer_horizontal=0.9,
        normalize_plurals=False,
        regexp=None,  # keep full phrases as tokens (we already split by delimiter)
    ).generate_from_frequencies(freqs)

    # Save and/or show
    if save_path:
        wc.to_file(save_path)

    if show:
        dpi = 100
        plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return wc, freqs


from collections import Counter
from typing import Iterable, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt


def plot_top_terms_from_column(
    df: pd.DataFrame,
    column: str = "activities",
    *,
    delimiter: str = ";",
    normalize_case: bool = True,
    stopwords: Optional[Iterable[str]] = None,
    min_count: int = 1,
    top_n: int = 25,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Create a Top-N bar chart of term frequencies from a DataFrame column whose
    cells may contain multiple entries separated by a delimiter (default: ';').

    Returns (freq_df, fig, ax) where freq_df has columns ['term', 'count'].
    """

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    # --- tokenize
    series = df[column].dropna().astype(str)
    tokens = []
    for cell in series:
        for raw in cell.split(delimiter):
            tok = raw.strip()
            if not tok:
                continue
            if normalize_case:
                tok = tok.lower()
            tokens.append(tok)

    # --- stopwords & counts
    if stopwords is not None:
        stop_set = {s.lower() if normalize_case else s for s in stopwords}
        tokens = [t for t in tokens if t not in stop_set]

    counts = Counter(tokens)
    if min_count > 1:
        counts = {k: v for k, v in counts.items() if v >= min_count}

    if not counts:
        raise ValueError(
            "No tokens to plot after processing. "
            "Check column values, delimiter, stopwords/min_count."
        )

    # --- build frequency DataFrame
    freq_df = (
        pd.DataFrame(list(counts.items()), columns=["term", "count"])
        .sort_values("count", ascending=False, kind="stable")
        .head(top_n)
        .reset_index(drop=True)
    )

    # --- plot (horizontal bars; ascending order for nice stacking)
    plot_df = freq_df.sort_values("count", ascending=True, kind="stable")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(plot_df))))
    ax.barh(plot_df["term"], plot_df["count"])   # no explicit colors
    ax.set_xlabel("Count")
    #ax.set_ylabel("Term")
    ax.set_title(f"Top {len(plot_df)} terms in '{column}'")
    for i, v in enumerate(plot_df["count"].to_numpy()):
        ax.text(v, i, f" {v}", va="center")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return freq_df, fig, ax


# =========================
# MGF SURVEY
# =========================
# Standard library
import os
import re
import itertools
from collections import OrderedDict as OD
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    Union,
)

# Third-party libraries
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Mass spectrometry I/O
from pyteomics import mgf as mgf_reader

# Statistics / ML
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances


# =========================
# Safe MGF loading
#    - uses pyteomics.mgf
#    - standardizes batch names: basename without extension
# =========================
def load_mgf_spectra(directory_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Safely load all .mgf files from a directory using pyteomics.mgf without indexing (use_index=False),
    which avoids the 'empty index' warning on some files. The batch name is standardized as the file
    basename without extension.

    Parameters
    ----------
    directory_path : str
        Path to the folder containing .mgf files.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        A mapping: {batch_name -> list of spectra}, where each spectrum is a dict with:
        - "params": header parameters
        - "m/z array": list[float]
        - "intensity array": list[float]
    """
    spectra: Dict[str, List[Dict[str, Any]]] = {}
    for fn in os.listdir(directory_path):
        if not fn.lower().endswith(".mgf"):
            continue
        batch = os.path.splitext(fn)[0]
        spectra[batch] = []
        path = os.path.join(directory_path, fn)

        # No indexing to avoid empty-index warnings
        for spec in mgf_reader.read(path, use_index=False, convert_arrays=True):
            mz = spec.get("m/z array", [])
            ii = spec.get("intensity array", [])
            params = spec.get("params", {})
            spectra[batch].append({
                "params": params,
                "m/z array": list(map(float, mz)) if len(mz) else [],
                "intensity array": list(map(float, ii)) if len(ii) else [],
            })
    return spectra

import pandas as pd

def _normalize_inchikey_value(x: str | None) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.upper()
    # insert dashes if missing and length is 27 (14-10-1)
    if "-" not in s and len(s) == 27:
        s = f"{s[:14]}-{s[14:24]}-{s[24:]}"
    return s

def _ensure_inchikey_column(df: pd.DataFrame, preferred: str = "InchiKey") -> pd.DataFrame:
    """
    Coalesce any InChIKey-like columns into a single 'InchiKey' column (Series),
    then normalize values.
    """
    df = df.copy()
    aliases = ["InchiKey", "InChIKey", "inchi_key", "inchi-key"]
    hits = [c for c in df.columns if c.lower() in {a.lower() for a in aliases}]
    if not hits:
        raise KeyError("No InChIKey-like column found. Expected one of: InchiKey, InChIKey, inchi_key, inchi-key")

    # Take first non-empty value across all aliases
    subset = df[hits].astype("string")
    subset = subset.replace({r"^\s*$": pd.NA, "nan": pd.NA, "NaN": pd.NA}, regex=True)
    s = subset.bfill(axis=1).iloc[:, 0]  # Series

    # Normalize values (upper, dash pattern)
    s = s.map(_normalize_inchikey_value)

    # Drop old columns and set the preferred one
    df = df.drop(columns=hits, errors="ignore")
    df[preferred] = s
    return df


def _normalize_inchikey_value(x: str | None) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.upper()
    # insert dashes if missing and length is 27 (14-10-1)
    if "-" not in s and len(s) == 27:
        s = f"{s[:14]}-{s[14:24]}-{s[24:]}"
    return s

def _ensure_inchikey_column(df: pd.DataFrame, preferred: str = "InchiKey") -> pd.DataFrame:
    """
    Coalesce any InChIKey-like columns into a single 'InchiKey' column (Series),
    then normalize values.
    """
    df = df.copy()
    aliases = ["InchiKey", "InChIKey", "inchi_key", "inchi-key"]
    hits = [c for c in df.columns if c.lower() in {a.lower() for a in aliases}]
    if not hits:
        raise KeyError("No InChIKey-like column found. Expected one of: InchiKey, InChIKey, inchi_key, inchi-key")

    # Take first non-empty value across all aliases
    subset = df[hits].astype("string")
    subset = subset.replace({r"^\s*$": pd.NA, "nan": pd.NA, "NaN": pd.NA}, regex=True)
    s = subset.bfill(axis=1).iloc[:, 0]  # Series

    # Normalize values (upper, dash pattern)
    s = s.map(_normalize_inchikey_value)

    # Drop old columns and set the preferred one
    df = df.drop(columns=hits, errors="ignore")
    df[preferred] = s
    return df

# --- helpers ---------------------------------------------------------------
def _normalize_inchikey_value(x: str | None) -> str | None:
    """Uppercase, strip, and insert dashes if missing (27 chars total)."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.upper()
    if "-" not in s and len(s) == 27:
        s = f"{s[:14]}-{s[14:24]}-{s[24:]}"
    return s

def _ensure_inchikey_column(df: pd.DataFrame, preferred: str = "InchiKey") -> pd.DataFrame:
    """
    Find any InChIKey-like column and normalize it into a single column named `preferred`.
    Accepted aliases: InchiKey, InChIKey, inchi_key, inchi-key.
    """
    df = df.copy()
    aliases = {"InchiKey", "InChIKey", "inchi_key", "inchi-key"}
    hit = None
    for c in df.columns:
        if c.lower() in {a.lower() for a in aliases}:
            hit = c
            break
    if hit is None:
        raise KeyError("No InChIKey-like column found. Expected one of: InchiKey, InChIKey, inchi_key, inchi-key")
    if hit != preferred:
        df = df.rename(columns={hit: preferred})
    df[preferred] = df[preferred].map(_normalize_inchikey_value)
    return df

def _coalesce_smiles_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a single 'SMILES' column:
      - If both 'SMILES' and 'SMILES_nodes' exist, fill nulls in left with right and drop the suffix.
      - If only 'SMILES_nodes' exists, rename it to 'SMILES'.
    """
    df = df.copy()
    if "SMILES" in df.columns and "SMILES_nodes" in df.columns:
        df["SMILES"] = df["SMILES"].astype("string").fillna(df["SMILES_nodes"])
        df = df.drop(columns=["SMILES_nodes"])
    elif "SMILES_nodes" in df.columns and "SMILES" not in df.columns:
        df = df.rename(columns={"SMILES_nodes": "SMILES"})
    return df

# --- main merge ------------------------------------------------------------
def merge_on_inchikey(left: pd.DataFrame, right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    """
    Merge two DataFrames on the column 'InchiKey' (this exact casing).
    - Normalizes/renames any InChIKey-like columns to 'InchiKey'
    - Coalesces SMILES into a single column
    - Ensures only one 'InchiKey' column remains
    """
    L = _ensure_inchikey_column(left, preferred="InchiKey")
    R = _ensure_inchikey_column(right, preferred="InchiKey")

    out = pd.merge(L, R, on="InchiKey", how=how, suffixes=("", "_nodes"))

    # Remove any accidental duplicate key columns (defensive)
    dups = [c for c in out.columns if c.lower() in {"inchikey_nodes", "inchikey_x", "inchikey_y"}]
    if dups:
        out = out.drop(columns=dups)

    # SMILES coalesce
    out = _coalesce_smiles_column(out)

    return out

# ---- helpers ---------------------------------------------------------------
def _first_param(params, keys):
    for k in keys:
        if k in params:
            v = params[k]
            if isinstance(v, (list, tuple)):
                return v[0] if v else None
            return v
    return None

def get_precursor_mz(params: dict) -> float | None:
    """Extract precursor m/z from PEPMASS (or fallbacks)."""
    # Try PEPMASS first (pyteomics often keeps the original case)
    for key in list(params.keys()):
        if key.lower() == "pepmass":
            v = params[key]
            # v can be: tuple/list (mz, intensity) OR string "mz intensity" OR a single float
            if isinstance(v, (tuple, list)):
                try: return float(v[0])
                except Exception: pass
            elif isinstance(v, str):
                toks = v.replace(",", " ").split()
                if toks:
                    try: return float(toks[0])
                    except Exception: pass
            else:
                try: return float(v)
                except Exception: pass

    # Fallback field names seen in the wild
    for alt in ("precursor_mz", "precursorMz", "parentmass", "ParentMass", "ms2precursor", "MS2Precursor"):
        if alt in params:
            v = params[alt]
            if isinstance(v, (tuple, list)):
                try: return float(v[0])
                except Exception: pass
            elif isinstance(v, str):
                toks = v.replace(",", " ").split()
                if toks:
                    try: return float(toks[0])
                    except Exception: pass
            else:
                try: return float(v)
                except Exception: pass
    return None

def extract_scans_fields(params: dict) -> tuple[str | None, int | None]:
    """Return ('scans' as-is, scan_number as int if parseable)."""
    scans_val = _first_param(params, ["SCANS","scans","scan","SCAN","scan_number","SCAN_NUMBER","FEATURE_ID","feature_id"])
    scan_number = None
    if scans_val is not None:
        s = str(scans_val).strip()
        m = re.search(r"\d+", s)
        if m:
            try:
                scan_number = int(m.group(0))
            except Exception:
                scan_number = None
    return scans_val, scan_number

def select_fragments(
    mzs: list[float] | np.ndarray,
    intens: list[float] | np.ndarray,
    *,
    top_n: int = 6,
    min_rel_pct: float = 1.0,
    mz_decimals: int = 4,
    rel_decimals: int = 1,
) -> str:
    """Return semicolon-separated 'mz:rel%' for peaks ≥ min_rel_pct, keeping top_n by rel intensity."""
    if mzs is None or intens is None:
        return ""
    if len(mzs) == 0 or len(intens) == 0 or len(mzs) != len(intens):
        return ""

    arr = np.column_stack([np.asarray(mzs, float), np.asarray(intens, float)])
    if arr.size == 0:
        return ""

    base = arr[:, 1].max()
    if not np.isfinite(base) or base <= 0:
        return ""

    rel = (arr[:, 1] / base) * 100.0
    keep = rel >= float(min_rel_pct)
    if not np.any(keep):
        return ""

    kept = np.column_stack([arr[keep, 0], rel[keep]])
    # sort by rel desc, tie-break by m/z asc
    order = np.lexsort((kept[:, 0], -kept[:, 1]))
    kept = kept[order][:top_n]

    parts = [f"{round(mz, mz_decimals)}:{round(r, rel_decimals)}" for mz, r in kept]
    return ";".join(parts)

# ---- main function ---------------------------------------------------------
def spectra_to_dataframe(
    spectra_by_batch: dict[str, list[dict]],
    *,
    top_n: int = 6,
    min_rel_pct: float = 1.0,
) -> pd.DataFrame:
    """
    Build a DataFrame from spectra with:
      - 'batch', 'scans', 'scan_number'
      - 'precursor_mass'
      - 'fragments' (top_n peaks with ≥ min_rel_pct, formatted 'mz:rel%')
      - 'n_fragments'
    """
    rows = []
    for batch, spectra in spectra_by_batch.items():
        for spec in spectra:
            params = spec.get("params", {}) or {}
            mzs = spec.get("m/z array") or []
            intens = spec.get("intensity array") or []

            scans, scan_number = extract_scans_fields(params)
            precursor = get_precursor_mz(params)
            frag_str = select_fragments(mzs, intens, top_n=top_n, min_rel_pct=min_rel_pct)

            rows.append({
                "batch": batch,
                "scans": scans,
                "scan_number": scan_number,
                "precursor_mass": precursor,
                "fragments": frag_str,
                "n_fragments": frag_str.count(";") + 1 if frag_str else 0,
            })
    return pd.DataFrame(rows)



from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

def graphml_to_dataframe(
    filename: str | Path,
    *,
    resources_dir: str | Path | None = None,
    cast_numeric: bool = True,
    write_csv: bool = False,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    GraphML → DataFrame (nodes). Accepts absolute path or just a filename.
    If only a filename is given, searches in resources_dir (default: ../resources),
    first directly, then recursively with rglob.
    """
    # --- resolve path robustly ---
    fn = Path(filename)
    if fn.is_absolute() and fn.exists():
        path = fn
    else:
        base = Path(resources_dir) if resources_dir else (Path.cwd().parent / "resources")
        direct = (base / fn.name)
        if direct.exists():
            path = direct
        else:
            matches = list(base.rglob(fn.name))  # recursive search under resources_dir
            if matches:
                path = matches[0]
            else:
                raise FileNotFoundError(
                    "GraphML not found.\n"
                    f"Tried:\n - {direct}\n"
                    f"And recursive under: {base}\n"
                    "Tip: pass an absolute path or fix RESOURCES_DIR."
                )

    # --- mapping d# → columns ---
    extract_keys = {
        "d0":"d0","d1":"d1","d2":"d2","d3":"d3","d4":"d4","d5":"d5",
        "d6":"SMILES","d7":"InChI",
        "d8":"d8","d9":"d9","d10":"d10","d11":"d11","d12":"d12","d13":"d13",
    }

    rows = []
    for event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag.endswith("node"):
            row = {"id": elem.get("id")}
            for data_el in elem:
                if not data_el.tag.endswith("data"):
                    continue
                k = data_el.get("key")
                if k in extract_keys:
                    txt = (data_el.text or "").strip() if data_el.text else ""
                    val = np.nan if txt == "" or txt.lower() == "nan" else txt
                    row[extract_keys[k]] = val
            rows.append(row)
            elem.clear()

    df = pd.DataFrame(rows)
    # nice order
    desired = ["id","d0","d1","d2","d3","d4","d5","SMILES","InChI","d8","d9","d10","d11","d12","d13"]
    df = df[[c for c in desired if c in df.columns]]

    if cast_numeric:
        for c in ["d0","d1","d2","d3","d4","d5","d8","d9","d10","d11","d12","d13"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    if write_csv:
        out_csv = Path(out_csv) if out_csv else path.with_name(path.stem + "_nodes_extracted.csv")
        df.to_csv(out_csv, index=False)

    return df

# Requires RDKit with InChI support
from rdkit import Chem
from rdkit.Chem import inchi
import pandas as pd

def add_inchikey_from_structures(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    inchi_col: str = "InChI",
    out_col: str = "InchiKey",
) -> pd.DataFrame:
    df = df.copy()
    if out_col not in df.columns:
        df[out_col] = pd.NA

    def key_from_smiles(s):
        if not isinstance(s, str) or not s.strip():
            return None
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        try:
            return inchi.MolToInchiKey(m)
        except Exception:
            return None

    def key_from_inchi(s):
        if not isinstance(s, str) or not s.strip():
            return None
        try:
            return inchi.InchiToInchiKey(s)
        except Exception:
            return None

    # 1) If you have InChI, use it first (fast & canonical)
    if inchi_col in df.columns:
        mask = df[out_col].isna()
        df.loc[mask, out_col] = df.loc[mask, inchi_col].apply(key_from_inchi)

    # 2) Fill remaining via SMILES
    if smiles_col in df.columns:
        mask = df[out_col].isna()
        df.loc[mask, out_col] = df.loc[mask, smiles_col].apply(key_from_smiles)

    # Normalize casing
    df[out_col] = df[out_col].astype("string").str.strip().str.upper()

    return df

import pandas as pd

# --- reuse the same normalizer you already used ---
def _normalize_inchikey_value(x: str | None) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.upper()
    if "-" not in s and len(s) == 27:  # insert dashes if compact form
        s = f"{s[:14]}-{s[14:24]}-{s[24:]}"
    return s

def _ensure_inchikey_column(df: pd.DataFrame, preferred: str = "InchiKey") -> pd.DataFrame:
    df = df.copy()
    aliases = ["InchiKey", "InChIKey", "inchi_key", "inchi-key"]
    hits = [c for c in df.columns if c.lower() in {a.lower() for a in aliases}]
    if not hits:
        raise KeyError("No InChIKey-like column found. Expected one of: InchiKey, InChIKey, inchi_key, inchi-key")
    subset = df[hits].astype("string").replace({r"^\s*$": pd.NA, "nan": pd.NA, "NaN": pd.NA}, regex=True)
    s = subset.bfill(axis=1).iloc[:, 0].map(_normalize_inchikey_value)  # first non-null across aliases
    df = df.drop(columns=hits, errors="ignore")
    df[preferred] = s
    return df

def inchikey_match_report(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str = "merged_df",
    right_name: str = "df_nodes",
    show_examples: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    # --- internal helpers (self-contained) ---
    def _normalize_inchikey_value(x):
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.lower() == "nan":
            return None
        s = s.upper()
        if "-" not in s and len(s) == 27:
            s = f"{s[:14]}-{s[14:24]}-{s[24:]}"
        return s

    def _ensure_inchikey_column(df: pd.DataFrame, preferred: str = "InchiKey") -> pd.DataFrame:
        df = df.copy()
        aliases = ["InchiKey", "InChIKey", "inchi_key", "inchi-key"]
        hits = [c for c in df.columns if c.lower() in {a.lower() for a in aliases}]
        if not hits:
            raise KeyError("No InChIKey-like column found. Expected one of: InchiKey, InChIKey, inchi_key, inchi-key")
        subset = df[hits].astype("string").replace({r"^\s*$": pd.NA, "nan": pd.NA, "NaN": pd.NA}, regex=True)
        s = subset.bfill(axis=1).iloc[:, 0].map(_normalize_inchikey_value)
        df = df.drop(columns=hits, errors="ignore")
        df["InchiKey"] = s
        return df

    # --- normalize keys on both sides ---
    L = _ensure_inchikey_column(left, preferred="InchiKey")
    R = _ensure_inchikey_column(right, preferred="InchiKey")

    lkeys = L["InchiKey"].dropna()
    rkeys = R["InchiKey"].dropna()
    lset, rset = set(lkeys), set(rkeys)
    inter = lset & rset
    union = lset | rset

    # row-level matches in left (this equals “entries merged” for a left join)
    row_match_mask = L["InchiKey"].isin(rset)
    n_row_matches = int(row_match_mask.sum())

    # duplicates per side
    ldups = lkeys.value_counts()
    ldups = ldups[ldups > 1]
    rdups = rkeys.value_counts()
    rdups = rdups[rdups > 1]

    summary = pd.DataFrame({
        "metric": [
            f"{left_name} rows",
            f"{right_name} rows",
            f"{left_name} unique InchiKey",
            f"{right_name} unique InchiKey",
            "unique matches (intersection)",
            "unique union",
            f"row-level matches in {left_name}",
            f"entries merged (left join {left_name}→{right_name})",   # <- added
            f"unique match rate vs {left_name}",
            f"unique match rate vs {right_name}",
            f"row-level match rate in {left_name}",
            f"entries merged rate in {left_name}",                    # <- added
            f"{left_name} duplicated InchiKey (keys)",
            f"{right_name} duplicated InchiKey (keys)",
        ],
        "value": [
            len(L),
            len(R),
            len(lset),
            len(rset),
            len(inter),
            len(union),
            n_row_matches,
            n_row_matches,  # same as row-level matches, explicit “merged” count
            (len(inter) / len(lset)) if lset else 0.0,
            (len(inter) / len(rset)) if rset else 0.0,
            (n_row_matches / len(L)) if len(L) else 0.0,
            (n_row_matches / len(L)) if len(L) else 0.0,  # explicit merged rate
            int(ldups.shape[0]),
            int(rdups.shape[0]),
        ],
    })

    sample_missing_in_left  = sorted(list(lset - rset))[:show_examples]
    sample_missing_in_right = sorted(list(rset - lset))[:show_examples]

    return summary, ldups, rdups, sample_missing_in_left, sample_missing_in_right

import re
import pandas as pd
import numpy as np

def merge_nodes_with_mgf(
    merged_df_plus_nodes: pd.DataFrame,
    mgfdata_df: pd.DataFrame,
    *,
    dedupe_mgf: bool = True,
    prefer_mgf_fragments: bool = True,
    drop_nan_fragments: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Merge node table (with 'id') and MGF table (with 'scan_number').

    Steps:
      1) Normalize join keys:
         - nodes:  from 'id'        → string digits (e.g., "94850")
         - mgf:    from 'scan_number' → string digits
      2) Optional dedup of mgf rows with same scan_number:
         keep highest n_fragments, tie-break by higher precursor_mass.
      3) Left-join nodes ← mgf   on normalized keys.
      4) Coalesce Fragments (prefer MGF if prefer_mgf_fragments=True).
      5) Optional: drop rows where final Fragments is NaN/empty.
      6) Return (merged_df, summary_dict).

    Returns
    -------
    merged : pd.DataFrame
    summary : dict
        {
          'left_rows', 'right_rows',
          'unique_ids', 'unique_scans',
          'matched_rows', 'match_rate_rows',
          'unmatched_id_examples': [... up to 5 keys ...]
        }
    """

    def _norm_id_to_str(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        m = re.search(r"\d+", s)
        return m.group(0) if m else pd.NA

    def _norm_scan_to_str(x):
        if pd.isna(x):
            return pd.NA
        try:
            return str(int(x))
        except Exception:
            s = str(x).strip()
            m = re.search(r"\d+", s)
            return m.group(0) if m else pd.NA

    nodes = merged_df_plus_nodes.copy()
    mgf   = mgfdata_df.copy()

    # --- build normalized keys ---
    nodes["_id_key"]  = nodes["id"].apply(_norm_id_to_str)
    mgf["_scan_key"]  = mgf["scan_number"].apply(_norm_scan_to_str)

    # --- optional dedupe on mgf by scan key ---
    if dedupe_mgf and mgf["_scan_key"].duplicated().any():
        # Replace NaNs for sorting
        mgf["n_fragments"]   = pd.to_numeric(mgf.get("n_fragments"), errors="coerce").fillna(0)
        mgf["precursor_mass"] = pd.to_numeric(mgf.get("precursor_mass"), errors="coerce").fillna(-np.inf)
        mgf = (mgf
               .sort_values(["_scan_key", "n_fragments", "precursor_mass"],
                            ascending=[True, False, False])
               .drop_duplicates(subset=["_scan_key"], keep="first"))

    # --- merge ---
    merged = pd.merge(
        nodes,
        mgf,
        left_on="_id_key",
        right_on="_scan_key",
        how="left",
        suffixes=("", "_mgf")
    )

    # --- coalesce Fragments (prefer mgf if requested) ---
    # Normalize empties to NA before coalescing
    if "Fragments" in merged.columns:
        merged["Fragments"] = merged["Fragments"].replace("", pd.NA)
    if "Fragments_mgf" in merged.columns:
        merged["Fragments_mgf"] = merged["Fragments_mgf"].replace("", pd.NA)

    if "Fragments_mgf" in merged.columns:
        if prefer_mgf_fragments:
            merged["Fragments"] = merged["Fragments_mgf"].combine_first(merged.get("Fragments"))
        else:
            merged["Fragments"] = merged.get("Fragments").combine_first(merged["Fragments_mgf"])
        merged = merged.drop(columns=["Fragments_mgf"])

    # --- optional: drop rows with missing/empty Fragments after coalescing ---
    if drop_nan_fragments and "Fragments" in merged.columns:
        merged = merged[merged["Fragments"].notna()].copy()

    # --- cleanup helper keys ---
    merged = merged.drop(columns=["_id_key", "_scan_key"], errors="ignore")

    # --- summary ---
    left_rows  = len(nodes)
    right_rows = len(mgfdata_df)
    unique_ids = nodes["_id_key"].dropna().nunique()
    unique_scans = mgf["_scan_key"].dropna().nunique()
    matched_rows = merged["scan_number"].notna().sum()  # row-level match on left
    match_rate_rows = (matched_rows / left_rows) if left_rows else 0.0

    # Sample unmatched id keys (up to 5)
    left_keys = set(nodes["_id_key"].dropna())
    right_keys = set(mgf["_scan_key"].dropna())
    unmatched_id_examples = sorted(list(left_keys - right_keys))[:5]

    summary = dict(
        left_rows=left_rows,
        right_rows=right_rows,
        unique_ids=unique_ids,
        unique_scans=unique_scans,
        matched_rows=int(matched_rows),
        match_rate_rows=float(match_rate_rows),
        unmatched_id_examples=unmatched_id_examples,
    )

    return merged, summary