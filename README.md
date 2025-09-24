# DBsimilarity_v3


This repository provides a modular Python environment for chemical and metabolomic data analysis, built around RDKit and Mordred.
It includes tools for:

Molecule handling: Import SMILES/SDF/CSV, generate molecular fingerprints, calculate descriptors, and compute molecular formulas.

Similarity analysis: Tanimoto/cosine similarity, clustering, and chemotaxonomic indexing.

Dimensionality reduction: PCA, t-SNE, and interactive scatter plots with Plotly.

Visualization: Publication-ready interactive figures (2D and 3D projections, heatmaps, radar charts).

Custom workflows: Designed for metabolomics, natural products, and chemical space exploration, with easy integration of metadata.

The codebase is optimized for Jupyter Notebooks but can also be run as standalone scripts.
It aims to serve researchers working with natural products, secondary metabolites, or large chemical libraries, offering a reproducible and flexible pipeline for data-driven discovery.



# Quick usage guide (DBsimilarity)

## 1) Prepare your inputs

* **Target table** (CSV/TSV): must have at least
  `["SMILES", Compound index]`. e.g.: Compound index = "Compound name"
* (Optional) **Reference table** with same/similar columns to cross-check overlaps.
* Keep files in the project folder (same place as the notebook/script).

## 2) Typical workflows (pick what you need)

### A) Annotate structures (formula, exact mass, adduct m/z, InChI/Key, logP)

```python
import pandas as pd
import dbsimilarity_2 as dbs

df = pd.read_csv("target_compounds.csv", sep=";")  # or "," as needed
annot = dbs.annotate_smiles_table(
    df, smiles_col="SMILES",
    compute_inchi=True, compute_inchikey=True,
    compute_formula=True, compute_exact_mass=True,
    compute_adduct_columns=True, compute_logp=True,
    decimals=6, drop_invalid_smiles=True, inplace=False
)
annot.to_csv("annotated_structures.csv", index=False)
```

### B) Generate MassQL queries

**MS1 (adduct filters):**

```python
ms1_list = dbs.generate_massql_queries(
    annot, ppm=10, intensity_percent=1, decimals=5,
    separate_adducts=False,  # True = one query per adduct
    name_col="Compound name", mass_col="MolWeight"
)
Path("massql_ms1.txt").write_text("\n".join(ms1_list), encoding="utf-8")
```

**MS2 (precursor constraints only):**

```python
ms2_prec_list = dbs.generate_massqlPrec_queries(
    annot, ppm=10, intensity_percent=1, decimals=5,
    separate_adducts=False, name_col="Compound name", mass_col="Monoisotopic mass"
)
Path("massql_ms2_prec.txt").write_text("\n".join(ms2_prec_list), "utf-8")
```

**MS2 with fragment constraints** (if you have a `Fragments` column like `"120.0813; 138.066"`):

```python
ms2_frag_list = dbs.generate_massql_ms2_queries(
    annot, name_col="Compound name", mass_col="Monoisotopic mass",
    fragments_col="Fragments", ppm_prec=10, ppm_prod=10,
    intensity_percent=5, cardinality_min=1, cardinality_max=3
)
Path("massql_ms2_frag.txt").write_text("\n".join(ms2_frag_list), "utf-8")
```

### C) Build a similarity network (Morgan FP Dice)

```python
simM, fps = dbs.morgan_similarity_matrix(
    annot, mol_col="Molecule", id_col="InchiKey",
    radius=2, nbits=2048, metric="dice", return_fps=True
)

edges = dbs.build_similarity_network(
    simM, threshold=0.85, project="MyProj", metadata_df=annot,
    id_col="InchiKey", identity_col="file_path",
    save_csv=True, out_name="similarity_edges.csv"
)
```

### D) Add Δmass/ppm to similarity edges (from metadata)

```python
edges2 = dbs.add_massdiff_from_metadata(
    edges, annot, id_col="InchiKey",
    weight_col="MolWeight", source_col="SOURCE", target_col="TARGET",
    add_ppm=True
)
edges2.to_csv("similarity_edges_massdiff.csv", index=False)
```

### E) Compute Mordred descriptors, cluster, and visualize

```python
# Descriptors
desc = dbs.compute_mordred_descriptors(
    annot, smiles_col="SMILES", project="MyProj",
    save_csv=True, out_sep=";", out_path="df_descriptors.csv"
)

# Dendrogram + (optional) cluster CSV
_ = dbs.dendrogram_and_cluster_descriptors(
    desc, labels=annot["Compound name"] if "Compound name" in annot else None,
    color_threshold=250000.0, orientation="right",
    save_dir="images", filename="dendrogram.png",
    run_agglomerative=True, metadata_df=annot,
    id_col="InchiKey", identity_col="file_path",
    project="MyProj", save_cluster_csv=True, cluster_csv_name="clusters.csv"
)

# t-SNE (interactive HTML)
dbs.tsne_projection_plot(
    desc, metadata_df=annot,
    hover_cols=("file_path","cluster","Compound name"),
    perplexity=20, random_state=0, save_html=True,
    out_html="images/t-sne.html",
    # Optional highlight of activity terms (triangles overlay)
    activities_df=annot, activities_col="activities",
    highlight_activity="antibacterial",  # or "any" or None
    highlight_mode="overlay"
)
```

### F) Quick activity text mining (word cloud & top terms)

```python
dbs.wordcloud_from_semicolon_column(
    annot, column="activities", delimiter=";", min_count=1,
    save_path="images/wordcloud.png", show=False
)
dbs.plot_top_terms_from_column(
    annot, column="activities", delimiter=";", top_n=25,
    save_path="images/top_terms.png", show=False
)
```

### G) Optional: MZmine custom DB export (POS)

```python
dbs.make_mzmine_custom_db(
    annot, mz_col="MolWeight+1H", formula_col="MolFormula",
    identity_col="file_path", default_rt=0.0,
    project="MyProj", mode="POS", start_id=1,
    save_csv=True, out_path="mzmine_custom_db.csv"
)
```

## 3) Running via the notebook

* Open `DBsimilarity_py.ipynb`.
* Execute cells in order (the outline/table we made maps each step to its outputs).
* Key artifacts saved to `images/` and root folder:
  `annotated_structures.csv`, `massql_*.txt`, `df_descriptors.csv`, `images/dendrogram.png`, `images/t-sne.html`, `images/wordcloud.png`, `images/top_terms.png`, `clusters.csv`.

## 4) Column names & aliases that “just work”

* IDs: `SMILES`, `InchiKey`, `Compound name`
* Mass fields: `MolWeight` (neutral), `Monoisotopic mass`
* Optional: `Fragments` (semicolon list), `activities` (semicolon list), `file_path` for provenance
* If your columns differ, use `rename_by_aliases(df, aliases)` before calls, or pass the right `*_col` parameters.

## 5) Common gotchas (and fixes)

* **Mixed separators**: wrong delimiter → `pd.read_csv(..., sep=";")` vs `","`.
* **Invalid SMILES**: enable `drop_invalid_smiles=True` in annotation.
* **t-SNE perplexity**: keep `≤ (n−1)/3`; small sets (5–15), larger (20–50).
* **Empty fragments**: set `ignore_zero_fragments=True` for MS2 queries.
* **Similarity threshold**: start at `0.85`, adjust to balance sparsity vs. detail.
