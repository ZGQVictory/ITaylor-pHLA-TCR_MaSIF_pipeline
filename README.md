# ITaylor-pHLA-TCR_MaSIF_pipeline: pHLA/TCR surface fingerprint preparation

This repository provides the surface-feature preprocessing pipeline for the surface expert in [ITaylor](https://github.com/ZGQVictory/ITaylor). 

ITaylor predicts pHLA–TCR recognition by combining paired sequence information with surface features derived independently from monomeric pHLA and TCR structures. This pipeline generates those residue-centred molecular-surface features for the ITaylor surface expert.

The surface-processing code is adapted from [MaSIF](https://github.com/LPDI-EPFL/masif). Please refer to the original MaSIF repository for the method, software background, and installation details.

This GitHub repository contains the source code and the `1AO7` reference example. The packed Conda environment and the external `software/` stack are not stored in this repository.

## Workflow

For each pHLA or TCR structure, ITaylor-pHLA-TCR_MaSIF_pipeline performs three steps:

1. **Surface preparation**
   - protonates the PDB with Reduce;
   - generates a molecular surface with MSMS;
   - regularizes the mesh with PyMesh; and
   - computes hydrogen-bond, hydrophobicity, shape, and APBS electrostatic features.

2. **Residue-to-surface mapping**
   - maps peptide residues to pHLA surface vertices; or
   - maps CDR3 alpha/beta core residues to TCR surface vertices.

3. **Fingerprint extraction**
   - extracts a 12 Å patch around each selected surface centre;
   - retains at most 200 vertices per patch; and
   - saves geometric coordinates, masks, indices, and surface features as NumPy arrays under `03-fp_datafile/` for use as MaSIF-derived inputs to the ITaylor surface expert.

The main scripts are under `source/data_preparation_fpscore/`.

## Installation

Clone this GitHub repository, then configure the Conda environment and external software described below.

### 1. Download the Conda environment from Zenodo

The prepared environment is named `immuno_masif` and is distributed on Zenodo as `ITaylor-pHLA-TCR_MaSIF_pipeline_env.tar.gz`.

> **Zenodo record:** [10.5281/zenodo.22074539](https://doi.org/10.5281/zenodo.22074539) 

Restore and activate it with:

```bash
mkdir -p /path/to/envs/immuno_masif
tar -xzf ITaylor-pHLA-TCR_MaSIF_pipeline_env.tar.gz -C /path/to/envs/immuno_masif
source /path/to/envs/immuno_masif/bin/activate
conda-unpack
```

`conda-unpack` is required only after the first extraction. For later sessions, run only:

```bash
source /path/to/envs/immuno_masif/bin/activate
```

### 2. Install the external software

The following programs are not included in the Conda archive. Obtain them separately, following the [MaSIF software prerequisites](https://github.com/LPDI-EPFL/masif#software-prerequisites) and each program's license.

| Software     | Tested version         | Purpose                                       |
| ------------ | ---------------------- | --------------------------------------------- |
| Reduce       | MaSIF-compatible build | PDB protonation                               |
| MSMS         | 2.6.1                  | Molecular-surface generation                  |
| PDB2PQR      | 3.6.1                  | Atomic charges/radii and APBS input           |
| APBS         | 1.5                    | Electrostatic potential calculation           |
| `multivalue` | APBS 1.5 distribution  | Potential interpolation onto surface vertices |
| PyMesh       | 0.3 for CPython 3.8    | Mesh repair, curvature, and PLY processing    |

MSMS must be obtained independently under its own distribution terms. All external programs remain subject to their respective licenses.

### 3. Configure the `software/` directory

The simplest configuration is to place the downloaded or compiled programs under the repository root using the layout expected by `setup_env.sh`:

```text
software/
├── APBS-1.5-linux64/
│   ├── bin/apbs
│   └── share/apbs/tools/bin/multivalue
├── pdb2pqr_env/
│   └── bin/pdb2pqr
├── msms/
│   └── msms_i86_64Linux2_2.6.1/msms.x86_64Linux2.2.6.1
├── reduce/
│   ├── reduce_src/reduce
│   └── reduce_wwPDB_het_dict.txt
└── PyMesh/
    └── python/pymesh/
```

With this exact layout, the existing `setup_env.sh` paths can be used directly. Check that each executable exists, then load the configuration:

```bash
source setup_env.sh
```

If the software is installed elsewhere, edit `setup_env.sh` so that these variables point to the actual locations:

```bash
export APBS_BIN=/path/to/apbs
export MULTIVALUE_BIN=/path/to/multivalue
export PDB2PQR_BIN=/path/to/pdb2pqr
export MSMS_BIN=/path/to/msms

export REDUCE_HOME=/path/to/reduce-directory
export REDUCE_HET_DICT=/path/to/reduce_wwPDB_het_dict.txt
export PATH="$REDUCE_HOME:$PATH"

export PYMESH_PATH=/path/to/PyMesh
export PYTHONPATH="$PYMESH_PATH/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$PYMESH_PATH/python/pymesh/third_party/lib:$PYMESH_PATH/python/pymesh/lib:$LD_LIBRARY_PATH"
```

Finally, set `PYTHON` near the top of the two test launchers to the restored Zenodo environment:

```bash
PYTHON=/path/to/envs/immuno_masif/bin/python
```

Files to update:

- `source/test_surface_prep.sh`
- `source/test_surface_prep_single.sh`

## Reference example: 1AO7

The bundled example processes PDB `1AO7` using:

- pHLA chains `ABC`, with chain `C` treated as the peptide;
- TCR chains `DE`, in alpha-then-beta order; and
- CDR3 sequences from `source/data_preparation_fpscore/humanPDB_cdr3_core8.csv`.

Run from the repository root:

```bash
bash source/test_surface_prep.sh \
  --pmhc_pdb data_preparation_fpscore/00-raw_pdbs/1AO7.pdb \
  --pmhc_id 1AO7 \
  --pmhc_chains ABC \
  --tcr_pdb data_preparation_fpscore/00-raw_pdbs/1AO7.pdb \
  --tcr_id 1AO7 \
  --tcr_chains DE \
  --output_dir ./test_output/1AO7
```

The launcher changes its working directory to `source/`, so the input paths above are relative to `source/`. Results are written to:

```text
source/test_output/1AO7/
├── 01-pdbs/            # protonated chain PDB files
├── 01-surfaces/        # feature-rich PLY surfaces
├── 02-resid_idx/       # residue-to-vertex mappings
├── 03-fp_datafile/     # MaSIF-derived inputs for the ITaylor surface expert
├── 03-pymol_ply/       # fingerprint visualization surfaces
├── fp-precomputation/  # surface coordinates
└── tmp/                # temporary files
```

The files under `03-fp_datafile/` are the final outputs of this pipeline and are used as the surface-feature inputs to the MaSIF component of the ITaylor model.

## Citation

If you use this pipeline, please cite both ITaylor and the original MaSIF publication.

**ITaylor:**

> Citation to be added after publication.

**MaSIF:**

> Gainza, P. et al. Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning. *Nature Methods* 17, 184–192 (2020). https://doi.org/10.1038/s41592-019-0666-6

## License

The source code in this repository is released under the [Apache License 2.0](LICENSE), consistent with the upstream MaSIF project. External software and data remain subject to their own licenses and terms.
