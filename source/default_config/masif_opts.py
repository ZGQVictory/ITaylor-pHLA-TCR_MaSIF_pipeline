import tempfile

masif_opts = {}
# Default directories
#masif_opts["raw_pdb_dir"] = "/public/SHARE/xmy_immu/tcr_only_tFold/" ## 26 Jan
#masif_opts["raw_pdb_dir"] = "data_preparation_fpscore/00-raw_pdbs/fromzgq_datafile_clear_pandora/structure_pdbs/" ## 26 Jan
#masif_opts["raw_pdb_dir"] = "/public/SHARE/zgq_immu/negative_PANDORA_pdb_output/" ## 26 Jan, negative pHLA
masif_opts["raw_pdb_dir"] = "./data_preparation_fpscore/00-raw_pdbs/"
#masif_opts["tmp_dir"] = tempfile.gettempdir()
masif_opts["tmp_dir"] = "./tmp_test/"


# Surface features
masif_opts["use_hbond"] = True
masif_opts["use_hphob"] = True
masif_opts["use_apbs"] = True
'''
masif_opts["compute_iface"] = True
''' ## SC, For pMHC-case
masif_opts["compute_iface"] = False
# Mesh resolution. Everything gets very slow if it is lower than 1.0
masif_opts["mesh_res"] = 1.0
masif_opts["feature_interpolation"] = True


# Coords params
masif_opts["radius"] = 12.0

# ===========================================
## 20251030 Surface-based TCR-pMHC Binding Score :)

masif_opts["cdr3_csv"] = "./data_preparation_fpscore/tfold_cdr3_core8.csv" ## Same within both-side, 25.11.07
masif_opts["fp-ply_chain_dir"] = "data_preparation_fpscore/01-benchmark_surfaces/"
masif_opts["fp-ply_file_template"] = masif_opts["fp-ply_chain_dir"] + "/{}_{}.ply" ## {dir_prefix}_{chainsid}, 25 Dec

## ^Note: ['p1'] for pmhc-side; ['p2'] for tcr-side
masif_opts["fingerprint_score"] = {}
masif_opts["fingerprint_score"][
    "fp_precomputation_dir"
] = "data_preparation_fpscore/fp-precomputation_12A/pmhc/" ## 26 Jan ## 25 Dec
masif_opts["fingerprint_score"]["max_shape_size"] = 200
masif_opts["fingerprint_score"]["max_distance"] = 12.0
masif_opts["fingerprint_score"]["resid_idx_dir"] = "data_preparation_fpscore/02-resid_idx/"


masif_opts["fingerprint_score"]["fp_pymol_ply"] = "data_preparation_fpscore/03-pymol_ply/pmhc/" ## pmhc-fp, 25 Dec
#masif_opts["fingerprint_score"]["fp_datafile"] = "/public/SHARE/surf-fp_sc2zgq/imfp/pmhc_negative/nonered/" ## 26 Jan, negative pmhcs
masif_opts["fingerprint_score"]["fp_datafile"] = "./data_preparation_fpscore/03-fp_datafile/pmhc/"
'''
masif_opts["fingerprint_score"]["fp_pymol_ply"] = "data_preparation_fpscore/03-pymol_ply/tcr/" ## tcr-fp, 26 Jan
masif_opts["fingerprint_score"]["fp_datafile"] = "data_preparation_fpscore/03-fp_datafile/tcr-add2/" ## 26 Jan
'''

# Allow environment variable overrides for test/local runs
import os as _os
for _key, _env in [
    ("tmp_dir",                                    "MASIF_TMP_DIR"),
    ("fp-pdb_chain_dir",                           "MASIF_FP_PDB_DIR"),
    ("fp-ply_chain_dir",                           "MASIF_FP_PLY_DIR"),
    ("cdr3_csv",                                   "MASIF_CDR3_CSV"),
]:
    if _os.environ.get(_env):
        masif_opts[_key] = _os.environ[_env]

# Keep template in sync with overridden ply dir
masif_opts["fp-ply_file_template"] = masif_opts["fp-ply_chain_dir"] + "/{}_{}.ply"

for _key, _env in [
    ("fp_precomputation_dir", "MASIF_FP_PRECOMP_DIR"),
    ("resid_idx_dir",         "MASIF_RESID_IDX_DIR"),
    ("fp_datafile",           "MASIF_FP_DATAFILE_DIR"),
    ("fp_pymol_ply",          "MASIF_FP_PYMOL_PLY_DIR"),
]:
    if _os.environ.get(_env):
        masif_opts["fingerprint_score"][_key] = _os.environ[_env]
