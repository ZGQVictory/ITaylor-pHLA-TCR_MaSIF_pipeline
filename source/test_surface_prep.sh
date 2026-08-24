#!/bin/bash
# test_surface_prep.sh
# Test script for pMHC and TCR surface fingerprint preparation.
#
# Usage:
#   ./test_surface_prep.sh \
#       --pmhc_pdb <pdb_file> --pmhc_id <id> --pmhc_chains <chains> \
#       --tcr_pdb  <pdb_file> --tcr_id  <id> --tcr_chains  <chains> \
#       --output_dir <output_dir>
#
# Example (same complex PDB):
#   ./test_surface_prep.sh \
#       --pmhc_pdb data_preparation_fpscore/00-raw_pdbs/1AO7.pdb --pmhc_id 1AO7 --pmhc_chains ABC \
#       --tcr_pdb  data_preparation_fpscore/00-raw_pdbs/1AO7.pdb --tcr_id  1AO7 --tcr_chains  DE \
#       --output_dir ./test_case/1AO7
#
# Example (separate PDB files):
#   ./test_surface_prep.sh \
#       --pmhc_pdb /path/to/pmhc.pdb --pmhc_id mypmhc --pmhc_chains ABC \
#       --tcr_pdb  /path/to/tcr.pdb  --tcr_id  mytcr  --tcr_chains  DE \
#       --output_dir ./test_output/

set -e
cd "$(dirname "$0")"  # source/

# ============================================================
# [用户配置区] 迁移到新机器时只需修改此区域
# ============================================================

# Python 解释器路径（conda 环境 immuno_masif）
PYTHON=/home/zgq/data/software/anaconda3/envs/immuno_masif/bin/python

# TCR CDR3 序列 CSV 文件路径
# 格式：pdb.id, cdr3b, cdr3a
# 新结构需在此文件中添加对应条目：PDBID_tcr,CDR3B_SEQ,CDR3A_SEQ,,,
export MASIF_CDR3_CSV="./data_preparation_fpscore/humanPDB_cdr3_core8.csv"

# ============================================================

# ---------- parse arguments ----------
PMHC_PDB=""; PMHC_ID=""; PMHC_CHAINS=""
TCR_PDB="";  TCR_ID="";  TCR_CHAINS=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pmhc_pdb)    PMHC_PDB="$2";    shift 2 ;;
        --pmhc_id)     PMHC_ID="$2";     shift 2 ;;
        --pmhc_chains) PMHC_CHAINS="$2"; shift 2 ;;
        --tcr_pdb)     TCR_PDB="$2";     shift 2 ;;
        --tcr_id)      TCR_ID="$2";      shift 2 ;;
        --tcr_chains)  TCR_CHAINS="$2";  shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$PMHC_PDB" || -z "$PMHC_ID" || -z "$PMHC_CHAINS" || \
      -z "$TCR_PDB"  || -z "$TCR_ID"  || -z "$TCR_CHAINS"  || \
      -z "$OUTPUT_DIR" ]]; then
    echo "Error: all arguments are required."
    sed -n '3,20p' "$0"
    exit 1
fi

# ---------- environment ----------
source "$(dirname "$0")/../setup_env.sh"

# ---------- test output dirs (isolated from original results) ----------
export MASIF_TMP_DIR="${OUTPUT_DIR}/tmp"
export MASIF_FP_PLY_DIR="${OUTPUT_DIR}/01-surfaces"
export MASIF_FP_PDB_DIR="${OUTPUT_DIR}/01-pdbs"
export MASIF_FP_PRECOMP_DIR="${OUTPUT_DIR}/fp-precomputation"
export MASIF_RESID_IDX_DIR="${OUTPUT_DIR}/02-resid_idx"
export MASIF_FP_DATAFILE_DIR="${OUTPUT_DIR}/03-fp_datafile/"
export MASIF_FP_PYMOL_PLY_DIR="${OUTPUT_DIR}/03-pymol_ply"

mkdir -p "$MASIF_TMP_DIR" "$MASIF_FP_PLY_DIR" "$MASIF_FP_PDB_DIR" \
         "$MASIF_FP_PRECOMP_DIR" "$MASIF_RESID_IDX_DIR" \
         "$MASIF_FP_DATAFILE_DIR" "$MASIF_FP_PYMOL_PLY_DIR"

SCRIPT_DIR="./data_preparation_fpscore"

# ---------- step 1: surface mesh + physicochemical features ----------
# pdb_id uses name_dummy format to match 02/03 scripts' parsing
PMHC_ARG="${PMHC_ID}_pmhc_${PMHC_CHAINS}"
TCR_ARG="${TCR_ID}_tcr_${TCR_CHAINS}"
PMHC_FULL_ID="${PMHC_ID}_pmhc"
TCR_FULL_ID="${TCR_ID}_tcr"

echo "=== [1/3] pMHC surface (${PMHC_FULL_ID}_${PMHC_CHAINS}) ==="
$PYTHON -W ignore::FutureWarning \
    ${SCRIPT_DIR}/01-test-pmhc_mesh_phch.py "$PMHC_PDB" "$PMHC_FULL_ID" "$PMHC_CHAINS"

echo "=== [1/3] TCR surface (${TCR_FULL_ID}_${TCR_CHAINS}) ==="
$PYTHON -W ignore::FutureWarning \
    ${SCRIPT_DIR}/01-test-tcr_mesh_phch.py "$TCR_PDB" "$TCR_FULL_ID" "$TCR_CHAINS"

# ---------- step 2: residue index mapping ----------
# Format: name_dummy_chains (e.g., 1AO7_pmhc_ABC, 1AO7_tcr_DE)
PMHC_ARG="${PMHC_ID}_pmhc_${PMHC_CHAINS}"
TCR_ARG="${TCR_ID}_tcr_${TCR_CHAINS}"

echo "=== [2/3] pMHC resid -> idx (${PMHC_ARG}) ==="
$PYTHON -W ignore::FutureWarning \
    ${SCRIPT_DIR}/02-pep-resid_to_idx.py "${PMHC_ARG}" fp_score

echo "=== [2/3] TCR resid -> idx (${TCR_ARG}) ==="
$PYTHON -W ignore::FutureWarning \
    ${SCRIPT_DIR}/02-tcr-resid_to_idx.py "${TCR_ARG}" fp_score

# ---------- step 3: geometric datafile ----------
echo "=== [3/3] pMHC geo datafile ==="
$PYTHON -W ignore::FutureWarning \
    ${SCRIPT_DIR}/03-pep-geo_datafile.py "${PMHC_ARG}" fp_score

echo "=== [3/3] TCR geo datafile ==="
$PYTHON -W ignore::FutureWarning \
    ${SCRIPT_DIR}/03-tcr-geo_datafile.py "${TCR_ARG}" fp_score

echo ""
echo "=== Done. Test outputs saved to: ${OUTPUT_DIR} ==="
