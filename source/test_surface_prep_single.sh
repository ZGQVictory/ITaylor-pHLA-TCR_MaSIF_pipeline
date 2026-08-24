#!/bin/bash
# test_surface_prep_single.sh
# Process a single PDB (either pmhc or tcr) through MaSIF surface prep.
#
# Usage:
#   ./test_surface_prep_single.sh \
#       --mode    pmhc|tcr \
#       --pdb     <pdb_file> \
#       --id      <id>       \
#       --chains  <chains>   \
#       --output_dir <output_dir>

set -e
cd "$(dirname "$0")"

PYTHON=/home/zgq/data/software/anaconda3/envs/immuno_masif/bin/python
export MASIF_CDR3_CSV="./data_preparation_fpscore/humanPDB_cdr3_core8.csv"

MODE=""; PDB=""; ID=""; CHAINS=""; OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)       MODE="$2";       shift 2 ;;
        --pdb)        PDB="$2";        shift 2 ;;
        --id)         ID="$2";         shift 2 ;;
        --chains)     CHAINS="$2";     shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODE" || -z "$PDB" || -z "$ID" || -z "$CHAINS" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: all arguments required."
    sed -n '3,10p' "$0"
    exit 1
fi

source "$(dirname "$0")/../setup_env.sh"

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

if [[ "$MODE" == "pmhc" ]]; then
    FULL_ID="${ID}_pmhc"
    ARG="${ID}_pmhc_${CHAINS}"
    echo "=== [1/3] pMHC surface (${ARG}) ==="
    $PYTHON -W ignore::FutureWarning \
        ${SCRIPT_DIR}/01-test-pmhc_mesh_phch.py "$PDB" "$FULL_ID" "$CHAINS"
    echo "=== [2/3] pMHC resid -> idx ==="
    $PYTHON -W ignore::FutureWarning \
        ${SCRIPT_DIR}/02-pep-resid_to_idx.py "${ARG}" fp_score
    echo "=== [3/3] pMHC geo datafile ==="
    $PYTHON -W ignore::FutureWarning \
        ${SCRIPT_DIR}/03-pep-geo_datafile.py "${ARG}" fp_score

elif [[ "$MODE" == "tcr" ]]; then
    FULL_ID="${ID}_tcr"
    ARG="${ID}_tcr_${CHAINS}"
    echo "=== [1/3] TCR surface (${ARG}) ==="
    $PYTHON -W ignore::FutureWarning \
        ${SCRIPT_DIR}/01-test-tcr_mesh_phch.py "$PDB" "$FULL_ID" "$CHAINS"
    echo "=== [2/3] TCR resid -> idx ==="
    $PYTHON -W ignore::FutureWarning \
        ${SCRIPT_DIR}/02-tcr-resid_to_idx.py "${ARG}" fp_score
    echo "=== [3/3] TCR geo datafile ==="
    $PYTHON -W ignore::FutureWarning \
        ${SCRIPT_DIR}/03-tcr-geo_datafile.py "${ARG}" fp_score

else
    echo "Error: --mode must be pmhc or tcr"
    exit 1
fi

echo ""
echo "=== Done. Output: ${OUTPUT_DIR} ==="
