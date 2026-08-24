#!/usr/bin/env python3
"""
Test version of 01-pep-chains_mesh_phch.py
Directly accepts pdb_file path instead of looking up CSV.

Usage:
    python3 01-test-pmhc_mesh_phch.py <pdb_file> <pdb_id> <pmhc_chains> [out_dir]

Example:
    python3 01-test-pmhc_mesh_phch.py 00-raw_pdbs/1AO7.pdb 1AO7 ABC
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # source/
sys.path.append(BASE_DIR)

import pymesh
print("PyMesh imported successfully! Version:", getattr(pymesh, "__version__", "unknown"))

from sklearn.neighbors import KDTree
from triangulation.compute_normal import compute_normal
from triangulation.computeAPBS import computeAPBS
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeHydrophobicity import computeHydrophobicity
from input_output.protonate import protonate
from input_output.save_ply import save_ply
from input_output.extractPDB import extractPDB
from triangulation.fixmesh import fix_mesh
from triangulation.computeMSMS import computeMSMS
from default_config.masif_opts import masif_opts
import numpy as np
import shutil

if len(sys.argv) < 4:
    print("Usage: python3 {} <pdb_file> <pdb_id> <pmhc_chains> [out_dir]".format(sys.argv[0]))
    sys.exit(1)

pdb_filename = sys.argv[1]
pdb_id       = sys.argv[2]
chain_ids1   = sys.argv[3]

# Allow overriding output dirs via env vars (set by test_surface_prep.sh)
if os.environ.get("MASIF_TMP_DIR"):
    masif_opts["tmp_dir"] = os.environ["MASIF_TMP_DIR"]
if os.environ.get("MASIF_FP_PLY_DIR"):
    masif_opts["fp-ply_chain_dir"] = os.environ["MASIF_FP_PLY_DIR"]
if os.environ.get("MASIF_FP_PDB_DIR"):
    masif_opts["fp-pdb_chain_dir"] = os.environ["MASIF_FP_PDB_DIR"]
if os.environ.get("MASIF_FP_PRECOMP_DIR"):
    masif_opts["fingerprint_score"]["fp_precomputation_dir"] = os.environ["MASIF_FP_PRECOMP_DIR"]

params  = masif_opts["fingerprint_score"]
tmp_dir = masif_opts["tmp_dir"]
os.makedirs(tmp_dir, exist_ok=True)

# Protonate
protonated_file = os.path.join(tmp_dir, pdb_id + ".pdb")
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

# Extract chains
out_filename1 = os.path.join(tmp_dir, pdb_id + "_" + chain_ids1)
extractPDB(pdb_filename, out_filename1 + ".pdb", chain_ids1)

# Compute MSMS surface
try:
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1 + ".pdb", protonate=True)
except Exception as e:
    print("computeMSMS() error:", e)
    raise

if masif_opts["use_hbond"]:
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)
if masif_opts["use_hphob"]:
    vertex_hphobicity = computeHydrophobicity(names1)

mesh = pymesh.form_mesh(vertices1, faces1)
regular_mesh = fix_mesh(mesh, masif_opts["mesh_res"])
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

if masif_opts["use_hbond"]:
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts)
if masif_opts["use_hphob"]:
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)
if masif_opts["use_apbs"]:
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1 + ".pdb", out_filename1)

# Save .ply
os.makedirs(masif_opts["fp-ply_chain_dir"], exist_ok=True)
os.makedirs(masif_opts["fp-pdb_chain_dir"], exist_ok=True)
save_ply(out_filename1 + ".ply", regular_mesh.vertices, regular_mesh.faces,
         normals=vertex_normal, charges=vertex_charges, normalize_charges=True,
         hbond=vertex_hbond, hphob=vertex_hphobicity)
shutil.copy(out_filename1 + ".ply", masif_opts["fp-ply_chain_dir"])
shutil.copy(out_filename1 + ".pdb", masif_opts["fp-pdb_chain_dir"])

# Save surface vertex coords
save_dir = os.path.join(params["fp_precomputation_dir"], pdb_id)
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, "p1_X.npy"), regular_mesh.vertices[:, 0])
np.save(os.path.join(save_dir, "p1_Y.npy"), regular_mesh.vertices[:, 1])
np.save(os.path.join(save_dir, "p1_Z.npy"), regular_mesh.vertices[:, 2])

print("pMHC surface done. Saved to:", save_dir)
