#! /usr/bin/env python3
#!/usr/bin/python
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # source/
sys.path.append(BASE_DIR)

# SC
# ===  导入 pymesh 模块 ===
import pymesh
# 测试导入
print("PyMesh imported successfully! Version:", getattr(pymesh, "__version__", "unknown"))


from sklearn.neighbors import KDTree
from triangulation.compute_normal import compute_normal
from triangulation.computeAPBS import computeAPBS
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeHydrophobicity import computeHydrophobicity
from input_output.protonate import protonate
from input_output.read_ply import read_ply
from input_output.save_ply import save_ply
from input_output.extractPDB import extractPDB
from triangulation.fixmesh import fix_mesh
from triangulation.computeMSMS import computeMSMS
from default_config.masif_opts import masif_opts
import numpy as np
import Bio
import shutil
from Bio.PDB import *
import importlib
from IPython.core.debugger import set_trace

import pandas as pd


# Local includes

## 25.Dec.30: exp: GILGFVFTL_MBP_t1
## 26.Jan.21: exp: AAFKRSCLK_HLA-A*01:01_MBP

if len(sys.argv) <= 1:
    '''
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    '''  # SC
    print("Usage: {config} " + sys.argv[0] + " PDBID_AC")
    print("AC are the chains to include in pMHC surface.")
    sys.exit(1)

masif_app = sys.argv[2]

if masif_app == 'fp_score':
    params = masif_opts['fingerprint_score']

# Save the chains as separate files.
in_fields = sys.argv[1].split("_")
pep_seq = in_fields[0]
hla_allele = in_fields[1]  # Index, 25.Dec
pdb_id = pep_seq + "_" + hla_allele ## dir prefix, 26 Jan
chain_ids1 = in_fields[2]  # pMHC three chains(eg. ABC), 25.Nov:)

'''
if (len(sys.argv)>2) and (sys.argv[2]=='masif_ligand'):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+".pdb")
else:
    pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+".pdb"
'''  # SC

# ---------- 获取并调用结构文件 ----------
info_csv = pd.read_csv(
    "/public/SHARE/zgq_immu/negative_phla_pandora_with_ids.csv"
)

# 精确匹配 peptide + hla_allele
matched = info_csv[
    (info_csv["peptide"] == pep_seq) &
    (info_csv["hla_allele"] == hla_allele)
]

# 取唯一的 pdb_file
pdbfile_csv = matched.iloc[0]["pdb_file"].strip()

# 拼接成完整路径
pdb_filename = os.path.join(
    masif_opts["raw_pdb_dir"],
    pdbfile_csv
)



# 为什么又质子化一遍？
tmp_dir = masif_opts['tmp_dir']
os.makedirs(masif_opts["tmp_dir"], exist_ok=True)

protonated_file = tmp_dir + "/" + pdb_id + ".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file


# Extract chains of interest.
out_filename1 = tmp_dir + "/" + pdb_id + "_" + chain_ids1
extractPDB(pdb_filename, out_filename1 + ".pdb", chain_ids1)

'''
# Compute MSMS of surface w/hydrogens,
try:
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(
        out_filename1 + ".pdb", protonate=True)
except BaseException:
    set_trace()
'''

# Compute MSMS of surface w/hydrogens
try:
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1 + ".pdb",
                                                              protonate=True)
except Exception as e:
    print("computeMSMS() 出错：", e)
    print("出错文件:", out_filename1 + ".pdb")
    raise   # 让错误继续抛出，容易发现具体问题



# Compute "charged" vertices
if masif_opts['use_hbond']:
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)

# For each surface residue, assign the hydrophobicity of its amino acid.
if masif_opts['use_hphob']:
    vertex_hphobicity = computeHydrophobicity(names1)

# If protonate = false, recompute MSMS of surface, but without hydrogens
# (set radius of hydrogens to 0).
vertices2 = vertices1
faces2 = faces1

# Fix the mesh.
mesh = pymesh.form_mesh(vertices2, faces2)
regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])

# Compute the normals
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
# Assign charges on new vertices based on charges of old vertices (nearest
# neighbor)

if masif_opts['use_hbond']:
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,
                                          vertex_hbond, masif_opts)

if masif_opts['use_hphob']:
    vertex_hphobicity = assignChargesToNewMesh(
        regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)
# SC, For test  print(regular_mesh.vertices,out_filename1+".pdb",
# out_filename1)


if masif_opts['use_apbs']:
    vertex_charges = computeAPBS(
        regular_mesh.vertices,
        out_filename1 + ".pdb",
        out_filename1)

iface = np.zeros(len(regular_mesh.vertices))
if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
    # Compute the surface of the entire complex and from that compute the
    # interface.
    v3, f3, _, _, _ = computeMSMS(pdb_filename,
                                  protonate=True)
    # Regularize the mesh
    mesh = pymesh.form_mesh(v3, f3)
    # I believe It is not necessary to regularize the full mesh. This can
    # speed up things by a lot.
    full_regular_mesh = mesh
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and
    # those in the full complex.
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    # Square d, because this is how it was in the pyflann version.
    d = np.square(d)
    assert (len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    # Convert to ply and save.
    save_ply(
        out_filename1 + ".ply",
        regular_mesh.vertices,
        regular_mesh.faces,
        normals=vertex_normal,
        charges=vertex_charges,
        normalize_charges=True,
        hbond=vertex_hbond,
        hphob=vertex_hphobicity,
        iface=iface)

else:
    # Convert to ply and save.
    save_ply(
        out_filename1 + ".ply",
        regular_mesh.vertices,
        regular_mesh.faces,
        normals=vertex_normal,
        charges=vertex_charges,
        normalize_charges=True,
        hbond=vertex_hbond,
        hphob=vertex_hphobicity)
    # 25.Nov.这里就是已经计算好的mesh

if not os.path.exists(masif_opts['fp-ply_chain_dir']):
    os.makedirs(masif_opts['fp-ply_chain_dir'])
if not os.path.exists(masif_opts['fp-pdb_chain_dir']):
    os.makedirs(masif_opts['fp-pdb_chain_dir'])
# prefix: pdb_id+"_"+chain_ids1
shutil.copy(out_filename1 + '.ply', masif_opts['fp-ply_chain_dir'])
shutil.copy(out_filename1 + '.pdb', masif_opts['fp-pdb_chain_dir'])

# 25.Nov. Save all surface vertices' coords to form a complete mesh...

# regular_mesh.vertices 的形状为 (N, 3)，在后续处理中保持不变
# 构建保存目录
save_dir = os.path.join(params['fp_precomputation_dir'], pdb_id)
os.makedirs(save_dir, exist_ok=True)

# 保存三个方向的坐标
np.save(os.path.join(save_dir, 'p1_X.npy'), regular_mesh.vertices[:, 0])
np.save(os.path.join(save_dir, 'p1_Y.npy'), regular_mesh.vertices[:, 1])
np.save(os.path.join(save_dir, 'p1_Z.npy'), regular_mesh.vertices[:, 2])

print(f"已保存 pMHC surface vertices 的坐标分量到: {save_dir}")
