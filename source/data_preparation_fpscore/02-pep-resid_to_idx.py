import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # source/
sys.path.append(BASE_DIR)

from Bio.PDB import PDBParser
from collections import Counter
from default_config.masif_opts import masif_opts
import os
import numpy as np
import sys
from Bio.PDB import *
from biopandas.pdb import PandasPdb
from scipy import spatial
import pandas as pd
import csv

# ---------------------------------------
# 20241129 对pMHC圈出“Peptide框”
# 本脚本：质子化后，储存Pep（所有原子的 ##可按需改动）坐标：
# 后面需要依据Pep-cutoff筛选可训练的pMHC-patch
# ---------------------------------------



## --------------------------##
# 迭代部分（传入）&& 超参设置
## 25.Dec.30: exp: GILGFVFTL_MBP_t1
## 26.Jan.21: exp: AAFKRSCLK_HLA-A*01:01_MBP
## --------------------------##
pdb_info = sys.argv[1]
fields = pdb_info.split("_")
pep_seq = fields[0] 
pmhc_chain = fields[2]  # ABC, 25.Nov.
hla_allele= fields[1]  # Index, 25.Dec
dir_prefix = pep_seq + "_" + hla_allele
pdbid = dir_prefix

# 拆分pMHC
pmhc = list(pmhc_chain)
pep = pmhc[2]

params = masif_opts["fingerprint_score"]

# 质子化后的PDB文件：用于统计pep各号位氨基酸的原子坐标
# pdb_file = "/Users/shangchun/Desktop/MyMaSIF_tolinux/source/data_preparation_pmhc/00-raw_pdbs/1AO7.pdb"
# 25.Nov. 读取抽取好链的.pdb
pdb_file = masif_opts['fp-pdb_chain_dir'] + "/" + pdbid + "_" + \
    pmhc_chain + ".pdb"  # 241203, 确保加载的pdb经过质子化；eg.1AO7_AC.pdb
pep_chainid = pep  # "C"

if not os.path.exists(masif_opts["fingerprint_score"]["resid_idx_dir"]):
    os.mkdir(masif_opts["fingerprint_score"]["resid_idx_dir"])

# 预处理数据文件夹：用于加载三角剖分后pMHC上所有surface-vert的坐标，用于对应于各号位氨基酸的residue-specific patch的搜寻
# precom_dir = "/Users/shangchun/Desktop/MyMaSIF_tolinux/source/data_preparation_pmhc/04sc-precomputation_12A/precomputation/1AO7_AC_DE_A6"
precom_dir = params["fp_precomputation_dir"] + "/" + pdbid + "/" ## 25 Dec


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Nov 2025 :)

## =============================================================== ##
##     For peptide-side fingerprint patch index extraction         ##
##     平行于 TCR-side "core8-1to1_patchidx" 脚本                ##
##     By 尚醇, Nov 2025                                           ##
## =============================================================== ##


# ---------------- 参数区 ----------------
atom_cutoff = 7  # 选取 residue-specific patch cutoff (Å) ## 25.Dec
print(f"正在处理 {pdbid} 的 peptide patches ... :)\n")

# pep_chainid 通常为例如 "L" 或 "P" （可在上层脚本传入）
# pdb_file、precom_dir、masif_opts 在上层全局定义 (暂时延用IMPRINT，SC Nov 25)
parser = PDBParser(QUIET=True)
struct = parser.get_structure(pdbid, pdb_file)

# ---------------- 提取 peptide 链上所有残基及其原子坐标 ----------------
residue_atoms_dict = {}
residue_order_list = []

for model in struct:
    for chain in model:
        if chain.get_id() in pep_chainid:  # 指定的peptide链
            for res in chain.get_residues():
                rid = res.get_id()  # (hetfield, resseq, icode)
                key = (chain.get_id(), rid)
                coords = [atom.get_coord().tolist()
                          for atom in res.get_atoms()]
                residue_atoms_dict[key] = coords
                residue_order_list.append(key)

print(f"共提取到 {len(residue_atoms_dict)} 个 peptide 残基。")


'''
## 25.Nov. Underline the importance of side-chain atoms:)
# ---------------- 提取 peptide 链上所有残基及其侧链原子坐标 ----------------
residue_atoms_dict = {}
residue_order_list = []

# 定义 backbone 原子名称集合
backbone_atoms = {"N", "CA", "C", "O"}

for model in struct:
    for chain in model:
        if chain.get_id() in pep_chainid:  # 指定的 peptide 链
            for res in chain.get_residues():
                rid = res.get_id()  # (hetfield, resseq, icode)
                key = (chain.get_id(), rid)
                sidechain_coords = []

                for atom in res.get_atoms():
                    atom_name = atom.get_name().strip()
                    # 排除 backbone 原子，只保留侧链原子
                    if atom_name not in backbone_atoms:
                        sidechain_coords.append(atom.get_coord().tolist())

                # 若无侧链原子（例如 GLY），则退而使用 Cα 代表该残基
                if len(sidechain_coords) == 0:
                    ca_atom = res["CA"] if "CA" in res else None
                    if ca_atom is not None:
                        sidechain_coords.append(ca_atom.get_coord().tolist())
                    else:
                        print(f"[Warning] Residue {res} 缺失侧链及CA原子，跳过。")
                        continue

                residue_atoms_dict[key] = sidechain_coords
                residue_order_list.append(key)

print(f"共提取到 {len(residue_atoms_dict)} 个 peptide 残基（仅使用侧链原子）。")
'''


# ---------------- 加载 precomputed surface vertices ----------------
X = np.load(os.path.join(precom_dir, "p1_X.npy"))  # 25.Nov.
Y = np.load(os.path.join(precom_dir, "p1_Y.npy"))
Z = np.load(os.path.join(precom_dir, "p1_Z.npy"))
xyz_coords = np.vstack([X, Y, Z]).T
tree = spatial.KDTree(xyz_coords)
n_verts = xyz_coords.shape[0]
print(
    f"加载 peptide surface vertices：共有 {n_verts} 个顶点；atom_cutoff = {atom_cutoff} Å")

# ---------------- residue-specific surface vertex 选取 ----------------
residue_to_selected_vert = {}
residue_to_counts = {}

for key, atom_coords in residue_atoms_dict.items():
    vert_counter = Counter()
    for atom_coord in atom_coords:
        found = tree.query_ball_point(atom_coord, r=atom_cutoff)
        unique_found = set(found)
        for vid in unique_found:
            vert_counter[vid] += 1

    residue_to_counts[key] = vert_counter

    if len(vert_counter) == 0:
        residue_to_selected_vert[key] = None
        print(
            f"[Warning] Residue {key} 在 cutoff={atom_cutoff} Å 下没有找到 surface vert。")
        continue

    # 找到命中次数最多的 vertex
    max_count = max(vert_counter.values())
    candidate_verts = [
        vid for vid,
        cnt in vert_counter.items() if cnt == max_count]

    if len(candidate_verts) == 1:
        selected = candidate_verts[0]
    else:
        # 若并列，则计算距离均值以决策
        atom_array = np.array(atom_coords)
        avg_dists = {
            vid: np.mean(
                np.linalg.norm(
                    atom_array -
                    xyz_coords[vid],
                    axis=1)) for vid in candidate_verts}
        min_avg = min(avg_dists.values())
        best_vids = [
            vid for vid,
            ad in avg_dists.items() if np.isclose(
                ad,
                min_avg)]
        selected = int(sorted(best_vids)[0])

    residue_to_selected_vert[key] = int(selected)

# ---------------- 汇总与保存 ----------------
selected_list_in_order = [
    residue_to_selected_vert.get(
        k, None) for k in residue_order_list]

residue_key_to_idx = {}
for k, v in residue_to_selected_vert.items():
    chainid, rid = k
    keystr = f"{chainid}|{rid}"
    residue_key_to_idx[keystr] = v if v is None else int(v)

out_dir = masif_opts["fingerprint_score"]["resid_idx_dir"]
os.makedirs(out_dir, exist_ok=True)
out_fname = os.path.join(out_dir, f"{pdbid}_p1_idx.npy")
np.save(out_fname, residue_key_to_idx)
print(f"已保存 residue->selected_vert 映射到: {out_fname}")

# ---------------- 打印检查信息 ----------------
print("\n每个 peptide residue 的选中 vert 与对应命中次数（如 None 表示未找到）:")
for k in residue_order_list:
    chainid, rid = k
    keystr = f"{chainid}|{rid}"
    sel = residue_key_to_idx.get(keystr, None)
    counts = residue_to_counts.get(k, Counter())
    if sel is None:
        print(f"  {keystr} -> None (no verts within {atom_cutoff}Å)")
    else:
        print(f"  {keystr} -> vert {sel}  (count = {counts.get(sel,0)})")

# 额外统计信息
selected_vals = [v for v in residue_key_to_idx.values() if v is not None]
sel_counter = Counter(selected_vals)
print("\n被选为 residue-specific patch 的 vert 统计（vert_id: frequency）：")
print(dict(sel_counter))

out_counts_fname = os.path.join(out_dir, f"{pdbid}_p1_counts.npy")
np.save(out_counts_fname, {f"{k[0]}|{k[1]}": dict(v)
        for k, v in residue_to_counts.items()})
print(f"已保存 residue->vert-counts 到: {out_counts_fname}")
