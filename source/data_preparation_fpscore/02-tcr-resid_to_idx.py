import os
import numpy as np
import sys
from Bio.PDB import *
from biopandas.pdb import PandasPdb
from scipy import spatial
import pandas as pd
import csv

#### ---------------------------------------
## 20241129 对pMHC圈出“Peptide框”
##   本脚本：质子化后，储存Pep（所有原子的 ##可按需改动）坐标：
##         后面需要依据Pep-cutoff筛选可训练的pMHC-patch
#### ---------------------------------------

## from SBI.structure import PDB
## 通过sys导入自定义的default_config/input_output模块（识别标志：__init__.py文件）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # source/
sys.path.append(BASE_DIR)
from default_config.masif_opts import masif_opts


##--------------------------##
##  迭代部分（传入）&& 超参设置
##--------------------------##
## 26. Jan: eg. tcr_000001_DE
pdb_info = sys.argv[1] 
fields = pdb_info.split("_")
tcr_mark = fields[0] # "tcr"
t_index = fields[1]  # Index, 25.Dec
pdbid = tcr_mark + "_" + t_index  ## dir prefix, 25 Dec
tcr_chain = fields[2] ## DE

## 拆分TCR
tcr = list(tcr_chain)
alpha_chainid = tcr[0]
beta_chainid = tcr[1]

params = masif_opts["fingerprint_score"]

## 质子化后的PDB文件：用于统计pep各号位氨基酸的原子坐标
# pdb_file = "/Users/shangchun/Desktop/MyMaSIF_tolinux/source/data_preparation_pmhc/00-raw_pdbs/1AO7.pdb" 
pdb_file = masif_opts['fp-pdb_chain_dir']+"/"+pdbid+"_"+tcr_chain+".pdb" ## 241203, 确保加载的pdb经过质子化；eg.1AO7_AC.pdb

if not os.path.exists(masif_opts["fingerprint_score"]["resid_idx_dir"]):
    os.mkdir(masif_opts["fingerprint_score"]["resid_idx_dir"])

precom_dir = params["fp_precomputation_dir"]+"/"+pdbid+"/"

# ---------------- 参数区 ----------------
atom_cutoff = 5  # 选取 residue-specific patch cutoff (Å)
print(f"正在处理 {pdbid} 的 tcr patches ... :)\n")


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Nov 2025 :)

# -----------------------------
# 选取 CDR3 core-residue-specific patch（每个残基选一个最代表的 surf-vert index）
# -----------------------------
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from Bio.PDB import PDBParser, PPBuilder
from Bio import pairwise2
from scipy import spatial

# ---------- 读取 core8 CDR3 序列 ----------
cdr3_df = pd.read_csv(masif_opts['cdr3_csv'])
cdr3_dict = {row['pdb.id']: {'cdr3a': row['cdr3a'], 'cdr3b': row['cdr3b']} for _, row in cdr3_df.iterrows()}

# ---------- 解析结构并定位 core8 残基（返回 residue 对象列表） ----------
parser = PDBParser(QUIET=True)
struct = parser.get_structure(pdbid, pdb_file)

ppb = PPBuilder()
# 构建每个链的连续序列字符串（用于局部比对）
chain_seqs = {}
for model in struct:
    for chain in model:
        seq = "".join([str(pp.get_sequence()) for pp in ppb.build_peptides(chain)])
        chain_seqs[chain.id] = seq

cdr3a_seq = cdr3_dict[pdbid]['cdr3a']
cdr3b_seq = cdr3_dict[pdbid]['cdr3b']

def find_cdr3_residues(chain, seq_cdr3):
    """返回 residue 对象的列表（按在链中出现的顺序）"""
    seq_chain = chain_seqs.get(chain.id, "")
    if seq_chain == "":
        return None
    alignments = pairwise2.align.localms(seq_chain, seq_cdr3, 2, -1, -5, -1)
    if not alignments:
        return None
    best = alignments[0]
    start, end = best.start, best.end  # start inclusive, end exclusive in python string index convention
    residues = list(chain.get_residues())
    # 对齐位置可能对应于若干 peptide-segment 的残基数，确保不越界
    if end <= len(residues):
        return residues[start:end]
    else:
        # 若对齐偏移导致越界，尽量返回可用的切片
        return residues[start:len(residues)]

# 找到 alpha 链和 beta 链对象
try:
    chain_alpha = [ch for ch in struct.get_chains() if ch.id == alpha_chainid][0]
except IndexError:
    raise RuntimeError(f"Alpha chain {alpha_chainid} not found in {pdbid}")
try:
    chain_beta = [ch for ch in struct.get_chains() if ch.id == beta_chainid][0]
except IndexError:
    raise RuntimeError(f"Beta chain {beta_chainid} not found in {pdbid}")

residues_alpha = find_cdr3_residues(chain_alpha, cdr3a_seq)
residues_beta  = find_cdr3_residues(chain_beta,  cdr3b_seq)

if residues_alpha is None:
    print(f"[Warning] 未能在链 {alpha_chainid} 中定位到 cdr3a ({cdr3a_seq})")
    residues_alpha = []
if residues_beta is None:
    print(f"[Warning] 未能在链 {beta_chainid} 中定位到 cdr3b ({cdr3b_seq})")
    residues_beta = []

# ---------- 将每个 core-residue 的原子坐标单独存为字典 ----------
# key: (chainid, (hetfield, resseq, icode)) == residue.get_id()
# value: list of atom coordinate tuples [(x,y,z), ...]
residue_atoms_dict = {}  # 保存每个残基的原子坐标
residue_order_list = []  # 按顺序保存 (chainid, resid_id) 以便后续输出按顺序排列

for res in residues_alpha:
    rid = res.get_id()      # 三元组 (hetfield, resseq, icode)
    key = (alpha_chainid, rid)
    coords = [atom.get_coord().tolist() for atom in res.get_atoms()]
    residue_atoms_dict[key] = coords
    residue_order_list.append(key)

for res in residues_beta:
    rid = res.get_id()
    key = (beta_chainid, rid)
    coords = [atom.get_coord().tolist() for atom in res.get_atoms()]
    residue_atoms_dict[key] = coords
    residue_order_list.append(key)

print(f"已提取 {len(residue_atoms_dict)} 个 core-residue 的原子坐标。")

# ---------- 加载 pMHC surface vertices 并建立 KDTree ----------
X = np.load(os.path.join(precom_dir, "p2_X.npy"))
Y = np.load(os.path.join(precom_dir, "p2_Y.npy"))
Z = np.load(os.path.join(precom_dir, "p2_Z.npy"))
xyz_coords = np.vstack([X, Y, Z]).T
tree = spatial.KDTree(xyz_coords)
n_verts = xyz_coords.shape[0]
print(f"加载 TCR surface vertices：共有 {n_verts} 个顶点；atom_cutoff = {atom_cutoff} Å")

# ---------- 对每个残基，统计每个 vertex 被多少个原子命中，选取命中次数最多的 vertex ----------
residue_to_selected_vert = {}    # key: (chainid, resid_id) -> selected vert index (int) or None
residue_to_counts = {}           # key -> Counter of vert->count （供后续分析）
for key, atom_coords in residue_atoms_dict.items():
    # atom_coords: list of [x,y,z]
    vert_counter = Counter()
    atom2verts = []  # 保存每个原子找到的verts（用于后续距离计算）
    for atom_coord in atom_coords:
        found = tree.query_ball_point(atom_coord, r=atom_cutoff)
        atom2verts.append(found)
        # count each found vert once per atom (原子对某 vert 多次出现不应重复计数)
        unique_found = set(found)
        for vid in unique_found:
            vert_counter[vid] += 1

    residue_to_counts[key] = vert_counter

    if len(vert_counter) == 0:
        # 没有在 cut-off 内找到任何 surface vert
        residue_to_selected_vert[key] = None
        print(f"[Warning] Residue {key} 在 cutoff={atom_cutoff} Å 下没有找到任何 surface vert。")
        continue

    # 找到最高频次
    max_count = max(vert_counter.values())
    candidate_verts = [vid for vid, cnt in vert_counter.items() if cnt == max_count]

    if len(candidate_verts) == 1:
        selected = candidate_verts[0]
        residue_to_selected_vert[key] = int(selected)
    else:
        # 并列时使用 tiebreaker：计算 candidate vert 到该残基所有原子的平均距离，选择 avg distance 最小者
        avg_dists = {}
        atom_array = np.array(atom_coords)
        for vid in candidate_verts:
            vert_coord = xyz_coords[vid]
            dists = np.linalg.norm(atom_array - vert_coord, axis=1)
            avg_dists[vid] = float(np.mean(dists))
        # 选择最小 avg distance 的 vid（若仍并列，选择最小 vid）
        min_avg = min(avg_dists.values())
        best_vids = [vid for vid, ad in avg_dists.items() if np.isclose(ad, min_avg)]
        selected = int(sorted(best_vids)[0])
        residue_to_selected_vert[key] = selected

# ---------- 汇总与保存 ----------
# 1) 列表形式（按 residue_order_list 顺序）：selected vert id 或 None
selected_list_in_order = [residue_to_selected_vert.get(k, None) for k in residue_order_list]

# 2) dict 映射（更精确，包含 chainid + residue id）
#    这里将 residue id 三元组转换为可序列化字符串作为 key，例如 "A|(' ',45,' ')"
residue_key_to_idx = {}
for k, v in residue_to_selected_vert.items():
    chainid, rid = k
    keystr = f"{chainid}|{rid}"   # e.g. "A|(' ',45,' ')"
    residue_key_to_idx[keystr] = v if v is None else int(v)

# 保存为 npy（或你也可以改成 json）
out_fname = os.path.join(masif_opts["fingerprint_score"]["resid_idx_dir"], f"{pdbid}_p2_idx.npy")
np.save(out_fname, residue_key_to_idx)
print(f"已保存 residue->selected_vert 映射到: {out_fname}")

# 同时打印统计信息：每个 residue 的选中 vert 及对应被命中次数
print("\n每个 core residue 的选中 vert 与对应命中次数（如 None 表示未找到）:")
for k in residue_order_list:
    chainid, rid = k
    keystr = f"{chainid}|{rid}"
    sel = residue_key_to_idx.get(keystr, None)
    counts = residue_to_counts.get(k, Counter())
    if sel is None:
        print(f"  {keystr} -> None (no verts within {atom_cutoff}Å)")
    else:
        print(f"  {keystr} -> vert {sel}  (count = {counts.get(sel,0)})")

# 额外：统计所有被选中的 vert 的出现频次分布（方便查看是否集中）
selected_vals = [v for v in residue_key_to_idx.values() if v is not None]
sel_counter = Counter(selected_vals)
print("\n被选为 residue-specific patch 的 vert 统计（vert_id: frequency）：")
print(dict(sel_counter))

# 如果需要把每个 residue 对应的 atom-hit-distribution 也保存，可选择将 residue_to_counts 一起保存为 npy
out_counts_fname = os.path.join(masif_opts["fingerprint_score"]["resid_idx_dir"], f"{pdbid}_p2_counts.npy")
np.save(out_counts_fname, {f"{k[0]}|{k[1]}": dict(v) for k,v in residue_to_counts.items()})
print(f"已保存 residue->vert-counts 到: {out_counts_fname}")




