import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # source/
sys.path.append(BASE_DIR)

from masif_modules.read_pool_from_surface import read_fp_coverage, extract_pool, output_pocket_feat, output_fp_feat, read_patch_from_surface
from default_config.masif_opts import masif_opts
import sys
import time
import os
import numpy as np
from IPython.core.debugger import set_trace

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)



'''
import torch ## 25.Nov
'''
# 本程序用于：Decompose proteins into patches for input into the neural network.

# Configuration imports. Config should be in run_args.py

np.random.seed(0)

# Load training data (From many files)
# 溯源：从新编写的脚本中引入功能函数

print(sys.argv[2])  # masif_pip ## Apr 2025

masif_app = sys.argv[2]

if masif_app == 'fp_score':
    params = masif_opts['fingerprint_score']

## 26. Jan: eg. tcr_000001_DE
in_fields = sys.argv[1].split("_") 

tcr_mark = in_fields[0] # "tcr"
t_index = in_fields[1]  # Index, 25.Dec
dir_prefix = tcr_mark + "_" + t_index  ## dir prefix, 25 Dec

patch_index_dir = params["resid_idx_dir"]  # fp patches
# 可视化：TCR/pep residue-specific patches
ply_set_out_dir = params["fp_pymol_ply"]  # Nov 2025
if not os.path.exists(ply_set_out_dir):
    os.makedirs(ply_set_out_dir)


ppi_pair_list = [sys.argv[1]]

total_shapes = 0
total_ppi_pairs = 0
np.random.seed(0)
print('Reading data from input ply surface files.')
for ppi_pair_id in ppi_pair_list:

    all_list_desc = []
    all_list_coords = []
    all_list_shape_idx = []
    all_list_names = []
    idx_positives = []

    my_precomp_dir = params['fp_datafile'] + dir_prefix + '/'
    if not os.path.exists(my_precomp_dir):
        os.makedirs(my_precomp_dir)

    # Read directly from the ply file.
    fields = ppi_pair_id.split('_')
    # >>> Read .ply with 3 feats
    ply_file = {}
    # tcr 26.Jan
    ply_file['p2'] = masif_opts['fp-ply_file_template'].format(
        dir_prefix, fields[2])

    # Note that, the .ply here only contain 3 phch features. *Oct 2025*

    # 注意，fp的直接可视化引用了pip-pocket可视化的相关函数，25.Oct
    all_center_index = {}
    
    p2_npy_path = os.path.join(patch_index_dir, f"{dir_prefix}_p2_idx.npy")
    p2_residue_map = np.load(p2_npy_path, allow_pickle=True).item()
    p2_index = [v for v in p2_residue_map.values() if v is not None]
    print(f"{dir_prefix}: 已读取 {len(p2_index)} 个 TCR 有效中心点索引。")
    all_center_index['p2'] = p2_index

    out_ply_file = {}   
    out_ply_file['p2'] = ply_set_out_dir + \
        "/p2_{}_{}.ply".format(dir_prefix, fields[2])
    

    '''
    if len(fields) == 4 or fields[3] == '':
        pids = ['p1', 'p2']
    else:
        '''
    pids = ['p2'] ## 26.Jan

    # Compute shape complementarity between the two proteins.
    save_rho = {}
    save_theta = {}
    save_mask = {}
    save_feat = {}

    out_neigh_indices = {}
    out_verts = {}
    out_faces = {}
    out_norms = {}

    input_feat = {}
    fp_index = {}
    verts = {}
    faces = {}
    norms = {}

    for pid in pids:
        # 25.Nov Save residue-specific patches' datafiles...
        save_feat[pid], save_rho[pid], save_theta[pid], save_mask[pid], out_neigh_indices[pid], out_verts[
            pid], out_faces[pid], out_norms[pid] = read_patch_from_surface(ply_file[pid], params, all_center_index[pid])

        ## ================================================##
        ##         添加功能：TCR/pMHC FP可视化 By 尚醇        ##
        # 2025/11/04
        ## ================================================##

        input_feat[pid], fp_index[pid], verts[pid], faces[pid], norms[pid] = read_fp_coverage(
            ply_file[pid], all_center_index[pid], params)

        neigh_i = fp_index[pid]  # 取得pool vert索引
        feature_matrix = input_feat[pid]
        print(
    'Analyzing pdb info: ' +
    "{}_{}_{}".format(
        fields[0],
        fields[1],
         fields[2]))
        print('------- 5 feature layers -------')
        print('Extracting fingerprint.')
        # 实现：把fp当做“一块patch”构建，以适应masif原始定义的patch可视化函数，Apr 2025
        subv, subn, subf = extract_pool(
    verts[pid], faces[pid], norms[pid], neigh_i)
        print('Storing feature-riched fp.ply.')
        output_fp_feat(
    subv,
    subf,
    subn,
    neigh_i,
    feature_matrix,
     out_ply_file[pid])
        # output_pocket_feat(subv, subf, subn, neigh_i, feature_matrix,
        # out_ply_file[pid])

    # Save data only if everything went well. 25.Nov
    ''' ## 11.17

    for pid in pids:

        # 坐标部分（[n, max_vertices]）
        torch.save(torch.from_numpy(save_rho[pid]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_rho_wrt_center.pt"))
        torch.save(torch.from_numpy(save_theta[pid]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_theta_wrt_center.pt"))
        torch.save(torch.from_numpy(save_mask[pid]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_mask.pt"))

        # 五种特征分别保存（[n, max_vertices]）
        torch.save(torch.from_numpy(save_feat[pid][:, :, 0]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_input_feat_si.pt"))
        torch.save(torch.from_numpy(save_feat[pid][:, :, 1]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_input_feat_ddc.pt"))
        torch.save(torch.from_numpy(save_feat[pid][:, :, 2]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_input_feat_hbond.pt"))
        torch.save(torch.from_numpy(save_feat[pid][:, :, 3]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_input_feat_charge.pt"))
        torch.save(torch.from_numpy(save_feat[pid][:, :, 4]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_input_feat_hphob.pt"))

        # 邻接索引与几何信息
        torch.save(out_neigh_indices[pid],
                   os.path.join(my_precomp_dir, f"{pid}_list_indices.pt"))

        torch.save(torch.from_numpy(out_verts[pid][:, 0]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_X.pt"))
        torch.save(torch.from_numpy(out_verts[pid][:, 1]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_Y.pt"))
        torch.save(torch.from_numpy(out_verts[pid][:, 2]).float(),
                   os.path.join(my_precomp_dir, f"{pid}_Z.pt"))

        print(f"Saved {pid} patch data in .pt format.")
    '''

    for pid in pids:
        # 坐标部分
        np.save(
    my_precomp_dir +
    pid +
    '_rho_wrt_center.npy',
     save_rho[pid])    # [n, max_vertices]
        np.save(
    my_precomp_dir +
    pid +
    '_theta_wrt_center.npy',
     save_theta[pid])  # [n, max_vertices]
        np.save(my_precomp_dir + pid + '_mask.npy',
                save_mask[pid])   # [n, max_vertices]

        # 五种特征分别保存（[n, max_vertices]）
        np.save(my_precomp_dir + pid + '_input_feat_si.npy',
                save_feat[pid][:, :, 0]) ## 25.Dec: shape: (9, 200); type: float64
        np.save(my_precomp_dir + pid + '_input_feat_ddc.npy',
                save_feat[pid][:, :, 1])
        np.save(my_precomp_dir + pid + '_input_feat_hbond.npy',
                save_feat[pid][:, :, 2])
        np.save(my_precomp_dir + pid + '_input_feat_charge.npy',
                save_feat[pid][:, :, 3])
        np.save(my_precomp_dir + pid + '_input_feat_hphob.npy',
                save_feat[pid][:, :, 4])

        # 邻接索引与几何信息
        np.save(
    my_precomp_dir +
    pid +
    '_list_indices.npy',
     out_neigh_indices[pid])
        np.save(my_precomp_dir + pid + '_X.npy', out_verts[pid][:, 0])
        np.save(my_precomp_dir + pid + '_Y.npy', out_verts[pid][:, 1])
        np.save(my_precomp_dir + pid + '_Z.npy', out_verts[pid][:, 2])

        print(f"Saved {pid} patch data in .npy format.")





