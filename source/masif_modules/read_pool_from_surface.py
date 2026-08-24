# coding: utf-8
# ## Imports and helper functions
from IPython.core.debugger import set_trace

import os
import sys

# 导入 pymesh
import pymesh


import time
import numpy as np

## 通过sys导入自定义的模块（识别标志：__init__.py文件）
import sys
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source") ## For Linux

from geometry.compute_polar_coordinates import compute_polar_coordinates
from input_output.save_ply import save_ply

from sklearn import metrics


from scipy import spatial ## Apr 2025

####################################
##  本程序用于： 数据后处理
##      MaSIF_pMHC解释性：Trace patches on surface（在pMHC表面上对verts的重要程度可视化）
##      本脚本：提取patch_center_pool的信息
###   SC，2023/10/24
####################################

##================================================##
##         修改功能：For Patch溯源 By 尚醇          ##
##               2023/10/24
##================================================##
def read_pool_from_surface(ply_fn, params, pool_csv):
    ## 参数格式：
    ## ply_fn: data_preparation/01-benchmark_surfaces/1AO7_AC.ply
    ## pool_csv: 包含了TCR 3A-cutoff范围内的verts pool: 索引 & 其与标签关联程度 的csv表格的路径
    """
    # Read data from a ply file -- Extract CENTER POOL. 
    # Returns: 
    # list_desc: List of features per patch
    # list_coords: list of angular and polar coordinates.
    # list_indices: list of indices of neighbors in the patch.
    # list_sc_labels: list of shape complementarity labels (computed here).
    """
    mesh = pymesh.load_mesh(ply_fn)

    # Normals: 
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1,n2,n3], axis=1)

    '''
    # Compute the angular and radial coordinates. 
    rho, theta, neigh_indices, mask = compute_polar_coordinates(mesh, radius=params['max_distance'], max_vertices=params['max_shape_size'])
    #######################################
    ## rho/thera/mask shape均为：(n->即所有vert的数量,max_vertices->即设定好的计算量）
    #   即：每一行代表一个以（行索引为索引值）的vert为中心的patch，存储其neigh的verts的测地极坐标信息
    #  格式：极径；极角；neigh索引；掩码
    #######################################
    ''' ## 溯源时，不以patch为单位

    ###########################################
    ## 为每个vertice interpolate几何性质：shape index（来自于微分几何的概念）
    #   K：高斯曲率，是曲面“内在”度量：即其只依赖于曲面上长度和角度的测量，与曲面如何嵌入空间无关（例如：平面卷起来形成圆柱）
    #   H：平均曲率，反映曲面与环境的关系：曲面如何嵌入周围空间
    #   k1, k2：主曲率：（过曲面某点E处面法线）的（所有剖切平面）（与曲面相交形成的）平面曲线中，
    #              过E点最大和最小的曲率半径，为主曲率半径；数学上证明两个主曲率半径的方向是互相垂直的
    #   P.S. “convex”: K>0 & H非0；“concave”: K<0
    #   -- 在草稿本上进行了数学推导验证 √
    ###########################################
    # Compute the principal curvature components for the shape index. 
    mesh.add_attribute("vertex_mean_curvature") ## PyMesh自带；标量场
    H = mesh.get_attribute("vertex_mean_curvature") 
    mesh.add_attribute("vertex_gaussian_curvature") ##PyMesh自带；标量场
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem<0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)

    

    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)

    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")

    # Hydropathy features
    # Normalize hydropathy by dividing by 4.5
    ## 为了适应pymol插件可视化的颜色需求，这里不需要除以4.5
    ##hphob = mesh.get_attribute("vertex_hphob")/4.5
    hphob = mesh.get_attribute("vertex_hphob")

    # Iface labels (for ground truth only)     
    if "vertex_iface" in mesh.get_attribute_names():
        iface_labels = mesh.get_attribute("vertex_iface") 
    else:
        iface_labels = np.zeros_like(hphob)


    ############################### For 溯源
    ## 对csv文件的处理：
    import pandas as pd
    ## 提取索引列表
    csv_file = pool_csv ## 路径
    pool_index = pd.read_csv(csv_file, usecols=["vert_idx"])
    pool_index = np.array(pool_index)
    pool_index = np.squeeze(pool_index) ## array格式，数据类型为numpy.int64

    ## 关联程度列表
    pool_impot = pd.read_csv(csv_file, usecols=["vert_impo"])
    pool_impot = np.array(pool_impot)
    pool_impot = np.squeeze(pool_impot) ## array格式，数据类型为numpy.float64(实际有6位浮点数)

    ## 写入特征集
    input_feat = np.zeros((len(pool_index), 5))
    input_feat[:len(pool_index), 0] = si[pool_index] ## shape index
    input_feat[:len(pool_index), 1] = hbond[pool_index] ## 氢键供受体
    input_feat[:len(pool_index), 2] = charge[pool_index] ## apbs
    input_feat[:len(pool_index), 3] = hphob[pool_index] ## 疏水性
    input_feat[:len(pool_index), 4] = pool_impot ## 关联程度标度值列表

    return input_feat, pool_index, np.copy(mesh.vertices), np.copy(mesh.faces), np.copy(normals)

    ######################################################
    ## 溯源返回格式：
    ##   五种特征（ddc替换成了标度值）；pool vert索引；vertices/faces/normals拷贝 
    ######################################################


##================================================##
##    补充功能：For pocket表面五层特征可视化 By 尚醇    ##
##               2025/04/07
##================================================##
def read_pocket(ply_fn, cutoff, all_pip_coords, all_pip_types):
    ## 参数格式：
    ## ply_fn: data_preparation/01-benchmark_surfaces/1AO7_AC.ply

    mesh = pymesh.load_mesh(ply_fn)

    # Normals: 
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1,n2,n3], axis=1)

    # Compute the principal curvature components for the shape index. 
    mesh.add_attribute("vertex_mean_curvature") ## PyMesh自带；标量场
    H = mesh.get_attribute("vertex_mean_curvature") 
    mesh.add_attribute("vertex_gaussian_curvature") ##PyMesh自带；标量场
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem<0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)


    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)

    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")

    # Hydropathy features
    hphob = mesh.get_attribute("vertex_hphob")

    ############################### For pocket特征可视化
    ## 提取pocket verts索引列表
    X = (np.copy(mesh.vertices))[:,0]
    Y = (np.copy(mesh.vertices))[:,1]
    Z = (np.copy(mesh.vertices))[:,2]

    xyz_coords = np.vstack([X, Y, Z]).T
    tree = spatial.KDTree(xyz_coords)


        ## 处理格式是string的pip原子坐标列表：转为浮点型
    pip_coords = []
    for i, coord_str in enumerate(all_pip_coords):
            coord_float = coord_str.strip().split()
            pip_coords.append(list(map(float,coord_float)))


        #############################
        ## 打标签 --> verts on pMHC surface within 3A from corresponding TCR
        ##   给此结构中符合条件的vertice（patch center vert）打上对应的TCR种类数字标签
        ##   最终的pmhc_labels形状: (n_vertices, 1)
        #############################
    for j, structure_pip in enumerate(all_pip_types):
            ## 注意：此列表只有一个元素（tcr type唯一） ## pip type唯一，Apr 2025
            pocket_points = tree.query_ball_point(pip_coords, float(cutoff)) 
            ## 250719, 7.0A for element P ## 选出较完整的antibody表面：cutoff: 3 => 4, 25Jul
            ## 251030, 4.0A for cdr3fp

            ## 此时，要搜索其邻居（距离3.0以内）的tcr_coords为多个点：
            ##   返回的pmhc_points为：包含(每点)邻居索引列表的形状元组的对象数组
            pocket_points_flatten = list(set([pp for p in pocket_points for pp in p]))  ## 返回符合条件的所有patch center vert索引
            ## list(set())：可以将原列表去重，并按从小到大排序
            print("Labeled pocket surface verts: "+str(len(pocket_points_flatten)))  ## 检查错误 ~350 vertices

    pool_index = np.array(pocket_points_flatten)
    ## 写入特征集
    input_feat = np.zeros((len(pool_index), 4))
    input_feat[:len(pool_index), 0] = hbond[pool_index] ## 氢键供受体
    input_feat[:len(pool_index), 1] = charge[pool_index] ## apbs
    input_feat[:len(pool_index), 2] = hphob[pool_index] ## 疏水性
    input_feat[:len(pool_index), 3] = si[pool_index] ## shape index
    ## Apr 2025, ddc依赖于各个patch的语境计算，在对pocket进行特征可视化时，不进行计算


    return input_feat, pool_index, np.copy(mesh.vertices), np.copy(mesh.faces), np.copy(normals)

    ######################################################
    ## For pocket 返回格式：
    ##   四种特征；pool vert索引；vertices/faces/normals拷贝 
    ######################################################


##================================================##
## 补充功能：For TCR/pMHC-FP 覆盖范围与表面特征可视化 By 尚醇 ##
##               2025/11/04
##================================================##

def read_fp_coverage(ply_fn, core_center_index, params):
    ## 参数格式：
    ## ply_fn: data_preparation/01-benchmark_surfaces/1AO7_AC.ply

    mesh = pymesh.load_mesh(ply_fn)

    # Normals: 
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1,n2,n3], axis=1)

    # Compute the angular and radial coordinates. ## For TCR/pMHC-FP
    rho, theta, neigh_indices, mask = compute_polar_coordinates(mesh, radius=params['max_distance'], max_vertices=params['max_shape_size'])
    #######################################
    ## rho/thera/mask shape均为：(n->即所有vert的数量,max_vertices->即设定好的计算量-->200）
    #   即：每一行代表一个以（行索引为索引值）的vert为中心的patch，存储其neigh的verts的测地极坐标信息
    #  格式：极径；极角；neigh索引；掩码
    #######################################

    # Compute the principal curvature components for the shape index. 
    mesh.add_attribute("vertex_mean_curvature") ## PyMesh自带；标量场
    H = mesh.get_attribute("vertex_mean_curvature") 
    mesh.add_attribute("vertex_gaussian_curvature") ##PyMesh自带；标量场
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem<0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)


    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)

    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")

    # Hydropathy features
    hphob = mesh.get_attribute("vertex_hphob")

    ## ==================== Find all neigh verts for residue-specific center points ==================== ##
    center_index = np.array(core_center_index, dtype=int)

    # --- Step 1. 收集所有 center 对应的邻域 verts（含自身）
    all_neighs = set()
    for vix in center_index:
        neigh_vix = np.array(neigh_indices[vix])
        # 只保留有效邻域顶点（mask[vix]==1 或 neigh_vix!=-1）
        valid_idx = np.where(mask[vix] == 1.0)[0]
        valid_neighs = neigh_vix[valid_idx]
        all_neighs.update(valid_neighs.tolist())
        all_neighs.add(vix)  # 确保中心点自身也包含在 patch 中

    # --- Step 2. 去重并转换为 numpy 数组
    fp_index = np.array(sorted(all_neighs), dtype=int)

    # --- Step 3. 计算中心标识（center_score）
    # np.isin 是最简洁高效的判断方式
    center_mask = np.isin(fp_index, center_index).astype(float)  # 1 for center, 0 for neighbor

    # --- Step 4. 写入特征矩阵
    input_feat = np.zeros((len(fp_index), 5))
    input_feat[:, 0] = hbond[fp_index]   # 氢键供受体特征
    input_feat[:, 1] = charge[fp_index]  # 电荷特征（APBS）
    input_feat[:, 2] = hphob[fp_index]   # 疏水性
    input_feat[:, 3] = si[fp_index]      # shape index
    input_feat[:, 4] = center_mask       # 是否为中心顶点

    # --- Step 5. 返回结果
    return (
        input_feat,           # [N_verts, 5]
        fp_index,             # 选中的 vert 索引（包括中心与邻域）
        np.copy(mesh.vertices),
        np.copy(mesh.faces),
        np.copy(normals)
    )

    ######################################################
    ## For TCR/pMHC-FP 返回格式：
    ##   四种表面特征 + center score；pool vert索引；
    ##   vertices/faces/normals 拷贝
    ######################################################


##================================================##
## 补充功能：For TCR/pMHC-FP Scoring patch数据文件存储 By 尚醇 ##
##               2025/11/07
##================================================##

def read_patch_from_surface(ply_fn, params, patch_index_list=None):
    ## 参数格式：
    ## ply_fn: data_preparation/01-benchmark_surfaces/1AO7_ABC/DE.ply
    """
    # Read data from a ply file -- decompose into patches. 
    # Returns (after patch selection):
    #   input_feat: [n, max_vertices, 5]
    #   rho, theta, mask: [n, max_vertices]
    #   neigh_indices: list of lists (neighbors of each selected patch)
    #   iface_labels: interface label
    #   vertices, faces, normals: mesh copy (unchanged)
    """

    mesh = pymesh.load_mesh(ply_fn)

    # Normals: 
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1, n2, n3], axis=1)

    # Compute the angular and radial coordinates. 
    rho, theta, neigh_indices, mask = compute_polar_coordinates(
        mesh, radius=params['max_distance'], max_vertices=params['max_shape_size']
    )

    # Compute the principal curvature components for the shape index. 
    mesh.add_attribute("vertex_mean_curvature")
    H = mesh.get_attribute("vertex_mean_curvature") 
    mesh.add_attribute("vertex_gaussian_curvature")
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    si = (k1 + k2) / (k1 - k2)
    si = np.arctan(si) * (2 / np.pi)

    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)

    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")

    # Hydropathy features
    hphob = mesh.get_attribute("vertex_hphob")

    # Iface labels (for ground truth only)     
    if "vertex_iface" in mesh.get_attribute_names():
        iface_labels = mesh.get_attribute("vertex_iface") 
    else:
        iface_labels = np.zeros_like(hphob)

    n = len(mesh.vertices)
    input_feat = np.zeros((n, params['max_shape_size'], 5))

    # Compute the input features for each patch.
    for vix in range(n):
        neigh_vix = np.array(neigh_indices[vix])
        patch_v = mesh.vertices[neigh_vix]
        patch_n = normals[neigh_vix]
        patch_cp = np.where(neigh_vix == vix)[0][0]  # central point
        mask_pos = np.where(mask[vix] == 1.0)[0]
        patch_rho = rho[vix][mask_pos]
        ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)

        input_feat[vix, :len(neigh_vix), 0] = si[neigh_vix]
        input_feat[vix, :len(neigh_vix), 1] = ddc
        input_feat[vix, :len(neigh_vix), 2] = hbond[neigh_vix]
        input_feat[vix, :len(neigh_vix), 3] = charge[neigh_vix]
        input_feat[vix, :len(neigh_vix), 4] = hphob[neigh_vix]

    # =========================================================
    # === Modified part: filter outputs for selected patches ===
    # =========================================================
    if patch_index_list is not None:
        patch_index_list = list(patch_index_list)  # ensure iterable
        input_feat = input_feat[patch_index_list, :, :]
        rho = rho[patch_index_list, :]
        theta = theta[patch_index_list, :]
        mask = mask[patch_index_list, :]
        neigh_indices = [neigh_indices[i] for i in patch_index_list]
        #iface_labels = iface_labels[patch_index_list]

        print(f"[INFO] Selected {len(patch_index_list)} patches from total {n} vertices.")
    else:
        print(f"[INFO] No patch_index_list provided, returning all {n} patches.")

    # =========================================================
    # Return
    # =========================================================
    return (
        input_feat,  # [n, max_vertices, 5]
        rho,         # [n, max_vertices]
        theta,       # [n, max_vertices]
        mask,        # [n, max_vertices]
        neigh_indices, 
        np.copy(mesh.vertices),
        np.copy(mesh.faces),
        np.copy(normals),
    )

    ######################################################
    ## 返回格式：（对于patch的neigh）
    ##   五种特征；极径；极角；掩码；neigh索引；iface（我们没有用到）；vertices拷贝
    ######################################################




    ''' ## 溯源时，不存在patch的概念：下程序段中
    ##  n --> 1; neigh_indices（格式：每个数字均为int形式的array） --> pool_index
    # n: number of patches, equal to the number of vertices.
    n = len(mesh.vertices)
    
    input_feat = np.zeros((n, params['max_shape_size'], 5))

    # Compute the input features for each patch.
    for vix in range(n):
        # Patch members.
        neigh_vix = np.array(neigh_indices[vix])

        # Compute the distance-dependent curvature for all neighbors of the patch. 
        patch_v = mesh.vertices[neigh_vix]
        patch_n = normals[neigh_vix]
        patch_cp = np.where(neigh_vix == vix)[0][0] # central point
        mask_pos = np.where(mask[vix] == 1.0)[0] # nonzero elements
        patch_rho = rho[vix][mask_pos] # nonzero elements of rho
        ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)        
        
        input_feat[vix, :len(neigh_vix), 0] = si[neigh_vix]     ## shape index: 相对local的几何特征 --> For every neigh vert（下同）
        input_feat[vix, :len(neigh_vix), 1] = ddc               ## distance-dependent curvature: 在patch内的几何特征（所以不是由vert索引即[neigh_vix]取得）
        input_feat[vix, :len(neigh_vix), 2] = hbond[neigh_vix]  ## hbond: 氢键供受体；物化特征
        input_feat[vix, :len(neigh_vix), 3] = charge[neigh_vix] ## charge: apbs；物化特征
        input_feat[vix, :len(neigh_vix), 4] = hphob[neigh_vix]  ## hydro: 疏水性；物化特征
        
    ##为实现patch特征可视化，补充拷贝mesh
    ##return input_feat, rho, theta, mask, neigh_indices, iface_labels, np.copy(mesh.vertices)
    return input_feat, rho, theta, mask, neigh_indices, iface_labels, np.copy(mesh.vertices), np.copy(mesh.faces), np.copy(normals)

    ######################################################
    ## 返回格式：（对于patch的neigh）
    ##   五种特征；极径；极角；掩码；neigh索引；iface（我们没有用到）；vertices拷贝
    ######################################################
    '''

# From a full shape in a full protein, extract a patch around a vertex.
# If patch_indices = True, then store the indices of all neighbors.
def extract_patch_and_coord(
    vix, shape, coord, max_distance, max_vertices, patch_indices=False
):
    # Member vertices are nonzero elements
    i, j = coord[np.int(vix), : coord.shape[1] // 2].nonzero()


    # D = np.squeeze(np.asarray(coord[np.int(vix),j].todense()))
    D = np.squeeze(np.asarray(coord[np.int(vix), : coord.shape[1] // 2].todense()))
    j = np.where((D < max_distance) & (D > 0))[0]
    max_dist_tmp = max_distance
    old_j = len(j)
    while len(j) > max_vertices:
        max_dist_tmp = max_dist_tmp * 0.95
        j = np.where((D < max_dist_tmp) & (D > 0))[0]
    #    print('j = {} {}'.format(len(j), old_j))
    D = D[j]
    patch = {}
    patch["X"] = shape["X"][0][j]
    patch["Y"] = shape["Y"][0][j]
    patch["Z"] = shape["Z"][0][j]
    patch["charge"] = shape["charge"][0][j]
    patch["hbond"] = shape["hbond"][0][j]
    patch["normal"] = shape["normal"][:, j]
    patch["shape_index"] = shape["shape_index"][0][j]
    if "hphob" in shape:
        patch["hphob"] = shape["hphob"][0][j]

    patch["center"] = np.argmin(D)

    j_theta = j + coord.shape[1] // 2
    theta = np.squeeze(np.asarray(coord[np.int(vix), j_theta].todense()))
    coord = np.concatenate([D, theta], axis=0)

    if patch_indices:
        return patch, coord, j
    else:
        return patch, coord


from scipy.spatial import cKDTree
# neigh1 and neigh2 are the precomputed indices; rho1 and rho2 their distances.
def compute_shape_complementarity(ply_fn1, ply_fn2, neigh1, neigh2, rho1, rho2, mask1, mask2, params):
    """
        compute_shape_complementarity: compute the shape complementarity between all pairs of patches. 
        ply_fnX: path to the ply file of the surface of protein X=1 and X=2
        neighX, rhoX, maskX: (N,max_vertices_per_patch) matrices with the indices of the neighbors, the distances to the center 
                and the mask

        Returns: vX_sc (2,N,10) matrix with the shape complementarity (shape complementarity 25 and 50) 
        of each vertex to its nearest neighbor in the other protein, in 10 rings.
    """
    # Mesh 1
    mesh1 = pymesh.load_mesh(ply_fn1)
    # Normals: 
    nx = mesh1.get_attribute("vertex_nx")
    ny = mesh1.get_attribute("vertex_ny")
    nz = mesh1.get_attribute("vertex_nz")
    n1 = np.stack([nx,ny,nz], axis=1)

    # Mesh 2
    mesh2 = pymesh.load_mesh(ply_fn2)
    # Normals: 
    nx = mesh2.get_attribute("vertex_nx")
    ny = mesh2.get_attribute("vertex_ny")
    nz = mesh2.get_attribute("vertex_nz")
    n2 = np.stack([nx,ny,nz], axis=1)

    w = params['sc_w']
    int_cutoff = params['sc_interaction_cutoff']
    radius = params['sc_radius']
    num_rings = 10
    scales = np.arange(0, radius, radius/10)
    scales = np.append(scales, radius)

    v1 = mesh1.vertices
    v2 = mesh2.vertices

    v1_sc = np.zeros((2,len(v1), 10))
    v2_sc = np.zeros((2,len(v2), 10))

    # Find all interface vertices
    kdt = cKDTree(v2)
    d, nearest_neighbors_v1_to_v2 = kdt.query(v1)
    # Interface vertices in v1
    interface_vertices_v1 = np.where(d < int_cutoff)[0]

    # Go through every interface vertex. 
    for cv1_iiix in range(len(interface_vertices_v1)):
        cv1_ix = interface_vertices_v1[cv1_iiix]
        assert (d[cv1_ix] < int_cutoff)
        # First shape complementarity s1->s2 for the entire patch
        patch_idxs1 = np.where(mask1[cv1_ix]==1)[0]
        neigh_cv1 = np.array(neigh1[cv1_ix])[patch_idxs1]
        # Find the point cv2_ix in s2 that is closest to cv1_ix
        cv2_ix = nearest_neighbors_v1_to_v2[cv1_ix]
        patch_idxs2 = np.where(mask2[cv2_ix]==1)[0]
        neigh_cv2 = np.array(neigh2[cv2_ix])[patch_idxs2]

        patch_v1 = v1[neigh_cv1]
        patch_v2 = v2[neigh_cv2]
        patch_n1 = n1[neigh_cv1]
        patch_n2 = n2[neigh_cv2]

        patch_kdt = cKDTree(patch_v1)
        p_dists_v2_to_v1, p_nearest_neighbor_v2_to_v1 = patch_kdt.query(patch_v2)
        patch_kdt = cKDTree(patch_v2)
        p_dists_v1_to_v2, p_nearest_neighbor_v1_to_v2 = patch_kdt.query(patch_v1)
        
        # First v1->v2
        neigh_cv1_p = p_nearest_neighbor_v1_to_v2
        comp1 = [np.dot(patch_n1[x], -patch_n2[neigh_cv1_p][x]) for x in range(len(patch_n1))]
        comp1 = np.multiply(comp1, np.exp(-w * np.square(p_dists_v1_to_v2)))
        # Use 10 rings such that each ring has equal weight in shape complementarity
        comp_rings1_25 = np.zeros(num_rings)
        comp_rings1_50 = np.zeros(num_rings)

        patch_rho1 = np.array(rho1[cv1_ix])[patch_idxs1]
        for ring in range(num_rings):
            scale = scales[ring]
            members = np.where((patch_rho1 >= scales[ring]) & (patch_rho1 < scales[ring + 1]))
            if len(members[0]) == 0:
                comp_rings1_25[ring] = 0.0
                comp_rings1_50[ring] = 0.0
            else:
                comp_rings1_25[ring] = np.percentile(comp1[members], 25)
                comp_rings1_50[ring] = np.percentile(comp1[members], 50)
        
        # Now v2->v1
        neigh_cv2_p = p_nearest_neighbor_v2_to_v1
        comp2 = [np.dot(patch_n2[x], -patch_n1[neigh_cv2_p][x]) for x in range(len(patch_n2))]
        comp2 = np.multiply(comp2, np.exp(-w * np.square(p_dists_v2_to_v1)))
        # Use 10 rings such that each ring has equal weight in shape complementarity
        comp_rings2_25 = np.zeros(num_rings)
        comp_rings2_50 = np.zeros(num_rings)

        # Apply mask to patch rho coordinates. 
        patch_rho2 = np.array(rho2[cv2_ix])[patch_idxs2]
        for ring in range(num_rings):
            scale = scales[ring]
            members = np.where((patch_rho2 >= scales[ring]) & (patch_rho2 < scales[ring + 1]))
            if len(members[0]) == 0:
                comp_rings2_25[ring] = 0.0
                comp_rings2_50[ring] = 0.0
            else:
                comp_rings2_25[ring] = np.percentile(comp2[members], 25)
                comp_rings2_50[ring] = np.percentile(comp2[members], 50)

        v1_sc[0,cv1_ix,:] = comp_rings1_25
        v2_sc[0,cv2_ix,:] = comp_rings2_25
        v1_sc[1,cv1_ix,:] = comp_rings1_50
        v2_sc[1,cv2_ix,:] = comp_rings2_50


    return v1_sc, v2_sc


def normalize_electrostatics(in_elec):
    """
        Normalize electrostatics to a value between -1 and 1
    """
    elec = np.copy(in_elec)
    upper_threshold = 3
    lower_threshold = -3
    elec[elec > upper_threshold] = upper_threshold
    elec[elec < lower_threshold] = lower_threshold
    elec = elec - lower_threshold
    elec = elec / (upper_threshold - lower_threshold)
    elec = 2 * elec - 1
    return elec

def mean_normal_center_patch(D, n, r):
    """
        Function to compute the mean normal of vertices within r radius of the center of the patch.
    """
    c_normal = [n[i] for i in range(len(D)) if D[i] <= r]
    mean_normal = np.mean(c_normal, axis=0, keepdims=True).T
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    return np.squeeze(mean_normal)

def compute_ddc(patch_v, patch_n, patch_cp, patch_rho):
    """
        Compute the distance dependent curvature, Yin et al PNAS 2009
            patch_v: the patch vertices
            patch_n: the patch normals
            patch_cp: the index of the central point of the patch 
            patch_rho: the geodesic distance to all members.
        Returns a vector with the ddc for each point in the patch.
    """
    ## 注意：在MaSIF中，不需要像PNAS 2009中计算几何指纹；
    ##  作者仍给出了计算几何指纹的脚本，在文件夹/gif_discriptors中
    n = patch_n
    r = patch_v
    i = patch_cp
    # Compute the mean normal 2.5A around the center point
    ni = mean_normal_center_patch(patch_rho, n, 2.5)
    dij = np.linalg.norm(r - r[i], axis=1)
    # Compute the step function sf:
    sf = r + n
    sf = sf - (ni + r[i])
    sf = np.linalg.norm(sf, axis=1)
    sf = sf - dij
    sf[sf > 0] = 1   ## 表示target vert处的normol相对中心平均normol“外扩”
    sf[sf < 0] = -1  ## 表示target vert处的normol相对中心平均normol“内收”
    sf[sf == 0] = 0
    # Compute the curvature between i and j
    dij[dij == 0] = 1e-8
    kij = np.divide(np.linalg.norm(n - ni, axis=1), dij)
    kij = np.multiply(sf, kij)
    # Ignore any values greater than 0.7 and any values smaller than 0.7
    kij[kij > 0.7] = 0 
    kij[kij < -0.7] = 0
    ########################################
    ## 上代码段参照PNAS2009(Fig.S1(f))理解
    #     P.S.：n0处代表坐标为0处；ni处代表patch中心vert；nj处代表待求vert
    ###  kij>0：“外扩”； kij<0：“内收”
    ## 返回：所有vert相对于中心vert的ddc: distance dependent curvature
    ########################################

    return kij




##================================================##
##         添加功能：Patch特征可视化 By 尚醇          ##
##================================================##
def extract_patch(mesh_v, mesh_f, mesh_n, neigh, cv):
    ## 注意：mesh_v --> verts[pid] --> np.copy(mesh.vertices)
    ##      mesh_f --> faces[pid] --> np.copy(mesh.faces)
    ##      mesh_n --> norms[pid] --> np.copy(normals)
    ##      neigh --> neigh_indices[pid] --> neigh_indices
    ##      cv(center vert) --> 选定的中心索引i in [0,100,500,1000,1500,2000]
    """ 
    Extract a patch from the mesh.
        neigh: the neighboring vertices.
    """
    n = len(mesh_v)
    subverts = mesh_v[neigh]

    normals = mesh_n
    subn = normals[neigh]


    # Extract triangulation. 
    
    m = np.zeros(n,dtype=int)

    # -1 if not there.
    m = m - 1 
    for i in range(len(neigh)):
        m[neigh[i]] = i
    f = mesh_f.astype(int)
    nf = len(f)
    
    neigh = set(neigh) 
    subf = [[m[f[i][0]], m[f[i][1]], m[f[i][2]]] for i in range(nf) \
             if f[i][0] in neigh and f[i][1] in neigh and f[i][2] in neigh]
    
    subfaces = subf
    return np.array(subverts), np.array(subn), np.array(subf) 
    ## 返回：patch内：所有vertice的信息；所有vertice的法向；vertice组成的face

def output_patch_feat(subv, subf, subn, i, neigh_i, feat_matx): 
    ## 注意：neigh_i --> neigh_indices[pid] --> neigh_indices
    ##     feat_matx --> input_feat[pid] --> patch五种特征矩阵：shape: (verts sum number, max_shape_index:200, 5)
    """ 
        For debugging purposes, save a patch to visualize it.
    """ 
    
    mesh = pymesh.form_mesh(subv, subf)
    n1 = subn[:,0]
    n2 = subn[:,1]
    n3 = subn[:,2]
    mesh.add_attribute('vertex_nx')
    mesh.set_attribute('vertex_nx', n1)
    mesh.add_attribute('vertex_ny')
    mesh.set_attribute('vertex_ny', n2)
    mesh.add_attribute('vertex_nz')
    mesh.set_attribute('vertex_nz', n3)


    si = feat_matx[i, :len(neigh_i), 0]
    mesh.add_attribute('vertex_si')
    mesh.set_attribute('vertex_si', si)

    ddc = feat_matx[i, :len(neigh_i), 1]
    mesh.add_attribute('vertex_ddc')
    mesh.set_attribute('vertex_ddc', ddc)

    hbond = feat_matx[i, :len(neigh_i), 2]
    mesh.add_attribute('vertex_hbond')
    mesh.set_attribute('vertex_hbond', hbond)

    charge = feat_matx[i, :len(neigh_i), 3]
    mesh.add_attribute('vertex_charge')
    mesh.set_attribute('vertex_charge', charge)

    hphob = feat_matx[i, :len(neigh_i), 4]
    mesh.add_attribute('vertex_hphob')
    mesh.set_attribute('vertex_hphob', hphob)

    pymesh.save_mesh('v{}_5feat.ply'.format(i), mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)



##================================================##
##         添加功能：Patch溯源 By 尚醇          ##
##               2023/10/24
##================================================##
def extract_pool(mesh_v, mesh_f, mesh_n, neigh):
    ## 注意：mesh_v --> verts[pid] --> np.copy(mesh.vertices)
    ##      mesh_f --> faces[pid] --> np.copy(mesh.faces)
    ##      mesh_n --> norms[pid] --> np.copy(normals)
    ##      neigh --> pool_index[pid] --> pool_index
    """ 
    Extract a patch from the mesh.
        neigh: the neighboring vertices.
    """
    n = len(mesh_v)
    subverts = mesh_v[neigh]

    normals = mesh_n
    subn = normals[neigh]


    # Extract triangulation. 
    
    m = np.zeros(n,dtype=int)

    # -1 if not there.
    m = m - 1 
    for i in range(len(neigh)):
        m[neigh[i]] = i
    f = mesh_f.astype(int)
    nf = len(f)
    
    neigh = set(neigh) 
    subf = [[m[f[i][0]], m[f[i][1]], m[f[i][2]]] for i in range(nf) \
             if f[i][0] in neigh and f[i][1] in neigh and f[i][2] in neigh]
    
    subfaces = subf
    return np.array(subverts), np.array(subn), np.array(subf) 
    ## 返回：pool中：所有vertice的信息；所有vertice的法向；vertice组成的face

def output_pool_feat(subv, subf, subn, neigh_i, feat_matx, plyfile): 
    ## 注意：neigh_i --> pool_index[pid] --> pool_index
    ##     feat_matx --> input_feat[pid] --> patch五种特征矩阵：shape: (len(pool_index), 5)
    """ 
        For debugging purposes, save a patch to visualize it.
    """ 
    
    mesh = pymesh.form_mesh(subv, subf)
    n1 = subn[:,0]
    n2 = subn[:,1]
    n3 = subn[:,2]
    mesh.add_attribute('vertex_nx')
    mesh.set_attribute('vertex_nx', n1)
    mesh.add_attribute('vertex_ny')
    mesh.set_attribute('vertex_ny', n2)
    mesh.add_attribute('vertex_nz')
    mesh.set_attribute('vertex_nz', n3)


    si = feat_matx[:len(neigh_i), 0]
    mesh.add_attribute('vertex_si')
    mesh.set_attribute('vertex_si', si)

    hbond = feat_matx[:len(neigh_i), 1]
    mesh.add_attribute('vertex_hbond')
    mesh.set_attribute('vertex_hbond', hbond)

    charge = feat_matx[:len(neigh_i), 2]
    mesh.add_attribute('vertex_charge')
    mesh.set_attribute('vertex_charge', charge)

    hphob = feat_matx[:len(neigh_i), 3]
    mesh.add_attribute('vertex_hphob')
    mesh.set_attribute('vertex_hphob', hphob)

    ## 溯源：特征中去掉patch相关的ddc；新增标签关联程度
    impot = feat_matx[:len(neigh_i), 4]
    mesh.add_attribute('vertex_impot')
    mesh.set_attribute('vertex_impot', impot)

    pymesh.save_mesh(plyfile, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)


##================================================##
##    补充功能：For pocket表面五层特征可视化 By 尚醇    ##
##               2025/04/08
##================================================##
def output_pocket_feat(subv, subf, subn, neigh_i, feat_matx, plyfile): 
    ## 注意：neigh_i --> pool_index[pid] --> pool_index
    ##     feat_matx --> input_feat[pid] --> patch五种特征矩阵：shape: (len(pool_index), 5)
    """ 
        For debugging purposes, save a patch to visualize it.
    """ 
    
    mesh = pymesh.form_mesh(subv, subf)
    n1 = subn[:,0]
    n2 = subn[:,1]
    n3 = subn[:,2]
    mesh.add_attribute('vertex_nx')
    mesh.set_attribute('vertex_nx', n1)
    mesh.add_attribute('vertex_ny')
    mesh.set_attribute('vertex_ny', n2)
    mesh.add_attribute('vertex_nz')
    mesh.set_attribute('vertex_nz', n3)

    hbond = feat_matx[:len(neigh_i), 0]
    mesh.add_attribute('vertex_hbond')
    mesh.set_attribute('vertex_hbond', hbond)

    charge = feat_matx[:len(neigh_i), 1]
    mesh.add_attribute('vertex_charge')
    mesh.set_attribute('vertex_charge', charge)

    hphob = feat_matx[:len(neigh_i), 2]
    mesh.add_attribute('vertex_hphob')
    mesh.set_attribute('vertex_hphob', hphob)
    
    si = feat_matx[:len(neigh_i), 3]
    mesh.add_attribute('vertex_si')
    mesh.set_attribute('vertex_si', si)

    pymesh.save_mesh(plyfile, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)


##================================================##
## 补充功能：For TCR/pMHC-FP 覆盖范围与表面特征可视化 By 尚醇 ##
##               2025/11/04
##================================================##
def output_fp_feat(subv, subf, subn, neigh_i, feat_matx, plyfile): 
    ## 注意：neigh_i --> pool_index[pid] --> pool_index
    ##     feat_matx --> input_feat[pid] --> patch五种特征矩阵：shape: (len(pool_index), 5)
    """ 
        For debugging purposes, save a patch to visualize it.
    """ 
    
    mesh = pymesh.form_mesh(subv, subf)
    n1 = subn[:,0]
    n2 = subn[:,1]
    n3 = subn[:,2]
    mesh.add_attribute('vertex_nx')
    mesh.set_attribute('vertex_nx', n1)
    mesh.add_attribute('vertex_ny')
    mesh.set_attribute('vertex_ny', n2)
    mesh.add_attribute('vertex_nz')
    mesh.set_attribute('vertex_nz', n3)

    hbond = feat_matx[:len(neigh_i), 0]
    mesh.add_attribute('vertex_hbond')
    mesh.set_attribute('vertex_hbond', hbond)

    charge = feat_matx[:len(neigh_i), 1]
    mesh.add_attribute('vertex_charge')
    mesh.set_attribute('vertex_charge', charge)

    hphob = feat_matx[:len(neigh_i), 2]
    mesh.add_attribute('vertex_hphob')
    mesh.set_attribute('vertex_hphob', hphob)
    
    si = feat_matx[:len(neigh_i), 3]
    mesh.add_attribute('vertex_si')
    mesh.set_attribute('vertex_si', si)

    ## fp coverage可视化：特征中去掉patch相关的ddc；新增
    center_mask = feat_matx[:len(neigh_i), 4] # 1 for center, 0 for neighbor
    mesh.add_attribute('vertex_ct')
    mesh.set_attribute('vertex_ct', center_mask)

    pymesh.save_mesh(plyfile, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)

