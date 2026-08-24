#!/bin/bash
# setup_env.sh


# SC 25.Dec
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


# ===== APBS 1.5 =====
APBS_ROOT="$BASE_DIR/software/APBS-1.5-linux64"


# 可执行文件
export APBS_BIN="$APBS_ROOT/bin/apbs"
export MULTIVALUE_BIN="$APBS_ROOT/share/apbs/tools/bin/multivalue"

# 动态库（关键！！）
export LD_LIBRARY_PATH="$APBS_ROOT/lib:$LD_LIBRARY_PATH"

# （可选）如果你有脚本里直接用 apbs 命令
export PATH="$APBS_ROOT/bin:$PATH"


# PDB2PQR
export PDB2PQR_BIN=$BASE_DIR/software/pdb2pqr_env/bin/pdb2pqr

# MSMS
export MSMS_BIN=$BASE_DIR/software/msms/msms_i86_64Linux2_2.6.1/msms.x86_64Linux2.2.6.1


# ======================
# Reduce (local build)
# ======================

export REDUCE_HOME="$BASE_DIR/software/reduce/reduce_src"
export REDUCE_BIN="$REDUCE_HOME/reduce"
# 可选：如果你后续用到 het dict
export REDUCE_HET_DICT="$BASE_DIR/software/reduce/reduce_wwPDB_het_dict.txt"
# 把 reduce 放进 PATH
export PATH="$REDUCE_HOME:$PATH"


# ======================
# PyMesh (local build)
# ======================
export PYMESH_PATH="$BASE_DIR/software/PyMesh"
# Python 模块路径（包含 __init__.py 的目录）
export PYTHONPATH=$PYMESH_PATH/python:$PYTHONPATH ## 包含包名的父目录
# C++ 扩展库依赖路径
export LD_LIBRARY_PATH=$PYMESH_PATH/python/pymesh/third_party/lib:$PYMESH_PATH/python/pymesh/lib:$BASE_DIR/software/compat_libs:$LD_LIBRARY_PATH


echo "[✔] Environment variables configured successfully."

