import os
import sys
import numpy
from subprocess import Popen, PIPE

# 导入 pymesh
import pymesh

from default_config.global_vars import apbs_bin, pdb2pqr_bin, multivalue_bin
import random

"""
computeAPBS.py: Wrapper function to compute the Poisson Boltzmann electrostatics for a surface using APBS.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

def computeAPBS(vertices, pdb_file, tmp_file_base):
    ## 参数格式：所有PyMesh vertices的3D坐标；/tmp/1AO7_AC.pdb；/tmp/1AO7_AC
    """
        Calls APBS, pdb2pqr, and multivalue and returns the charges per vertex
    """
    ## pdb_file="/tmp/1AO7_AC.pdb"
    ## tmp_file_base="/tmp/1AO7_AC"
    fields = tmp_file_base.split("/")[0:-1]
    directory = "/".join(fields) + "/"
    filename_base = tmp_file_base.split("/")[-1]
    pdbname = pdb_file.split("/")[-1]
    ## 原参数格式不适用pdb2pqr 3.6.1版本：
    ## args = [pdb2pqr_bin, "--ff=parse", "--whitespace",\
    ##          "--noopt", "--apbs-input", pdbname, filename_base]
    
    args = [pdb2pqr_bin, "--ff=PARSE", "--whitespace", "--noopt",\
                "--apbs-input", filename_base + ".in", "--whitespace",\
                  pdbname, filename_base]   
    '''
    ## 25.12.14
    args = [
    pdb2pqr_bin,
    "--ff=PARSE",
    "--whitespace",
    "--noopt",
    "--apbs-input",
    pdbname,
    filename_base
]
'''

    ## 在python中调试：
    ## args = [pdb2pqr_bin, "--ff=PARSE", "--whitespace", "--noopt",\
    ##            "--apbs-input", "/tmp/1AO7_AC.in", "--whitespace",\
    ##             "/tmp/1AO7_AC.pdb", "/tmp/1AO7_AC" ]
    ## ！注意：在命令行python下，要调用另一程序需要使用Popen
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory) 
    stdout, stderr = p2.communicate()

    in_path = os.path.join(directory, filename_base + ".in")
    if not os.path.exists(in_path):
        raise RuntimeError(
            f"pdb2pqr did not generate APBS input file: {in_path}"
            )



    args = [apbs_bin, filename_base + ".in"]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    vertfile = open(directory + "/" + filename_base + ".csv", "w")
    for vert in vertices:
        vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))
    vertfile.close()

    args = [
        multivalue_bin,
        filename_base + ".csv",
        filename_base + ".dx",
        filename_base + "_out.csv",
    ]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    # Read the charge file
    chargefile = open(tmp_file_base + "_out.csv")
    charges = numpy.array([0.0] * len(vertices))
    for ix, line in enumerate(chargefile.readlines()):
        charges[ix] = float(line.split(",")[3])

    remove_fn = os.path.join(directory, filename_base)
    os.remove(remove_fn)
    os.remove(remove_fn+'.csv')
    os.remove(remove_fn+'.dx')
    os.remove(remove_fn+'.in')
    ## os.remove(remove_fn+'-input.p')
    os.remove(remove_fn+'_out.csv')

    return charges
