import numpy as np
import open3d as o3d
from utils import to_o3d_pcd
vs = """
v(0): 5 0 0
v(1): 63 9 0

"""

fs = """

"""

# for q in range(0, 10):
#     path = "./log_3d_buildings/%d.txt" % q
#     with open(path, "r") as f:
#         ls = f.readlines()
#     i, j = 0, 0
#     while i < len(ls):
#         if ls[i-1] == "<|im_start|>assistant\n" and ls[i] == "vertices:\n":
#             break
#         i += 1
#     j = i + 1
#     while j < len(ls):
#         if ls[j] == "triangles:\n":
#             break
#         j += 1
#     vs, fs = "", ""
#     for k in range(i+1, j):
#         vs += ls[k]
#     v_num = j - i - 1
#     for k in range(j+1, len(ls)):
#         fs += ls[k].replace("<|im_end|>", "")
#     print("item: %d" %q)
#     print("v: %d" % v_num)
#     print(vs)
#     print("f:")
#     print(fs)
#     # 找pcd字符串
#     i, j = 0, 0
#     while i < len(ls):
#         if ls[i] == "Please create a 3D building model based on this point cloud:\n":
#             break
#         i += 1
#     j = i + 1
#     while j < len(ls):
#         if ls[j] == "<|im_start|>assistant\n":
#             break
#         j += 1
#     pcd_s = ""
#     for k in range(i + 1, j):
#         pcd_s += ls[k].replace("<|im_end|>", "")
#     pts = [[float(x) for x in l.split(" ") if x != "" and not x.endswith(":")] for l in pcd_s.split("\n") if l != ""]
#     pts = np.array(pts)
#
#     x = [[float(x) for x in l.split(" ") if x != "" and not x.endswith(":")] for l in vs.split("\n") if l != ""]
#
#     v = np.array(x)
#
#     # v_pcd = to_o3d_pcd(v)
#     #
#     # o3d.visualization.draw_geometries([v_pcd], width=1000, height=800, window_name="pcd")
#
#     pcd = to_o3d_pcd(pts)
#
#     x = [[int(x) for x in l.split(" ") if x != "" and not x.endswith(":")] for l in fs.split("\n") if l != ""]
#     tri = np.array(x, dtype=np.int64)
#
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(v)
#     mesh.triangles = o3d.utility.Vector3iVector(tri)
#     mesh.compute_vertex_normals()
#     o3d.visualization.draw_geometries([pcd], width=1000, height=800, window_name="pcd")
#     o3d.visualization.draw_geometries([mesh, pcd], width=1000, height=800, window_name="pcd")

# path = "./log_3d_buildings_first/0.txt"
path = "./log_imgs/5.txt"
with open(path, "r") as f:
    ls = f.readlines()
i, j = 0, 0
while i < len(ls):
    if ls[i-1] == "<|im_start|>assistant\n" and ls[i] == "vertices:\n":
        break
    i += 1
j = i + 1
while j < len(ls):
    if ls[j] == "triangles:\n":
        break
    j += 1
vs, fs = "", ""
for k in range(i+1, j):
    vs += ls[k]
v_num = j - i - 1
for k in range(j+1, len(ls)):
    fs += ls[k].replace("<|im_end|>", "")
print("item: %d")
print("v: %d" % v_num)
print(vs)
print("f:")
print(fs)
# 找pcd字符串
i, j = 0, 0
while i < len(ls):
    if ls[i] == "Please create a 3D building model based on this point cloud:\n":
        break
    i += 1
j = i + 1
while j < len(ls):
    if ls[j] == "<|im_start|>assistant\n":
        break
    j += 1
pcd_s = ""
for k in range(i + 1, j):
    pcd_s += ls[k].replace("<|im_end|>", "")
pts = [[float(x) for x in l.split(" ") if x != "" and not x.endswith(":")] for l in pcd_s.split("\n") if l != ""]
pts = np.array(pts)

x = [[float(x) for x in l.split(" ") if x != "" and not x.endswith(":")] for l in vs.split("\n") if l != ""]
print(x)
v = np.array(x)

# v_pcd = to_o3d_pcd(v)
#
# o3d.visualization.draw_geometries([v_pcd], width=1000, height=800, window_name="pcd")

pcd = to_o3d_pcd(pts)

x = [[int(x) for x in l.split(" ") if x != "" and not x.endswith(":")] for l in fs.split("\n") if l != ""]
tri = np.array(x, dtype=np.int64)

# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(v)
# mesh.triangles = o3d.utility.Vector3iVector(tri)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([pcd], width=1000, height=800, window_name="pcd")
# o3d.visualization.draw_geometries([mesh, pcd], width=1000, height=800, window_name="pcd")

import torch
print(torch.__version__)

