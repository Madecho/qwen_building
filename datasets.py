import json

import numpy as np
import open3d as o3d
from utils import to_o3d_pcd, yellow, blue
from utils import discretize, undiscretize
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from prompts import system_prompt, system_prompt_pcd
import torch
import random



def pcd_and_mesh(root, item, compute_pcd_normals=False, compute_mesh_normals=False):
    pcd = o3d.io.read_point_cloud("%s/pointclouds/%d.xyz" % (root, item))
    mesh = o3d.io.read_triangle_mesh("%s/building_mesh/%d.obj" % (root, item))

    vertices = np.asarray(mesh.vertices)
    v_min = np.min(vertices, axis=0, keepdims=True)
    vertices -= v_min

    xyz = np.asarray(pcd.points)
    xyz -= v_min

    if compute_pcd_normals:
        pcd.estimate_normals()

    if compute_mesh_normals:
        mesh.compute_vertex_normals()

    return pcd, mesh


def norm_pcd_and_mesh(pcd, mesh):
    v = np.asarray(mesh.vertices)
    xyz = np.asarray(pcd.points)

    v_min = np.min(v, axis=0, keepdims=True)
    v -= v_min
    xyz -= v_min

    fac = np.max(v, axis=0).max().item()+0.0001

    v /= fac
    xyz /= fac


def clip_pcd_by_mesh(pcd, mesh):
    v = np.asarray(mesh.vertices)
    x_min, x_max = v[:, 0].min().item(), v[:, 0].max().item()
    y_min, y_max = v[:, 1].min().item(), v[:, 1].max().item()
    z_min, z_max = v[:, 2].min().item(), v[:, 2].max().item()
    xyz = np.asarray(pcd.points)
    valid_inds = (x_min <= xyz[:, 0]) & (xyz[:, 0] <= x_max) & (y_min <= xyz[:, 1]) & (xyz[:, 1] <= y_max) & (z_min <= xyz[:, 2]) & (xyz[:, 2] <= z_max)

    new_xyz = xyz[valid_inds]
    new_normals = None
    if pcd.normals is not None and np.asarray(pcd.normals).shape[0] != 0:
        new_normals = np.asarray(pcd.normals)[valid_inds]

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_xyz)
    if new_normals is not None:
        new_pcd.normals = o3d.utility.Vector3dVector(new_normals)

    return new_pcd


def discretize_normalized_xyz(normed_xyz, bin_num=32):
    d_xyz = discretize(normed_xyz, continuous_range=[0, 1], num_discrete=bin_num)
    occupy = np.zeros((bin_num ** 3,))
    occupy[d_xyz[:, 2] * bin_num ** 2 + d_xyz[:, 1] * bin_num + d_xyz[:, 0]] = 1
    occupied = np.nonzero(occupy)[0].reshape(-1)

    z = occupied // bin_num ** 2
    y = occupied // bin_num % bin_num
    x = occupied % bin_num
    d_xyz = np.stack([x, y, z], axis=1)

    # print("xyz discretize, pts num: %d -> %d" % (normed_xyz.shape[0], d_xyz.shape[0]))
    return d_xyz


def discretize_normalized_mesh(normed_mesh, bin_num=32):
    d_xyz = discretize(np.asarray(normed_mesh.vertices), continuous_range=[0, 1], num_discrete=bin_num)
    occupy = np.zeros((bin_num ** 3,))
    occupy[d_xyz[:, 2] * bin_num ** 2 + d_xyz[:, 1] * bin_num + d_xyz[:, 0]] = 1
    occupied = np.nonzero(occupy)[0].reshape(-1)

    z = occupied // bin_num ** 2
    y = occupied // bin_num % bin_num
    x = occupied % bin_num
    no_repeat_d_xyz = np.stack([x, y, z], axis=1)

    xyz_to_id = -np.ones((bin_num ** 3,))
    xyz_to_id[no_repeat_d_xyz[:, 2] * bin_num ** 2 + no_repeat_d_xyz[:, 1] * bin_num + no_repeat_d_xyz[:, 0]] = np.arange(no_repeat_d_xyz.shape[0])

    # 修改每个三角形的顶点id
    e = np.array(normed_mesh.triangles)
    d_pt1 = d_xyz[e[:, 0]]
    d_pt2 = d_xyz[e[:, 1]]
    d_pt3 = d_xyz[e[:, 2]]

    pt1_new_id = xyz_to_id[d_pt1[:, 2] * bin_num ** 2 + d_pt1[:, 1] * bin_num + d_pt1[:, 0]]
    pt2_new_id = xyz_to_id[d_pt2[:, 2] * bin_num ** 2 + d_pt2[:, 1] * bin_num + d_pt2[:, 0]]
    pt3_new_id = xyz_to_id[d_pt3[:, 2] * bin_num ** 2 + d_pt3[:, 1] * bin_num + d_pt3[:, 0]]

    # 顶点排序
    new_e = np.stack([pt1_new_id, pt2_new_id, pt3_new_id], axis=1)
    min_ind = np.argmin(new_e, axis=1)
    min_ind_eq_1 = (min_ind == 1)
    min_ind_eq_2 = (min_ind == 2)

    new_e[min_ind_eq_1] = np.concatenate([new_e[min_ind_eq_1, 1:], new_e[min_ind_eq_1, 0:1]], axis=1)
    new_e[min_ind_eq_2] = np.concatenate([new_e[min_ind_eq_2, 2:], new_e[min_ind_eq_2, 0:2]], axis=1)
    new_e = new_e.astype(np.int64)
    #
    # # 三角面去重
    #
    v_num = no_repeat_d_xyz.shape[0]
    e_occupy = np.zeros((v_num ** 3,))
    e_occupy[new_e[:, 0] * v_num ** 2 + new_e[:, 1] * v_num + new_e[:, 2]] = 1
    e_occupied = np.nonzero(e_occupy)[0].reshape(-1)

    pt1_id = e_occupied // v_num ** 2
    pt2_id = e_occupied // v_num % v_num
    pt3_id = e_occupied % v_num
    new_e = np.stack([pt1_id, pt2_id, pt3_id], axis=1).astype(np.int64)

    # 顶点不能重复
    valid_inds = (new_e[:, 0] != new_e[:, 1]) & (new_e[:, 0] != new_e[:, 2]) & (new_e[:, 1] != new_e[:, 2])
    new_e = new_e[valid_inds]

    # print(new_e)
    # print(no_repeat_d_xyz)

    # print("mesh discretize, pts num: %d -> %d, triangle num: %d -> %d" % (d_xyz.shape[0], no_repeat_d_xyz.shape[0], e.shape[0], new_e.shape[0]))
    # print("max e id: %d" % (new_e.max().item()))

    return no_repeat_d_xyz, new_e


def random_rotate_pcd_and_mesh_around_zaxis(pcd, mesh):
    v = np.asarray(mesh.vertices)
    xyz = np.asarray(pcd.points)

    euler_ab = np.random.rand(3) * np.pi * 2
    euler_ab[1] = 0
    euler_ab[2] = 0
    rot_ab = Rotation.from_euler("zyx", euler_ab).as_matrix()

    rotated_xyz = np.matmul(rot_ab, xyz.T).T
    rotated_v = np.matmul(rot_ab, v.T).T

    xyz[:] = rotated_xyz
    v[:] = rotated_v



class BuildingDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.items = np.load("face_num_less_64_items.npy").tolist()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        pcd, mesh = pcd_and_mesh(self.root, self.items[item], compute_pcd_normals=False, compute_mesh_normals=False)
        pcd = clip_pcd_by_mesh(pcd, mesh)

        random_rotate_pcd_and_mesh_around_zaxis(pcd, mesh)
        norm_pcd_and_mesh(pcd, mesh)

        bin_num = 32

        d_v, tri = discretize_normalized_mesh(mesh, bin_num)

        xyz = np.asarray(pcd.points)
        d_xyz = discretize_normalized_xyz(xyz, bin_num=bin_num)

        return d_xyz, d_v, tri

        # d_mesh = o3d.geometry.TriangleMesh()
        # d_mesh.vertices = o3d.utility.Vector3dVector(d_v.astype(np.float64))
        # d_mesh.triangles = o3d.utility.Vector3iVector(tri)
        # d_mesh.compute_vertex_normals()
        #
        # d_pcd = to_o3d_pcd(d_xyz.astype(np.float64))
        # d_pcd.estimate_normals()
        #
        # print(d_xyz)
        # print(d_v)
        #
        # o3d.visualization.draw_geometries([d_mesh], width=1000, height=800, window_name="mesh")
        # o3d.visualization.draw_geometries([d_pcd, d_mesh], width=1000, height=800, window_name="pcd, mesh")


class PretrainBuildingDataset(Dataset):
    def __init__(self, root, tokenizer):
        self.ds = BuildingDataset(root)
        self.tokenizer = tokenizer
        # self.column_names = ['input_ids', 'labels']

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        _, v, tri = self.ds[item]
        v_num = v.shape[0]
        if np.random.rand(1).item() < 0.2:
            user_prompt = "Please randomly create a 3D building model."
        else:
            user_prompts = [
                "Please randomly create a 3D building model with %d vertices." % v_num,
                "Please help me randomly create a 3D building, n=%d." % v_num,
                "Please randomly create a 3D building, n=%d." % v_num,
                "Please randomly create a 3D building, the number of vertices n is %d." % v_num,
            ]
            user_prompt = random.choice(user_prompts)
        q_text = self.tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
            tokenize=False,
            add_generation_prompt=True
        )
        a_text = self.get_3d_building_text(v, tri) + self.tokenizer.eos_token
        # print(q_text+a_text)

        q_inp_inds = self.tokenizer(q_text)["input_ids"]
        a_inp_inds = self.tokenizer(a_text)["input_ids"]
        # print(len(q_inp_inds), len(a_inp_inds))

        input_inds = q_inp_inds + a_inp_inds
        labels = [self.tokenizer.pad_token_id] * len(q_inp_inds) + a_inp_inds

        input_inds = input_inds[:-1]
        labels = labels[1:]

        return input_inds, labels

    def get_3d_building_text(self, v, tri):
        text = "vertices:\n"
        for i in range(v.shape[0]):
            vi = v[i]
            # text += "v(%d): %d %d %d\n" % (i, vi[0].item(), vi[1].item(), vi[2].item())
            text = "v=" + json.dumps(v.tolist(), separators=(',', ':')) + "\n"
        text += "triangles:\n"
        for i in range(tri.shape[0]):
            fi = tri[i]
            # text += "f(%d): %d %d %d" % (i, fi[0].item(), fi[1].item(), fi[2].item())
            text += "f=" + json.dumps(tri.tolist(), separators=(',', ':'))
            if i < tri.shape[0] - 1:
                text += "\n"
        return text

class PretrainDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        max_len = max(len(input_inds) for input_inds, labels in batch)
        input_ids = []
        labels = []
        for input_inds, label in batch:
            input_ids.append(input_inds + [self.tokenizer.pad_token_id] * (max_len - len(input_inds)))
            labels.append(label + [self.tokenizer.pad_token_id] * (max_len - len(label)))

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class PCDBuildingDataset(Dataset):
    def __init__(self, root, tokenizer):
        self.ds = BuildingDataset(root)
        self.tokenizer = tokenizer
        self.pcd_max_pts_num = 128

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        xyz, v, tri = self.ds[item]
        if xyz.shape[0] > self.pcd_max_pts_num:
            random_inds = np.random.permutation(xyz.shape[0])[:self.pcd_max_pts_num]
            xyz = xyz[random_inds]

        # d_mesh = o3d.geometry.TriangleMesh()
        # d_mesh.vertices = o3d.utility.Vector3dVector(v.astype(np.float64))
        # d_mesh.triangles = o3d.utility.Vector3iVector(tri)
        # d_mesh.compute_vertex_normals()
        #
        # d_pcd = to_o3d_pcd(xyz.astype(np.float64))
        # d_pcd.estimate_normals()
        #
        # o3d.visualization.draw_geometries([d_pcd], width=1000, height=800, window_name="mesh")
        # o3d.visualization.draw_geometries([d_pcd, d_mesh], width=1000, height=800, window_name="pcd, mesh")
        # v_pcd = to_o3d_pcd(v.astype(np.float64))
        # geos = [v_pcd]
        # print("v")
        # print(v)
        # print("tri")
        # print(tri)
        # for i in range(tri.shape[0]):
        #     fv = np.zeros((3, 3))
        #     fv[0, :] = v[tri[i, 0].item()].astype(np.float64)
        #     fv[1, :] = v[tri[i, 1].item()].astype(np.float64)
        #     fv[2, :] = v[tri[i, 2].item()].astype(np.float64)
        #     print(fv)
        #     f = o3d.geometry.TriangleMesh()
        #     f.vertices = o3d.utility.Vector3dVector(fv)
        #     f.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]).astype(np.int64))
        #     f.compute_vertex_normals()
        #     geos.append(f)
        #     o3d.visualization.draw_geometries(geos, width=1000, height=800, window_name="face one by one")

        xyz_txt = self.get_pcd_text(xyz)
        user_prompt = "Please create a 3D building model based on this point cloud:\n" + xyz_txt

        q_text = self.tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt_pcd},
            {"role": "user", "content": user_prompt}
        ],
            tokenize=False,
            add_generation_prompt=True
        )
        a_text = self.get_3d_building_text(v, tri) + self.tokenizer.eos_token
        # print(q_text+a_text)

        q_inp_inds = self.tokenizer(q_text)["input_ids"]
        a_inp_inds = self.tokenizer(a_text)["input_ids"]
        # print(len(q_inp_inds), len(a_inp_inds))

        input_inds = q_inp_inds + a_inp_inds
        # print("token length: %d" % len(input_inds))
        labels = [self.tokenizer.pad_token_id] * len(q_inp_inds) + a_inp_inds

        input_inds = input_inds[:-1]
        labels = labels[1:]

        return input_inds, labels

    #这里还是用原版
    # def get_3d_building_text(self, v, tri):
    #     text = "vertices:\n"
    #     for i in range(v.shape[0]):
    #         vi = v[i]
    #         text += "%d: %d %d %d\n" % (i, vi[0].item(), vi[1].item(), vi[2].item())
    #     text += "triangles:\n"
    #     for i in range(tri.shape[0]):
    #         fi = tri[i]
    #         text += "%d %d %d" % (fi[0].item(), fi[1].item(), fi[2].item())
    #         if i < tri.shape[0] - 1:
    #             text += "\n"
    #     return text

    def get_3d_building_text(self, v, tri):
        text = "vertices:\n"
        for i in range(v.shape[0]):
            vi = v[i]
            text += "v(%d): %d %d %d\n" % (i, vi[0].item(), vi[1].item(), vi[2].item())
        text += "triangles:\n"
        for i in range(tri.shape[0]):
            fi = tri[i]
            text += "f(%d): %d %d %d" % (i, fi[0].item(), fi[1].item(), fi[2].item())
            if i < tri.shape[0] - 1:
                text += "\n"
        return text

    def get_pcd_text(self, xyz):
        text = ""
        for i in range(xyz.shape[0]):
            pti = xyz[i]
            text += "%d %d %d" % (pti[0].item(), pti[1].item(), pti[2].item())
            if i < xyz.shape[0] - 1:
                text += "\n"
        return text


if __name__ == '__main__':
    # ds = BuildingDataset("G:/Building3D")
    # random_inds = np.random.permutation(len(ds))
    # for i in range(len(ds)):
    #     item = random_inds[i]
    #     ds[item]

    from transformers import Qwen2Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained("./models/Qwen/Qwen3-0___6B-Base")
    # ds = PretrainBuildingDataset("G:/Building3D", tokenizer)
    # print(len(ds))
    # random_inds = np.random.permutation(len(ds))
    # for i in range(len(ds)):
    #     item = random_inds[i]
    #     ds[item]

    print("在读数据了！！！！")

    ds = PCDBuildingDataset("./dataset", tokenizer)
    random_inds = np.random.permutation(len(ds))
    for i in range(len(ds)):
        item = random_inds[i]
        ds[item]