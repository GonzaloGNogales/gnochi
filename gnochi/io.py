import json
import os
import typing

import numpy as np
import torch
from scipy.spatial.transform import Rotation

DEFAULT_SMPL_ROTATION_MATRIX = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0.0]])
DEFAULT_SIMULATOR_ROTATION_MATRIX = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]])
DEFAULT_NUM_BETAS = 10
DEFAULT_NUM_JOINTS = 24


def save_json(path: typing.Union[str, os.PathLike], data: typing.Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: typing.Union[str, os.PathLike]) -> typing.Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_subject(subject: dict) -> dict:
    return {
        "betas": torch.tensor(subject.get("betas", [0.0] * DEFAULT_NUM_BETAS), dtype=torch.float32).reshape(-1)[:DEFAULT_NUM_BETAS],
        "global_orient": torch.tensor(subject.get("global_orient", [0.0, 0.0, 0.0]), dtype=torch.float32).reshape(3),
        "transl": torch.tensor(subject.get("transl", [0.0, 0.0, 0.0]), dtype=torch.float32).reshape(3),
        "body_pose": torch.tensor(subject["body_pose"], dtype=torch.float32).reshape(23 * 3),
    }


def read_subjects(path: typing.Union[str, os.PathLike]) -> typing.List[dict]:
    loaded = load_json(path)
    if "body_pose" in loaded:
        return [normalize_subject(loaded)]

    num_avatar = int(loaded.get("numAvatar", 2))
    num_frames = int(loaded.get("numFrames", 1))
    num_betas = int(loaded.get("numBetas", DEFAULT_NUM_BETAS))
    num_joints = int(loaded.get("numJoints", DEFAULT_NUM_JOINTS))

    poses = torch.tensor(loaded["poses"], dtype=torch.float32).reshape(num_avatar, num_frames, num_joints, 3)
    transforms = torch.tensor(loaded.get("transforms", [0.0] * (num_avatar * num_frames * 3)), dtype=torch.float32)
    transforms = transforms.reshape(num_avatar, num_frames, 3)
    betas = torch.tensor(loaded.get("betas", [0.0] * (num_avatar * num_betas)), dtype=torch.float32)
    betas = betas.reshape(num_avatar, num_betas)

    subjects = []
    for subject_id in range(num_avatar):
        frame_id = 0
        subjects.append(
            {
                "betas": betas[subject_id].reshape(-1),
                "global_orient": poses[subject_id, frame_id, 0].reshape(3),
                "body_pose": poses[subject_id, frame_id, 1:].reshape(23 * 3),
                "transl": transforms[subject_id, frame_id].reshape(3),
            }
        )
    return subjects


def condition_local_subject(condition_subject: dict) -> dict:
    return {
        "betas": condition_subject["betas"].detach().cpu().reshape(-1),
        "global_orient": torch.zeros(3, dtype=torch.float32),
        "transl": torch.zeros(3, dtype=torch.float32),
        "body_pose": condition_subject["body_pose"].detach().cpu().reshape(-1),
    }


def subject_to_lists(subject: dict) -> dict:
    return {
        key: value.detach().cpu().reshape(-1).tolist()
        for key, value in subject.items()
    }


def smpl_to_simulator_subject(subject: dict) -> typing.Tuple[typing.List[float], typing.List[float], typing.List[float]]:
    change_reference = (
        Rotation.from_matrix(DEFAULT_SIMULATOR_ROTATION_MATRIX)
        * Rotation.from_matrix(DEFAULT_SMPL_ROTATION_MATRIX).inv()
    )

    global_orient = subject["global_orient"].detach().cpu().reshape(3).numpy()
    transl = subject["transl"].detach().cpu().reshape(3).numpy()
    body_pose = subject["body_pose"].detach().cpu().reshape(-1).numpy()

    sim_global_orient = (change_reference * Rotation.from_rotvec(global_orient)).as_rotvec()
    sim_transl = change_reference.apply(transl)
    sim_pose = np.concatenate([sim_global_orient, body_pose], axis=0)
    return sim_transl.tolist(), sim_pose.tolist(), subject["betas"].detach().cpu().reshape(-1).tolist()


def build_simulator_scene(condition_subject: dict, generated_subject: dict) -> dict:
    transforms = []
    poses = []
    betas = []

    for subject in [condition_subject, generated_subject]:
        subject_transforms, subject_poses, subject_betas = smpl_to_simulator_subject(subject)
        transforms.extend(subject_transforms)
        poses.extend(subject_poses)
        betas.extend(subject_betas)

    return {
        "numAvatar": 2,
        "numFrames": 1,
        "numBetas": DEFAULT_NUM_BETAS,
        "betas": betas,
        "numTransformAxes": 3,
        "transforms": transforms,
        "poses": poses,
        "objs": [],
        "gender": ["N", "N"],
        "numJoints": DEFAULT_NUM_JOINTS,
    }


def save_obj(path: typing.Union[str, os.PathLike], vertices: torch.Tensor, faces: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = []
    for vertex in vertices.detach().cpu():
        x, y, z = vertex.tolist()
        lines.append(f"v {x} {y} {z}\n")
    for face in faces.detach().cpu() + 1:
        x, y, z = face.tolist()
        lines.append(f"f {x} {y} {z}\n")
    with open(path, "w", encoding="utf-8") as file:
        file.writelines(lines)


def save_obj_scene(
        path: typing.Union[str, os.PathLike],
        meshes: typing.Iterable[typing.Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = []
    vertex_offset = 0
    for vertices, faces in meshes:
        vertices_cpu = vertices.detach().cpu()
        faces_cpu = faces.detach().cpu()
        for vertex in vertices_cpu:
            x, y, z = vertex.tolist()
            lines.append(f"v {x} {y} {z}\n")
        for face in faces_cpu + 1 + vertex_offset:
            x, y, z = face.tolist()
            lines.append(f"f {x} {y} {z}\n")
        vertex_offset += len(vertices_cpu)
    with open(path, "w", encoding="utf-8") as file:
        file.writelines(lines)
