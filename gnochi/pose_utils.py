import torch


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rotmat_to_sixdof(rotmat: torch.Tensor) -> torch.Tensor:
    rotsixdof = rotmat[..., :2]
    return torch.flatten(rotsixdof.transpose(-1, -2), start_dim=-2)


def rotsixdof_to_mat(rotsixdof: torch.Tensor) -> torch.Tensor:
    a1 = rotsixdof[..., :3]
    a2 = rotsixdof[..., 3:]

    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3_raw = torch.linalg.cross(b1, a2, dim=-1)
    b3 = b3_raw / b3_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b2 = torch.linalg.cross(b3, b1, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def move_subject_to_device(subject: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in subject.items()}


def tensor_dict_to_lists(subject: dict) -> dict:
    return {key: value.detach().cpu().reshape(-1).tolist() for key, value in subject.items()}
