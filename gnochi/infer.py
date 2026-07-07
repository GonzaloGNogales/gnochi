import argparse
import datetime
import os
import typing
from pathlib import Path

import roma
import torch
import tqdm

from .io import (
    build_simulator_scene,
    condition_local_subject,
    read_subjects,
    save_json,
    save_obj,
    save_obj_scene,
    subject_to_lists,
)
from .model import GnocchiModel, LATENT_SIZE
from .pose_utils import default_device, move_subject_to_device, rotmat_to_sixdof, rotsixdof_to_mat

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CVAE_WEIGHTS = RELEASE_ROOT / "assets" / "weights" / "cvae.torch"
DEFAULT_CAPFIX_WEIGHTS = RELEASE_ROOT / "assets" / "weights" / "capfix.torch"
DEFAULT_SMPL_MODEL_PATH = RELEASE_ROOT / "assets" / "smpl"
DEFAULT_OUTPUT_DIR = RELEASE_ROOT / "outputs" / "inference"


def has_smpl_assets(path: Path) -> bool:
    return path.exists() and any(path.glob("*.pkl"))


def choose_generated_betas(subjects: typing.List[dict], condition_subject: int, target_subject: int) -> torch.Tensor:
    if 0 <= target_subject < len(subjects):
        return subjects[target_subject]["betas"].detach().cpu().clone()
    return subjects[condition_subject]["betas"].detach().cpu().clone()


def generate_reacting_subject(
        *,
        model: GnocchiModel,
        condition_subject: dict,
        latent: torch.Tensor,
        generated_betas: torch.Tensor,
        use_capfix: bool = True,
) -> typing.Tuple[dict, dict, dict]:
    condition_device = move_subject_to_device(condition_subject, model.device)

    condition_pose_mat = roma.rotvec_to_rotmat(condition_device["body_pose"].view(-1, 3))
    condition_pose_sixdof = rotmat_to_sixdof(condition_pose_mat)
    latent = latent.to(model.device).reshape(1, LATENT_SIZE)

    with torch.no_grad():
        generated_pose_sixdof_raw, generated_transl_raw, generated_orient_sixdof_raw = model.cvae.decode(
            latent,
            condition_pose_sixdof[None],
        )

        generated_pose_sixdof = generated_pose_sixdof_raw.clone()
        capfix_offset = torch.zeros_like(generated_pose_sixdof_raw)
        if use_capfix:
            if model.capfix is None:
                raise RuntimeError("CapFix was requested, but no CapFix weights were loaded.")
            capfix_offset = model.capfix(
                input_pose_sixdof=generated_pose_sixdof,
                input_global_orient_sixdof=generated_orient_sixdof_raw,
                input_transl=generated_transl_raw,
                conditional_pose_sixdof=condition_pose_sixdof[None],
            )
            generated_pose_sixdof = generated_pose_sixdof + capfix_offset

        generated_pose_mat_raw = rotsixdof_to_mat(generated_pose_sixdof_raw)
        generated_pose_mat = rotsixdof_to_mat(generated_pose_sixdof)
        generated_orient_mat = rotsixdof_to_mat(generated_orient_sixdof_raw)

    raw_subject = {
        "betas": generated_betas.detach().cpu().reshape(-1),
        "body_pose": roma.rotmat_to_rotvec(generated_pose_mat_raw[0]).detach().cpu().reshape(-1),
        "global_orient": roma.rotmat_to_rotvec(generated_orient_mat[0]).detach().cpu().reshape(-1),
        "transl": generated_transl_raw[0].detach().cpu().reshape(-1),
    }
    final_subject = {
        "betas": generated_betas.detach().cpu().reshape(-1),
        "body_pose": roma.rotmat_to_rotvec(generated_pose_mat[0]).detach().cpu().reshape(-1),
        "global_orient": roma.rotmat_to_rotvec(generated_orient_mat[0]).detach().cpu().reshape(-1),
        "transl": generated_transl_raw[0].detach().cpu().reshape(-1),
    }
    metadata = {
        "latent": latent.detach().cpu().reshape(-1).tolist(),
        "used_capfix": bool(use_capfix),
        "capfix_offset_sixdof": capfix_offset.detach().cpu().reshape(-1).tolist(),
    }
    return raw_subject, final_subject, metadata


def smpl_vertices(model: GnocchiModel, subject: dict) -> torch.Tensor:
    subject_device = move_subject_to_device(subject, model.device)
    body_pose_mat = roma.rotvec_to_rotmat(subject_device["body_pose"].view(-1, 3))
    global_orient_mat = roma.rotvec_to_rotmat(subject_device["global_orient"].view(1, 3))
    with torch.no_grad():
        vertices = model.smpl.forward(
            betas=subject_device["betas"][None],
            body_pose=body_pose_mat[None],
            global_orient=global_orient_mat,
            transl=subject_device["transl"][None],
            return_full_pose=True,
        ).vertices[0]
    return vertices.detach().cpu()


def infer_input_file(
        *,
        input_path: Path,
        output_dir: Path,
        model: GnocchiModel,
        latents: torch.Tensor,
        condition_subject_index: int,
        target_subject_index: int,
        use_capfix: bool,
) -> dict:
    subjects = read_subjects(input_path)
    if condition_subject_index < 0 or condition_subject_index >= len(subjects):
        raise ValueError(f"{input_path} has {len(subjects)} subject(s); condition index {condition_subject_index} is invalid.")

    condition_world = subjects[condition_subject_index]
    condition_local = condition_local_subject(condition_world)
    generated_betas = choose_generated_betas(subjects, condition_subject_index, target_subject_index)

    sample_dir = output_dir / input_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_json(sample_dir / "condition_subject.json", subject_to_lists(condition_local))

    faces = None
    condition_vertices = None
    if model.has_smpl:
        faces = torch.as_tensor(model.smpl.faces.astype("int64"))
        condition_vertices = smpl_vertices(model, condition_local)
        save_obj(sample_dir / "condition_body.obj", condition_vertices, faces)

    generations = []
    for variation_id, latent in enumerate(latents):
        raw_subject, generated_subject, metadata = generate_reacting_subject(
            model=model,
            condition_subject=condition_world,
            latent=latent,
            generated_betas=generated_betas,
            use_capfix=use_capfix,
        )

        suffix = f"{variation_id:03d}"
        scene = build_simulator_scene(condition_local, generated_subject)
        save_json(sample_dir / f"generated_subject_{suffix}.json", subject_to_lists(generated_subject))
        save_json(sample_dir / f"generated_subject_raw_{suffix}.json", subject_to_lists(raw_subject))
        save_json(sample_dir / f"generated_scene_{suffix}.json", scene)
        save_json(sample_dir / f"metadata_{suffix}.json", metadata)

        generation_summary = {
            "variation_id": variation_id,
            "generated_subject_file": f"generated_subject_{suffix}.json",
            "generated_scene_file": f"generated_scene_{suffix}.json",
            "metadata_file": f"metadata_{suffix}.json",
        }

        if model.has_smpl:
            generated_vertices = smpl_vertices(model, generated_subject)
            save_obj(sample_dir / f"generated_body_{suffix}.obj", generated_vertices, faces)
            save_obj_scene(
                sample_dir / f"scene_generated_{suffix}.obj",
                [(condition_vertices, faces), (generated_vertices, faces)],
            )
            generation_summary["generated_body_file"] = f"generated_body_{suffix}.obj"
            generation_summary["scene_generated_file"] = f"scene_generated_{suffix}.obj"

        generations.append(generation_summary)

    sample_summary = {
        "input_file": str(input_path),
        "condition_subject": condition_subject_index,
        "target_subject_for_betas": target_subject_index if 0 <= target_subject_index < len(subjects) else None,
        "num_generations": len(generations),
        "used_capfix": use_capfix,
        "has_mesh_outputs": model.has_smpl,
        "generations": generations,
    }
    save_json(sample_dir / "sample_summary.json", sample_summary)
    return sample_summary


def collect_input_files(input_path: typing.Optional[str], input_dir: typing.Optional[str]) -> typing.List[Path]:
    input_files = []
    if input_path:
        input_files.append(Path(input_path))
    if input_dir:
        input_files.extend(sorted(Path(input_dir).glob("*.json")))
    if not input_files:
        input_files.append(RELEASE_ROOT / "examples" / "ood_hiphop_frame000002.json")
    return input_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reacting/interacting SMPL poses with GNOCHI.")
    parser.add_argument("--input", default=None, help="Conditioning JSON file. Defaults to one bundled example.")
    parser.add_argument("--input-dir", default=None, help="Directory of conditioning JSON files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where outputs are written.")
    parser.add_argument("--num-generations", type=int, default=10, help="Number of reacting poses to sample per input.")
    parser.add_argument("--seed", type=int, default=20260408, help="Latent sampling seed.")
    parser.add_argument("--condition-subject", type=int, default=0, help="Subject index used as conditioning avatar.")
    parser.add_argument("--target-subject", type=int, default=1, help="Subject index used only for generated avatar betas when present.")
    parser.add_argument("--cvae-weights", default=str(DEFAULT_CVAE_WEIGHTS), help="Path to CVAE .torch weights.")
    parser.add_argument("--capfix-weights", default=str(DEFAULT_CAPFIX_WEIGHTS), help="Path to CapFix .torch weights.")
    parser.add_argument("--no-capfix", action="store_true", help="Disable CapFix and use raw CVAE samples.")
    parser.add_argument("--smpl-model-path", default=str(DEFAULT_SMPL_MODEL_PATH), help="Folder containing SMPL .pkl files.")
    parser.add_argument("--skip-meshes", action="store_true", help="Skip OBJ export even if SMPL assets are available.")
    parser.add_argument("--device", default=str(default_device()), help="Torch device, e.g. cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = collect_input_files(args.input, args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cvae_weights = Path(args.cvae_weights)
    capfix_weights = None if args.no_capfix else Path(args.capfix_weights)
    if not cvae_weights.exists():
        raise FileNotFoundError(f"CVAE weights not found: {cvae_weights}")
    if capfix_weights and not capfix_weights.exists():
        raise FileNotFoundError(f"CapFix weights not found: {capfix_weights}")

    smpl_model_path = None
    candidate_smpl = Path(args.smpl_model_path)
    if not args.skip_meshes and has_smpl_assets(candidate_smpl):
        smpl_model_path = candidate_smpl
    elif not args.skip_meshes:
        print(f"SMPL assets were not found in {candidate_smpl}. Inference will save JSON outputs only.")

    model = GnocchiModel(
        cvae_weights_path=cvae_weights,
        capfix_weights_path=capfix_weights,
        smpl_model_path=smpl_model_path,
        device=args.device,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    latents = torch.randn(args.num_generations, LATENT_SIZE, generator=generator, dtype=torch.float32)

    summaries = []
    for input_file in tqdm.tqdm(input_files, desc="GNOCHI inputs", unit="input"):
        summaries.append(
            infer_input_file(
                input_path=input_file,
                output_dir=output_dir,
                model=model,
                latents=latents,
                condition_subject_index=args.condition_subject,
                target_subject_index=args.target_subject,
                use_capfix=not args.no_capfix,
            )
        )

    run_summary = {
        "created_at": datetime.datetime.now().isoformat(),
        "num_inputs": len(input_files),
        "num_generations_per_input": args.num_generations,
        "seed": args.seed,
        "device": args.device,
        "cvae_weights": str(cvae_weights),
        "capfix_weights": str(capfix_weights) if capfix_weights else None,
        "used_capfix": not args.no_capfix,
        "smpl_model_path": str(smpl_model_path) if smpl_model_path else None,
        "samples": summaries,
    }
    save_json(output_dir / "summary.json", run_summary)
    print(f"Saved GNOCHI outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
