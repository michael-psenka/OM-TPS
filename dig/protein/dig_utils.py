import torch
import numpy as np
from numpy.linalg import svd
from rmsd import kabsch_rmsd, kabsch

# Reference vectors in the standard residue coordinate system
N_ref = np.array([1.45597958, 0.0, 0.0])
C_ref = np.array([-0.533655602, 1.42752619, 0.0])


def convert_to_CANC(tr, rot_mat):
    tr, rot_mat = tr.cpu(), rot_mat.cpu()
    CA = tr
    N_ref = torch.tensor([1.45597958, 0.0, 0.0])
    C_ref = torch.tensor([-0.533655602, 1.42752619, 0.0])
    N = torch.matmul(rot_mat.transpose(-1, -2), N_ref) + CA
    C = torch.matmul(rot_mat.transpose(-1, -2), C_ref) + CA
    return CA, N, C


def rotation_matrix_to_axis_angle(rot_mat):
    """
    Converts a batch of rotation matrices to axis-angle (rotation vector) representation.

    Args:
        rot_mat: Tensor of shape [batch_size, n_protein_residues, 3, 3] representing the rotation matrices.

    Returns:
        A tensor of shape [batch_size, n_protein_residues, 3] representing the rotation vectors (axis-angle).
    """
    batch_size, n_residues, _, _ = rot_mat.shape

    # Trace of the matrix
    trace = rot_mat[..., 0, 0] + rot_mat[..., 1, 1] + rot_mat[..., 2, 2]

    # Angle of rotation
    theta = torch.acos((trace - 1) / 2)

    # Avoid division by zero
    sin_theta = torch.sin(theta)
    sin_theta[sin_theta == 0] = 1e-8  # Small value to prevent division by zero

    # Rotation axis components
    r1 = (rot_mat[..., 2, 1] - rot_mat[..., 1, 2]) / (2 * sin_theta)
    r2 = (rot_mat[..., 0, 2] - rot_mat[..., 2, 0]) / (2 * sin_theta)
    r3 = (rot_mat[..., 1, 0] - rot_mat[..., 0, 1]) / (2 * sin_theta)

    rotation_vector = torch.stack((r1, r2, r3), dim=-1) * theta.unsqueeze(-1)
    return rotation_vector


def axis_angle_to_rotation_matrix(axis_angle):
    """
    Converts a batch of axis-angle vectors to rotation matrices using the Rodrigues formula.

    Args:
        axis_angle: Tensor of shape [batch_size, n_protein_residues, 3] representing the rotation vectors.

    Returns:
        A tensor of shape [batch_size, n_protein_residues, 3, 3] representing the rotation matrices.
    """
    batch_size, n_residues, _ = axis_angle.shape
    theta = torch.norm(axis_angle, dim=-1, keepdim=True).clamp_min(
        1e-8
    )  # Avoid division by zero
    k = axis_angle / theta

    # Compute the Rodrigues' rotation formula components
    kx = k[..., 0].unsqueeze(-1)
    ky = k[..., 1].unsqueeze(-1)
    kz = k[..., 2].unsqueeze(-1)

    K = torch.zeros((batch_size, n_residues, 3, 3), device=axis_angle.device)
    K[..., 0, 1] = -kz.squeeze()
    K[..., 0, 2] = ky.squeeze()
    K[..., 1, 0] = kz.squeeze()
    K[..., 1, 2] = -kx.squeeze()
    K[..., 2, 0] = -ky.squeeze()
    K[..., 2, 1] = kx.squeeze()

    I = torch.eye(3, device=axis_angle.device).unsqueeze(0).unsqueeze(0)
    rot_mat = (
        I
        + torch.sin(theta).unsqueeze(-1) * K
        + (1 - torch.cos(theta).unsqueeze(-1)) * torch.matmul(K, K)
    )

    return rot_mat


def slerp_rotation_matrices(rot_mat1, rot_mat2, path_length):
    """
    Perform spherical linear interpolation (SLERP) between two sets of rotation matrices using axis-angle representation.

    Args:
        rot_mat1: Tensor of shape [batch_size, n_protein_residues, 3, 3] representing the first rotation matrix.
        rot_mat2: Tensor of shape [batch_size, n_protein_residues, 3, 3] representing the second rotation matrix.
        path_length: Number of steps in the interpolation.

    Returns:
        A tensor of shape [path_length, batch_size, n_protein_residues, 3, 3] representing interpolated rotation matrices.
    """
    # Convert rotation matrices to axis-angle (rotation vector) representation

    n_protein_residues = rot_mat1.shape[1]

    axis_angle1 = rotation_matrix_to_axis_angle(rot_mat1)
    axis_angle2 = rotation_matrix_to_axis_angle(rot_mat2)

    # Generate interpolation steps (alphas) from 0 to 1
    alphas = torch.linspace(0, 1, path_length, device=rot_mat1.device).view(-1, 1, 1, 1)

    # Interpolate rotation vectors
    interpolated_vectors = (1 - alphas) * axis_angle1.unsqueeze(
        0
    ) + alphas * axis_angle2.unsqueeze(0)

    # Convert the interpolated rotation vectors back to rotation matrices
    noised_rot_mats = axis_angle_to_rotation_matrix(
        interpolated_vectors.reshape(-1, n_protein_residues, 3)
    ).reshape(path_length, -1, n_protein_residues, 3, 3)

    return noised_rot_mats


def pdb_to_tr_rots(pdb_file):
    """
    Converts an all-atom PDB structure to the positions of the alpha carbons (CA)
    and the rotation matrices for each residue, assuming standard amino acid backbone geometry.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        tr (torch.Tensor): Tensor of shape (L, 3), where L is the number of residues,
                           containing the positions of the Cα atoms.
        rot_mats (torch.Tensor): Tensor of shape (L, 3, 3), containing the rotation
                                 matrices for each residue.
    """

    # Dictionaries to hold atom coordinates
    residues = {}
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_seq = int(line[22:26])
                key = (chain_id, res_seq)
                if key not in residues:
                    residues[key] = {}
                if atom_name in ["N", "CA", "C"]:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    residues[key][atom_name] = np.array([x, y, z])

    # Lists to store translations and rotation matrices
    tr_list = []
    rot_mat_list = []

    all_N = []
    all_C = []
    all_CA = []

    for key in sorted(residues.keys()):
        residue = residues[key]
        if all(atom in residue for atom in ["N", "CA", "C"]):
            N = residue["N"]
            CA = residue["CA"]
            C = residue["C"]
            all_N.append(N)
            all_C.append(C)
            all_CA.append(CA)

            # Translation vector (position of the Cα atom)
            tr = CA
            tr_list.append(tr)

            # Local coordinate vectors
            N_rel = N - CA
            C_rel = C - CA

            # Reference vectors
            A = np.stack([N_ref, C_ref], axis=0)  # Shape: (2, 3)
            B = np.stack([N_rel, C_rel], axis=0)  # Shape: (2, 3)
            R_matrix = kabsch(A, B)  # Compute rotation matrix
            rot_mat_list.append(R_matrix)
        else:
            # If any backbone atom is missing, skip this residue
            print(f"Warning: Missing backbone atoms in residue {key}. Skipping.")
            continue

    # Convert lists to tensors
    tr = torch.from_numpy(np.stack(tr_list)).float()
    rot_mats = torch.from_numpy(np.stack(rot_mat_list)).float()

    all_CA = np.stack(all_CA)
    all_N = np.stack(all_N)
    all_C = np.stack(all_C)

    recon_CA, recon_N, recon_C = convert_to_CANC(tr, rot_mats)

    mol = torch.from_numpy(np.concatenate([all_CA, all_N, all_C], axis=0))
    recon_mol = torch.cat([recon_CA, recon_N, recon_C], axis=0)

    print(
        f"RMSD between original and reconstructed PDB endpoint: {kabsch_rmsd(mol.numpy(), recon_mol.numpy())} A)"
    )

    return tr, rot_mats, mol
