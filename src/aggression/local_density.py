import torch


def apply_local_density(
    dist_matrix,
    reference_features,
    calc_dist_matrix,
    k=16,
    eps=1e-12,
):
    """Scale test-to-reference distances by reference local density.

    System 2 LD uses one combined reference bank, without source/target
    separation. The local density for each reference item is the summed distance
    to its K nearest reference neighbors, excluding itself.
    """
    if k <= 0:
        raise ValueError(f"LD k must be positive; got {k}.")
    if reference_features.size(0) <= k:
        raise ValueError(
            "LD k must be smaller than the reference bank size; "
            f"got k={k}, n_ref={reference_features.size(0)}."
        )

    ref_ref_dist = calc_dist_matrix(reference_features, reference_features)
    topk_ref, _ = torch.topk(ref_ref_dist, k=k + 1, dim=1, largest=False)
    local_density = torch.sum(topk_ref[:, 1:], dim=1) + eps
    return dist_matrix / local_density.unsqueeze(0), local_density
