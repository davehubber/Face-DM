import os
import torch
import torch.nn.functional as F

DATASET_ROOT = "encoded_ffhq256_semantic_split"
SAVE_DIR = os.path.join(DATASET_ROOT, "semantic_stats")
os.makedirs(SAVE_DIR, exist_ok=True)

# Match your current normalization
GLOBAL_MEAN_SCALAR = -0.04999
GLOBAL_STD_SCALAR = 0.28236

def load_split(split_name: str):
    path = os.path.join(DATASET_ROOT, "semantic", f"{split_name}_zsem.pt")
    pack = torch.load(path, map_location="cpu")

    z = pack["z_sem"].float()
    z = (z - GLOBAL_MEAN_SCALAR) / GLOBAL_STD_SCALAR

    return {
        "z": z,
        "sample_ids": list(pack["sample_ids"]),
        "source_paths": list(pack["source_paths"]),
        "relative_paths": list(pack.get("relative_paths", [""] * len(pack["sample_ids"]))),
        "split": split_name,
    }

def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if denom.item() == 0:
        return float("nan")
    return ((x * y).mean() / denom).item()

def rankdata_torch(x: torch.Tensor) -> torch.Tensor:
    # Simple rank transform without tie correction sophistication
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(x), dtype=torch.float32)
    return ranks

def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = rankdata_torch(x)
    ry = rankdata_torch(y)
    return pearson_corr(rx, ry)

def main():
    train = load_split("train")
    val = load_split("val")

    X = torch.cat([train["z"], val["z"]], dim=0)   # [N, D]
    N, D = X.shape

    split_names = (["train"] * len(train["z"])) + (["val"] * len(val["z"]))
    sample_ids = train["sample_ids"] + val["sample_ids"]
    source_paths = train["source_paths"] + val["source_paths"]
    relative_paths = train["relative_paths"] + val["relative_paths"]

    # Global mean and normalized mean direction
    mu = X.mean(dim=0)                              # [D]
    mu_dir = F.normalize(mu, dim=0)                 # [D]

    # PCA on centered data
    Xc = X - mu.unsqueeze(0)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    pc1 = F.normalize(Vh[0], dim=0)                 # [D]

    # Fix sign for consistency: make it point roughly toward +mean direction
    if torch.dot(pc1, mu_dir) < 0:
        pc1 = -pc1

    # Scores
    cosine_to_mean = F.cosine_similarity(X, mu_dir.unsqueeze(0), dim=1)  # [N]
    pc1_score = Xc @ pc1                                                 # [N]

    # Variance explained
    eigvals = (S ** 2) / max(N - 1, 1)
    explained_var_ratio_pc1 = (eigvals[0] / eigvals.sum()).item()

    # Correlations
    pearson = pearson_corr(cosine_to_mean, pc1_score)
    spearman = spearman_corr(cosine_to_mean, pc1_score)

    # Pairwise agreement test
    g = torch.Generator().manual_seed(1234)
    num_pairs = 10000
    idx1 = torch.randint(0, N, (num_pairs,), generator=g)
    idx2 = torch.randint(0, N - 1, (num_pairs,), generator=g)
    idx2 = idx2 + (idx2 >= idx1).long()

    cos_pick_1 = cosine_to_mean[idx1] >= cosine_to_mean[idx2]
    pc1_pick_1 = pc1_score[idx1] >= pc1_score[idx2]
    pair_agreement = (cos_pick_1 == pc1_pick_1).float().mean().item()

    # Extremes
    topk = 10
    cos_top = torch.topk(cosine_to_mean, k=topk).indices.tolist()
    cos_bottom = torch.topk(-cosine_to_mean, k=topk).indices.tolist()
    pc1_top = torch.topk(pc1_score, k=topk).indices.tolist()
    pc1_bottom = torch.topk(-pc1_score, k=topk).indices.tolist()

    def pack_rows(indices):
        rows = []
        for i in indices:
            rows.append({
                "global_index": i,
                "split": split_names[i],
                "sample_id": sample_ids[i],
                "source_path": source_paths[i],
                "relative_path": relative_paths[i],
                "cosine_to_mean": float(cosine_to_mean[i]),
                "pc1_score": float(pc1_score[i]),
            })
        return rows

    report = {
        "num_samples": N,
        "embedding_dim": D,
        "explained_var_ratio_pc1": explained_var_ratio_pc1,
        "pearson_cosine_vs_pc1": pearson,
        "spearman_cosine_vs_pc1": spearman,
        "pairwise_agreement": pair_agreement,
        "cosine_top": pack_rows(cos_top),
        "cosine_bottom": pack_rows(cos_bottom),
        "pc1_top": pack_rows(pc1_top),
        "pc1_bottom": pack_rows(pc1_bottom),
    }

    stats = {
        "global_mean_scalar": GLOBAL_MEAN_SCALAR,
        "global_std_scalar": GLOBAL_STD_SCALAR,
        "mu": mu,                   # [D]
        "mu_dir": mu_dir,           # [D]
        "pc1": pc1,                 # [D]
        "singular_values": S,       # optional, useful
        "explained_var_ratio_pc1": explained_var_ratio_pc1,
        "cosine_to_mean": cosine_to_mean,
        "pc1_score": pc1_score,
        "sample_ids": sample_ids,
        "source_paths": source_paths,
        "relative_paths": relative_paths,
        "splits": split_names,
        "report": report,
    }

    torch.save(stats, os.path.join(SAVE_DIR, "global_pc1_stats.pt"))

    print(f"N={N}, D={D}")
    print(f"Explained variance ratio (PC1): {explained_var_ratio_pc1:.6f}")
    print(f"Pearson(cosine_to_mean, pc1_score): {pearson:.6f}")
    print(f"Spearman(cosine_to_mean, pc1_score): {spearman:.6f}")
    print(f"Pairwise agreement: {pair_agreement:.6f}")
    print(f"Saved stats to: {os.path.join(SAVE_DIR, 'global_pc1_stats.pt')}")

if __name__ == "__main__":
    main()