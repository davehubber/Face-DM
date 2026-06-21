import argparse
from collections import defaultdict
from pathlib import Path


def parse_morph_filename(filename: str) -> tuple:
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) >= 3 and parts[0] == "morphed":
        return parts[1], parts[2]

    return None, None


def discover_images(root_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        p
        for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--morph-root",
        type=str,
        default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/m15k_t/",
    )
    
    parser.add_argument(
        "--allowed-connectivity",
        type=int,
        default=1000,
        help="Maximum number of crossing morph files allowed between the sets (Damage Budget).",
    )

    parser.add_argument(
        "--max-fraction",
        type=float,
        default=0.20,
        help="Maximum fraction of total identities allowed in the reserved set (default: 0.20)",
    )

    args = parser.parse_args()

    morph_root = Path(args.morph_root)
    morph_paths = discover_images(morph_root)

    parsed_morphs = []
    all_bonafides = set()
    invalid_files = []

    for morph_path in morph_paths:
        src_a, src_b = parse_morph_filename(morph_path.name)

        if src_a is None or src_b is None:
            invalid_files.append(morph_path.name)
            continue

        parsed_morphs.append((morph_path, src_a, src_b))
        all_bonafides.add(src_a)
        all_bonafides.add(src_b)

    all_bonafides = sorted(all_bonafides)
    total_identities = len(all_bonafides)

    if total_identities == 0:
        print("No valid identities found.")
        return

    # -----------------------------------------------------------------
    # Build a Weighted Graph (Weights = Number of Morph Files)
    # -----------------------------------------------------------------
    identity_graph = defaultdict(lambda: defaultdict(int))
    total_weights = defaultdict(int)

    for _, src_a, src_b in parsed_morphs:
        if src_a != src_b:
            identity_graph[src_a][src_b] += 1
            identity_graph[src_b][src_a] += 1
            total_weights[src_a] += 1
            total_weights[src_b] += 1

    # Find peripheral seeds (identities with the fewest total morph files)
    seeds = sorted(all_bonafides, key=lambda x: total_weights[x])[:50]

    best_reserved_bonafides = set()
    best_internal_morphs = -1
    max_allowed_identities = int(total_identities * args.max_fraction)

    print(f"Searching for a cluster causing <= {args.allowed_connectivity} crossing morphs...")

    for seed in seeds:
        current_set = {seed}
        boundary = set(identity_graph[seed].keys())
        
        current_crossing = total_weights[seed]
        current_internal = 0
        
        # Check if a single seed already satisfies the budget
        if current_crossing <= args.allowed_connectivity:
            if current_internal > best_internal_morphs:
                best_internal_morphs = current_internal
                best_reserved_bonafides = set(current_set)
        
        while boundary and len(current_set) < max_allowed_identities:
            best_candidate = None
            best_resulting_crossing = float('inf')
            best_w_to_S = 0
            
            # Greedily look for the neighbor that keeps crossing morphs lowest
            for candidate in boundary:
                w_to_S = sum(identity_graph[candidate][n] for n in identity_graph[candidate] if n in current_set)
                w_to_outside = total_weights[candidate] - w_to_S
                resulting_crossing = current_crossing - w_to_S + w_to_outside
                
                if resulting_crossing < best_resulting_crossing:
                    best_resulting_crossing = resulting_crossing
                    best_candidate = candidate
                    best_w_to_S = w_to_S

            if best_candidate is None:
                break
                
            boundary.remove(best_candidate)
            current_set.add(best_candidate)
            
            for neighbor in identity_graph[best_candidate]:
                if neighbor not in current_set:
                    boundary.add(neighbor)
            
            current_internal += best_w_to_S
            current_crossing = best_resulting_crossing
            
            # If this cluster state is within our budget, track it!
            if current_crossing <= args.allowed_connectivity:
                # We want to maximize the number of morphs we successfully isolate
                if current_internal > best_internal_morphs:
                    best_internal_morphs = current_internal
                    best_reserved_bonafides = set(current_set)

    reserved_bonafides = best_reserved_bonafides
    remaining_bonafides = set(all_bonafides) - reserved_bonafides

    # -----------------------------------------------------------------
    # Final Evaluation Loop
    # -----------------------------------------------------------------
    reserved_morphs = 0
    remaining_morphs = 0
    mixed_morphs = 0

    for _, src_a, src_b in parsed_morphs:
        src_a_reserved = src_a in reserved_bonafides
        src_b_reserved = src_b in reserved_bonafides

        if src_a_reserved and src_b_reserved:
            reserved_morphs += 1
        elif not src_a_reserved and not src_b_reserved:
            remaining_morphs += 1
        else:
            mixed_morphs += 1

    # -----------------------------------------------------------------
    # Reporting Results
    # -----------------------------------------------------------------
    print("\n==================================================")
    print(f"   RESULTS FOR ALLOWED CONNECTIVITY: {args.allowed_connectivity}   ")
    print("==================================================")
    print(f"Total unique identities: {total_identities}")
    print(f"Reserved identities:     {len(reserved_bonafides)}")
    print(f"Remaining identities:    {len(remaining_bonafides)}")
    print()
    print(f"Morphs fully WITHIN reserved group:  {reserved_morphs}")
    print(f"Morphs fully WITHIN remaining group: {remaining_morphs}")
    print(f"Morphs CROSSING groups (The Damage): {mixed_morphs}")
    print("==================================================")


if __name__ == "__main__":
    main()
