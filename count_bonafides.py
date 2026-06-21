import argparse
import random
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
        p for p in root_dir.rglob("*")
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
        "--reserved-identities",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    if args.reserved_identities > len(all_bonafides):
        raise ValueError(
            f"Requested {args.reserved_identities} reserved identities, "
            f"but only found {len(all_bonafides)} unique bona fide names."
        )

    rng = random.Random(args.seed)
    reserved_bonafides = set(rng.sample(all_bonafides, args.reserved_identities))
    remaining_bonafides = set(all_bonafides) - reserved_bonafides

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

    print(f"Total morph files found: {len(morph_paths)}")
    print(f"Valid morph filenames parsed: {len(parsed_morphs)}")
    print(f"Invalid morph filenames: {len(invalid_files)}")
    print(f"Total unique bona fide names: {len(all_bonafides)}")
    print(f"Reserved bona fide names: {len(reserved_bonafides)}")
    print(f"Remaining bona fide names: {len(remaining_bonafides)}")
    print()
    print(f"Morphs fully inside reserved group: {reserved_morphs}")
    print(f"Morphs fully inside remaining group: {remaining_morphs}")
    print(f"Morphs crossing reserved/remaining groups: {mixed_morphs}")
    print()
    print(f"Sanity check total: {reserved_morphs + remaining_morphs + mixed_morphs}")


if __name__ == "__main__":
    main()