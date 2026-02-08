import argparse
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download


FILES = {
    "id_test.json": "timeomni1_id_test.json",
    "ood_test.json": "timeomni1_ood_test.json",
}


def main():
    parser = argparse.ArgumentParser(description="Download TimeOmni-1 testbed")
    parser.add_argument("--repo", type=str, default="anton-hugging/timeomni-1-testbed")
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src, dst in FILES.items():
        target = out_dir / dst
        if target.exists():
            print(f"skip {target}")
            continue
        cache_path = hf_hub_download(repo_id=args.repo, repo_type="dataset", filename=src)
        shutil.copyfile(cache_path, target)
        print(f"saved {target}")


if __name__ == "__main__":
    main()
