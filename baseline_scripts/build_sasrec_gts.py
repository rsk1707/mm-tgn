# baseline_scripts/build_sasrec_gts.py

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from baseline_scripts.data_loader import load_gts_dataset  # uses GTS splits


def main():
    # Canonical name (no "-gts" here)
    dataset_name = "amazon-sports"
    gts_name = dataset_name + "-gts"

    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name=dataset_name)

    print("num_users:", dataset.num_users)
    print("num_items:", dataset.num_items)

    # Build per-user (timestamp, item) lists from GTS *train* split
    user_events = {u: [] for u in range(dataset.num_users)}
    for inter in dataset.splits["train"]:  # Interaction(user, item, timestamp)
        user_events[inter.user].append((inter.timestamp, inter.item))

    # Where SASRec expects data/%s.txt (relative to sasrec/main.py)
    out_dir = os.path.join(PROJECT_ROOT, "sasrec", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gts_name}.txt")

    with open(out_path, "w") as f:
        for u in range(dataset.num_users):
            events = user_events[u]
            if not events:
                continue
            # sort by time so User[u] is in chronological order
            events.sort(key=lambda x: x[0])
            for _, item in events:
                u_1 = u + 1          # SASRec is 1-based
                i_1 = item + 1
                f.write(f"{u_1} {i_1}\n")

    print("Wrote SASRec data file:", out_path)
    print("  format: one interaction per line, 'user item' (1-based IDs)")
    print("  interactions come ONLY from GTS train split")


if __name__ == "__main__":
    main()
