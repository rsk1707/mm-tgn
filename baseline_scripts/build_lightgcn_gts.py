import os
import sys

# python baseline_scripts/build_lightgcn_gts.py

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from baseline_scripts.data_loader import load_gts_dataset


def write_user_item_txt(path, user_items_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for u, items in user_items_dict.items():
            if not items:
                continue
            line = str(u) + " " + " ".join(str(i) for i in items)
            f.write(line + "\n")


def main():
    dataset_name = "amazon-sports" # CHANGE TO WHAT YOU WANT
    gts_name = dataset_name + "-gts"

    dataset = load_gts_dataset(
        root_dir=PROJECT_ROOT,
        dataset_name=dataset_name,
    )

    print("num_users:", getattr(dataset, "num_users", None))
    print("num_items:", getattr(dataset, "num_items", None))

    def stats(name, user_items):
        num_users = len(user_items)
        num_interactions = sum(len(v) for v in user_items.values())
        print(f"{name}: users={num_users}, interactions={num_interactions}")

    stats("train", dataset.user_train_items)
    stats("val", dataset.user_val_items)
    stats("test", dataset.user_test_items)

    out_dir = os.path.join(PROJECT_ROOT, "lightgcn", "Data", gts_name)
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.txt")
    write_user_item_txt(train_path, dataset.user_train_items)

    test_path = os.path.join(out_dir, "test.txt")
    write_user_item_txt(test_path, dataset.user_test_items)

    val_path = os.path.join(out_dir, "val.txt")
    write_user_item_txt(val_path, dataset.user_val_items)

    print("Wrote LightGCN data to:", out_dir)
    print("  train.txt from GTS train")
    print("  test.txt  from GTS test")
    print("  val.txt   from GTS val (not used by code, just for you)")


if __name__ == "__main__":
    main()
