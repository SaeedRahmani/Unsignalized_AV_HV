import pickle
import logging
import argparse
from tqdm import tqdm
from src.util import lyft_loader


def find_lyft_intersection_scene_ids(
    dataset_type: str, intersection_id: int
):
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')

    intersection_id = 'WTgZ' if intersection_id == 0 else 'sGK1'
    dataset = lyft_loader(
        dataset_type=dataset_type,
        intersection_id=intersection_id
    )
    logging.info(f"Find dataset `{dataset_type}`")
    logging.info(f"Filter intersection `{intersection_id}`")
    logging.info(f"#Frames {len(dataset)}")
    logging.info(f"#Fcenes {len(dataset.cumulative_sizes)}")
    print(dataset)

    frame_indices = []
    scene_indices = []

    for frame_index, frame in tqdm(
            enumerate(dataset), total=len(dataset), unit="frame", desc="frames"):
        if frame["is_intersection_included"]:
            scene_indices.append(frame["scene_index"])
            frame_indices.append(frame_index)

    pickle_path = f'./processed/lyft/filtered_intersection_scenes-{dataset_type}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(set(scene_indices), f)
        logging.info(f"Pickle filtered intersection scenes into path `{pickle_path}`")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter intersection scenes from Lyft dataset"
    )

    parser.add_argument("--type",
        default="sample", type=str, required=True, choices=["sample", "train", "validate"],
        help="Type of dataset to load, choose from 'sample', 'train', 'validate'",
    )
    parser.add_argument("--id",
        default=0, type=int, required=True, choices=[0, 1],
        help="ID of dataset to load, 0 for 'WTgZ' and 1 for 'sGK1'"
    )
    args = parser.parse_args()
    find_lyft_intersection_scene_ids(args.type, args.id)