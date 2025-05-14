import os
import pickle
import logging
from tqdm import tqdm
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from src.datasets.lyft.rasterizer import build_intersection_rasterizer
from src.datasets.lyft.dataset import IntersectionDataset


def filter_all_unsignalized_intersections(
    dataset_type: str, intersection_id: int
):
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')

    intersection_id = 'WTgZ' if intersection_id == 0 else 'sGK1'
    dataset = create_intersection_dataset(
        dataset_type=dataset_type,
        intersection_id=intersection_id
    )
    logging.info(f"Find dataset `{dataset_type}`")
    logging.info(f"Filter datasets `{intersection_id}`")
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
        logging.info(f"Pickle filtered datasets scenes into path `{pickle_path}`")


def create_intersection_dataset(dataset_type: str = "sample", intersection_id: str = "WTgZ"):
    """ 
    Create the datasets dataset from Lyft Level 5 dataset.

    Args
        dataset_type: str, "sample", "train" or "validate"
        intersection_id: str, the id of the chosen datasets dataset.
    Returns
        intersection_dataset: IntersectionDataset
    """
    assert dataset_type in ["sample", "training", "validate"], f"Got unexpected dataset named {dataset_type}."

    os.environ["L5KIT_DATA_FOLDER"] = "/home/gavin/DEV/Unsignalized_AV_HV/"
    zarr_data_path: str = f"./raw_data/lyft/scenes/{dataset_type}.zarr"
    config_yaml_path: str = f"raw_data/lyft/configs/config-{dataset_type}.yaml"

    # load zarr dataset
    dm = LocalDataManager()
    zarr_data_path = dm.require(key=zarr_data_path) # validate the path
    chunk_dataset = ChunkedDataset(path=zarr_data_path)
    chunk_dataset.open()
    cfg = load_config_data(path=os.environ["L5KIT_DATA_FOLDER"] + config_yaml_path)
    rast = build_intersection_rasterizer(cfg=cfg, data_manager=dm, intersection_id=intersection_id)
    return IntersectionDataset(cfg=cfg, zarr_dataset=chunk_dataset, rasterizer=rast)