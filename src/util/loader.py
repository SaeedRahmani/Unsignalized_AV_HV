import os
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from src.util.lyft.rasterizer import build_intersection_rasterizer
from src.util.lyft.dataset import IntersectionDataset


def waymo_loader():
    return None

def lyft_loader(
    dataset_type: str = "sample",
    intersection_id: str = "WTgZ"
):
    """ 
    Load lyft sample/train/validate datasets, respectively.
    """
    assert dataset_type in ["sample", "train", "validate"], f"Got unexpected dataset named {dataset_type}."
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


if __name__ == "__main__":
    dataset = lyft_loader()
    print(dataset)