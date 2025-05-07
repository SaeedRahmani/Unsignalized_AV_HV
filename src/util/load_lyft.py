import os
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data

def load_lyft_scenes(dataset_type: str = "sample") -> EgoDataset:
    """ 
    Load lyft sample/train/validate datasets, respectively.
    """
    assert dataset_type in ["sample", "train", "validate"], f"Got unexpected dataset named {dataset_type}."
    os.environ["L5KIT_DATA_FOLDER"] = "/home/gavin/DEV/Unsignalized_AV_HV/"    
    zarr_data_path: str = f"./raw_data/lyft/scenes/{dataset_type}.zarr"
    config_yaml_path: str = f"./raw_data/lyft/configs/config-{dataset_type}.yaml"

    # load config
    cfg = load_config_data(path=config_yaml_path)  

    # load zarr dataset
    dm = LocalDataManager()
    zarr_data_path = dm.require(key=zarr_data_path) # validate the path
    chunk_dataset = ChunkedDataset(path=zarr_data_path)
    chunk_dataset.open()
    rast = build_rasterizer(cfg=cfg, data_manager=dm)
    ego_dataset = EgoDataset(cfg=cfg, zarr_dataset=chunk_dataset, rasterizer=rast)
    
    return ego_dataset

if __name__ == "__main__":
    dataset = load_lyft_scenes()
    print(dataset)