import pickle
from pprint import pprint
from typing import Tuple, List, Dict, Any

from l5kit.configs.config import load_metadata, load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer

HVHV_conflict = {
	"cross": {
		"turnleft&turnleft": [],  # HVHV case 1
		"straight&turnleftRight": [],  # HVHV case 2
		"straight&turnleftBottom": [],  # HVHV case 3
	},
	"merge": {
		"turnleft&turnright": [],  # HVHV case 4
		"straight&turnright": [],  # HVHV case 5
		"straight&turnleft": [],  # HVHV case 6
	},
}

AVHV_conflict = {
	"merge": {
		"turnleft&turnright": [],
		"straight&turnright": [],
	},
	"cross": {
		"turnleft&turnleft": [],
		"straight&turnleft": [],
	}
}

# Junction 2
HVHV_conflict_v2 = {
	"cross": {
		"turnLeftFromLeft&turnLeftFromTop": [],  	# HVHV case 1
		"goLeft&turnLeftFromLeft": [],  				# HVHV case 2
		"goLeft&turnLeftFromTop": [],  			# HVHV case 3
	},
	"merge": {
		"goLeft&turnRightFromTop": [],  			# HVHV case 4
		"goRight&turnLeftFromTop": [],  			# HVHV case 5
	},
}

AVHV_conflict_v2 = {
	"merge": {
		"goLeft&turnRightFromTop": [],
		"goRight&turnLeftFromTop": [],
	},
	"cross": {
		"goLeft&turnLeftFromLeft": [],
		"goLeft&turnLeftFromTop": [],
	}
}


def init_dataset(dataset_name: str = "train1") -> EgoDataset:
	"""
	Initialize the EgoDataset object, reading the dataset [train1, train2 or validate]
	:param dataset_name: name of the dataset
	:return: EgoDataset object
	"""
	cfg = load_config_data(f"./yaml_config/visualisation_config_{dataset_name}.yaml")
	dm = LocalDataManager()
	dataset_path = dm.require(cfg["val_data_loader"]["key"])
	zarr_dataset = ChunkedDataset(dataset_path)
	zarr_dataset.open()
	rast = build_rasterizer(cfg, dm)
	dataset = EgoDataset(cfg, zarr_dataset, rast)
	return dataset


def load_pickle(dataset_name: str = "train2") -> Tuple[List[Any], Any]:
	"""
	Load the pickle file [train1, train2 or validate]
	:param dataset_name: name of the dataset
	:return: a list containing the scene indices and a list containing the frame indices
	"""
	with open(f"./pickle_backup/scenes/junction1/frame/{dataset_name}_frames_including_intersection.pkl", 'rb') as file:
		frame_indices = pickle.load(file)
	with open(f"./pickle_backup/scenes/junction1/scene/{dataset_name}_scenes_including_intersection.pkl", 'rb') as file:
		scene_indices = pickle.load(file)

	return list(scene_indices), frame_indices

def load_pickle_v2(dataset_name: str = "train2") -> Tuple[List[Any], Any]:
	"""
	Load the pickle file [train2 or validate]
	:param dataset_name: name of the dataset
	:return: a list containing the scene indices and a list containing the frame indices
	"""
	with open(f"./pickle_backup/scenes/junction2/frame/{dataset_name}_frames_including_intersection_v2.pkl", 'rb') as file:
		frame_indices = pickle.load(file)
	with open(f"./pickle_backup/scenes/junction2/scene/{dataset_name}_scenes_including_intersection_v2.pkl", 'rb') as file:
		scene_indices = pickle.load(file)

	return list(scene_indices), frame_indices

def save_pickle(obj, name: str):
	with open(f"./pickle_backup/{name}.pkl", 'wb') as file:
		pickle.dump(obj, file)

# def save_pickle_v2(obj, name: str):
# 	with open(f"./pickle_backup/{name}.pkl", 'wb') as file:
# 		pickle.dump(obj, file)

def report_AVHV_conflicts(collection: dict):
	assert isinstance(collection, dict)
	pprint({
		"merge": {
			"#turnleft&turnright": len(collection["merge"]["turnleft&turnright"]),
			"#straight&turnright": len(collection["merge"]["straight&turnright"])},
		"cross": {
			"#turnleft&turnleft": len(collection["cross"]["turnleft&turnleft"]),
			"#straight&turnleft": len(collection["cross"]["straight&turnleft"])},
	})
	num_AVHV_conflicts = (
			len(collection["merge"]["turnleft&turnright"]) +
			len(collection["merge"]["straight&turnright"]) +
			len(collection["cross"]["turnleft&turnleft"]) +
			len(collection["cross"]["straight&turnleft"]))
	print(f"# AV-HV conflicts in total: {num_AVHV_conflicts}")


def report_HVHV_conflicts(collection: dict):
	assert isinstance(collection, dict)
	pprint({
		"merge": {
			"#turnleft&turnright": len(collection["merge"]["turnleft&turnright"]),
			"#straight&turnright": len(collection["merge"]["straight&turnright"]),
			"#straight&turnleft": len(collection["merge"]["straight&turnleft"])},
		"cross": {
			"#turnleft&turnleft": len(collection["cross"]["turnleft&turnleft"]),
			"#straight&turnleftRight": len(collection["cross"]["straight&turnleftRight"]),
			"#straight&turnleftBottom": len(collection["cross"]["straight&turnleftBottom"])}
	})

	num_HVHV_conflicts = (
			len(collection["merge"]["turnleft&turnright"]) +
			len(collection["merge"]["straight&turnright"]) +
			len(collection["merge"]["straight&turnleft"]) +
			len(collection["cross"]["turnleft&turnleft"]) +
			len(collection["cross"]["straight&turnleftRight"]) +
			len(collection["cross"]["straight&turnleftBottom"]))
	print(f"# HV-HV conflicts in total: {num_HVHV_conflicts}")


def load_potential_conflict_pickle_junction1(
		dataset_type: str,
		dataset_name: str,
		delta_time: int = 10,
) -> Dict:
	assert dataset_type in ["AVHV", "HVHV"], f"Invalid dataset type: {dataset_type}"
	assert dataset_name in ["train2", "validate"], f"Invalid dataset name: {dataset_name}"
	assert delta_time == 5 or delta_time == 10, f"Invalid delta time: {delta_time}"

	pkl_path = f"./pickle_backup/conflicts/junction1/{dataset_type}_conflict_{dataset_name}_pet{delta_time}s_buffer2m.pkl"
	with open(pkl_path, "rb") as f:
		potential_conflict_dict = pickle.load(f)

	assert isinstance(potential_conflict_dict, dict)

	return potential_conflict_dict

def load_potential_conflict_pickle_junction2(
		dataset_type: str,
		dataset_name: str,
		split: int = 1,
		delta_time: int = 10,
) -> Dict:
	assert dataset_type in ["AVHV", "HVHV"], f"Invalid dataset type: {dataset_type}"
	assert dataset_name in ["train2", "validate"], f"Invalid dataset name: {dataset_name}"
	assert delta_time == 5 or delta_time == 10, f"Invalid delta time: {delta_time}"
	assert split >= 1 and split <= 5 and type(split) == int
    
	# specify the file path for the pickle.
	if dataset_name == "train2":
		# need to specify the split version (from 1 to 5)
		pkl_path = f"./pickle_backup/conflicts/junction2/{dataset_type}_conflict_{dataset_name}_pet{delta_time}s_buffer2m_{split}e3.pkl"
	else:
		pkl_path = f"./pickle_backup/conflicts/junction2/{dataset_type}_conflict_{dataset_name}_pet{delta_time}s_buffer2m.pkl"
  
	# open and load the pickle
	with open(pkl_path, "rb") as f:
		potential_conflict_dict = pickle.load(f)
	assert isinstance(potential_conflict_dict, dict)

	return potential_conflict_dict