import numpy as np
from typing import List, Dict, Tuple

from .conflict import Conflict
from l5kit_conflict.pickle.io import ( 
    load_potential_conflict_pickle_junction1,
    load_potential_conflict_pickle_junction2,
)

def load_l5kit_potential_conflicts_junction2(delta_time: int = 10, split: int = 1, is_postprocess: bool = True) -> Tuple[List[Conflict], List[Conflict]]:
    """
    Load the potential conflicts in `junction 2` from pkl.
    """
    AVHV_val_potential_conflict_dataset = load_potential_conflict_pickle_junction2(
        dataset_type="AVHV", dataset_name="validate", delta_time=delta_time, split=split)
    HVHV_val_potential_conflict_dataset = load_potential_conflict_pickle_junction2(
        dataset_type="HVHV", dataset_name="validate", delta_time=delta_time, split=split)
    AVHV_train_potential_conflict_dataset = load_potential_conflict_pickle_junction2(
        dataset_type="AVHV", dataset_name="train2", delta_time=delta_time, split=split)
    HVHV_train_potential_conflict_dataset = load_potential_conflict_pickle_junction2(
        dataset_type="HVHV", dataset_name="train2", delta_time=delta_time, split=split)

    AVHV_val_potential_conflict_dataset = remove_non_continuous_AV_trajectory(
        AVHV_val_potential_conflict_dataset, threshold=10)
    AVHV_train_potential_conflict_dataset = remove_non_continuous_AV_trajectory(
        AVHV_train_potential_conflict_dataset, threshold=10)
    print(
        f"HVHV dataset #samples: {get_dataset_sample(HVHV_train_potential_conflict_dataset, HVHV_val_potential_conflict_dataset)}")
    print(
        f"AVHV dataset #samples: {get_dataset_sample(AVHV_train_potential_conflict_dataset, AVHV_val_potential_conflict_dataset)}")

    # %% store the conflicts as key-value pairs (k: scene_indices, v: conflict object)
    potential_conflict_dataset: Dict = get_dataset_as_dict(
        HVHV_train_potential_conflict_dataset, HVHV_val_potential_conflict_dataset,
        AVHV_train_potential_conflict_dataset, AVHV_val_potential_conflict_dataset,
    )

    # %% post-process the scene with multiple potential conflicts
    print("4> Post-processing the complex potential conflicts ...")
    AVHV_conflict_dataset, HVHV_conflict_dataset = get_dataset_as_list(potential_conflict_dataset, is_postprocess)
    print(f"HVHV dataset #samples: {len(HVHV_conflict_dataset)}")
    print(f"AVHV dataset #samples: {len(AVHV_conflict_dataset)}")
    return AVHV_conflict_dataset, HVHV_conflict_dataset


def load_l5kit_potential_conflicts_junction1(delta_time: int = 10, is_postprocess: bool = True) -> Tuple[List[Conflict], List[Conflict]]:
    """ 
    Load the potential conflicts in `junction 1` from pkl. 
    """
    # %% read pickle files that saves all the potential conflicts under delta_time
    print("1> Loading pickle files ...")
    AVHV_val_potential_conflict_dataset = load_potential_conflict_pickle_junction1(
        dataset_type="AVHV", dataset_name="validate", delta_time=delta_time)
    HVHV_val_potential_conflict_dataset = load_potential_conflict_pickle_junction1(
        dataset_type="HVHV", dataset_name="validate", delta_time=delta_time)
    AVHV_train_potential_conflict_dataset = load_potential_conflict_pickle_junction1(
        dataset_type="AVHV", dataset_name="train2", delta_time=delta_time)
    HVHV_train_potential_conflict_dataset = load_potential_conflict_pickle_junction1(
        dataset_type="HVHV", dataset_name="train2", delta_time=delta_time)
    print(
        f"HVHV dataset #samples: {get_dataset_sample(HVHV_train_potential_conflict_dataset, HVHV_val_potential_conflict_dataset)}")
    print(
        f"AVHV dataset #samples: {get_dataset_sample(AVHV_train_potential_conflict_dataset, AVHV_val_potential_conflict_dataset)}")

    # %% remove unnecessary conflicts for AVHV/HVAV: only 1 crossing case
    print("2> Removing unnecessary conflicts ...")
    AVHV_val_potential_conflict_dataset["cross"].pop("turnleft&turnleft")
    AVHV_train_potential_conflict_dataset["cross"].pop("turnleft&turnleft")
    # %% remove unnecessary conflicts for HVHV: 2 cases one crossing, one merging
    HVHV_val_potential_conflict_dataset["cross"].pop("turnleft&turnleft")
    HVHV_train_potential_conflict_dataset["cross"].pop("turnleft&turnleft")
    HVHV_val_potential_conflict_dataset["merge"].pop("straight&turnleft")
    HVHV_train_potential_conflict_dataset["merge"].pop("straight&turnleft")
    print(
        f"HVHV dataset #samples: {get_dataset_sample(HVHV_train_potential_conflict_dataset, HVHV_val_potential_conflict_dataset)}")
    print(
        f"AVHV dataset #samples: {get_dataset_sample(AVHV_train_potential_conflict_dataset, AVHV_val_potential_conflict_dataset)}")

    # %% remove AVHV conflicts with non-continuous AV trajectories
    print("3> Removing AVHV conflicts with non-continuous AV trajectories ...")
    AVHV_val_potential_conflict_dataset = remove_non_continuous_AV_trajectory(
        AVHV_val_potential_conflict_dataset, threshold=10)
    AVHV_train_potential_conflict_dataset = remove_non_continuous_AV_trajectory(
        AVHV_train_potential_conflict_dataset, threshold=10)
    print(
        f"HVHV dataset #samples: {get_dataset_sample(HVHV_train_potential_conflict_dataset, HVHV_val_potential_conflict_dataset)}")
    print(
        f"AVHV dataset #samples: {get_dataset_sample(AVHV_train_potential_conflict_dataset, AVHV_val_potential_conflict_dataset)}")

    # %% store the conflicts as key-value pairs (k: scene_indices, v: conflict object)
    potential_conflict_dataset: Dict = get_dataset_as_dict(
        HVHV_train_potential_conflict_dataset, HVHV_val_potential_conflict_dataset,
        AVHV_train_potential_conflict_dataset, AVHV_val_potential_conflict_dataset,
    )

    # %% post-process the scene with multiple potential conflicts
    print("4> Post-processing the complex potential conflicts ...")
    AVHV_conflict_dataset, HVHV_conflict_dataset = get_dataset_as_list(potential_conflict_dataset, is_postprocess)
    print(f"HVHV dataset #samples: {len(HVHV_conflict_dataset)}")
    print(f"AVHV dataset #samples: {len(AVHV_conflict_dataset)}")
    return AVHV_conflict_dataset, HVHV_conflict_dataset


def identify_complex_potential_conflicts(list_potential_conflicts: List[Conflict]) -> List[Conflict]:
    # retrieve a list of pairs of (leader id, follower id)
    list_2pair = [(conflict.first_id, conflict.second_id) for conflict in list_potential_conflicts]
    # retrieve a list of tuples of (leader id, follower id, time of leader reaching conflict point)
    list_3tuple = [(conflict.first_id, conflict.second_id, conflict.first_time_at_conflict) for conflict in list_potential_conflicts]
    # sort this list of tuples based on the time of leader vehicle reaching conflict time
    list_3tuple = sorted(list_3tuple, key=lambda x: x[2])

    # construct the table
    table = []
    for element in list_3tuple:
        table.append(list(element[:2]))
    table = np.array(table).T
    assert table.shape[0] == 2
    table = table.reshape((-1))
    # vehicle sequences
    sequences = []
    for element in table:
        if element not in sequences:
            sequences.append(element)
        else:
            pass

    list_conflicts = []
    list_ids = []
    for element in range(len(sequences)-1):
        id_pair = (sequences[element], sequences[element+1])
        if id_pair in list_2pair:
            list_ids.append(id_pair)
            position = list_2pair.index(id_pair)
            list_conflicts.append(list_potential_conflicts[position])
    # print(list_potential_conflicts[0].scene_indices, list_2pair)
    # print(list_ids, '\n')
    return list_conflicts


def get_dataset_as_list(dataset, is_post_process: bool = True) -> Tuple[List[Conflict], List[Conflict]]:
    conflicts_list = []
    for scene_indices, list_conflicts in dataset.items():
        if len(list_conflicts) == 1:
            conflicts_list.append(list_conflicts[0])
        else:
            conflicts_list += identify_complex_potential_conflicts(list_conflicts) if is_post_process else list_conflicts

    AVHV_conflicts_list, HVHV_conflicts_list = [], []
    for conflict in conflicts_list:
        if conflict.is_first_AV or conflict.is_second_AV:
            AVHV_conflicts_list.append(conflict)
        else:
            HVHV_conflicts_list.append(conflict)

    return AVHV_conflicts_list, HVHV_conflicts_list


def get_dataset_as_dict(
        HVHV_train_dataset,
        HVHV_val_dataset,
        AVHV_train_dataset,
        AVHV_val_dataset,
) -> Dict:
    conflicts_dictionary = {}
    for index, dataset in enumerate([HVHV_train_dataset, HVHV_val_dataset, AVHV_train_dataset, AVHV_val_dataset]):
        for category in dataset:
            for direction in dataset[category]:
                for item in dataset[category][direction]:
                    # construct the Conflict object from value
                    scene_indices, conflict = list(item.keys())[0], list(item.values())[0]
                    c = Conflict(
                        first_trajectory=conflict.first_agent_trajectory,
                        second_trajectory=conflict.second_agent_trajectory,
                        is_first_AV=True if conflict.first_agent_trajectory_id == -1 else False,
                        is_second_AV=True if conflict.second_agent_trajectory_id == -1 else False,
                        PET=conflict.delta_time,
                        first_time_at_conflict=conflict.first_agent_conflict_time,
                        second_time_at_conflict=conflict.second_agent_conflict_time,
                        first_id=conflict.first_agent_trajectory_id if conflict.first_agent_trajectory_id is not None else -1,
                        second_id=conflict.second_agent_trajectory_id if conflict.second_agent_trajectory_id is not None else -1,
                        category=category,
                        direction=direction,
                        dataset="train2" if index % 2 == 0 else "validate",
                        scene_indices=scene_indices,
                    )
                    # retrieve the scene indices from key
                    if scene_indices not in conflicts_dictionary:
                        conflicts_dictionary[scene_indices] = [c, ]
                    else:
                        conflicts_dictionary[scene_indices].append(c)
    return conflicts_dictionary


def get_dataset_sample(train_dataset, val_dataset) -> int:
    num_samples = 0
    for dataset in [train_dataset, val_dataset]:
        for category in dataset:
            for direction in dataset[category]:
                for item in dataset[category][direction]:
                    num_samples += 1
    return num_samples


def remove_non_continuous_AV_trajectory(dataset, threshold: int = 5):
    for category in dataset.keys():
        for direction in dataset[category].keys():
            for index, conflict in enumerate(dataset[category][direction]):
                # retrieve the conflict
                conflict = list(conflict.values())[0]
                AV_trajectory = conflict.first_agent_trajectory.trajectory_xy if conflict.first_agent_trajectory_id is None \
                    else conflict.second_agent_trajectory.trajectory_xy
                AV_distance_ts = np.linalg.norm(np.diff(AV_trajectory, axis=0), axis=1)
                if np.any(AV_distance_ts > threshold):
                    dataset[category][direction].remove(dataset[category][direction][index])
    return dataset
