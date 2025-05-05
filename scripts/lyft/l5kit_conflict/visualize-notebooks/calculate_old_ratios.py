import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString, Polygon
from l5kit_conflict.pickle.io import load_potential_conflict_pickle, report_AVHV_conflicts, report_HVHV_conflicts
from l5kit_conflict.filter.helper import multiline_to_single_line, multi2singleLineString
from l5kit_conflict.analysis.conflict import Conflict, Trajectory
from l5kit_conflict.analysis.post_process import post_process, compute_position_based_velocities


np.set_printoptions(suppress=True)
plt.style.use("ggplot")

# %% load datasets.
delta_time = 10  # 10 seconds for now [April 3rd, meeting];
AVHV_val_conflict_dataset = load_potential_conflict_pickle(dataset_type="AVHV", dataset_name="validate",
                                                           delta_time=delta_time)
HVHV_val_conflict_dataset = load_potential_conflict_pickle(dataset_type="HVHV", dataset_name="validate",
                                                           delta_time=delta_time)

AVHV_train_conflict_dataset = load_potential_conflict_pickle(dataset_type="AVHV", dataset_name="train2",
                                                             delta_time=delta_time)
HVHV_train_conflict_dataset = load_potential_conflict_pickle(dataset_type="HVHV", dataset_name="train2",
                                                             delta_time=delta_time)

# %% define variables (numerators and denominators).
# numerator
num_AVHV, num_HVAV = 0, 0
num_AVHV_merge, num_AVHV_cross = 0, 0
num_HVAV_merge, num_HVAV_cross = 0, 0

# denominator
num_AVHV_noTA, num_HVAV_noTA = 0, 0
num_AVHV_noTA_merge, num_AVHV_noTA_cross = 0, 0
num_HVAV_noTA_merge, num_HVAV_noTA_cross = 0, 0

num_discard = 0

if __name__ == '__main__':

    print("Post-processing")
    conflicts = []
    conflicts = conflicts + post_process(AVHV_val_conflict_dataset, "val")
    conflicts = conflicts + post_process(AVHV_train_conflict_dataset, "train")
    # 646 AVHV/HVAV conflicts

    print("Calculate the TA ratio")
    for index, conflict in enumerate(conflicts):

        first_xyt = np.hstack(
            [conflict.first_trajectory.xy, conflict.first_trajectory.t.reshape((-1,1))])
        second_xyt = np.hstack(
            [conflict.second_trajectory.xy, conflict.second_trajectory.t.reshape((-1,1))])

        # extract the trajectory within the study rectangle
        first_xyt = first_xyt[
            (first_xyt[:, 0] <= Trajectory.X_MAX) &
            (first_xyt[:, 0] >= Trajectory.X_MIN) &
            (first_xyt[:, 1] <= Trajectory.Y_MAX) &
            (first_xyt[:, 1] >= Trajectory.Y_MIN)
        ]
        second_xyt = second_xyt[
            (second_xyt[:, 0] <= Trajectory.X_MAX) &
            (second_xyt[:, 0] >= Trajectory.X_MIN) &
            (second_xyt[:, 1] <= Trajectory.Y_MAX) &
            (second_xyt[:, 1] >= Trajectory.Y_MIN)
        ]

        # take the first timestamp that two vehicles co-exist
        coexist_start_time = max(first_xyt[0, 2], second_xyt[0, 2])
        first_xyt = first_xyt[
            (conflict.first_time_at_conflict > first_xyt[:, 2]) &
            (first_xyt[:, 2] >= coexist_start_time)
            ]    
        second_xyt = second_xyt[
            (conflict.first_time_at_conflict > second_xyt[:, 2]) &
            (second_xyt[:, 2] >= coexist_start_time)
            ]
        
        if first_xyt.shape[0] !=  second_xyt.shape[0]:
            print("Non-equal two trajectories", first_xyt.shape, second_xyt.shape)
        #     plt.figure(figsize=(3, 3))
        #     plt.plot(first_xyt[:, 0], first_xyt[:, 1], 'red')     
        #     plt.plot(second_xyt[:, 0], second_xyt[:, 1], 'green')     
        #     plt.savefig(f"./scene_gif/{index}.png", dpi=400)
        #     plt.close()

        if first_xyt.shape[0] < 3 or second_xyt.shape[0] < 3:
            num_discard += 1
            continue
        assert first_xyt[0, 2] == second_xyt[0, 2]

        len_first_traj = LineString(first_xyt[:, :2]).length  
        first_speed = compute_position_based_velocities(first_xyt[:, :2])[0]
        len_second_traj = LineString(second_xyt[:, :2]).length
        second_speed = compute_position_based_velocities(second_xyt[:, :2])[0]
      
    
        # increment num_AVHV or num_HVAV
        if conflict.is_first_AV:
            # check the denominator
            num_AVHV += 1
            if conflict.category == "merge":
                num_AVHV_merge += 1
            else:
                num_AVHV_cross += 1
    
            # check the numerator: TA
            if len_first_traj/first_speed > len_second_traj/second_speed:
                num_AVHV_noTA += 1
                if conflict.category == "merge":
                    num_AVHV_noTA_merge += 1
                else:
                    num_AVHV_noTA_cross += 1
        else:
            # check the denominator
            num_HVAV += 1
            if conflict.category == "merge":
                num_HVAV_merge += 1
            else:
                num_HVAV_cross += 1
    
            # check the numerator: TA
            if len_first_traj/first_speed > len_second_traj/second_speed:
                num_HVAV_noTA += 1
                if conflict.category == "merge":
                    num_HVAV_noTA_merge += 1
                else:
                    num_HVAV_noTA_cross += 1
    
    print(f"All   - AVHV: {num_AVHV / num_AVHV_noTA:.2f} {num_AVHV} {num_AVHV_noTA}")
    print(f"All   - HVAV: {num_HVAV / num_HVAV_noTA:.2f} {num_HVAV} {num_HVAV_noTA}")
    print(f"Merge - AVHV: {num_AVHV_merge / num_AVHV_noTA_merge:.2f} {num_AVHV_merge} {num_AVHV_noTA_merge}")
    print(f"Merge - HVAV: {num_HVAV_merge / num_HVAV_noTA_merge:.2f} {num_HVAV_merge} {num_HVAV_noTA_merge}")
    print(f"Cross - AVHV: {num_AVHV_cross / num_AVHV_noTA_cross:.2f} {num_AVHV_cross} {num_AVHV_noTA_cross}")
    print(f"Cross - HVAV: {num_HVAV_cross / num_HVAV_noTA_cross:.2f} {num_HVAV_cross} {num_HVAV_noTA_cross}")
    print(f"#Discard cases: {num_discard}")