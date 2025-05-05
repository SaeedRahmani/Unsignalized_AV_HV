import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString, Polygon
from l5kit_conflict.pickle.io import load_potential_conflict_pickle, report_AVHV_conflicts, report_HVHV_conflicts
from l5kit_conflict.filter.helper import multiline_to_single_line, multi2singleLineString
from l5kit_conflict.analysis.conflict import Conflict, Trajectory
from l5kit_conflict.analysis.utils import compute_position_based_velocities
# from l5kit_conflict.analysis.utils import post_process

np.set_printoptions(suppress=True)
plt.style.use("ggplot")


def calculate_fraction(conflict, truncate_frames: bool = False, num_frames: int = 40):
    """ Calculate the fraction of TA for AV and HV, respectively. """  
    
    # %% define variables
    HV_numerator, HV_denominator = 0, 0
    AV_numerator, AV_denominator = 0, 0

    # calculate the fraction of TA (new) for AV and HV, respectively.
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

        if first_xyt.shape[0] < 3 or second_xyt.shape[0] < 3:
            # num_discard += 1
            continue
        assert first_xyt[0, 2] == second_xyt[0, 2]

        if truncate_frames:
            if first_xyt.shape[0] > num_frames+2 and second_xyt.shape[0] > num_frames+2:
                first_distance = LineString(first_xyt[num_frames:, :2]).length  
                first_speed = compute_position_based_velocities(first_xyt[num_frames:, :2])[0]
                second_distance = LineString(second_xyt[num_frames:, :2]).length
                second_speed = compute_position_based_velocities(second_xyt[num_frames:, :2])[0]
                TA_first = first_distance / first_speed
                TA_second = second_distance / second_speed      
            else:
                continue      
        else:
            first_distance = LineString(first_xyt[:, :2]).length  
            first_speed = compute_position_based_velocities(first_xyt[:, :2])[0]
            second_distance = LineString(second_xyt[:, :2]).length
            second_speed = compute_position_based_velocities(second_xyt[:, :2])[0]
            TA_first = first_distance / first_speed
            TA_second = second_distance / second_speed

        # outer-loop: whether AV or HV's TA_{init} < 0   
        if (TA_first < TA_second and conflict.is_first_AV) or (TA_second < TA_first and conflict.is_second_AV):
            # this means AV has the time advantage 
            AV_denominator += 1
            if conflict.is_first_AV:
                AV_numerator += 1
        elif (TA_first < TA_second and not conflict.is_first_AV) or (TA_second < TA_first and not conflict.is_second_AV):
            # this means HV has the time advantage 
            HV_denominator += 1
            if not conflict.is_first_AV:
                HV_numerator += 1
        
    print(f"AV: {AV_numerator}, {AV_denominator}")
    print(f"HV: {HV_numerator}, {HV_denominator}")      
    print(f"AV: {AV_numerator / AV_denominator}")
    print(f"HV: {HV_numerator / HV_denominator}")
    
    # print(f"All   - AVHV: {num_AVHV / num_AVHV_noTA:.2f} {num_AVHV} {num_AVHV_noTA}")
    # print(f"All   - HVAV: {num_HVAV / num_HVAV_noTA:.2f} {num_HVAV} {num_HVAV_noTA}")
    # print(f"Merge - AVHV: {num_AVHV_merge / num_AVHV_noTA_merge:.2f} {num_AVHV_merge} {num_AVHV_noTA_merge}")
    # print(f"Merge - HVAV: {num_HVAV_merge / num_HVAV_noTA_merge:.2f} {num_HVAV_merge} {num_HVAV_noTA_merge}")
    # print(f"Cross - AVHV: {num_AVHV_cross / num_AVHV_noTA_cross:.2f} {num_AVHV_cross} {num_AVHV_noTA_cross}")
    # print(f"Cross - HVAV: {num_HVAV_cross / num_HVAV_noTA_cross:.2f} {num_HVAV_cross} {num_HVAV_noTA_cross}")
    # print(f"#Discard cases: {num_discard}")


# %% main function
if __name__ == '__main__':

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

    print("Post-processing")
    conflicts = []
    conflicts = conflicts + post_process(AVHV_val_conflict_dataset, "val")
    conflicts = conflicts + post_process(AVHV_train_conflict_dataset, "train")
    # 646 AVHV/HVAV conflicts

    calculate_fraction(conflict=conflicts)
    
    calculate_fraction(conflict=conflicts, truncate_frames=True)
    
    calculate_fraction(conflict=conflicts, truncate_frames=True, num_frames=20)