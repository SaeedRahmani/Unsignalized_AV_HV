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

# %% define the collections of TAs.
HVHV_TAs = []
AVHV_TAs = []
HVAV_TAs = []

if __name__ == '__main__':

    print("Post-processing")
    conflicts = []
    conflicts = conflicts + post_process(AVHV_val_conflict_dataset, "val")
    conflicts = conflicts + post_process(AVHV_train_conflict_dataset, "train")
    # 646 AVHV/HVAV conflicts

    print("Calculate the TA distribution")
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
        
        # %% discard the samples with short trajectory (unable to calculate position-based speed)
        if first_xyt.shape[0] < 3 or second_xyt.shape[0] < 3:
            continue
        assert first_xyt[0, 2] == second_xyt[0, 2], \
            f"Two vehicles' initial states were not in the same time instant."


        if first_xyt.shape[0] !=  second_xyt.shape[0]:
            print("Non-equal two trajectories", first_xyt.shape, second_xyt.shape)
            len_traj = min(first_xyt.shape[0], second_xyt.shape[0]) - 2
        else:
            len_traj = first_xyt.shape[0] - 2

        for jdx in range(len_traj):
            # %% calculate the length and speed -> to get TA
            len_first_traj = LineString(first_xyt[jdx:, :2]).length  
            first_speed = compute_position_based_velocities(first_xyt[jdx:, :2])[0]
            len_second_traj = LineString(second_xyt[jdx:, :2]).length
            second_speed = compute_position_based_velocities(second_xyt[jdx:, :2])[0]
            TA = len_second_traj / second_speed - len_first_traj / first_speed
            if conflict.is_first_AV:
                AVHV_TAs.append(TA)
            else:
                HVAV_TAs.append(TA)

# %% draw distribution
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)

sca0 = axs[0].hist(x=AVHV_TAs, bin=10)
sca1 = axs[1].hist(x=HVAV_TAs, bin=10)
# sca2 = axs[2].scatter(x=HVAV_PETs_cross, y=HVAV_average_speeds_first_cross, s=20, marker="o", c=HVAV_MRDs_cross, cmap="Reds", vmin=0, vmax=5)

# axs[0].set_title(r"$\bf{HV}$-HV crossing")
# axs[0].set_xlabel("Post encroachment time")
# axs[0].set_ylabel("Average speed")
# axs[0].set_ylim([0, 20])

# axs[1].set_title(r"$\bf{AV}$-HV crossing")
# axs[1].set_xlabel("Post encroachment time")
# axs[1].set_ylabel("Average speed")

# axs[2].set_title(r"$\bf{HV}$-AV crossing")
# axs[2].set_xlabel("Post encroachment time")
# axs[2].set_ylabel("Average speed")

# plt.colorbar(sca0,ax=axs[0])
# plt.colorbar(sca1,ax=axs[1])
# plt.colorbar(sca2,ax=axs[2])
plt.tight_layout()
plt.show()