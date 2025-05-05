import pickle

import numpy as np

with open("./pickle_backup/behaviour_identification/validate/AV_turnleft_trajectory_normal_validate.pkl", "rb") as f:
	Data1_trajectories = pickle.load(f)
with open("./pickle_backup/behaviour_identification/validate/AV_turnleft_trajectory_5s_validate.pkl", "rb") as f:
	Data2_trajectories = pickle.load(f)

print(f"Data1:, {len(Data1_trajectories)}")
print(f"Data2:, {len(Data2_trajectories)}")

Data1_speeds, Data1_accelerations = [], []
Data2_speeds, Data2_accelerations = [], []

for index, Data1_trajectory in enumerate(Data1_trajectories):
	Data1_speeds.append(Data1_trajectory.average_speed_intersection)
	Data1_accelerations.append(Data1_trajectory.average_acceleration_intersection)


for index, Data2_trajectory in enumerate(Data2_trajectories):
	Data2_speeds.append(Data2_trajectory.average_speed_intersection)
	Data2_accelerations.append(Data2_trajectory.average_acceleration_intersection)

Data2_speeds, Data2_accelerations = np.array(Data2_speeds), np.array(Data2_accelerations)

print(f"Data1 | VEL: MEAN: {np.mean(Data1_speeds):2.3f}, SD: {np.std(Data1_speeds):2.3f}")
print(f"Data1 | ACC: MEAN: {np.mean(Data1_accelerations):2.3f}, SD: {np.std(Data1_accelerations):2.3f}")

print(f"Data2 | VEL: MEAN: {np.mean(Data2_speeds):2.3f}, SD: {np.std(Data2_speeds):2.3f}")
print(f"Data2 | ACC: MEAN: {np.mean(Data2_accelerations):2.3f}, SD: {np.std(Data2_accelerations):2.3f}")

import matplotlib.pyplot as plt
#
# bins = 8
#
# plt.hist(Data1_speeds, bins=bins, alpha=0.5, label='Data1', color='blue')
# plt.hist(Data2_speeds, bins=bins, alpha=0.5, label='Data2', color='red')
# plt.legend()
# plt.show()

import scipy.stats as stats
t_stat, p_value = stats.ttest_ind(Data1_speeds, Data2_speeds, equal_var=False)
print("t-statistic:", t_stat)
print("p-value    :", p_value)

t_stat, p_value = stats.ttest_ind(Data1_accelerations, Data2_accelerations, equal_var=False)
print("t-statistic:", t_stat)
print("p-value    :", p_value)

