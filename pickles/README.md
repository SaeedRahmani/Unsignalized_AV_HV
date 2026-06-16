# README – Traffic Conflict Dataset: AV–HV Interactions at Unsignalized Intersections

## General Information

**Title:** Traffic Conflict Records at Unsignalized Intersections — Autonomous Vehicle (AV) and Human-driven Vehicle (HV) Interactions

**Authors:**
- Saeed Rahmani — Delft University of Technology
- Zhenlin (Gavin) Xu — Delft University of Technology
- Simeon Calvert — Delft University of Technology
- Bart van Arem — Delft University of Technology

**Description:**
This dataset contains traffic conflict records extracted from two publicly available naturalistic driving datasets: the Waymo Open Dataset and the Lyft Level 5 Prediction Dataset. Conflicts are identified at unsignalized intersections using surrogate safety measures. The data covers both AV–HV and HV–HV interaction pairs, and both crossing (CROSS) and merging (MERGE) conflict types.

Each pickle file stores a collection of conflict objects or dictionaries, each representing one identified traffic conflict including full trajectory data for both vehicles, the conflict point, and pre-computed safety metrics.

**Related repository:** https://github.com/SaeedRahmani/Unsignalized_AV_HV

---

## File Structure

```
pickles/
├── waymo/
│   ├── conflict_pet10s.pkl
│   ├── conflict_pet15s.pkl
│   ├── cross_conflict_pet10s.pkl
│   ├── cross_conflict_pet15s.pkl
│   ├── merge_conflict_pet10s.pkl
│   └── merge_conflict_pet15s.pkl
└── lyft/
    ├── AVHV_conflict_train2_10_2.0.pkl
    ├── AVHV_conflict_validate_10_2.0.pkl
    ├── HVHV_conflict_train2_10_2.0.pkl
    └── HVHV_conflict_validate_10_2.0.pkl
```

---

## File Format

All files are Python pickle files (binary format, `.pkl`). They can be loaded in Python as follows:

```python
import pickle

with open("conflict_pet10s.pkl", "rb") as f:
    data = pickle.load(f)
```

**Requirements:** Python ≥ 3.8, NumPy, Shapely

---

## Waymo Files (`pickles/waymo/`)

### Source Dataset

Extracted from the **Waymo Open Dataset** (v1.2), specifically the 20-second training split (`training_20s`).  
Source: https://waymo.com/open/

Scenarios were selected based on the presence of 3 or 4 stop signs within a 45-metre distance threshold of the ego vehicle. Only scenarios where the ego vehicle's trajectory intersects the inferred intersection polygon are included.

### File Naming Convention

| Filename | Contents |
|---|---|
| `conflict_pet10s.pkl` | All conflicts (CROSS + MERGE, AV–HV + HV–HV) with PET < 10 s |
| `conflict_pet15s.pkl` | All conflicts (CROSS + MERGE, AV–HV + HV–HV) with PET < 15 s |
| `cross_conflict_pet10s.pkl` | Crossing conflicts only, PET < 10 s |
| `cross_conflict_pet15s.pkl` | Crossing conflicts only, PET < 15 s |
| `merge_conflict_pet10s.pkl` | Merging conflicts only, PET < 10 s |
| `merge_conflict_pet15s.pkl` | Merging conflicts only, PET < 15 s |

### Data Structure

Each file is a **Python `list`** of **`dict`** objects, where each element represents one identified conflict.

#### Fields per conflict record

| Field | Type | Unit | Description |
|---|---|---|---|
| `conflict_type` | `str` | — | Type of conflict: `"CROSS"` or `"MERGE"` |
| `leader_is_av` | `bool` | — | `True` if the leader vehicle is the AV (ego vehicle) |
| `follower_is_av` | `bool` | — | `True` if the follower vehicle is the AV (ego vehicle) |
| `leader_id` | `int` | — | Track ID of the leader vehicle within the Waymo scenario |
| `follower_id` | `int` | — | Track ID of the follower vehicle within the Waymo scenario |
| `leader_index` | `int` | — | Trajectory array index for the leader in the scenario |
| `follower_index` | `int` | — | Trajectory array index for the follower in the scenario |
| `leader_states` | `np.ndarray`, shape `(T, 6)` | see below | Full trajectory of the leader vehicle |
| `follower_states` | `np.ndarray`, shape `(T, 6)` | see below | Full trajectory of the follower vehicle |
| `leader_time_at_conflict` | `float` | s | Timestamp when the leader vehicle passes the conflict point |
| `follower_time_at_conflict` | `float` | s | Timestamp when the follower vehicle passes the conflict point |
| `PET` | `float` | s | Post-Encroachment Time (absolute value) |
| `center` | `tuple(float, float)` | m | (x, y) coordinates of the inferred intersection centre in the Waymo map coordinate system |
| `radius` | `float` | m | Radius of the intersection circle used to define the conflict zone |
| `tfrecord_index` | `str` | — | Index string of the TFRecord file (e.g., `"00123"`) in the Waymo training split |
| `scenario_index` | `int` | — | Scenario index within the TFRecord file |
| `scene_index` | `int` | — | Conflict index within the scenario (relevant for multi-conflict scenarios) |

#### Columns of `leader_states` and `follower_states` arrays (shape `(T, 6)`)

| Column index | Variable | Unit | Description |
|---|---|---|---|
| 0 | x | m | Position in the x direction in the Waymo map coordinate system |
| 1 | y | m | Position in the y direction in the Waymo map coordinate system |
| 2 | timestamp | s | Absolute timestamp (= frame index × 0.1 s) |
| 3 | heading | rad | Heading angle of the vehicle |
| 4 | vx | m/s | Velocity component in the x direction |
| 5 | vy | m/s | Velocity component in the y direction |

Trajectory data is sampled at **10 Hz** (0.1 s intervals).

---

## Lyft Files (`pickles/lyft/`)

### Source Dataset

Extracted from the **Lyft Level 5 Prediction Dataset** (also known as the Woven by Toyota Prediction Dataset).  
Source: https://woven-planet.github.io/l5kit/

Conflicts were identified at a T-junction (Junction 2) in the Lyft dataset map. The ego vehicle in the Lyft dataset is always treated as the AV.

### File Naming Convention

Pattern: `{VehiclePair}_conflict_{Split}_{DeltaTime}_{Margin}.pkl`

| Segment | Meaning |
|---|---|
| `VehiclePair` | `AVHV` = AV–HV pair; `HVHV` = HV–HV pair |
| `Split` | Dataset split: `train2` or `validate` (Lyft Level 5 dataset splits) |
| `DeltaTime` | `10` — maximum time window for conflict identification (= 100 frames × 0.1 s = 10 s) |
| `Margin` | `2.0` — buffer distance in metres used for merge conflict zone detection |

| Filename | Contents |
|---|---|
| `AVHV_conflict_train2_10_2.0.pkl` | AV–HV conflicts, training split (train2) |
| `AVHV_conflict_validate_10_2.0.pkl` | AV–HV conflicts, validation split |
| `HVHV_conflict_train2_10_2.0.pkl` | HV–HV conflicts, training split (train2) |
| `HVHV_conflict_validate_10_2.0.pkl` | HV–HV conflicts, validation split |

### Data Structure

Each file is a **Python `dict`** with the following nested structure:

```python
{
    "cross": {
        "goLeft&turnLeftFromLeft":   [ {scene_tuple: Conflict}, ... ],
        "goLeft&turnLeftFromTop":    [ {scene_tuple: Conflict}, ... ],
        # HVHV files also contain:
        "turnLeftFromLeft&turnLeftFromTop": [ {scene_tuple: Conflict}, ... ],
    },
    "merge": {
        "goLeft&turnRightFromTop":   [ {scene_tuple: Conflict}, ... ],
        "goRight&turnLeftFromTop":   [ {scene_tuple: Conflict}, ... ],
    }
}
```

The top-level keys are conflict types (`"cross"`, `"merge"`). Second-level keys are sub-scenario labels encoding the movement directions of the two vehicles. Each list element is a **single-entry `dict`** mapping a `scene_tuple` (a tuple of Lyft scene indices used to extract the conflict) to a `Conflict` object.

#### Sub-scenario label codes (second-level keys)

| Label | Present in | Conflict type | Vehicle A movement | Vehicle B movement |
|---|---|---|---|---|
| `goLeft&turnLeftFromLeft` | AVHV and HVHV files | CROSS | AV/HV goes straight (leftward) | HV turns left from the left arm |
| `goLeft&turnLeftFromTop` | AVHV and HVHV files | CROSS | AV/HV goes straight (leftward) | HV turns left from the top arm |
| `turnLeftFromLeft&turnLeftFromTop` | HVHV files only | CROSS | HV turns left from the left arm | HV turns left from the top arm |
| `goLeft&turnRightFromTop` | AVHV and HVHV files | MERGE | AV/HV goes straight (leftward) | HV turns right from the top arm |
| `goRight&turnLeftFromTop` | AVHV and HVHV files | MERGE | AV/HV goes straight (rightward) | HV turns left from the top arm |

#### `Conflict` object attributes

Each `Conflict` object (class `l5kit_conflict.objects.conflict.Conflict`) has the following attributes:

| Attribute | Type | Unit | Description |
|---|---|---|---|
| `first_agent_trajectory` | `Trajectory` | — | Trajectory of the first vehicle (the one that reaches the conflict point first) |
| `second_agent_trajectory` | `Trajectory` | — | Trajectory of the second vehicle (the follower) |
| `first_agent_trajectory_id` | `int` or `None` | — | Agent identifier: `-1` for the AV (ego vehicle); `None` for HVs |
| `second_agent_trajectory_id` | `int` or `None` | — | Agent identifier: `-1` for the AV (ego vehicle); `None` for HVs |
| `first_agent_conflict_time` | `float` | frames | Frame index at which the first vehicle passes the conflict point (divide by 10 for seconds) |
| `second_agent_conflict_time` | `float` | frames | Frame index at which the second vehicle passes the conflict point (divide by 10 for seconds) |
| `delta_time` | `float` | frames | Post-Encroachment Time in frame units: `second_agent_conflict_time − first_agent_conflict_time` (divide by 10 for seconds) |

#### `Trajectory` object attributes

Each `Trajectory` object (class `l5kit_conflict.objects.trajectory.Trajectory`) embedded inside a `Conflict` has the following attributes:

| Attribute | Type | Unit | Description |
|---|---|---|---|
| `trajectory_xy` | `np.ndarray`, shape `(T, 2)` | m | Vehicle positions: columns are [x, y] in the Lyft map coordinate system |
| `trajectory_t` | `np.ndarray`, shape `(T,)` | frames | Frame indices (integer, 10 Hz; divide by 10 for seconds) |
| `agent_index` | `int` or `None` | — | `-1` for the AV (ego vehicle); `None` for HV agents |
| `scene_indices` | `int` or `tuple` | — | Lyft dataset scene index (or tuple of indices for concatenated scenes) |
| `dataset` | `str` | — | Dataset split the trajectory belongs to: `"train2"` or `"validate"` |

---

## Codes and Symbols

| Symbol / Code | Meaning |
|---|---|
| `AV` | Autonomous Vehicle (ego vehicle in both datasets) |
| `HV` | Human-driven Vehicle (non-ego agent) |
| `AVHV` | Conflict pair where one vehicle is AV and the other is HV |
| `HVHV` | Conflict pair where both vehicles are HVs |
| `CROSS` | Crossing conflict: vehicles approach from different inbound lanes and depart to different outbound lanes, with intersecting trajectories |
| `MERGE` | Merging conflict: vehicles approach from different inbound lanes but converge to the same outbound lane |
| `PET` | Post-Encroachment Time: time elapsed between the moment the first vehicle clears the conflict zone and the moment the second vehicle enters it |
| `-1` | Agent index value (`agent_index` / `first_agent_trajectory_id`) used in the Lyft files to identify the AV (ego vehicle) |
| `None` | Agent index value used in the Lyft files to identify HV agents |
| `NaN` | Not a Number: value is undefined for this record |

---

## Signal Processing Notes

Speed values are **not pre-computed** in the Lyft pickle files. Trajectory data is stored as raw (x, y) positions with frame indices; speeds must be derived numerically from positions if needed.

In the Waymo files, velocity components (`vx`, `vy`) are taken directly from the Waymo Open Dataset proto fields and are stored as-is in the trajectory arrays.

---

## Coordinate Systems

- **Waymo:** Positions are in the Waymo map coordinate system (metric, arbitrary origin per map). Units are metres.
- **Lyft:** Positions are in the Lyft dataset coordinate system (metric). Units are metres. The study area (Junction 2, a T-junction) is bounded by x ∈ [−200, −50] m and y ∈ [−915, −865] m in the Lyft map frame.

---

## How to Use

Example: loading a Waymo conflict list and accessing a record

```python
import pickle

with open("pickles/waymo/conflict_pet10s.pkl", "rb") as f:
    conflicts = pickle.load(f)

c = conflicts[0]  # first conflict record (dict)
print(c["conflict_type"])       # "CROSS" or "MERGE"
print(c["PET"])                 # Post-Encroachment Time in seconds
print(c["leader_states"].shape) # e.g., (200, 6): columns = [x, y, timestamp, heading, vx, vy]
```

Example: loading a Lyft conflict file and iterating conflicts

```python
import pickle

with open("pickles/lyft/AVHV_conflict_train2_10_2.0.pkl", "rb") as f:
    data = pickle.load(f)

# Each list element is a single-entry dict: {scene_tuple: Conflict}
for entry in data["cross"]["goLeft&turnLeftFromLeft"]:
    scene_tuple = list(entry.keys())[0]
    conflict = list(entry.values())[0]
    # PET in seconds (stored in frame units at 10 Hz):
    pet_seconds = conflict.delta_time * 0.1
    # AV is identified by agent_index == -1
    is_first_av = (conflict.first_agent_trajectory_id == -1)
    print(scene_tuple, pet_seconds, is_first_av)
```

---

## Related Publications

If you use this dataset, please cite the associated publication (details to be added upon publication).

---

## License

Please refer to the `LICENSE` file in the root of this repository.

The underlying source data are subject to the licenses of their respective providers:
- Waymo Open Dataset: https://waymo.com/open/terms/
- Lyft Level 5 Prediction Dataset: https://woven-planet.github.io/l5kit/
