import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from io import BytesIO
from shapely import Point, LineString, Polygon

from waymo_open_dataset.protos import scenario_pb2, map_pb2
from waymo_open_dataset.utils.sim_agents.visualizations import add_road_edge, add_road_line

from waymo_devkit.query import get_intersection_stopSigns, get_intersection_circle, get_intersection_lanes


def visualize_traj(scenario: scenario_pb2.Scenario, tfrecord_index, scenario_index,
                   distance_threshold=45):
    INTERSECTION_CENTER_SIZE = 50
    INTERSECTION_CENTER_COLOR = "red"
    INTERSECTION_CENTER_MARKER = "*"
    INTERSECTION_RADIUS = 20

    IDENTIFIED_STOP_SIGN_COLOR = "orange"
    STOP_SIGN_COLOR = "red"
    STOP_SIGN_SIZE = 50
    STOP_SIGN_MARKER = "."
    
    fig, ax = plt.subplots()

    for mf in scenario.map_features:
        if mf.WhichOneof('feature_data') == 'road_edge':
            x, y = zip(*[(p.x, p.y) for p in mf.road_edge.polyline])
            ax.plot(x, y, "black")
        elif mf.WhichOneof('feature_data') == 'road_line':
            x, y = zip(*[(p.x, p.y) for p in mf.road_line.polyline])
            ax.plot(x, y, 'gray')
        # elif mf.WhichOneof('feature_data') == 'lane':
        #     x, y = zip(*[(p.x, p.y) for p in mf.lane.polyline])
        #     plt.plot(x, y)        
        elif mf.WhichOneof('feature_data') == 'stop_sign':
            x, y = mf.stop_sign.position.x, mf.stop_sign.position.y
            ax.scatter(x, y, c=STOP_SIGN_COLOR, s=STOP_SIGN_SIZE)
            
    # draw the center of intersection
    stopSigns = get_intersection_stopSigns(scenario, distance_threshold)
    for ss in stopSigns:
        ax.scatter(ss[1][0], ss[1][1], s=STOP_SIGN_SIZE, c=IDENTIFIED_STOP_SIGN_COLOR)
        
    intersection_centerCoordinate, intersection_radius = get_intersection_circle(stopSigns, aggregation="max", buffer=2)
    # display the center of the intersection
    ax.scatter(intersection_centerCoordinate[0], intersection_centerCoordinate[1], 
                s=INTERSECTION_CENTER_SIZE, c=INTERSECTION_CENTER_COLOR, marker=INTERSECTION_CENTER_MARKER)
    
    # draw the circular studying area 
    theta = np.linspace(0, 2 * np.pi, 100) 
    intersection_circle_x = intersection_centerCoordinate[0] + intersection_radius * np.cos(theta)  
    intersection_circle_y = intersection_centerCoordinate[1] + intersection_radius * np.sin(theta) 
    ax.plot(intersection_circle_x, intersection_circle_y, INTERSECTION_CENTER_COLOR)

    intersection_area_coords = np.vstack([intersection_circle_x, intersection_circle_y]).T
    assert intersection_area_coords.shape[1] == 2
    intersection_area = Polygon(intersection_area_coords)

    # draw the lane centers
    inbound_lanes, outbound_lanes = get_intersection_lanes(scenario, intersection_area)
    for lane in inbound_lanes:
        ax.plot(lane[2][:,0], lane[2][:,1], "blue")  
    for lane in outbound_lanes:
        ax.plot(lane[2][:,0], lane[2][:,1], "green")  

    len_t = len(scenario.timestamps_seconds)
    for track_index, track in enumerate(scenario.tracks):
        if track.object_type == 1:
            traj = []
            # ego
            if track_index == scenario.sdc_track_index:
                for t in range(len_t):
                    state = track.states[t]
                    traj.append((state.center_x, state.center_y))
                traj = np.array(traj)
                ax.scatter(traj[:,0], traj[:,1], c='blue', marker="s", s=1)
            # agents
            else:
                for t in range(len_t):
                    state = track.states[t]
                    if state.valid:
                        traj.append((state.center_x, state.center_y))
                traj = np.array(traj)
                ax.scatter(traj[:,0], traj[:,1], c='green', marker="s", s=1)
    
    ax.set_xlim([intersection_centerCoordinate[0] - 50, intersection_centerCoordinate[0] + 50])
    ax.set_ylim([intersection_centerCoordinate[1] - 50, intersection_centerCoordinate[1] + 50])
    plt.savefig(f'./outputs/trajectories/{tfrecord_index}-{int(scenario_index)}.png')
    plt.close(fig)    
    
    
def visualize_gif(scenario: scenario_pb2.Scenario, tfrecord_index, scenario_index, 
                  duration: float = 0.1, distance_threshold=45):

    INTERSECTION_CENTER_SIZE = 50
    INTERSECTION_CENTER_COLOR = "red"
    INTERSECTION_CENTER_MARKER = "*"
    INTERSECTION_RADIUS = 20

    IDENTIFIED_STOP_SIGN_COLOR = "orange"
    STOP_SIGN_COLOR = "red"
    STOP_SIGN_SIZE = 50
    STOP_SIGN_MARKER = "." 

    len_t = len(scenario.timestamps_seconds)
    all_frames = []

    for t_index in range(len_t):
        fig, ax = plt.subplots()

        for mf in scenario.map_features:
            if mf.WhichOneof('feature_data') == 'road_edge':
                x, y = zip(*[(p.x, p.y) for p in mf.road_edge.polyline])
                ax.plot(x, y, "black")
            elif mf.WhichOneof('feature_data') == 'road_line':
                x, y = zip(*[(p.x, p.y) for p in mf.road_line.polyline])
                ax.plot(x, y, 'gray')
            # elif mf.WhichOneof('feature_data') == 'lane':
            #     x, y = zip(*[(p.x, p.y) for p in mf.lane.polyline])
            #     plt.plot(x, y)        
            elif mf.WhichOneof('feature_data') == 'stop_sign':
                x, y = mf.stop_sign.position.x, mf.stop_sign.position.y
                ax.scatter(x, y, c=STOP_SIGN_COLOR, s=STOP_SIGN_SIZE)
                
        # draw the center of intersection
        stopSigns = get_intersection_stopSigns(scenario, distance_threshold)
        for ss in stopSigns:
            ax.scatter(ss[1][0], ss[1][1], s=STOP_SIGN_SIZE, c=IDENTIFIED_STOP_SIGN_COLOR)
            
        intersection_centerCoordinate, intersection_radius = get_intersection_circle(stopSigns, aggregation="max", buffer=2)
        # display the center of the intersection
        ax.scatter(intersection_centerCoordinate[0], intersection_centerCoordinate[1], 
                    s=INTERSECTION_CENTER_SIZE, c=INTERSECTION_CENTER_COLOR, marker=INTERSECTION_CENTER_MARKER)
        
        # draw the circular studying area 
        theta = np.linspace(0, 2 * np.pi, 100) 
        intersection_circle_x = intersection_centerCoordinate[0] + intersection_radius * np.cos(theta)  
        intersection_circle_y = intersection_centerCoordinate[1] + intersection_radius * np.sin(theta) 
        ax.plot(intersection_circle_x, intersection_circle_y, INTERSECTION_CENTER_COLOR)

        intersection_area_coords = np.vstack([intersection_circle_x, intersection_circle_y]).T
        assert intersection_area_coords.shape[1] == 2
        intersection_area = Polygon(intersection_area_coords)

        # draw the lane centers
        inbound_lanes, outbound_lanes = get_intersection_lanes(scenario, intersection_area)
        for lane in inbound_lanes:
            ax.plot(lane[2][:,0], lane[2][:,1], "blue")  
        for lane in outbound_lanes:
            ax.plot(lane[2][:,0], lane[2][:,1], "green")  

        for track_index, track in enumerate(scenario.tracks):
            # ego 
            if track.object_type == 1:
                if track_index == scenario.sdc_track_index:
                    state = track.states[t_index]
                    center_x, center_y = state.center_x, state.center_y
                    ax.scatter(state.center_x, state.center_y, c='blue', marker="s")
                # agents
                else:
                    state = track.states[t_index]
                    if state.valid:
                        ax.scatter(state.center_x, state.center_y, c='green', marker="s")
        
        OFFSET = 80
        ax.set_xlim([center_x - OFFSET, center_x + OFFSET])
        ax.set_ylim([center_y - OFFSET, center_y + OFFSET])
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        all_frames.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    imageio.mimsave(f'./outputs/gifs/{tfrecord_index}-{int(scenario_index)}.gif', all_frames, duration=duration)

def visualize_map(scenario: scenario_pb2.Scenario, tfrecord_id, scenario_id, n_legs=None, distance_threshold=45, buffer=20):
    # CONSTANTS
    FIGSIZE_WIDTH, FIGSIZE_HEIGHT = 5, 5
    
    INTERSECTION_CENTER_SIZE = 50
    INTERSECTION_CENTER_COLOR = "red"
    INTERSECTION_CENTER_MARKER = "*"
    INTERSECTION_RADIUS = 20

    IDENTIFIED_STOP_SIGN_COLOR = "orange"
    STOP_SIGN_COLOR = "red"
    STOP_SIGN_SIZE = 100
    STOP_SIGN_MARKER = "." 
    
    # init the figure
    fig = plt.figure(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    
    for mf in scenario.map_features:
        if mf.WhichOneof('feature_data') == 'road_edge':
            x, y = zip(*[(p.x, p.y) for p in mf.road_edge.polyline])
            plt.plot(x, y, "black")
        elif mf.WhichOneof('feature_data') == 'road_line':
            x, y = zip(*[(p.x, p.y) for p in mf.road_line.polyline])
            plt.plot(x, y, 'gray')
        # elif mf.WhichOneof('feature_data') == 'lane':
        #     x, y = zip(*[(p.x, p.y) for p in mf.lane.polyline])
        #     plt.plot(x, y)        
        elif mf.WhichOneof('feature_data') == 'stop_sign':
            x, y = mf.stop_sign.position.x, mf.stop_sign.position.y
            plt.scatter(x, y, c=STOP_SIGN_COLOR, s=STOP_SIGN_SIZE)
        
    # draw the center of intersection
    stopSigns = get_intersection_stopSigns(scenario, distance_threshold)
    for ss in stopSigns:
        plt.scatter(ss[1][0], ss[1][1], s=STOP_SIGN_SIZE, c=IDENTIFIED_STOP_SIGN_COLOR, label="stop sign")
        
    intersection_centerCoordinate, intersection_radius = get_intersection_circle(stopSigns, aggregation="max", buffer=buffer)
    # display the center of the intersection
    plt.scatter(intersection_centerCoordinate[0], intersection_centerCoordinate[1], 
                s=INTERSECTION_CENTER_SIZE, c=INTERSECTION_CENTER_COLOR, marker=INTERSECTION_CENTER_MARKER, label="center")
    
    # draw the circular studying area 
    theta = np.linspace(0, 2 * np.pi, 100) 
    intersection_circle_x = intersection_centerCoordinate[0] + intersection_radius * np.cos(theta)  
    intersection_circle_y = intersection_centerCoordinate[1] + intersection_radius * np.sin(theta) 
    plt.plot(intersection_circle_x, intersection_circle_y, INTERSECTION_CENTER_COLOR)

    intersection_area_coords = np.vstack([intersection_circle_x, intersection_circle_y]).T
    assert intersection_area_coords.shape[1] == 2
    intersection_area = Polygon(intersection_area_coords)

    # draw the lane centers
    inbound_lanes, outbound_lanes = get_intersection_lanes(scenario, intersection_area)
    for lane in inbound_lanes:
        plt.plot(lane[2][:,0], lane[2][:,1], "blue", label="inbound lane")  
    for lane in outbound_lanes:
        plt.plot(lane[2][:,0], lane[2][:,1], "green", label="outbound lane") 

    # set the display border
    OFFSET = 40
    plt.xlim([intersection_centerCoordinate[0] - OFFSET, intersection_centerCoordinate[0] + OFFSET])
    plt.ylim([intersection_centerCoordinate[1] - OFFSET, intersection_centerCoordinate[1] + OFFSET])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12, loc="lower right")

    # remove ticks
    plt.xticks([])
    plt.yticks([])
    
    # plt.show()
    plt.savefig(f"./outputs/maps/{n_legs}_stop_signs/{tfrecord_id}_{scenario_id}.png", dpi=300, bbox_inches="tight")
    plt.close()