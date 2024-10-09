"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import yaml
import numpy as np
import torch as th
import torch.distributed as dist
from improved_diffusion.trajectory_datasets import load_data
import conf_mgt
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString, LinearRing
from argoverse.map_representation.map_api import ArgoverseMap
import math
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from visualize_30hz_benchmark_data_on_map import DatasetOnMapVisualizer
##set root_dir to the correct path to your dataset folder
root_dir = '/mnt/sdb/jianghanhu/ArgoversePrediction/val/data'

afl = ArgoverseForecastingLoader(root_dir)
CITYS = ['PIT','MIA']

def get_xy_from_nt_seq(nt_seq: np.ndarray,
                       centerlines) -> np.ndarray:
    """Convert n-t coordinates to x-y, i.e., convert from centerline curvilinear coordinates to map coordinates.

    Args:
        nt_seq (numpy array): Array of shape (num_tracks x seq_len x 2) where last dimension has 'n' (offset from centerline) and 't' (distance along centerline)
        centerlines (list of numpy array): Centerline for each track
    Returns:
        xy_seq (numpy array): Array of shape (num_tracks x seq_len x 2) where last dimension contains coordinates in map frame

    """
    seq_len = nt_seq.shape[1]

    # coordinates obtained by interpolating distances on the centerline
    xy_seq = np.zeros(nt_seq.shape)
    for i in range(nt_seq.shape[0]):
        curr_cl = centerlines[0]
        line_string = LineString(curr_cl)
        for time in range(seq_len):

            # Project nt to xy
            offset_from_cl = nt_seq[i][time][0]
            dist_along_cl = nt_seq[i][time][1]
            x_coord, y_coord = get_xy_from_nt(offset_from_cl, dist_along_cl,
                                              curr_cl)
            xy_seq[i, time, 0] = x_coord
            xy_seq[i, time, 1] = y_coord

    return xy_seq

def get_xy_from_nt(n: float, t: float,
                   centerline: np.ndarray) :
    """Convert a single n-t coordinate (centerline curvilinear coordinate) to absolute x-y.

    Args:
        n (float): Offset from centerline
        t (float): Distance along the centerline
        centerline (numpy array): Centerline coordinates
    Returns:
        x1 (float): x-coordinate in map frame
        y1 (float): y-coordinate in map frame

    """
    line_string = LineString(centerline)

    # If distance along centerline is negative, keep it to the start of line
    point_on_cl = line_string.interpolate(
        t) if t > 0 else line_string.interpolate(0)
    local_ls = None

    # Find 2 consective points on centerline such that line joining those 2 points
    # contains point_on_cl
    for i in range(len(centerline) - 1):
        pt1 = centerline[i]
        pt2 = centerline[i + 1]
        ls = LineString([pt1, pt2])
        if ls.distance(point_on_cl) < 1e-8:
            local_ls = ls
            break

    assert local_ls is not None, "XY from N({}) T({}) not computed correctly".format(
        n, t)

    pt1, pt2 = local_ls.coords[:]
    x0, y0 = point_on_cl.coords[0]

    # Determine whether the coordinate lies on left or right side of the line formed by pt1 and pt2
    # Find a point on either side of the line, i.e., (x1_1, y1_1) and (x1_2, y1_2)
    # If the ring formed by (pt1, pt2, (x1_1, y1_1)) is counter clockwise, then it lies on the left

    # Deal with edge cases
    # Vertical
    if pt1[0] == pt2[0]:
        m = 0
        x1_1, x1_2 = x0 + n, x0 - n
        y1_1, y1_2 = y0, y0
    # Horizontal
    elif pt1[1] == pt2[1]:
        m = float("inf")
        x1_1, x1_2 = x0, x0
        y1_1, y1_2 = y0 + n, y0 - n
    # General case
    else:
        ls_slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        m = -1 / ls_slope

        x1_1 = x0 + n / math.sqrt(1 + m**2)
        y1_1 = y0 + m * (x1_1 - x0)
        x1_2 = x0 - n / math.sqrt(1 + m**2)
        y1_2 = y0 + m * (x1_2 - x0)

    # Rings formed by pt1, pt2 and coordinates computed above
    lr1 = LinearRing([pt1, pt2, (x1_1, y1_1)])
    lr2 = LinearRing([pt1, pt2, (x1_2, y1_2)])

    # If ring is counter clockwise
    if lr1.is_ccw:
        x_ccw, y_ccw = x1_1, y1_1
        x_cw, y_cw = x1_2, y1_2
    else:
        x_ccw, y_ccw = x1_2, y1_2
        x_cw, y_cw = x1_1, y1_1

    # If offset is positive, coordinate on the left
    if n > 0:
        x1, y1 = x_ccw, y_ccw
    # Else, coordinate on the right
    else:
        x1, y1 = x_cw, y_cw

    return x1, y1


def get_abs_traj(
        input_: np.ndarray,
        output: np.ndarray,
        helpers,
        start_idx: int = None,
):
    """Get absolute trajectory reverting all the transformations.

    Args:
        input_ (numpy array): Input Trajectory with shape (num_tracks x obs_len x 2)
        output (numpy array): Predicted Trajectory with shape (num_tracks x pred_len x 2)
        args (Argparse): Config parameters
        helpers (dict):Data helpers
        start_id (int): Start index of the current batch (used in joblib). If None, then no batching.
    Returns:            
        input_ (numpy array): Input Trajectory in map frame with shape (num_tracks x obs_len x 2)
        output (numpy array): Predicted Trajectory in map frame with shape (num_tracks x pred_len x 2)

    """
    # print(input_.shape,output.shape)
    obs_len = input_.shape[1]
    pred_len = output.shape[1]

    # if start_idx is None:
    s = 0
    e = output.shape[0]
    # else:
    #     print(f"Abs Traj Done {start_idx}/{input_.shape[0]}")
    #     s = start_idx
    #     e = start_idx + args.joblib_batch_size

    input_ = input_.copy()[s:e]
    output = output.copy()[s:e]
    input_[:, :, 1]*=3
    output[:, :, 1]*=3

    # Convert relative to absolute
    # if args.use_delta:
    if True:
        reference = np.array(helpers[7]).copy()[start_idx]
        input_[:, 0, :2] = reference
        for i in range(1, obs_len):
            input_[:, i, :2] = input_[:, i, :2] + input_[:, i - 1, :2]

        output[:, 0, :2] = output[:, 0, :2] + input_[:, -1, :2]
        for i in range(1, pred_len):
            output[:, i, :2] = output[:, i, :2] + output[:, i - 1, :2]

    # Convert centerline frame (n,t) to absolute frame (x,y)
    # if args.use_map:
    if True:
        centerlines=[]
        centerlines.append(np.array(helpers[9][start_idx,:helpers[10][start_idx],:]))
        # print(centerlines)
        centerlines = np.array(centerlines)
        input_[:, :, :2] = get_xy_from_nt_seq(input_[:, :, :2], centerlines)
        output[:, :, :2] = get_xy_from_nt_seq(output[:, :, :2], centerlines)

    # Denormalize trajectory
    # elif args.normalize and not args.use_map:
    #     translation = helpers["TRANSLATION"].copy()[s:e]
    #     rotation = helpers["ROTATION"].copy()[s:e]
    #     input_[:, :, :2] = normalized_to_map_coordinates(
    #         input_[:, :, :2], translation, rotation)
    #     output[:, :, :2] = normalized_to_map_coordinates(
    #         output[:, :, :2], translation, rotation)
    return input_, output

def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()
def swap_left_and_right(
    condition: np.ndarray, left_centerline: np.ndarray, right_centerline: np.ndarray
):
    """
    Swap points in left and right centerline according to condition.

    Args:
       condition: Numpy array of shape (N,) of type boolean. Where true, swap the values in the left and
                   right centerlines.
       left_centerline: The left centerline, whose points should be swapped with the right centerline.
       right_centerline: The right centerline.

    Returns:
       left_centerline
       right_centerline
    """

    right_swap_indices = right_centerline[condition]
    left_swap_indices = left_centerline[condition]

    left_centerline[condition] = right_swap_indices
    right_centerline[condition] = left_swap_indices
    return left_centerline, right_centerline

def centerline_to_polygon(
    centerline: np.ndarray, width_scaling_factor: float = 1.0
) -> np.ndarray:
    """
    Convert a lane centerline polyline into a rough polygon of the lane's area.

    On average, a lane is 3.8 meters in width. Thus, we allow 1.9 m on each side.
    We use this as the length of the hypotenuse of a right triangle, and compute the
    other two legs to find the scaled x and y displacement.

    Args:
       centerline: Numpy array of shape (N,2).
       width_scaling_factor: Multiplier that scales 3.8 meters to get the lane width.
       visualize: Save a figure showing the the output polygon.

    Returns:
       polygon: Numpy array of shape (2N+1,2), with duplicate first and last vertices.
    """
    # eliminate duplicates
    _, inds = np.unique(centerline, axis=0, return_index=True)
    # does not return indices in sorted order
    inds = np.sort(inds)
    centerline = centerline[inds]

    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])

    # compute the normal at each point
    slopes = dy / dx
    inv_slopes = -1.0 / slopes

    thetas = np.arctan(inv_slopes)
    x_disp = 3 * width_scaling_factor / 2.0 * np.cos(thetas)
    y_disp = 3 * width_scaling_factor / 2.0 * np.sin(thetas)

    displacement = np.hstack([x_disp[:, np.newaxis], y_disp[:, np.newaxis]])
    right_centerline = centerline + displacement
    left_centerline = centerline - displacement

    # right centerline position depends on sign of dx and dy
    subtract_cond1 = np.logical_and(dx > 0, dy < 0)
    subtract_cond2 = np.logical_and(dx > 0, dy > 0)
    subtract_cond = np.logical_or(subtract_cond1, subtract_cond2)
    left_centerline, right_centerline = swap_left_and_right(subtract_cond, left_centerline, right_centerline)

    # right centerline also depended on if we added or subtracted y
    neg_disp_cond = displacement[:, 1] > 0
    left_centerline, right_centerline = swap_left_and_right(neg_disp_cond, left_centerline, right_centerline)

    # return the polygon
    # return convert_lane_boundaries_to_polygon(right_centerline, left_centerline)
    return right_centerline,left_centerline


def convert_lane_boundaries_to_polygon(right_lane_bounds: np.ndarray, left_lane_bounds: np.ndarray) -> np.ndarray:
    """
    Take a left and right lane boundary and make a polygon of the lane segment, closing both ends of the segment.

    These polygons have the last vertex repeated (that is, first vertex == last vertex).

    Args:
       right_lane_bounds: Right lane boundary points. Shape is (N, 2).
       left_lane_bounds: Left lane boundary points.

    Returns:
       polygon: Numpy array of shape (2N+1,2)
    """
    assert right_lane_bounds.shape[0] == left_lane_bounds.shape[0]
    polygon = np.vstack([right_lane_bounds, left_lane_bounds[::-1]])
    polygon = np.vstack([polygon, right_lane_bounds[0]])
    return polygon
def draw_car(start_point, end_point):
    # 计算线的斜率
    slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
    # 定义长方形的宽和高
    width = 1.8
    height = 4
    if end_point[0]-start_point[0]<0:
        direction = -1
    else:
        direction = 1
    x1,y1=height / 2* np.cos(np.arctan(slope))-width / 2* np.sin(np.arctan(slope)),height / 2* np.sin(np.arctan(slope))+width / 2* np.cos(np.arctan(slope))
    x2,y2=height / 2* np.cos(np.arctan(slope))+width / 2* np.sin(np.arctan(slope)),height / 2* np.sin(np.arctan(slope))-width / 2* np.cos(np.arctan(slope))
    x3,y3=3.5 *np.cos(np.arctan(slope)),3.5 *np.sin(np.arctan(slope))
    # 计算长方形的四个顶点
    if direction>0:
        rectangle_points = [(end_point[0] - x1, end_point[1] - y1),
                            (end_point[0] + x2, end_point[1] + y2),
                            (end_point[0] + x3, end_point[1] + y3),
                            (end_point[0] + x1, end_point[1] + y1),
                            (end_point[0] - x2, end_point[1] - y2),
                            (end_point[0] - x1, end_point[1] - y1)]
    else:
        rectangle_points = [(end_point[0] - x1, end_point[1] - y1),
                            (end_point[0] + x2, end_point[1] + y2),
                            (end_point[0] + x1, end_point[1] + y1),
                            (end_point[0] - x2, end_point[1] - y2),
                            (end_point[0] - x3, end_point[1] - y3),
                            (end_point[0] - x1, end_point[1] - y1)]

    return rectangle_points
    # # 绘制线
    # plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]])

    # # 绘制长方形
    # plt.fill([point[0] for point in rectangle_points], [point[1] for point in rectangle_points], 'r', alpha=0.3)

def viz_predictions(
        input_: np.ndarray,
        output: np.ndarray,
        target: np.ndarray,
        centerlines: np.ndarray,
        city_names: np.ndarray,
        idx,
        fileID,
        show: bool = True,
) -> None:
    """Visualize predicted trjectories.

    Args:
        input_ (numpy array): Input Trajectory with shape (num_tracks x obs_len x 2)
        output (numpy array of list): Top-k predicted trajectories, each with shape (num_tracks x pred_len x 2)
        target (numpy array): Ground Truth Trajectory with shape (num_tracks x pred_len x 2)
        centerlines (numpy array of list of centerlines): Centerlines (Oracle/Top-k) for each trajectory
        city_names (numpy array): city names for each trajectory
        show (bool): if True, show

    """
    seq_path = f"{root_dir}/"+str(int(fileID))+".csv"
    argoverse_forecasting_data = afl.get(seq_path).seq_df
    
    avm = ArgoverseMap()
    dis = np.max([np.linalg.norm(input_[0,-1] - output[i][0,-1]) for i in range(len(output))])+10
    num_tracks = input_.shape[0]
    obs_len = input_.shape[1]
    pred_len = target.shape[1]
    print(city_names)
    plt.figure(idx, figsize=(8, 7),dpi=300)
    object_type_tracker={"OTHERS":False,"AV":False,"Predicted Future Trajectory":False}
    city_name = city_names[0]
    lane_centerlines = None
    if lane_centerlines is None:
        # Get API for Argo Dataset map
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    x_min = input_[0,-1,0]-dis
    x_max = input_[0,-1,0]+dis
    y_min = input_[0,-1,1]-dis
    y_max = input_[0,-1,1]+dis
    # draw_circle = plt.Circle((input_[0,-1,0], input_[0,-1,1]), dis)
    # plt.gcf().gca().add_artist(draw_circle)
    if lane_centerlines is None:

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        lane_centerlines = []
        # Get lane centerlines which lie within the range of trajectories
        for lane_id, lane_props in seq_lane_props.items():

            lane_cl = lane_props.centerline

            if (
                np.min(lane_cl[:, 0]) < x_max
                and np.min(lane_cl[:, 1]) < y_max
                and np.max(lane_cl[:, 0]) > x_min
                and np.max(lane_cl[:, 1]) > y_min
            ):
                lane_centerlines.append(lane_cl)

    for lane_cl in lane_centerlines:
        lane_cl = lane_cl[(lane_cl[:, 0] > x_min) & (lane_cl[:, 0] < x_max)]
        lane_cl = lane_cl[(lane_cl[:, 1] > y_min) & (lane_cl[:, 1] < y_max)]
        if len(lane_cl)<2:
            continue
        l1,l2 = centerline_to_polygon(lane_cl)
        plt.plot(
            l1[:, 0],
            l1[:, 1],
            "-",
            color="grey",
            alpha=1,
            linewidth=1,
            zorder=0,
        )
        plt.plot(
            l2[:, 0],
            l2[:, 1],
            "-",
            color="grey",
            alpha=1,
            linewidth=1,
            zorder=0,
        )
        
    frames = argoverse_forecasting_data.groupby("TRACK_ID")
    for group_name, group_data in frames:
        # print(group_data) 
        object_type = group_data["OBJECT_TYPE"].values[0]

        cor_x = group_data["X"].values
        cor_y = group_data["Y"].values
        final_x = cor_x[-1]
        final_y = cor_y[-1]
        distance = np.linalg.norm(input_[0,-1] - [final_x, final_y]) 
        if object_type == "OTHERS" and distance > 25:
            continue
        # for w in range(len(cor_x)):
        #     lane_ids = avm.get_lane_ids_in_xy_bbox(
        #         cor_x[w],
        #         cor_y[w],
        #         city_names[0],
        #         query_search_range_manhattan=2.5,
        #     )
        # [avm.draw_lane(lane_id, city_names[0]) for lane_id in lane_ids]
        # if object_type == "OTHERS" and len(cor_x)>3:
        #     # plt.plot(
        #     #     cor_x,
        #     #     cor_y,
        #     #     "-",
        #     #     color="#d3e8ef",
        #     #     label="OTHERS" if not object_type_tracker["OTHERS"] else '',
        #     #     alpha=1,
        #     #     linewidth=1,
        #     #     zorder=5,
        #     # )
        #     # points = draw_car([cor_x[-4],cor_y[-4]],[cor_x[-1],cor_y[-1]])
        #     # plt.plot([point[0] for point in points], [point[1] for point in points], '#d3e8ef', alpha=1,zorder=5)
        #     plt.plot(
        #         final_x,
        #         final_y,
        #         's',
        #         color="#DDA0DD",
        #         label="OTHERS" if not object_type_tracker["OTHERS"] else '',
        #         alpha=1,
        #         markersize=5,
        #         zorder=5,
        #     )
        #     object_type_tracker["OTHERS"]=True
            
        if object_type == "AV":
            plt.plot(
                cor_x[:20],
                cor_y[:20],
                "-",
                color="#78a7bb",
                label="Ego Car History Trajectory",
                alpha=1,
                linewidth=1,
                zorder=10,
            )
            plt.plot(
                cor_x[20:],
                cor_y[20:],
                "--",
                color="#78a7bb",
                label="Ego Car Future Trajectory",
                alpha=1,
                linewidth=1,
                zorder=10,
            )
            points = draw_car([cor_x[15],cor_y[15]],[cor_x[20],cor_y[20]])
            # print(points)
            plt.fill([point[0] for point in points], [point[1] for point in points], '#78a7bb', alpha=1,zorder=10,label="Ego Car")
            # plt.plot(
            #     cor_x[20],
            #     cor_y[20],
            #     'v',
            #     color="blue",
            #     label="",
            #     alpha=1,
            #     markersize=7,
            #     zorder=10,
            # )
    for i in range(num_tracks):
        plt.plot(
            input_[i, :, 0],
            input_[i, :, 1],
            color="#ECA154",
            label="Observed Trajectory" ,
            alpha=1,
            linewidth=2,
            zorder=15,
        )
        ocpoints = draw_car(input_[i, -5, :2],input_[i, -1, :2])
        plt.fill([point[0] for point in ocpoints], [point[1] for point in ocpoints], '#ECA154', alpha=1,zorder=16,label="Vehicle")
        # plt.plot(
        #     input_[i, -1, 0],
        #     input_[i, -1, 1],
        #     "*",
        #     color="#ECA154",
        #     label="Target Vehicle",
        #     alpha=1,
        #     linewidth=2,
        #     zorder=15,
        #     markersize=12,
        # )
        plt.plot(
            target[i, :, 0],
            target[i, :, 1],
            color="red",
            label="GroundTruth Future Trajectory ",
            alpha=1,
            linewidth=2,
            zorder=17,
        )
        plt.plot(
            target[i, -1, 0],
            target[i, -1, 1],
            "o",
            color="red",
            label="",
            alpha=1,
            linewidth=2,
            zorder=16,
            markersize=4,
        )

    

                

        # for j in range(len(centerlines[i])):
        #     plt.plot(
        #         centerlines[i][j][:, 0],
        #         centerlines[i][j][:, 1],
        #         "--",
        #         color="grey",
        #         alpha=1,
        #         linewidth=1,
        #         zorder=0,
        #     )
        minindex = 0
        mindiff = 9999999999
        for j in range(len(output[0])):
            diff = np.mean((target[i,:] - output[i][j]) ** 2)
            if diff < mindiff:
                mindiff = diff
                minindex = j
        for j in range(len(output[0])):
            # print(output[j])
            if(minindex==j):
                plt.plot(
                    output[i][j][:, 0],
                    output[i][j][:, 1],
                    color="blue",
                    label="Best Predicted Future Trajectory",
                    alpha=1,
                    linewidth=2,
                    zorder=17,
                )
                plt.plot(
                    output[i][j][-1, 0],
                    output[i][j][-1, 1],
                    "o",
                    color="blue",
                    label="" ,
                    alpha=1,
                    linewidth=2,
                    zorder=17,
                    markersize=4,
                )
            else:
                plt.plot(
                    output[i][j][:, 0],
                    output[i][j][:, 1],
                    color="#4682B4",
                    label="Predicted Future Trajectory" if not object_type_tracker["Predicted Future Trajectory"] else '',
                    alpha=0.8,
                    linewidth=0.6,
                    zorder=15,
                )
                plt.plot(
                    output[i][j][-1, 0],
                    output[i][j][-1, 1],
                    "o",
                    color="#4682B4",
                    label="" ,
                    alpha=0.8,
                    linewidth=1,
                    zorder=15,
                    markersize=4,
                )
                plt.text(output[i][j][-1, 0], output[i][j][-1, 1], output[i][j][-1, 1], ha='center', va='bottom')
                object_type_tracker["Predicted Future Trajectory"] = True
        
        # for j in range(obs_len):
        #     lane_ids = avm.get_lane_ids_in_xy_bbox(
        #         input_[i, j, 0],
        #         input_[i, j, 1],
        #         city_names[i],
        #         query_search_range_manhattan=10,
        #     )
        #     # print(lane_ids)
        #     [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]
        # for j in range(pred_len):
        #     lane_ids = avm.get_lane_ids_in_xy_bbox(
        #         target[i, j, 0],
        #         target[i, j, 1],
        #         city_names[i],
        #         query_search_range_manhattan=10,
        #     )
        #     [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]
    
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    # by_label = OrderedDict(zip(labels, handles))
    if show:
        plt.savefig(str(idx)+'_5.png', bbox_inches='tight')
    plt.close()

def smooth(res_in,res_out):
    for i in range(len(res_out)):
        res_out[i]=res_in[0,-2:]+res_out[i]
        for j in range(2,len(res_out[i])-2):
            if np.linalg.norm(res_out[i,j]-np.mean(res_out[i,j-2:j+3],axis=0))<1:
                res_out[i,j] = np.mean(res_out[i,[j-2,j-1,j+1,j+2]],axis=0)           
            res_out[i,j] = np.mean(res_out[i,j-2:j+3],axis=0)
        res_out[i]=res_out[i,2:]
    return res_out

def main():
    args = create_argparser().parse_args()
    conf = conf_mgt.conf_base.Default_Conf()
    conf.update(yaml.safe_load(txtread("/mnt/sdc/jianghanhu/improved-diffusion/improved_diffusion/trajectory.yml")))
    dist_util.setup_dist()
    logger.configure()
    device = dist_util.dev()
    show_progress = conf.show_progress
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        mode = "val",
        deterministic= False
    )
    logger.log("sampling...")
    # all_images = []
    # all_labels = []
    # while len(all_images) * args.batch_size < args.num_samples:
    forecasted_trajectories={}
    gt_trajectories={}
    city_names={}
    count = 1
    for batch,mask,batch_helper in iter(data):
        model_kwargs = {}
        # if args.class_cond:
        #     classes = th.randint(
        #         low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        #     )
        #     model_kwargs["y"] = classes
        batch = batch.to(device)
       
        mask = mask.to(device)
        
        model_kwargs['gt'] = batch
        
        model_kwargs['gt_keep_mask'] = mask
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        result={"pred_xstart":np.zeros([args.batch_size,args.num_samples,50,5]),"gt":np.zeros([args.batch_size,1,50,5])}
        for j in range(args.num_samples):
            sample = sample_fn(
                model,
                (args.batch_size, 50, 5),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=device,
                progress=show_progress,
                return_all=True,
                conf=conf
            )
           
            for k in range(args.batch_size):
                result["pred_xstart"][k,j,:,:]=sample["pred_xstart"][k,:,:].to(th.device("cpu"))
            if(j==0):
                result["gt"][:,j,:,:]=batch[:,:,:].to(th.device("cpu"))
        for j in range(args.batch_size):           
            res_in,res_out = get_abs_traj(np.array(result["gt"][j,:,:20,:2]),np.array(result["pred_xstart"][j,:,20:50,:2]),batch_helper,j)
            _,gt = get_abs_traj(np.array(result["gt"][j,:,:20,:2]),np.array(result["gt"][j,:,20:50,:2]),batch_helper,j)
            forecasted_trajectories[count]=res_out
            gt_trajectories[count]=gt[0]
            city_names[count]=CITYS[int(batch_helper[1][j][0])]
            
            if count<2:
                viz_predictions(res_in,[res_out],gt,batch_helper[0][j],np.array([CITYS[int(batch_helper[1][j][0])]]),j,batch_helper[8][j])
                # for k in range(30):
                #     print(result["pred_xstart"][j,:,20+k,:2],res_out[j,k,:2],result["gt"][j,:,20+k,:2],gt[j,k,:2])
            count+=1

        if count>=args.batch_size:
            break

    # compute_forecasting_metrics(
    #         forecasted_trajectories,
    #         gt_trajectories,
    #         city_names,
    #         20,
    #         30,
    #         args.num_samples
    #     )
    # compute_forecasting_metrics(
    #         forecasted_trajectories,
    #         gt_trajectories,
    #         city_names,
    #         20,
    #         20,
    #         args.num_samples
    #     )
    # compute_forecasting_metrics(
    #         forecasted_trajectories,
    #         gt_trajectories,
    #         city_names,
    #         20,
    #         10,
    #         args.num_samples
    #     )



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=5,
        batch_size=200,
        use_ddim=False,
        model_path="",
        data_dir=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
