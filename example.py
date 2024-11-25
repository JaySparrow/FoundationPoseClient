from sensor.realsense import RealsenseSensor
from segmentation.mask_select import MaskSelect
from client.pose_tracker_zmq_client import PoseTrackerZMQClient
from utils import visualize_pose_2d

import cv2
import numpy as np

server_ip = "172.16.71.50" # Arrakis
port = "8080"
sensor_device_id = "109422062805"
sam_checkpoint = "segmentation/sam2_b.pt"
object_names = "stick nut2"
cam_T_base = np.eye(4)

realsense = RealsenseSensor(device_id=sensor_device_id)
mask_selecter = MaskSelect(sam_checkpoint)
pose_tracker_client = PoseTrackerZMQClient(server_ip=server_ip, port=port)

camera_intrinsics = realsense.cam_K
initialized_object_dict = pose_tracker_client.request_init(object_names, camera_intrinsics)
initialized_object_names_str = " ".join(initialized_object_dict.keys())
if input(f"Objects with existing mesh files on Server: {initialized_object_names_str}. Continue? (y/[n]): ").lower() != "y":
    exit()

try:
    frame_idx = 0
    while True:
        rgb, depth, is_frame_received = realsense.get_aligned_frames(depth_processed=False)
        if not is_frame_received:
            continue

        if frame_idx == 0:
            masks_dict = mask_selecter.run_gui(rgb, list(initialized_object_dict.keys()))
            object_names = list(masks_dict.keys())
            masks = np.stack(list(masks_dict.values()), axis=0)
            poses = pose_tracker_client.request_estimate(rgb, depth, object_names, masks)
        else:
            poses = pose_tracker_client.request_track(rgb, depth, object_names)

        vis = rgb.copy()
        for obj_name, pose in zip(object_names, poses):
            bboxCenter_T_localOrigin = initialized_object_dict[obj_name]["bboxCenter_T_localOrigin"]
            bbox = initialized_object_dict[obj_name]["bbox"]
            vis = visualize_pose_2d(vis, pose, realsense.cam_K, draw_bbox=True, bbox=bbox, bboxCenter_T_localOrigin=bboxCenter_T_localOrigin)
        cv2.imshow("vis", vis[..., ::-1])
        cv2.waitKey(1)

        frame_idx += 1

except KeyboardInterrupt:
    realsense.stop()
    cv2.destroyAllWindows()
    pose_tracker_client.close()
