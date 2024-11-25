from sensor.realsense import RealsenseSensor
from segmentation.mask_select import MaskSelect
from client.pose_tracker_zmq_client import PoseTrackerZMQClient

from utils import visualize_pose_2d

import numpy as np
import cv2

class PoseTracker:
    r"""
    A class to track the pose of objects in the scene
    """

    def __init__(self, object_names: list, 
                 device_id: str, 
                 sam_checkpoint: str,
                 server_ip: str, 
                 port: str, 
                 cam_T_base: np.ndarray=np.eye(4, dtype=np.float64)):
        r"""
        Initialize the Pose Tracker
        """
        ## initialize classes
        self.realsense = RealsenseSensor(device_id=device_id)
        self.mask_selecter = MaskSelect(sam_checkpoint)
        self.pose_tracker_client = PoseTrackerZMQClient(server_ip=server_ip, port=port)

        ## inputs
        self.cam_T_base = cam_T_base # (4, 4)

        ## initialize the Pose Tracker in the Server
        # object_shape_dict: {"object_name": {"bboxCenter_T_localOrigin": ndarry(4, 4), "bbox": ndarray(2, 3)}}
        #   these are requested objects available in the Server
        self.object_shape_dict = self.pose_tracker_client.request_init(" ".join(object_names), self.realsense.cam_K)
        if input(f"---\nObjects with existing mesh files on Server:\n{self.object_shape_dict.keys()}.\nContinue? (y/[n]): ").lower() != "y":
            exit()

        ## segment the objects and estimate the poses
        # object_mask_dict: {"object_name": mask}
        #   these are the objects with masks, among those in object_shape_dict
        rgb, depth = self.get_aligned_frame()
        self.object_mask_dict = self.get_masks(rgb, list(self.object_shape_dict.keys()))
        self.object_names = list(self.object_mask_dict.keys())
        masks = np.stack(list(self.object_mask_dict.values()), axis=0)
        poses_in_cam = self.pose_tracker_client.request_estimate(rgb, depth, self.object_names, masks)

        self.rgb = rgb
        self.depth = depth
        self.poses_in_cam = poses_in_cam

    def get_aligned_frame(self):
        r"""
        Get the frame from the RealSense Sensor
        """
        max_frames = 100
        for i in range(max_frames):
            rgb, depth, is_frame_received = self.realsense.get_aligned_frames(depth_processed=False)
            if is_frame_received:
                return rgb, depth
        raise Exception(f"Failed to get the frame within {max_frames} frames")
    
    def get_masks(self, rgb: np.ndarray, object_names) -> dict:
        r"""
        Get the masks for the objects in the scene
        ---
        return:
            masks_dict: {"object_name": mask}
                mask: ndarray(h, w) {0: background, 1: object}
        """
        masks_dict = self.mask_selecter.run_gui(rgb, object_names)
        return masks_dict

    def get_poses(self) -> dict:
        rgb, depth = self.get_aligned_frame()
        poses_in_cam = self.pose_tracker_client.request_track(rgb, depth, self.object_names)
        self.rgb = rgb
        self.depth = depth
        self.poses_in_cam = poses_in_cam
        
        poses_in_base = self.cam_T_base @ poses_in_cam
        object_pose_dict = dict(zip(self.object_names, poses_in_base))
        return object_pose_dict

    def draw_poses(self, rgb: np.ndarray=None) -> np.ndarray:
        r"""
        Draw the poses of the objects in the scene
        ---
        return:
            vis: ndarray(h, w, 3), RGB
        """
        if rgb is None:
            vis = self.rgb.copy()
        else:
            vis = rgb.copy()
            
        for obj_name, pose in zip(self.object_names, self.poses_in_cam):
            bboxCenter_T_localOrigin = self.object_shape_dict[obj_name]["bboxCenter_T_localOrigin"]
            bbox = self.object_shape_dict[obj_name]["bbox"]
            vis = visualize_pose_2d(vis, pose, self.realsense.cam_K, draw_bbox=True, bbox=bbox, bboxCenter_T_localOrigin=bboxCenter_T_localOrigin)
        return vis

    def close(self):
        r"""
        Close the Pose Tracker
        """
        cv2.destroyAllWindows()
        self.realsense.stop()
        self.pose_tracker_client.close()
        
if __name__ == "__main__":
    server_ip = "172.16.71.50" # Arrakis
    port = "8080"
    sensor_device_id = "109422062805"
    sam_checkpoint = "segmentation/sam2_b.pt"
    object_names = ["nut2", "stick"]
    cam_T_base = np.eye(4)

    pose_tracker = PoseTracker(object_names, sensor_device_id, sam_checkpoint, server_ip, port, cam_T_base)
    
    try:
        while True:
            object_pose_dict = pose_tracker.get_poses()
            vis = pose_tracker.draw_poses()
            cv2.imshow("vis", vis[..., ::-1])
            cv2.waitKey(1)
    
    except KeyboardInterrupt:
        pose_tracker.close()
