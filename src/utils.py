import numpy as np
import cv2
import open3d as o3d

def draw_3d_bounding_box(image, pose, intrinsics, object_model_path, color=(0, 255, 0)):
    """Draws a 3D bounding box on a 2D image."""
    mesh = o3d.io.read_triangle_mesh(object_model_path)
    bbox_3d = mesh.get_axis_aligned_bounding_box()
    corners_3d = np.asarray(bbox_3d.get_box_points())

    corners_3d_hom = np.hstack((corners_3d, np.ones((8, 1))))
    transformed_corners = (pose @ corners_3d_hom.T).T
    projected_points = (intrinsics @ transformed_corners[:, :3].T).T
    projected_points_2d = projected_points[:, :2] / projected_points[:, 2:]
    projected_points_2d = projected_points_2d.astype(int)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        start_point = tuple(projected_points_2d[edge[0]])
        end_point = tuple(projected_points_2d[edge[1]])
        cv2.line(image, start_point, end_point, color, 2)

    return image

class_id_to_name = {
    1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box", 
    4: "005_tomato_soup_can", 5: "006_mustard_bottle", 6: "007_tuna_fish_can", 
    7: "008_pudding_box", 8: "009_gelatin_box", 9: "010_potted_meat_can", 
    10: "011_banana", 11: "019_pitcher_base", 12: "021_bleach_cleanser", 
    13: "024_bowl", 14: "025_mug", 15: "035_power_drill", 16: "036_wood_block", 
    17: "037_scissors", 18: "040_large_marker", 19: "051_large_clamp", 
    20: "052_extra_large_clamp", 21: "061_foam_brick"
}