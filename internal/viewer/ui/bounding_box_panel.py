import viser
import viser.transforms as vtf
import torch
import numpy as np
import os

from internal.transform_utils.alignment import *
class BoundingBoxPanel:
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            tab,
    ):
        self.server = server
        self.viewer = viewer
        self.tab = tab

        self._setup_bounding_box_ui()

    def _setup_bounding_box_ui(self):
        server = self.server
        with server.gui.add_folder("Bounding Box"):
            self.box_size = server.gui.add_vector3(
                "Size (Real World Units)",
                initial_value=(1.0, 1.0, 1.0),
                min=(0.01, 0.01, 0.01),
                step=0.01,
            )

            # Create six grid meshes for the bounding box faces
            self.box_faces = {}
            face_names = ["floor", "ceiling", "front", "back", "left", "right"]
            for name in face_names:
                self.box_faces[name] = server.add_grid(
                    name=f"/bounding_box/{name}",
                    width=1.0, # Initial size, will be updated
                    height=1.0, # Initial size, will be updated
                )

            # Add transform controls only to the floor grid
            self.floor_transform_controls = server.add_transform_controls(
                name="/bounding_box/floor_transform",
                position=(0, 0, 0), # Initial position
                wxyz=(1, 0, 0, 0), # Initial rotation (identity)
                scale=1.0, # Initial scale
            )

            # Link the floor transform controls to the floor grid
            @self.floor_transform_controls.on_update
            def _(_):
                if "floor" in self.box_faces:

                    # Update other faces relative to the floor
                    self._update_other_faces()

            # Initial update of other faces
            # Initialize gaussian_scene_size before the first update
            self.gaussian_scene_size = np.array(self.box_size.value)
            self._update_box_faces_size_and_position() # Update sizes based on initial value
            self._update_other_faces() # Update positions based on initial size


        self.sensor_data = {} # To store loaded sensor data
        self.real_to_gaussian_transform = None # Attribute to store the calculated transform
        # self.gaussian_scene_size is now initialized above

        # Link box_size input to update the bounding box face sizes and positions
        @self.box_size.on_update
        def _(event: viser.GuiEvent):
            # Recalculate gaussian_scene_size based on current real_world_size and stored transform
            if self.real_to_gaussian_transform is not None:
                 real_world_size = np.array(self.box_size.value)
                 scale_x = np.linalg.norm(self.real_to_gaussian_transform[:3, 0])
                 scale_y = np.linalg.norm(self.real_to_gaussian_transform[:3, 1])
                 scale_z = np.linalg.norm(self.real_to_gaussian_transform[:3, 2])
                 average_scale = (scale_x + scale_y + scale_z) / 3.0
                 self.gaussian_scene_size = real_world_size * average_scale
            else:
                 # If transform is not loaded, assume 1:1 scale for visual feedback
                 self.gaussian_scene_size = np.array(self.box_size.value)

            self._update_box_faces_size_and_position()
            self._update_other_faces() # Update positions based on new size


        with server.gui.add_folder("Segment Loading"):
            load_segments_button = server.gui.add_button("Load Sensor Segments")

            @load_segments_button.on_click
            def _(event: viser.GuiEvent):
                self.sensor_data = {} # Clear previous data
                segment_dir = self.tab.segment_result_save_dir # Get the segment directory from ViewerOptions
                os.makedirs(segment_dir, exist_ok=True) # Ensure the directory exists

                print(segment_dir)

                for i in range(4): # Check for sensor0 to sensor3
                    file_path = os.path.join(segment_dir, f"sensor{i}.npz") # Use the segment directory
                    if os.path.exists(file_path):
                        try:
                            with np.load(file_path) as data:
                                if 'transform' in data and 'coordinates' in data:
                                    self.sensor_data[f"sensor{i}"] = {
                                        "transform": data['transform'],
                                        "coordinates": data['coordinates']
                                    }
                                    print(f"Loaded data from {file_path}")
                                else:
                                    print(f"Warning: {file_path} is missing 'transform' or 'coordinates' keys.")
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                    else:
                        print(f"File not found: {file_path}")

                if self.sensor_data:
                    print("All available sensor data loaded.")
                    # Trigger automatic floor plane configuration and bounding box update
                    self._configure_floor_plane_and_box()
                else:
                    print("No sensor data files found (sensor0.npz to sensor3.npz).")

        with server.gui.add_folder("Gaussian Visibility"):
            toggle_gaussians_button = server.gui.add_button("Toggle Gaussians Outside Box")

            @toggle_gaussians_button.on_click
            def _(event: viser.GuiEvent):
                self._toggle_gaussians_outside_box()

    def _update_box_faces_size_and_position(self):
        # Update the size of the grid meshes based on gaussian_scene_size
        if self.box_faces:
            # Floor and Ceiling: width = x size, height = y size
            self.box_faces["floor"].width = self.gaussian_scene_size[0]
            self.box_faces["floor"].height = self.gaussian_scene_size[1]
            self.box_faces["ceiling"].width = self.gaussian_scene_size[0]
            self.box_faces["ceiling"].height = self.gaussian_scene_size[1]

            # Front and Back: width = x size, height = z size
            self.box_faces["front"].width = self.gaussian_scene_size[0]
            self.box_faces["front"].height = self.gaussian_scene_size[2]
            self.box_faces["back"].width = self.gaussian_scene_size[0]
            self.box_faces["back"].height = self.gaussian_scene_size[2]

            # Left and Right: width = y size, height = z size
            # Corrected: Left/Right should have width = z size, height = y size
            self.box_faces["left"].width = self.gaussian_scene_size[1]
            self.box_faces["left"].height = self.gaussian_scene_size[2]
            self.box_faces["right"].width = self.gaussian_scene_size[1]
            self.box_faces["right"].height = self.gaussian_scene_size[2]

            # Update positions and orientations of other faces relative to the floor
            self._update_other_faces()


    def _update_other_faces(self):
        if not self.box_faces or self.floor_transform_controls is None:
            return

        
        self.box_faces["floor"].position = self.floor_transform_controls.position
        self.box_faces["floor"].wxyz = self.floor_transform_controls.wxyz
        # Get floor transform
        floor_position = np.array(self.floor_transform_controls.position)
        floor_rotation = vtf.SO3(self.floor_transform_controls.wxyz)
        floor_transform_matrix = np.eye(4)
        floor_transform_matrix[:3, :3] = floor_rotation.as_matrix()
        floor_transform_matrix[:3, 3] = floor_position



        # Calculate positions and orientations relative to the floor
        half_size = self.gaussian_scene_size / 2.0

        # Ceiling: translate up by height
        ceiling_relative_position = np.array([0, 0, self.gaussian_scene_size[2]])
        ceiling_position_world = (floor_transform_matrix @ np.array([*ceiling_relative_position, 1]))[:3]
        self.box_faces["ceiling"].position = ceiling_position_world
        self.box_faces["ceiling"].wxyz = floor_rotation.wxyz # Same rotation as floor

        # Front face: translate forward by half y size, rotate to be vertical
        front_relative_position = np.array([0, half_size[1], half_size[2]])
        front_position_world = (floor_transform_matrix @ np.array([*front_relative_position, 1]))[:3]
        front_rotation_relative = vtf.SO3.from_x_radians(np.pi / 2) # Rotate 90 degrees around x-axis
        front_rotation_world = floor_rotation @ front_rotation_relative
        self.box_faces["front"].position = front_position_world
        self.box_faces["front"].wxyz = front_rotation_world.wxyz

        # Back face: translate backward by half y size, rotate to be vertical
        back_relative_position = np.array([0, -half_size[1], half_size[2]])
        back_position_world = (floor_transform_matrix @ np.array([*back_relative_position, 1]))[:3]
        back_rotation_relative = vtf.SO3.from_x_radians(np.pi / 2) # Rotate 90 degrees around x-axis
        back_rotation_world = floor_rotation @ back_rotation_relative
        self.box_faces["back"].position = back_position_world
        self.box_faces["back"].wxyz = back_rotation_world.wxyz

        # Left face: translate left by half x size, rotate to be vertical
        left_relative_position = np.array([-half_size[0], 0, half_size[2]])
        left_position_world = (floor_transform_matrix @ np.array([*left_relative_position, 1]))[:3]
        # Corrected rotation for left/right faces
        left_rotation_relative = vtf.SO3.from_y_radians(np.pi / 2) @ vtf.SO3.from_z_radians(np.pi / 2) # Rotate 90 around y, then 90 around z
        left_rotation_world = floor_rotation @ left_rotation_relative
        self.box_faces["left"].position = left_position_world
        self.box_faces["left"].wxyz = left_rotation_world.wxyz

        # Right face: translate right by half x size, rotate to be vertical
        right_relative_position = np.array([half_size[0], 0, half_size[2]])
        right_position_world = (floor_transform_matrix @ np.array([*right_relative_position, 1]))[:3]
        # Corrected rotation for left/right faces
        right_rotation_relative = vtf.SO3.from_y_radians(np.pi / 2) @ vtf.SO3.from_z_radians(np.pi / 2) # Rotate 90 around y, then 90 around z
        right_rotation_world = floor_rotation @ right_rotation_relative
        self.box_faces["right"].position = right_position_world
        self.box_faces["right"].wxyz = right_rotation_world.wxyz


    def _configure_floor_plane_and_box(self):
        print("Configuring floor plane and bounding box based on loaded sensor data...")
        if not self.sensor_data:
            print("No sensor data available for configuration.")
            return

        # Collect all sensor Gaussian and real-world coordinates
        gaussian_points = []
        real_points = []
        for sensor_name, data in self.sensor_data.items():
            # The translation part of align_transform is the sensor center in Gaussian scene
            gaussian_points.append(data["transform"][:3, 3])
            # The coordinates are the sensor center in the real measured system
            real_points.append(data["coordinates"])

        if not gaussian_points or not real_points or len(gaussian_points) != len(real_points) or len(gaussian_points) < 3:
            print("Insufficient sensor data (need at least 3 corresponding points) to estimate transformation.")
            # Fallback or error handling if not enough data
            self.real_to_gaussian_transform = None
            # Optionally reset bounding box to a default state or hide it
            return

        gaussian_points = np.vstack(gaussian_points)
        real_points = np.vstack(real_points)

        # Estimate the transformation from real world to Gaussian scene
        real_to_gaussian_transform = estimate_similarity_transform(real_points, gaussian_points)

        if real_to_gaussian_transform is None:
            print("Failed to estimate transformation.")
            self.real_to_gaussian_transform = None
            # Optionally reset bounding box to a default state or hide it
            return

        # Store the calculated transform
        self.real_to_gaussian_transform = real_to_gaussian_transform

        # 1) 从变换中提取旋转+缩放子矩阵，并正交化成纯旋转
        RS = real_to_gaussian_transform[:3, :3]
        U, S_vals, Vt = np.linalg.svd(RS)
        D = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            D[2, 2] = -1
        rotation_matrix_gaussian = U @ D @ Vt

        # 2) 计算均匀缩放因子
        var_real = np.sum((real_points - real_points.mean(axis=0))**2) / real_points.shape[0]
        scale = np.sum(S_vals * np.diag(D)) / var_real

        # 3) 将真实世界 z=0 平面映射到高斯场景
        point_on_plane_real = np.array([0, 0, 0, 1])
        point_on_plane_gaussian = real_to_gaussian_transform @ point_on_plane_real
        point_on_plane_gaussian = point_on_plane_gaussian[:3] / point_on_plane_gaussian[3]

        # 4) 计算平面法线（纯旋转作用下的 z 轴向量）
        normal_gaussian = rotation_matrix_gaussian @ np.array([0, 0, 1])
        normal_gaussian /= np.linalg.norm(normal_gaussian)

        # 5) 计算 floor plane 的中心：把传感器中心平均点投影到平面上
        mean_sensor_center_gaussian = np.mean(gaussian_points, axis=0)
        plane_point = point_on_plane_gaussian
        plane_normal = normal_gaussian
        vector_to_mean_sensor = mean_sensor_center_gaussian - plane_point
        projection_on_normal = (np.dot(vector_to_mean_sensor, plane_normal)
                                / np.dot(plane_normal, plane_normal)) * plane_normal
        floor_plane_center_gaussian = mean_sensor_center_gaussian - projection_on_normal

        # 6) 更新 floor 控件的位置和朝向
        if self.floor_transform_controls is not None:
            self.floor_transform_controls.position = floor_plane_center_gaussian
            self.floor_transform_controls.wxyz = vtf.SO3.from_matrix(rotation_matrix_gaussian).wxyz

        # 7) 计算并存储均匀缩放后的 bounding box 大小
        real_world_size = np.array(self.box_size.value)
        self.gaussian_scene_size = real_world_size * scale

        # 8) 更新所有 box faces
        self._update_box_faces_size_and_position()
        self._update_other_faces()


    def _estimate_transform(self, source_points: np.ndarray, target_points: np.ndarray):
        """
        Estimates a similarity transformation (translation, rotation, and uniform scaling)
        from source_points to target_points using Procrustes analysis.

        Args:
            source_points: A numpy array of shape (N, 3) representing the source points (real-world coordinates).
            target_points: A numpy array of shape (N, 3) representing the target points (Gaussian scene coordinates).

        Returns:
            A 4x4 numpy array representing the estimated transformation matrix from source to target,
            or None if estimation fails.
        """
        if source_points.shape != target_points.shape or source_points.shape[0] < 3:
            print("Error: Need at least 3 corresponding points with matching shapes for transformation estimation.")
            return None

        # Center the point sets
        centroid_source = np.mean(source_points, axis=0)
        centroid_target = np.mean(target_points, axis=0)
        centered_source = source_points - centroid_source
        centered_target = target_points - centroid_target

        # Calculate the covariance matrix
        covariance_matrix = centered_target.T @ centered_source

        # Perform SVD on the covariance matrix
        U, _, Vt = np.linalg.svd(covariance_matrix)

        # Calculate the optimal rotation matrix
        rotation_matrix = U @ Vt

        # Handle reflection
        if np.linalg.det(rotation_matrix) < 0:
            Vt[2, :] *= -1
            rotation_matrix = U @ Vt

        # Calculate the optimal scaling factor
        scale = np.sum(np.linalg.norm(centered_target, axis=1)) / np.sum(np.linalg.norm(centered_source, axis=1))

        # Construct the transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = scale * rotation_matrix
        transform_matrix[:3, 3] = centroid_target - (scale * rotation_matrix @ centroid_source)

        return transform_matrix


    def _get_selected_gaussians_mask(self):
        if not self.box_faces or self.floor_transform_controls is None or self.real_to_gaussian_transform is None:
            # If bounding box is not set up or transform is not loaded, consider all gaussians "selected" (inside)
            if self.viewer.gaussian_model is not None:
                 return torch.ones(self.viewer.gaussian_model.get_xyz.shape[0], dtype=torch.bool, device=self.viewer.gaussian_model.get_xyz.device)
            else:
                 return None # Cannot determine mask without gaussian model


        # Get floor transform (Gaussian scene to floor local)
        floor_position = np.array(self.floor_transform_controls.position)
        floor_wxyz = self.floor_transform_controls.wxyz
        floor_rotation_matrix = vtf.SO3(floor_wxyz).as_matrix()

        # Combine rotation and translation into a 4x4 transform matrix
        floor_transform_gaussian = np.eye(4)
        floor_transform_gaussian[:3, :3] = floor_rotation_matrix
        floor_transform_gaussian[:3, 3] = floor_position

        # Inverse transform to get from Gaussian scene to floor local coordinates
        gaussian_to_floor_transform = np.linalg.inv(floor_transform_gaussian)

        # Get Gaussian coordinates
        if self.viewer.gaussian_model is None:
            return None # Cannot determine mask without gaussian model
        xyz_gaussian = self.viewer.gaussian_model.get_xyz.cpu().numpy() # Get as numpy for matrix multiplication

        # Transform Gaussian coordinates to floor local coordinates
        # Add a column of ones for homogeneous coordinates
        xyz_gaussian_homogeneous = np.hstack([xyz_gaussian, np.ones((xyz_gaussian.shape[0], 1))])
        xyz_floor_local_homogeneous = (gaussian_to_floor_transform @ xyz_gaussian_homogeneous.T).T
        xyz_floor_local = xyz_floor_local_homogeneous[:, :3] / xyz_floor_local_homogeneous[:, 3:] # Convert back from homogeneous

        # Get bounding box size in Gaussian scene units
        gaussian_scene_size = self.gaussian_scene_size # Use the stored size

        # Check if points are within the bounding box bounds relative to the floor
        # Assuming floor is at z=0 in floor local coordinates, and box extends in +z
        is_inside = (xyz_floor_local[:, 0] >= -gaussian_scene_size[0] / 2.0) & (xyz_floor_local[:, 0] <= gaussian_scene_size[0] / 2.0) & \
                    (xyz_floor_local[:, 1] >= -gaussian_scene_size[1] / 2.0) & (xyz_floor_local[:, 1] <= gaussian_scene_size[1] / 2.0) & \
                    (xyz_floor_local[:, 2] >= 0) & (xyz_floor_local[:, 2] <= gaussian_scene_size[2]) # Z from 0 to height


        return torch.from_numpy(is_inside).to(self.viewer.gaussian_model.get_xyz.device)


    def _toggle_gaussians_outside_box(self):
        if self.viewer.gaussian_model is None:
            print("Gaussian model not available.")
            return

        mask_inside = self._get_selected_gaussians_mask()

        if mask_inside is None:
            print("Could not determine bounding box mask.")
            return

        # Invert the mask to get gaussians outside the box
        mask_outside = ~mask_inside

        # Store the mask for gaussians inside the global box
        self._global_box_mask = mask_inside

        # Toggle visibility/opacity of gaussians outside the box

        # Let's add a state variable to track if gaussians outside are hidden
        if not hasattr(self, '_gaussians_outside_hidden'):
            self._gaussians_outside_hidden = False

        if self._gaussians_outside_hidden:
            # Gaussians outside are currently hidden, show all
            print("Showing all gaussians.")
            all_gaussians_mask = torch.ones(self.viewer.gaussian_model.get_xyz.shape[0], dtype=torch.bool, device=self.viewer.gaussian_model.get_xyz.device)
            self.viewer.gaussian_model.select(~all_gaussians_mask)
            self._gaussians_outside_hidden = False
        else:
            # Gaussians outside are currently visible, hide them
            print("Hiding gaussians outside bounding box.")
            self.viewer.gaussian_model.select(mask_outside)
            self._gaussians_outside_hidden = True

        # Trigger a rerender to update the display
        self.viewer.rerender_for_all_client()
