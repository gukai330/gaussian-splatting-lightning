import viser
import viser.transforms as vtf
import torch
import numpy as np
import os

from internal.transform_utils.alignment import *
from internal.viewer.ui.bounding_box_panel import BoundingBoxPanel # Import the base class
from internal.utils.create_occupancy_grid import get_occupancy_grid, save_to_npz

class MultipleBoundingBoxesPanel(BoundingBoxPanel):
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            tab,
    ):
        # Call the parent constructor
        super().__init__(server, viewer, tab)

        # Initialize a list to store information about additional bounding boxes
        self.additional_boxes = []
        self._showing_single_box_index = None # Track which single box is being shown
        self._global_box_selected_mask = None # Store the mask from the global box

        # The global bounding box UI and logic are set up by the parent class (BoundingBoxPanel)
        # through the super().__init__ call.
        # We only need to set up the UI for additional bounding boxes here.

        self._setup_multiple_boxes_ui()

    def _setup_multiple_boxes_ui(self):
        server = self.server
        # The "Bounding Box" folder is already added in ViewerOptions.
        # We will add the "Additional Bounding Boxes" folder inside it.
        with server.gui.add_folder("Additional Bounding Boxes"):
            add_box_button = server.gui.add_button("Add Box")

            @add_box_button.on_click
            def _(event: viser.GuiEvent):
                self._add_new_box()

        # add input field for configuring the box coarse and fine spacing
        self.box_spacing = server.gui.add_vector2(
            "Box Spacing (Real World Units)",
            initial_value=(0.1, 0.05),
            step=0.01
        )
        
        
        export_button = server.gui.add_button("Export All Boxes")
        @export_button.on_click
        def _(event: viser.GuiEvent):
            self.export_all_boxes()

        

    def _add_new_box(self):
        server = self.server
        box_index = len(self.additional_boxes)
        box_name = f"Box {box_index + 1}"

        with server.gui.add_folder(box_name) as box_folder:
            # Add UI for size adjustment
            box_size_gui = server.gui.add_vector3(
                "Size (Real World Units)",
                initial_value=(1.0, 1.0, 1.0),
                min=(0.01, 0.01, 0.01),
                step=0.01,
            )

            # Add toggle visibility button
            toggle_visibility_button = server.gui.add_button("Toggle Visibility")

            @toggle_visibility_button.on_click
            def _(event: viser.GuiEvent):
                self._toggle_single_box_visibility(box_index)

            # Add delete button
            delete_button = server.gui.add_button("Delete Box")

            @delete_button.on_click
            def _(event: viser.GuiEvent):
                self._delete_box(box_index)

            # Create six grid meshes for the bounding box faces
            box_faces = {}
            face_names = ["floor", "ceiling", "front", "back", "left", "right"]
            for name in face_names:
                box_faces[name] = server.add_grid(
                    name=f"/additional_boxes/{box_name}/{name}",
                    width=1.0, # Initial size, will be updated
                    height=1.0, # Initial size, will be updated
                )

            # Add transform controls to the center of the box
            # Initial position will be at the origin, will be updated by user
            transform_controls = server.add_transform_controls(
                name=f"/additional_boxes/{box_name}/transform",
                position=(0, 0, 0), # Initial position
                wxyz=self.floor_transform_controls.wxyz, # Initialize with global rotation
                scale=1.0, # Initial scale
            )

            # Store the box information
            self.additional_boxes.append({
                "name": box_name,
                "size_gui": box_size_gui,
                "delete_button": delete_button,
                "toggle_visibility_button": toggle_visibility_button,
                "faces": box_faces,
                "transform_controls": transform_controls,
                "folder": box_folder,
                "gaussian_scene_size": np.array(box_size_gui.value), # Initialize gaussian size
            })

            # Add event listeners for size_gui and transform_controls
            @box_size_gui.on_update
            def _(event: viser.GuiEvent):
                self._update_single_box_display(box_index)

            @transform_controls.on_update
            def _(event: viser.GuiEvent):
                self._update_single_box_display(box_index)

    def _update_single_box_display(self, box_index):
        if box_index < 0 or box_index >= len(self.additional_boxes):
            print(f"Error: Invalid box index {box_index}")
            return

        box_info = self.additional_boxes[box_index]
        box_size_gui = box_info["size_gui"]
        transform_controls = box_info["transform_controls"]
        box_faces = box_info["faces"]

        # Update gaussian_scene_size for this box based on real_world_size and global transform
        if self.real_to_gaussian_transform is not None:
             real_world_size = np.array(box_size_gui.value)
             scale_x = np.linalg.norm(self.real_to_gaussian_transform[:3, 0])
             scale_y = np.linalg.norm(self.real_to_gaussian_transform[:3, 1])
             scale_z = np.linalg.norm(self.real_to_gaussian_transform[:3, 2])
             average_scale = (scale_x + scale_y + scale_z) / 3.0
             box_info["gaussian_scene_size"] = real_world_size * average_scale
        else:
             # If transform is not loaded, assume 1:1 scale for visual feedback
             box_info["gaussian_scene_size"] = np.array(box_size_gui.value)


        self._update_single_box_faces_size(box_info)
        self._update_single_box_faces_position(box_info)


    def _update_single_box_faces_size(self, box_info):
        # Update the size of the grid meshes based on gaussian_scene_size for a single box
        box_faces = box_info["faces"]
        gaussian_scene_size = box_info["gaussian_scene_size"]

        if box_faces:
            # Floor and Ceiling: width = x size, height = y size
            box_faces["floor"].width = gaussian_scene_size[0]
            box_faces["floor"].height = gaussian_scene_size[1]
            box_faces["ceiling"].width = gaussian_scene_size[0]
            box_faces["ceiling"].height = gaussian_scene_size[1]

            # Front and Back: width = x size, height = z size
            box_faces["front"].width = gaussian_scene_size[0]
            box_faces["front"].height = gaussian_scene_size[2]
            box_faces["back"].width = gaussian_scene_size[0]
            box_faces["back"].height = gaussian_scene_size[2]

            # Left and Right: width = y size, height = z size
            # Corrected: Left/Right should have width = z size, height = y size
            box_faces["left"].width = gaussian_scene_size[1]
            box_faces["left"].height = gaussian_scene_size[2]
            box_faces["right"].width = gaussian_scene_size[1]
            box_faces["right"].height = gaussian_scene_size[2]


    def _update_single_box_faces_position(self, box_info):
        # Update positions and orientations of box faces relative to the transform controls (center)
        box_faces = box_info["faces"]
        transform_controls = box_info["transform_controls"]
        gaussian_scene_size = box_info["gaussian_scene_size"]

        if not box_faces or transform_controls is None:
            return

        # Get box center transform
        center_position = np.array(transform_controls.position)
        # Use the global box rotation
        global_rotation = vtf.SO3(self.floor_transform_controls.wxyz)
        center_transform_matrix = np.eye(4)
        center_transform_matrix[:3, :3] = global_rotation.as_matrix()
        center_transform_matrix[:3, 3] = center_position

        # Calculate positions and orientations relative to the center
        half_size = gaussian_scene_size / 2.0

        # Floor face: translate down by half z size
        floor_relative_position = np.array([0, 0, -half_size[2]])
        floor_position_world = (center_transform_matrix @ np.array([*floor_relative_position, 1]))[:3]
        box_faces["floor"].position = floor_position_world
        box_faces["floor"].wxyz = global_rotation.wxyz # Same rotation as global floor

        # Ceiling face: translate up by half z size
        ceiling_relative_position = np.array([0, 0, half_size[2]])
        ceiling_position_world = (center_transform_matrix @ np.array([*ceiling_relative_position, 1]))[:3]
        box_faces["ceiling"].position = ceiling_position_world
        box_faces["ceiling"].wxyz = global_rotation.wxyz # Same rotation as global floor

        # Front face: translate forward by half y size, rotate to be vertical
        front_relative_position = np.array([0, half_size[1], 0])
        front_position_world = (center_transform_matrix @ np.array([*front_relative_position, 1]))[:3]
        front_rotation_relative = vtf.SO3.from_x_radians(np.pi / 2) # Rotate 90 degrees around x-axis
        front_rotation_world = global_rotation @ front_rotation_relative
        box_faces["front"].position = front_position_world
        box_faces["front"].wxyz = front_rotation_world.wxyz

        # Back face: translate backward by half y size, rotate to be vertical
        back_relative_position = np.array([0, -half_size[1], 0])
        back_position_world = (center_transform_matrix @ np.array([*back_relative_position, 1]))[:3]
        back_rotation_relative = vtf.SO3.from_x_radians(np.pi / 2) # Rotate 90 degrees around x-axis
        back_rotation_world = global_rotation @ back_rotation_relative
        box_faces["back"].position = back_position_world
        box_faces["back"].wxyz = back_rotation_world.wxyz

        # Left face: translate left by half x size, rotate to be vertical
        left_relative_position = np.array([-half_size[0], 0, 0])
        left_position_world = (center_transform_matrix @ np.array([*left_relative_position, 1]))[:3]
        # Corrected rotation for left/right faces
        left_rotation_relative = vtf.SO3.from_y_radians(np.pi / 2) @ vtf.SO3.from_z_radians(np.pi / 2) # Rotate 90 around y, then 90 around z
        left_rotation_world = global_rotation @ left_rotation_relative
        box_faces["left"].position = left_position_world
        box_faces["left"].wxyz = left_rotation_world.wxyz

        # Right face: translate right by half x size, rotate to be vertical
        right_relative_position = np.array([half_size[0], 0, 0])
        right_position_world = (center_transform_matrix @ np.array([*right_relative_position, 1]))[:3]
        # Corrected rotation for left/right faces
        right_rotation_relative = vtf.SO3.from_y_radians(np.pi / 2) @ vtf.SO3.from_z_radians(np.pi / 2) # Rotate 90 around y, then 90 around z
        right_rotation_world = global_rotation @ right_rotation_relative
        box_faces["right"].position = right_position_world
        box_faces["right"].wxyz = right_rotation_world.wxyz

    def _delete_box(self, box_index):
        if box_index < 0 or box_index >= len(self.additional_boxes):
            print(f"Error: Invalid box index {box_index}")
            return

        box_info = self.additional_boxes.pop(box_index)

        # Destroy viser objects
        for face in box_info["faces"].values():
            face.remove()
        box_info["transform_controls"].remove()

        # Destroy GUI elements
        box_info["size_gui"].remove()
        box_info["toggle_visibility_button"].remove()
        box_info["delete_button"].remove()
        # Remove the folder associated with the box
        if "folder" in box_info and box_info["folder"] is not None:
             box_info["folder"].remove()


        # Update indices and names of remaining boxes
        for i in range(box_index, len(self.additional_boxes)):
            self.additional_boxes[i]["name"] = f"Box {i + 1}"
            # TODO: Update the GUI folder name if possible with viser

    def _toggle_single_box_visibility(self, box_index):
        if box_index < 0 or box_index >= len(self.additional_boxes):
            print(f"Error: Invalid box index {box_index}")
            return

        box_info = self.additional_boxes[box_index]

        # Use the stored global box mask
        global_mask = self._global_box_mask
        if global_mask is None:
            # If global box mask is not stored (e.g., global visibility never toggled), calculate it now
            global_mask = self._get_selected_gaussians_mask()
            if global_mask is None:
                 print("Could not determine global bounding box mask.")
                 return

        if self._showing_single_box_index is not None and self._showing_single_box_index == box_index:
            # Currently showing this single box, switch back to global box visibility
            print(f"Showing gaussians based on global box.")
            self.viewer.gaussian_model.select(~global_mask) # Use the stored global mask
            self._showing_single_box_index = None
        else:
            # Not showing this single box, switch to showing only this box
            print(f"Showing gaussians inside {box_info['name']}.")
            mask_inside_single_box = self._get_selected_gaussians_mask_single_box(box_info)
            if mask_inside_single_box is None:
                 print("Could not determine single bounding box mask.")
                 return
            self.viewer.gaussian_model.select(~mask_inside_single_box) # Use the single box mask
            self._showing_single_box_index = box_index

        # Trigger a rerender to update the display
        self.viewer.rerender_for_all_client()

    def _get_selected_gaussians_mask_single_box(self, box_info):
        # Calculate the mask for gaussians inside a single additional box
        if self.viewer.gaussian_model is None:
            return None # Cannot determine mask without gaussian model
        xyz_gaussian = self.viewer.gaussian_model.get_xyz.cpu().numpy() # Get as numpy for matrix multiplication

        transform_controls = box_info["transform_controls"]
        gaussian_scene_size = box_info["gaussian_scene_size"]

        if transform_controls is None:
             return torch.zeros(self.viewer.gaussian_model.get_xyz.shape[0], dtype=torch.bool, device=self.viewer.gaussian_model.get_xyz.device)

        # Get box center transform
        center_position = np.array(transform_controls.position)
        # Use the global box rotation
        global_rotation = vtf.SO3(self.floor_transform_controls.wxyz)
        center_transform_matrix = np.eye(4)
        center_transform_matrix[:3, :3] = global_rotation.as_matrix()
        center_transform_matrix[:3, 3] = center_position

        # Inverse transform to get from Gaussian scene to box local coordinates
        gaussian_to_box_transform = np.linalg.inv(center_transform_matrix)

        # Transform Gaussian coordinates to box local coordinates
        # Add a column of ones for homogeneous coordinates
        xyz_gaussian_homogeneous = np.hstack([xyz_gaussian, np.ones((xyz_gaussian.shape[0], 1))])
        xyz_box_local_homogeneous = (gaussian_to_box_transform @ xyz_gaussian_homogeneous.T).T
        xyz_box_local = xyz_box_local_homogeneous[:, :3] / xyz_box_local_homogeneous[:, 3:] # Convert back from homogeneous

        # Check if points are within the bounding box bounds relative to the center
        half_size = gaussian_scene_size / 2.0
        is_inside = (xyz_box_local[:, 0] >= -half_size[0]) & (xyz_box_local[:, 0] <= half_size[0]) & \
                    (xyz_box_local[:, 1] >= -half_size[1]) & (xyz_box_local[:, 1] <= half_size[1]) & \
                    (xyz_box_local[:, 2] >= -half_size[2]) & (xyz_box_local[:, 2] <= half_size[2]) # Z from -half_size to +half_size


        return torch.from_numpy(is_inside).to(self.viewer.gaussian_model.get_xyz.device)


    # TODO: Override or extend methods from BoundingBoxPanel as needed
    pass
    def export_all_boxes(self):

        # get the global bounding box dimensions and the origin

        # get floor transform
        global_rotation = vtf.SO3(self.floor_transform_controls.wxyz)

        floor_center_position = np.array(self.floor_transform_controls.position)
        
        half_size = self.gaussian_scene_size/2.0

        # local origin is half_size[2] above (in floor z direction) the floor center 
        z_axis_world = global_rotation.as_matrix()[:, 2]  # 取旋转矩阵的第3列（z轴方向）

        # 2. 沿 z 轴上移 half_size[2]
        global_origin = floor_center_position + z_axis_world * half_size[2]

        global_transformation = np.eye(4)
        global_transformation[:3, :3] = global_rotation.as_matrix()
        global_transformation[:3, 3] = global_origin
        
        # get global scale
        scale_x = np.linalg.norm(self.real_to_gaussian_transform[:3, 0])
        scale_y = np.linalg.norm(self.real_to_gaussian_transform[:3, 1])
        scale_z = np.linalg.norm(self.real_to_gaussian_transform[:3, 2])
        average_scale = (scale_x + scale_y + scale_z) / 3.0

        box_spacing = np.array(self.box_spacing.value) * average_scale

        all_grid_info = {
            'global_transformation': global_transformation,
            'global_scene_size': self.gaussian_scene_size,       
            'spacing_coarse': box_spacing[0],
            'spacing_fine': box_spacing[1],
        }
        
        additional_boxes = []

        # get the local bounding box dimensions and the origin
        for box_info in self.additional_boxes:
            local_rotation = global_rotation
            local_origin = np.array(box_info["transform_controls"].position)
            local_transformation = np.eye(4)
            local_transformation[:3, :3] = local_rotation.as_matrix()
            local_transformation[:3, 3] = local_origin

            scene_size = box_info["gaussian_scene_size"]

            grid_info = {
                'origin': local_origin,
                'scene_size': scene_size,
            }

            additional_boxes.append(grid_info)
        
        all_grid_info['additional_boxes'] = additional_boxes

        source_path = self.tab.source_path
        output_path = os.path.join(self.tab.segment_result_save_dir, "occupancy_grid.npz")


        res = get_occupancy_grid(source_path, output_path, all_grid_info)

        return