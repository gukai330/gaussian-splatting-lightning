import torch

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from .renderer import Renderer
from .gsplat_renderer import DEFAULT_BLOCK_SIZE, DEFAULT_ANTI_ALIASED_STATUS
from ..cameras import Camera
from ..models.gaussian import GaussianModel

from gsplat.cuda._wrapper import (
    fully_fused_projection,
)


class GSplatContrastiveFeatureRenderer(Renderer):
    def __init__(self, feature_map_width: int = -1) -> None:
        super().__init__()

        self.block_size = DEFAULT_BLOCK_SIZE
        self.anti_aliased = DEFAULT_ANTI_ALIASED_STATUS
        self.feature_map_width = feature_map_width

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            semantic_features: torch.Tensor = None,
            **kwargs,
    ):
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())
        fx = viewpoint_camera.fx.item()
        fy = viewpoint_camera.fy.item()
        cx = viewpoint_camera.cx.item()
        cy = viewpoint_camera.cy.item()

        if self.feature_map_width > 0:
            feature_width = self.feature_map_width
            feature_height = int(feature_width * img_height / img_width)

            x_scale = feature_width / img_width
            y_scale = feature_height / img_height

            img_height = feature_height
            img_width = feature_width

            fx = fx * x_scale
            fy = fy * y_scale
            cx = cx * x_scale
            cy = cy * y_scale

        # use v1 interface manually
        viewmat = viewpoint_camera.world_to_camera.T[:3, :]
        # 3x4 -> 4x4
        if viewmat.shape[0] == 3:
            viewmat = torch.concat(
                [
                    viewmat,
                    torch.tensor(
                        [[0., 0., 0., 1.]],
                        dtype=viewmat.dtype,
                        device=viewmat.device,
                    ),
                ],
                dim=0,
            )

        Ks = torch.eye(3, device= pc.get_xyz.device)
        Ks[0, 0] = fx
        Ks[1, 1] = fy
        Ks[0, 2] = cx
        Ks[1, 2] = cy

        camera_type_id = viewpoint_camera.camera_type.item() 
        # 0 = pinhole, 1 = fisheye, 3 = panorama
        if camera_type_id == 0:
            camera_model = "pinhole"
        elif camera_type_id == 1:
            camera_model = "fisheye"
        elif camera_type_id == 3:
            camera_model = "panorama"
        else:
            raise ValueError(f"Unknown camera type {camera_type_id}.")

        radii, means2d, depths, conics, compensations = fully_fused_projection(
                means = pc.get_xyz,
                covars = None,
                quats= pc.get_rotation,
                scales = pc.get_scaling if scaling_modifier == 1.0 else pc.get_scaling * scaling_modifier,
                viewmats = viewmat.unsqueeze(0),
                Ks = Ks.unsqueeze(0),
                width = img_width,
                height = img_height,
                eps2d = 0.3,
                packed=False,
                calc_compensations=True,
                near_plane=0.01,
                camera_model=camera_model,
            )

        xys = means2d.squeeze(0)
        depths = depths.squeeze(0)
        radii = radii.squeeze(0)
        conics = conics.squeeze(0)
        comp = compensations.squeeze(0)
        num_tiles_hit = radii


        opacities = pc.get_opacity
        if self.anti_aliased is True:
            opacities = opacities * comp[:, None]

        rgb = rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            semantic_features,
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=self.block_size,
            background=bg_color,
            return_alpha=False,
        )  # type: ignore




        return {
            "render": rgb.permute(2, 0, 1),
            "viewspace_points": xys,
            "viewspace_points_grad_scale": 0.5 * max(img_height, img_width),
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def depth_forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
    ):
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())


        # use v1 interface manually
        viewmat = viewpoint_camera.world_to_camera.T[:3, :]
        # 3x4 -> 4x4
        if viewmat.shape[0] == 3:
            viewmat = torch.concat(
                [
                    viewmat,
                    torch.tensor(
                        [[0., 0., 0., 1.]],
                        dtype=viewmat.dtype,
                        device=viewmat.device,
                    ),
                ],
                dim=0,
            )

        Ks = torch.eye(3, device= pc.get_xyz.device)
        Ks[0, 0] = viewpoint_camera.fx.item()
        Ks[1, 1] = viewpoint_camera.fy.item()
        Ks[0, 2] = viewpoint_camera.cx.item()
        Ks[1, 2] = viewpoint_camera.cy.item()

        radii, means2d, depths, conics, compensations = fully_fused_projection(
                means = pc.get_xyz,
                covars = None,
                quats= pc.get_rotation,
                scales = pc.get_scaling,
                viewmats = viewmat.unsqueeze(0),
                Ks = Ks.unsqueeze(0),
                width = img_width,
                height = img_height,
                eps2d = 0.3,
                packed=False,
                calc_compensations=True,
                near_plane=0.01,
                camera_model="panorama",
            )

        xys = means2d.squeeze(0)
        depths = depths.squeeze(0)
        radii = radii.squeeze(0)
        conics = conics.squeeze(0)
        comp = compensations.squeeze(0)
        num_tiles_hit = radii


        opacities = pc.get_opacity
        if self.anti_aliased is True:
            opacities = opacities * comp[:, None]

        depth_im = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            depths.unsqueeze(-1),
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=self.block_size,
            background=torch.zeros((1,), dtype=torch.float, device=xys.device),
            return_alpha=False,
        )  # type: ignore
        depth_im = depth_im.permute(2, 0, 1)

        return depth_im



