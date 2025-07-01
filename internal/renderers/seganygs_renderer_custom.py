import os
import traceback
from typing import Dict

import re
import torch
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
from .renderer import RendererOutputTypes, RendererOutputInfo, Renderer
from .gsplat_renderer import GSPlatRenderer, DEFAULT_ANTI_ALIASED_STATUS, DEFAULT_BLOCK_SIZE
from .gsplat_contrastive_feature_renderer import GSplatContrastiveFeatureRenderer
from ..cameras import Camera
from ..models.gaussian import GaussianModel
from internal.utils.seganygs import ScaleGateUtils, SegAnyGSUtils

import trimesh
from ..mesh_alignment.mesh_utils import sample_mesh_surface, sample_mesh_volume, align_with_cpd_simple, remove_outliers
import numpy as np

import viser.transforms as vtf


def extract_uniform_scale(transform: np.ndarray) -> float:
    # 提取旋转缩放矩阵的列向量长度，取其平均作为统一 scale
    sx = np.linalg.norm(transform[:3, 0])
    sy = np.linalg.norm(transform[:3, 1])
    sz = np.linalg.norm(transform[:3, 2])
    return (sx + sy + sz) / 3.0

class SegAnyGSRenderer(Renderer):
    def __init__(
            self,
            semantic_features: torch.Tensor,
            scale_gate: torch.nn.Module,
            anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS,
    ):
        super().__init__()

        self.anti_aliased = anti_aliased

        self.initial_scale = 1.

        # move to cuda first
        self.semantic_features = semantic_features.cuda()
        self.scale_gate = ScaleGateUtils(scale_gate.cuda())

        self.scale_conditioned_semantic_features = SegAnyGSUtils.get_scale_conditioned_semantic_features(self.semantic_features, self.scale_gate, self.initial_scale)

        # PCA
        normalized_semantic_features = torch.nn.functional.normalize(self.semantic_features, dim=-1)
        self.pca_projection_matrix = SegAnyGSUtils.get_pca_projection_matrix(normalized_semantic_features)
        self.pca_colors = SegAnyGSUtils.get_pca_projected_colors(normalized_semantic_features, self.pca_projection_matrix)
        # scale conditioned PCA
        self.scale_conditioned_pca_projection_matrix = torch.nn.functional.normalize(self.pca_projection_matrix * self.scale_gate(self.initial_scale).unsqueeze(-1).to(self.pca_projection_matrix.device), dim=-1)
        self.scale_gated_pca_colors = SegAnyGSUtils.get_pca_projected_colors(
            self.scale_conditioned_semantic_features,
            self.scale_conditioned_pca_projection_matrix,
        )

        self.gaussians = None

        self.positive_mask = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.bool, device="cuda")
        self.negative_mask = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.bool, device="cuda")
        self.segment_mask = self.positive_mask & ~self.negative_mask

        self.similarities = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.float, device="cuda")

        self.cluster_color = None
        self.cluster_result = None

        # reduce CUDA memory consumption
        # self.semantic_features = self.semantic_features.cpu()  # slow scale update a little
        self.scale_conditioned_semantic_features = self.scale_conditioned_semantic_features.cpu()
        # torch.cuda.empty_cache()

        self.color_producers = {
            "rgb": self._shs_to_rgb,
            "depth": self._depth_as_color,
            "pca2d": self._semantic_features_as_color,
            "pca3d": self._pca_as_color,
            "scale_gated_pca2d": self._scale_gated_semantic_features_as_color,
            "scale_gated_pca3d": self._scale_gated_pca_as_color,
            "cluster3d": self._cluster_as_color,
            "segment3d": self._segment_as_color,
            "segment3d_out": self._segment_out,
            "segment3d_removed": self._segment_removed,
            "segment3d_similarities": self._segment_similarities_as_color,
        }

        self.available_output_types = {
            "rgb": "rgb",
            "depth": "depth",
            "pca2d": "semantic_features",
            "pca3d": "pca3d",
            "scale_gated_pca2d": "semantic_features_scale_gated",
            "scale_gated_pca3d": "pca3d_scale_gated",
            "cluster3d": "cluster3d",
            "segment3d": "segment3d",
            "segment3d_out": "segment3d_out",
            "segment3d_removed": "segment3d_removed",
            "segment3d_similarities": "segment3d_similarities",
        }

        self.output_post_processor = {
            "pca2d": self._get_pca_projected_color,
            "scale_gated_pca2d": self._get_scale_gated_pca_projected_color,
        }

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        project_results = GSPlatRenderer.project(
            means3D=pc.get_xyz,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
            viewpoint_camera=viewpoint_camera,
            scaling_modifier=scaling_modifier,
        )
        self.gaussians = pc
        opacities = pc.get_opacity
        if self.anti_aliased is True:
            comp = project_results[4]
            opacities = opacities * comp[:, None]

        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        outputs = {}

        for i in render_types:
            colors, rasterize_bg_color, new_opacities = self.color_producers[i](project_results, pc, viewpoint_camera, bg_color, opacities)
            outputs[self.available_output_types[i]] = self.rasterize(project_results, img_height=img_height, img_width=img_width, colors=colors, bg_color=rasterize_bg_color, opacities=new_opacities)
            output_processor = self.output_post_processor.get(i)
            if output_processor is not None:
                outputs[self.available_output_types[i]] = output_processor(outputs[self.available_output_types[i]])

        return outputs

    def rasterize(self, project_results, img_height, img_width, colors, bg_color, opacities):
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_results

        return rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            colors,
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=DEFAULT_BLOCK_SIZE,
            background=bg_color,
            return_alpha=False,
        ).permute(2, 0, 1)  # type: ignore

    def _shs_to_rgb(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        means3D = pc.get_xyz
        viewdirs = means3D.detach() - viewpoint_camera.camera_center  # (N, 3)
        # viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

        return rgbs, bg_color, opacities

    def _depth_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return project_results[1].unsqueeze(-1), torch.zeros((1,), dtype=torch.float, device=bg_color.device), opacities

    def _semantic_features_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return self.semantic_features, torch.zeros((self.semantic_features.shape[-1],), dtype=torch.float, device=bg_color.device), opacities

    def _get_pca_projected_color(self, feature_map):
        return SegAnyGSUtils.get_pca_projected_colors(
            semantic_features=torch.nn.functional.normalize(feature_map.permute(1, 2, 0).view(-1, feature_map.shape[0]), dim=-1),
            pca_projection_matrix=self.pca_projection_matrix,
        ).view(*feature_map.shape[1:], 3).permute(2, 0, 1)

    def _pca_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return self.pca_colors, bg_color, opacities

    def _scale_gated_semantic_features_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return self.scale_conditioned_semantic_features.to(bg_color.device), torch.zeros((self.scale_conditioned_semantic_features.shape[-1],), dtype=torch.float, device=bg_color.device), opacities

    def _get_scale_gated_pca_projected_color(self, feature_map):
        return SegAnyGSUtils.get_pca_projected_colors(
            semantic_features=torch.nn.functional.normalize(feature_map.permute(1, 2, 0).view(-1, feature_map.shape[0]), dim=-1),
            pca_projection_matrix=self.scale_conditioned_pca_projection_matrix,
        ).view(*feature_map.shape[1:], 3).permute(2, 0, 1)

    def _scale_gated_pca_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return self.scale_gated_pca_colors, bg_color, opacities

    def _cluster_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        if self.cluster_color is None:
            # TODO: fix cluster twice sometimes
            try:
                self.viewer_options.print_cluster_start_message()
            except:
                pass
            self.cluster_in_3d()
            try:
                self.viewer_options.print_cluster_finished_message()
            except:
                pass

        return self.cluster_color, bg_color, opacities

    def _segment_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        colors, bg_color, opacities = self._shs_to_rgb(project_results, pc, viewpoint_camera, bg_color, opacities)
        if self.segment_mask is not None:
            colors[self.segment_mask] = torch.tensor([0., 1., 1.], dtype=torch.float, device=bg_color.device)
        return colors, bg_color, opacities

    def _segment_out(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        colors, bg_color, opacities = self._shs_to_rgb(project_results, pc, viewpoint_camera, bg_color, opacities)
        if self.segment_mask is not None:
            opacities = opacities * self.segment_mask.unsqueeze(-1)
        return colors, bg_color, opacities

    def _segment_removed(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        colors, bg_color, opacities = self._shs_to_rgb(project_results, pc, viewpoint_camera, bg_color, opacities)
        if self.segment_mask is not None:
            opacities = opacities * (~self.segment_mask).unsqueeze(-1)
        return colors, bg_color, opacities

    def _segment_similarities_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        if self.similarities is not None:
            return self.similarities.unsqueeze(-1), torch.zeros((1, 1), dtype=torch.float, device=opacities.device), opacities
        return torch.zeros((pc.get_xyz.shape[0], 3), dtype=torch.float, device=opacities.device), bg_color, opacities

    def cluster_in_3d(self):
        self.cluster_result = SegAnyGSUtils.cluster_3d_as_dict(self.scale_conditioned_semantic_features)
        self.cluster_color = torch.tensor(self.cluster_result["point_colors"], dtype=torch.float, device="cuda")

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        with tabs.add_tab("Semantic"):
            self.viewer_options = ViewerOptions(self, viewer, server, initial_scale=self.initial_scale)

    def get_available_outputs(self) -> Dict:
        available_outputs = {}
        for i in self.available_output_types:
            available_outputs[i] = RendererOutputInfo(self.available_output_types[i], type=RendererOutputTypes.GRAY if self.is_type_depth_map(i) else RendererOutputTypes.RGB)

        return available_outputs

    def is_type_depth_map(self, t: str) -> bool:
        return t == "depth" or t == "segment3d_similarities"




class OptionCallbacks:
    def __init__(
            self,
            options,
    ):
        self.options = options

    @property
    def renderer(self) -> SegAnyGSRenderer:
        return self.options.renderer

    @property
    def viewer(self):
        return self.options.viewer

    @property
    def scale_gate(self):
        return self.renderer.scale_gate

    def get_update_scale_conditioned_features_callback(self, on_features_updated_callbacks):
        def update_scale_conditioned_features(scale):
            semantic_features = self.renderer.semantic_features.cuda()

            scale_conditioned_semantic_features = torch.nn.functional.normalize(
                semantic_features * self.scale_gate(scale).to(semantic_features.device),
                dim=-1,
            )
            scale_conditioned_pca_projection_matrix = torch.nn.functional.normalize(self.renderer.pca_projection_matrix * self.scale_gate(scale).to(semantic_features.device).unsqueeze(-1), dim=-1)

            self.renderer.scale_conditioned_semantic_features = scale_conditioned_semantic_features
            self.renderer.scale_conditioned_pca_projection_matrix = scale_conditioned_pca_projection_matrix
            for i in on_features_updated_callbacks:
                i(scale_conditioned_semantic_features)

            # move to cpu after all callback invoked (slow scale update a lot)
            # self.renderer.scale_conditioned_semantic_features = scale_conditioned_semantic_features.cpu()

        return update_scale_conditioned_features

    def update_scale_conditioned_pca_colors(self, scale_conditioned_semantic_features):
        self.renderer.scale_gated_pca_colors = SegAnyGSUtils.get_pca_projected_colors(
            scale_conditioned_semantic_features,
            self.renderer.scale_conditioned_pca_projection_matrix,
        )

    def get_update_selected_point_number_by_mask_callback(self, point_number):
        def update_point_number(mask):
            if mask is None:
                point_number.value = 0
            else:
                point_number.value = mask.sum().item()

        return update_point_number

    def update_segment_mask_on_scale_conditioned_feature_updated(self, *args, **kwargs):
        self.options._segment()


class ViewerOptions:
    def __init__(
            self,
            renderer: SegAnyGSRenderer,
            viewer, server,
            initial_scale: float,
    ):
        self.renderer = renderer
        self.viewer = viewer
        self.server = server

        # callback lists
        self.callbacks = OptionCallbacks(self)
        self._on_scale_updated_callbacks = []
        self._on_segment_mask_updated_callbacks = []
        self._on_scale_conditioned_features_updated_callbacks = [
            self.callbacks.update_scale_conditioned_pca_colors,
            self.callbacks.update_segment_mask_on_scale_conditioned_feature_updated,
        ]
        self._on_render_output_type_switched_callbacks = []

        self._on_scale_updated_callbacks.append(
            self.callbacks.get_update_scale_conditioned_features_callback(self._on_scale_conditioned_features_updated_callbacks),
        )

        # properties
        self.scale = initial_scale
        self.similarity_score = 0.9
        self.similarity_score_gamma = 1.

        self._feature_map = None
        self.feature_list = []

        model_dir = self.viewer.model_paths[0]
        print(f"Model dir: {model_dir}")

        self.segment_result_save_dir = os.path.join(
           model_dir,
            "segments",
        )

        self.cluster_result_save_dir = os.path.join(
           model_dir,
            "clusters",
        )
        self.mesh_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "meshes",
        )

        self.save_name = None
        self.align_transform = None
        self.transform_control = None

        # setup ui
        self._setup_output_type_dropdown()
        self._setup_scale_number()

        with server.gui.add_folder("分割选项"):
            self._setup_segment()

        with server.gui.add_folder("对齐选项"):
            self._setup_alignment()

        with server.gui.add_folder("聚类选项", expand_by_default= False):
            self._setup_cluster()
            server.gui.add_markdown("")
            with server.gui.add_folder("保存聚类"):
                self._setup_save_cluster()
            with server.gui.add_folder("加载聚类"):
                self._setup_load_cluster()

    @property
    def scale_gate(self) -> ScaleGateUtils:
        return self.renderer.scale_gate

    @property
    def semantic_features(self) -> torch.Tensor:
        return self.renderer.semantic_features

    @property
    def scale_conditioned_semantic_features(self) -> torch.Tensor:
        return self.renderer.scale_conditioned_semantic_features

    @property
    def segment_mask(self):
        return self.renderer.segment_mask

    @segment_mask.setter
    def segment_mask(self, value):
        self.renderer.segment_mask = value
        for i in self._on_segment_mask_updated_callbacks:
            i(value)

    @property
    def positive_mask(self):
        return self.renderer.positive_mask

    @positive_mask.setter
    def positive_mask(self, value):
        self.renderer.positive_mask = value
    
    @property
    def negative_mask(self):
        return self.renderer.negative_mask
    
    @negative_mask.setter
    def negative_mask(self, value):
        self.renderer.negative_mask = value

    @property
    def similarities(self):
        return self.renderer.similarities

    @similarities.setter
    def similarities(self, value):
        self.renderer.similarities = value

    @property
    def cluster_result(self):
        return self.renderer.cluster_result

    @cluster_result.setter
    def cluster_result(self, value):
        self.renderer.cluster_result = value
        if value is None:
            self.renderer.cluster_color = None
        else:
            self.renderer.cluster_color = torch.tensor(value["point_colors"], dtype=torch.float, device="cuda")

    def _setup_output_type_dropdown(self):
        render_type_dropdown = self.server.gui.add_dropdown(
            label="渲染模式",
            options=list(self.renderer.available_output_types.keys()),
        )

        @render_type_dropdown.on_update
        def _(event):
            if event.client is None:
                return
            self._switch_renderer_output_type(render_type_dropdown.value)

        def update_dropdown(value):
            render_type_dropdown.value = value

        self._on_render_output_type_switched_callbacks.append(update_dropdown)

    def _setup_scale_number(self):
        scale_slider = self.server.gui.add_slider(
            "分割粒度",
            min=0.,
            max=1.,
            step=0.001,
            initial_value=self.scale,
            hint="分割粒度越小，分割越精细",
        )
        self._scale_slider = scale_slider

        @scale_slider.on_update
        def _(event):
            if event.client is None:
                return
            with self.server.atomic():
                self.scale = scale_slider.value
                for i in self._on_scale_updated_callbacks:
                    i(scale_slider.value)
                self.viewer.rerender_for_all_client()

    """
    Segment
    """

    def _segment(self):
        if len(self.feature_list) == 0:
            self.positive_mask = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.bool, device="cuda")
            self.negative_mask = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.bool, device="cuda")
            self.segment_mask = self.positive_mask & ~self.negative_mask
            self.similarities = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.float, device="cuda")
            return

        scale_conditioned_semantic_features = self.scale_conditioned_semantic_features.cuda()

        positive_feature_list = [feature[0] for feature in self.feature_list if feature[1]]
        negative_feature_list = [feature[0] for feature in self.feature_list if not feature[1]]

        if len(positive_feature_list):
            mask, similarities = SegAnyGSUtils.get_segment_mask_by_raw_feature_list(
                scale_conditioned_semantic_features,
                positive_feature_list,
                self.scale_gate,
                self.scale,
                self.similarity_score,
                self.similarity_score_gamma,
                return_similarity_matrix=True,
            )
            self.positive_mask = mask
            similarities = torch.max(similarities, dim=-1).values
            self.similarities = similarities
        
        if len(negative_feature_list):

            semantic_features = self.renderer.semantic_features.cuda()

            scale_conditioned_semantic_features = torch.nn.functional.normalize(
                semantic_features * self.scale_gate(0.0).to(semantic_features.device),
                dim=-1,
            )

            mask, similarities = SegAnyGSUtils.get_segment_mask_by_raw_feature_list(
                scale_conditioned_semantic_features,
                negative_feature_list,
                self.scale_gate,
                0.0,
                self.similarity_score,
                self.similarity_score_gamma,
                return_similarity_matrix=True,
            )
            self.negative_mask = mask
        
        self.segment_mask = self.positive_mask & ~self.negative_mask

    def _add_segment_by_query_feature(self, query_feature):
        # current_mask = self.segment_mask
        # current_similarities = self.similarities
        # if current_mask is None:
        #     current_mask = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.bool, device="cuda")
        #     current_similarities = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.float, device="cuda")

        if query_feature[1] is False:
            scale = 0.0
            semantic_features = self.renderer.semantic_features.cuda()
            scale_conditioned_semantic_features = torch.nn.functional.normalize(
                semantic_features * self.scale_gate(scale).to(semantic_features.device),
                dim=-1,
            )

        else:
            scale = self.scale
            scale_conditioned_semantic_features = self.scale_conditioned_semantic_features.cuda()

        mask, similarities = SegAnyGSUtils.get_segment_mask_by_raw_feature_list(
            scale_conditioned_semantic_features,
            [query_feature[0]],
            self.scale_gate,
            scale,
            self.similarity_score,
            self.similarity_score_gamma,
            return_similarity_matrix=True,
        )
        similarities = torch.max(similarities, dim=-1).values

        
        if query_feature[1] is True:
            
            self.positive_mask |= mask

        else:
            self.negative_mask |= mask
            print(mask.sum(), self.negative_mask.sum(), scale)

        self.segment_mask = self.positive_mask & ~self.negative_mask
        self.similarities = torch.maximum(self.similarities, similarities)

    def _setup_segment(self):
        viewer, server = self.viewer, self.server

        from internal.viewer.client import ClientThread

        def switch_to_segment_output():
            if self.viewer.viewer_renderer.output_type_dropdown.value.startswith("segment3d_") is False:
                self._switch_renderer_output_type("segment3d")

        # setup feature map renderer
        feature_map_render = GSplatContrastiveFeatureRenderer()
        feature_map_render.anti_aliased = self.renderer.anti_aliased

        point_number = server.gui.add_number(
            label="提示点数目",
            initial_value=0,
            disabled=True,
        )
        selected_point_number = server.gui.add_number(
            label="选中目标数目",
            initial_value=0,
            disabled=True,
        )
        self._on_segment_mask_updated_callbacks.append(self.callbacks.get_update_selected_point_number_by_mask_callback(selected_point_number))

        similarity_score_number = server.gui.add_slider(
            label="相似度阈值",
            initial_value=self.similarity_score,
            min=0.,
            max=1.,
            step=0.001,
        )
        similarity_score_gamma = server.gui.add_slider(
            label="相似度Gamma",
            initial_value=self.similarity_score_gamma,
            min=0.,
            max=10.,
            step=0.01,
            hint="Gamma值越大，相似度阈值越严格",
        )

        @similarity_score_number.on_update
        def _(event):
            if event.client is None:
                return
            self.similarity_score = similarity_score_number.value
            with server.atomic():
                self._segment()
            viewer.rerender_for_all_client()

        @similarity_score_gamma.on_update
        def _(event):
            if event.client is None:
                return
            self.similarity_score_gamma = similarity_score_gamma.value
            with server.atomic():
                self._segment()
            viewer.rerender_for_all_client()

        positive_prompt_button = server.gui.add_button("添加正向提示点")
        disable_click_mode_button1 = server.gui.add_button("退出点击模式", visible=False, color="red")
        negative_prompt_button = server.gui.add_button("添加负向提示点", color="black")
        disable_click_mode_button2 = server.gui.add_button("退出点击模式", visible=False, color="red")      
        @positive_prompt_button.on_click
        def _(event):
            positive_prompt_button.visible = False
            negative_prompt_button.visible = True
            disable_click_mode_button1.visible = True
            disable_click_mode_button2.visible = False

            switch_to_segment_output()

            max_res = viewer.max_res_when_static.value
            camera = ClientThread.get_camera(
                event.client.camera,
                image_size=max_res,
            ).to_device(viewer.device)

            self._feature_map = feature_map_render(
                viewpoint_camera=camera,
                pc=viewer.viewer_renderer.gaussian_model,
                bg_color=torch.zeros((self.semantic_features.shape[-1],), dtype=torch.float, device=viewer.device),
                semantic_features=self.semantic_features.to(device=viewer.device),
            )["render"].permute(1, 2, 0)

            @server.on_scene_pointer(event_type="click")
            def on_scene_click(event):
                x, y = round(event.screen_pos[0][0] * (self._feature_map.shape[1] - 1)), round(event.screen_pos[0][1] * (self._feature_map.shape[0] - 1))
                print(f"x={x}, y={y}")

                feature = self._feature_map[y, x]
                self.feature_list.append((feature,True))
                self._add_segment_by_query_feature((feature,True))
                point_number.value += 1
                viewer.rerender_for_all_client()

        @negative_prompt_button.on_click
        def _(event):
            negative_prompt_button.visible = False
            positive_prompt_button.visible = True
            disable_click_mode_button2.visible = True
            disable_click_mode_button1.visible = False
            switch_to_segment_output()

            max_res = viewer.max_res_when_static.value
            camera = ClientThread.get_camera(
                event.client.camera,
                image_size=max_res,
            ).to_device(viewer.device)

            self._feature_map = feature_map_render(
                viewpoint_camera=camera,
                pc=viewer.viewer_renderer.gaussian_model,
                bg_color=torch.zeros((self.semantic_features.shape[-1],), dtype=torch.float, device=viewer.device),
                semantic_features=self.semantic_features.to(device=viewer.device),
            )["render"].permute(1, 2, 0)

            @server.on_scene_pointer(event_type="click")
            def on_scene_click(event):
                x, y = round(event.screen_pos[0][0] * (self._feature_map.shape[1] - 1)), round(event.screen_pos[0][1] * (self._feature_map.shape[0] - 1))
                print(f"x={x}, y={y}")

                feature = self._feature_map[y, x]
                self.feature_list.append((feature,False))
                self._add_segment_by_query_feature((feature,False))
                point_number.value += 1
                viewer.rerender_for_all_client()

        @disable_click_mode_button1.on_click
        def _(event):
            server.remove_scene_pointer_callback()
            self._feature_map = None
            positive_prompt_button.visible = True
            negative_prompt_button.visible = True
            disable_click_mode_button1.visible = False
        
        @disable_click_mode_button2.on_click
        def _(event):
            server.remove_scene_pointer_callback()
            self._feature_map = None
            positive_prompt_button.visible = True
            negative_prompt_button.visible = True
            disable_click_mode_button2.visible = False

        # clear points
        clear_prompt_point_button = server.gui.add_button("清除所有提示点", color="red")

        @clear_prompt_point_button.on_click
        def _(_):
            with server.atomic():
                self.feature_list.clear()
                self.positive_mask = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.bool, device="cuda")
                self.negative_mask = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.bool, device="cuda")
                self.segment_mask = self.positive_mask & ~self.negative_mask
                self.similarities = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.float, device="cuda")
                point_number.value = 0
            
            # remove aligned mesh and point
            try:
                server.remove_by_name("Aligned Mesh")
                server.remove_by_name("scene Points")
                # server.remove_by_name("Aligned Mesh Transform")
                self.transform_control.visible = False
            except:
                pass
            viewer.rerender_for_all_client()

        # pop a point
        pop_prompt_point_button = server.gui.add_button("删除上一个提示点", color="orange")

        @pop_prompt_point_button.on_click
        def _(_):
            with server.atomic():
                try:
                    self.feature_list.pop()
                    point_number.value -= 1
                    self._segment()
                except:
                    pass
            viewer.rerender_for_all_client()

        # save segment
        # server.gui.add_markdown("")
        # with server.gui.add_folder("Save Segment"):
        #     save_name = server.gui.add_text("Name", initial_value="")           
        #     save_button = server.gui.add_button("Save")

        #     @save_button.on_click
        #     def _(event):
        #         if self._filename_check(save_name.value) is False:
        #             self._show_message(event.client, "Invalid name")
        #             return
        #         if self.segment_mask is None:
        #             self._show_message(event.client, "Mask is empty")
        #             return
        #         save_to = os.path.join(self.segment_result_save_dir, f"{save_name.value}.pt")
        #         if os.path.exists(save_to) is True:
                    
        #             # remove existing file
        #             os.remove(save_to)

        #         os.makedirs(self.segment_result_save_dir, exist_ok=True)
        #         save_button.disabled = True
        #         with server.atomic():
        #             torch.save({
        #                 "mask": self.segment_mask,
        #                 "similarities": self.similarities,
        #                 "query_features": self.feature_list,
        #                 "scale": self.scale,
        #                 "similarity_score": self.similarity_score,
        #                 "similarity_score_gamma": self.similarity_score_gamma,
        #             }, save_to)

        #             # save ply
                    

                    
        #         save_button.disabled = False
        #         self._show_message(event.client, f"Saved to '{save_to}'")


        # # load segment
        # with server.gui.add_folder("Load Segment"):
        #     reload_file_list_button = server.gui.add_button(
        #         label="Refresh",
        #     )
        #     file_dropdown = server.gui.add_dropdown(
        #         label="File",
        #         options=self._scan_pt_files(self.segment_result_save_dir),
        #         initial_value="",
        #     )
        #     load_button = server.gui.add_button(
        #         label="Load",
        #     )

        #     @reload_file_list_button.on_click
        #     def _(_):
        #         file_dropdown.options = self._scan_pt_files(self.segment_result_save_dir)

        #     @load_button.on_click
        #     def _(event):
        #         if self._filename_check(file_dropdown.value) is False or file_dropdown.value.endswith(".pt") is False:
        #             self._show_message(event.client, "Invalid filename")
        #             return

        #         load_button.disabled = True
        #         try:
        #             segment = torch.load(os.path.join(self.segment_result_save_dir, file_dropdown.value), map_location=viewer.device)
        #             if segment["mask"].shape[0] != self.semantic_features.shape[0]:
        #                 self._show_message(event.client, "File does not match to current scene")
        #                 return

        #             with server.atomic():
        #                 previous_scale = self.scale

        #                 self.segment_mask = segment["mask"]
        #                 self.similarities = segment["similarities"]
        #                 self.feature_list = segment["query_features"]
        #                 self.scale = segment["scale"]
        #                 self.similarity_score = segment["similarity_score"]
        #                 self.similarity_score_gamma = segment["similarity_score_gamma"]

        #                 # update ui
        #                 self._scale_slider.value = segment["scale"]
        #                 similarity_score_number.value = segment["similarity_score"]
        #                 similarity_score_gamma.value = segment["similarity_score_gamma"]
        #                 point_number.value = len(segment["query_features"])

        #                 # invoke scale updated callbacks
        #                 if previous_scale != segment["scale"]:
        #                     for i in self._on_scale_updated_callbacks:
        #                         i(segment["scale"])

        #             switch_to_segment_output()
        #             viewer.rerender_for_all_client()
        #         finally:
        #             load_button.disabled = False



    def _setup_alignment(self):
        viewer, server = self.viewer, self.server
        # reload_mesh_list_button = server.gui.add_button("Refresh")
        mesh_file_dropdown = server.gui.add_dropdown(
            label="标准模型文件",
            options=self._scan_obj_files(self.mesh_dir),
            initial_value="",
        )

        # set default
        if self._scan_obj_files(self.mesh_dir):
            mesh_file_dropdown.value = self._scan_obj_files(self.mesh_dir)[0]

        sample_mode_dropdown = server.gui.add_dropdown(
            label="模型采样模式",
            options=["Surface", "Volume"],
            initial_value="Volume",
        )


        align_button = server.gui.add_button("对齐模型")

        scale_slider = self.server.gui.add_slider(
            "缩放比例",
            min=0.5,
            max=1.5,
            step=0.001,
            initial_value=1,
            
        )

        scale_slider.visible = False
        
        measured_coords = server.gui.add_vector3("测得坐标", initial_value=(0, 0, 0), step=0.01)
        # add text box for adding coordinates
        
        # @reload_mesh_list_button.on_click
        # def _(_):
        #     mesh_file_dropdown.options = self._scan_obj_files(self.mesh_dir)
        transform_control = server.scene.add_transform_controls(
            name="Aligned Mesh Transform",
            scale=0.5,
            disable_sliders=False,
            visible=False,
        )
        self.transform_control = transform_control
        @align_button.on_click
        def _(event):
            if self._filename_check(mesh_file_dropdown.value) is False or not mesh_file_dropdown.value.endswith(".obj"):
                self._show_message(event.client, "不支持的文件名")
                return

            mesh_path = os.path.join(self.mesh_dir, mesh_file_dropdown.value)
            if not os.path.exists(mesh_path):
                self._show_message(event.client, f"文件不存在: {mesh_path}")
                return

            align_button.disabled = True
            try:
                mode = sample_mode_dropdown.value.lower()
                if mode == "surface":
                    source_points, mesh = sample_mesh_surface(mesh_path, num_points=6000)
                elif mode == "volume":
                    source_points, mesh = sample_mesh_volume(mesh_path, num_points=6000)
                else:
                    self._show_message(event.client, f"Unsupported sample mode: {mode}")
                    return

                means3D = self.renderer.gaussians.get_xyz
                target_points = means3D[self.segment_mask]
                # remove outliers
                target_points = remove_outliers(target_points.numpy(force=True), nb_neighbors=50, std_ratio=1.0)
                print("means3d", means3D[:10])
                print("center", target_points.mean(axis=0))
                if target_points.shape[0] == 0:
                    self._show_message(event.client, "选取的分割内点数目为0")
                    return

                
                transform = align_with_cpd_simple(target_points, source_points)
                # transform[:3, 1:3] *= -1
                self.align_transform = transform
                print("before adjustment", transform)

                scene_transform = np.linalg.inv(self.viewer.camera_transform.cpu().numpy())
                
                
                # mesh.apply_transform(transform)                    
                # mesh.apply_transform(scene_transform)
               
                # vertices = np.asarray(mesh.vertices, dtype=np.float32)
                # vertices = np.dot(vertices, scene_transform[:3, :3].T) + scene_transform[:3, 3]
                final_transform = scene_transform @ transform
                scale = extract_uniform_scale(final_transform)
                rotation = vtf.SO3.from_matrix(final_transform[:3, :3]/scale)
                position = final_transform[:3, 3]
                
                rgba = np.tile(np.array([[255, 0, 0, 20]], dtype=np.uint8), (len(mesh.faces), 1))

                # 应用颜色到 mesh
                mesh.visual.face_colors = rgba

                # 添加 mesh（用 trimesh 版本支持位姿）
                aligned_mesh = server.add_mesh_trimesh(
                    name="Aligned Mesh",
                    mesh=mesh,
                    position=position,
                    wxyz=rotation.wxyz,
                    scale=scale,
                    visible=True,
                    # wireframe=True,
                    # opacity=0.8,
                    # color=(1.0, 0.0, 0.0),
                )

                transformed_target_points = np.dot(target_points, scene_transform[:3, :3].T) + scene_transform[:3, 3]
                server.add_point_cloud(
                    name = "scene Points",
                    points = transformed_target_points,
                    point_size = 0.001,
                    colors=(0.0,1.0,0.0),  # green
                )
                final_transform = scene_transform @ transform

                # # 提取旋转四元数
                # R = vtf.SO3.from_matrix(final_transform[:3, :3])
                # position = final_transform[:3, 3]

                # 添加 transform control
                transform_control.position = position
                transform_control.wxyz = rotation.wxyz
                transform_control.visible = True
    
                scale_slider.visible = True

                def on_user_update():
                    # process the transform
                    user_so3 = vtf.SO3(transform_control.wxyz)
                    user_position = np.array(transform_control.position)
                    user_scale = scale_slider.value * scale  # 当前用户可调的最终 scale

                    # 构造旋转 + 缩放矩阵
                    rot_scaled = user_so3.as_matrix() * user_scale

                    # 拼成 4x4 仿射矩阵
                    aligned_transform = np.eye(4)
                    aligned_transform[:3, :3] = rot_scaled
                    aligned_transform[:3, 3] = user_position

                    aligned_transform = np.linalg.inv(scene_transform) @ aligned_transform

                    self.align_transform = aligned_transform

                    # print("after adjustment", aligned_transform)

                @scale_slider.on_update
                def _(_):
                    if event.client is None:
                        return
                    with self.server.atomic():
                        aligned_mesh.scale = scale_slider.value * scale

                        on_user_update()
    

                # 同步控制器 → mesh
                @transform_control.on_update
                def _(_):
                    transform_rotation = vtf.SO3(transform_control.wxyz)
                    aligned_mesh.position = transform_control.position
                    aligned_mesh.wxyz = transform_rotation.wxyz
                    # aligned_mesh.scale = transform_control.scale/100

                    on_user_update()
                self._show_message(event.client, f"Aligned '{mesh_file_dropdown.value}' with current segment")
            except Exception as e:
                self._show_message(event.client, f"Error during alignment: {str(e)}")
            finally:
                align_button.disabled = False
        
        save_name = server.gui.add_text("物体名称", initial_value="")
        save_button = server.gui.add_button("保存物体（及其位姿）")

    
        
        @save_button.on_click
        def _(event):
            if self._filename_check(save_name.value) is False:
                self._show_message(event.client, "请输入正确的文件名")
                return
            # if self.align_transform is None:
            #     self._show_message(event.client, "Please align first")
            #     return
            save_to = os.path.join(self.segment_result_save_dir, f"{save_name.value}.npz")
            save_button.disabled = True
            
            with server.atomic():

                # replace if exists
                if os.path.exists(save_to) is True:
                    os.remove(save_to)

                os.makedirs(self.segment_result_save_dir, exist_ok=True)

                
                # save the transform and the mask                        

                np.savez_compressed(save_to, transform=self.align_transform, mask=self.segment_mask.cpu().numpy(), 
                                    coordinates=measured_coords.value)
                

                
            save_button.disabled = False

            self._show_message(event.client, f"Saved to '{save_to}'")

            # remove aligned mesh
            try:
                server.remove_by_name("Aligned Mesh")
                server.remove_by_name("scene Points")
                # server.remove_by_name("Aligned Mesh Transform")
                self.transform_control.visible = False
            except:
                pass

            server.scene.add_label(
                name=save_name.value,
                text=save_name.value,
                position=self._get_label_position(),
                )
            scale_slider.visible = False
            
            self.align_transform = None
            
            measured_coords.value = [0, 0, 0]
            save_name.value = ""
            transform_control.visible = False



        
            
    def _get_label_position(self):
        means3D = self.renderer.gaussians.get_xyz
        raw_points = means3D[self.segment_mask]

        # remove outliers
        raw_points = remove_outliers(raw_points.numpy(force=True), nb_neighbors=50, std_ratio=1.0)

        if raw_points.shape[0] == 0:
            # fallback to scene center
            return self.viewer.camera_center

        # compute bounding box in up direction
        up = np.asarray(self.viewer.up_direction)
        up = up / np.linalg.norm(up)

        # 投影所有点到 up 方向
        projections = raw_points @ up
        max_proj = np.max(projections)

        # 找到对应的最大投影点
        max_point = raw_points[np.argmax(projections)]

        # 设置 label 的偏移距离（可调）
        offset = 0.05 * np.linalg.norm(np.max(raw_points, axis=0) - np.min(raw_points, axis=0))

        label_position = max_point + up * offset

        return label_position
    def _scan_obj_files(self, dir_path):
        if not os.path.exists(dir_path):
            return []
        return sorted([f for f in os.listdir(dir_path) if f.endswith(".obj")])
    """
    Cluster
    """

    def _setup_cluster(self):
        viewer, server = self.viewer, self.server

        clustering_button = server.gui.add_button(
            label="Clustering...",
            disabled=True,
            visible=False,
        )
        cluster_button = server.gui.add_button(
            label="Re-Cluster in 3D",
        )

        @cluster_button.on_click
        def _(event):
            cluster_button.visible = False
            clustering_button.visible = True
            self.print_cluster_start_message(event.client)
            with server.atomic():
                self.renderer.cluster_in_3d()
            cluster_button.visible = True
            clustering_button.visible = False
            self.print_cluster_finished_message(event.client)

            # switch output type to cluster3d
            self._switch_renderer_output_type("cluster3d")

        reshuffle_color_button = server.gui.add_button("Reshuffle Color", color="green")

        @reshuffle_color_button.on_click
        def _(event):
            if self.cluster_result is None:
                self._show_message(event.client, "Please click 'Re-Cluster in 3D' first")
                return
            cluster_result = self.cluster_result
            with server.atomic():
                new_color = SegAnyGSUtils.cluster_label2colors(cluster_result["seg_score"])
                cluster_result["point_colors"] = new_color
                self.cluster_result = cluster_result
            viewer.rerender_for_all_client()

    def _setup_save_cluster(self):
        viewer, server = self.viewer, self.server

        save_name_text = server.gui.add_text(label="Name", initial_value="")
        save_cluster_button = server.gui.add_button(label="Save")

        @save_cluster_button.on_click
        def _(event):
            save_cluster_button.disabled = True
            with server.atomic():
                try:
                    output_path = self.save_cluster_results(save_name_text.value)
                    message_text = f"Saved to '{output_path}'"
                except Exception as e:
                    message_text = str(e)
                    traceback.print_exc()
            save_cluster_button.disabled = False
            self._show_message(event.client, message_text)

    def _setup_load_cluster(self):
        viewer, server = self.viewer, self.server

        reload_file_list_button = server.gui.add_button(
            label="Refresh",
        )
        cluster_result_file_dropdown = server.gui.add_dropdown(
            label="File",
            options=self._scan_cluster_files(),
            initial_value="",
        )
        load_cluster_button = server.gui.add_button(
            label="Load",
        )

        @reload_file_list_button.on_click
        def _(_):
            cluster_result_file_dropdown.options = self._scan_cluster_files()

        @load_cluster_button.on_click
        def _(event):
            match = re.search(r"^[a-zA-Z0-9_\-.]+\.pt$", cluster_result_file_dropdown.value)
            if not match:
                self._show_message(event.client, "Invalid filename")
                return

            loaded = False

            load_cluster_button.disabled = True
            with server.atomic():
                cluster_result = torch.load(os.path.join(self.cluster_result_save_dir, cluster_result_file_dropdown.value))
                if isinstance(cluster_result, dict) is False or "point_colors" not in cluster_result:
                    self._show_message(event.client, "Invalid file content")
                elif cluster_result["point_colors"].shape[0] == self.semantic_features.shape[0]:
                    self.cluster_result = cluster_result
                    loaded = True
                else:
                    self._show_message(event.client, "File not match to current scene")

            # switch output type to cluster3d
            if loaded is True:
                self._switch_renderer_output_type("cluster3d")
            load_cluster_button.disabled = False

    def _scan_cluster_files(self):
        return self._scan_pt_files(self.cluster_result_save_dir)

    def _scan_pt_files(self, path):
        file_list = []
        try:
            for i in os.listdir(path):
                if i.endswith(".pt"):
                    file_list.append(i)
        except:
            pass
        return file_list

    def save_cluster_results(self, name):
        if self.cluster_result is None:
            raise RuntimeError("Please click 'Re-Cluster in 3D' first")

        match = re.search(r"^[a-zA-Z0-9_\-.]+$", name)
        if match:
            output_path = os.path.join(self.cluster_result_save_dir, f"{name}.pt")
            if os.path.exists(output_path):
                raise RuntimeError("File already exists")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(self.cluster_result, output_path)

            return output_path
        else:
            raise RuntimeError("Invalid name")

    def _switch_renderer_output_type(self, type):
        output_type_info = self.renderer.get_available_outputs().get(type, None)
        if output_type_info is None:
            return

        viewer = self.viewer
        viewer.viewer_renderer.output_type_dropdown.value = type
        viewer.viewer_renderer._set_output_type(type, output_type_info)

        for i in self._on_render_output_type_switched_callbacks:
            i(type)

        viewer.rerender_for_all_client()

    def print_cluster_start_message(self, client=None):
        message = "Cluster takes some time. The viewer will not response any requests during this process (may including the 'Close' button below), please be patient...<br/>You will be noticed when it is completed."
        print(message)
        self._show_message(client, message)

    def print_cluster_finished_message(self, client=None):
        message = f"Cluster completed: {len(self.cluster_result['cluster_labels'])} labels"
        print(message)
        self._show_message(client, message)

    def _show_message(self, client, message: str):
        target = client
        if target is None:
            target = self.server

        with target.gui.add_modal("Message") as modal:
            target.gui.add_markdown(message)
            close_button = target.gui.add_button("Close")

            @close_button.on_click
            def _(_) -> None:
                try:
                    modal.close()
                except:
                    pass

    def _filename_check(self, name) -> bool:
        return re.search(r"^[a-zA-Z0-9_\-.]+$", name) is not None
