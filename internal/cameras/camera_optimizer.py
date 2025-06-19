from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import torch
import torch.nn as nn
from .cameras import Camera, Cameras, CameraType

@dataclass
class CameraOptimizerConfig:
    """相机优化器配置"""
    mode: Literal["off", "SO3xR3", "SE3"] = "SO3xR3"
    """相机姿态优化策略"""
    
    trans_l2_penalty: float = 1e-2
    """平移参数的L2惩罚"""
    
    rot_l2_penalty: float = 1e-3
    """旋转参数的L2惩罚"""
    
    learning_rate: float = 1e-4
    """学习率"""
    
    weight_decay: float = 1e-5
    """权重衰减"""

class CameraOptimizer(nn.Module):
    """相机参数优化器"""
    def __init__(
        self,
        config: CameraOptimizerConfig,
        cameras: Cameras,
    ):
        super().__init__()
        self.config = config
        self.cameras = cameras
        
        # 初始化可优化参数
        if self.config.mode == "off":
            self.pose_adjustment = None
        else:
            # 使用6维向量表示相机姿态调整
            # 前3维表示平移，后3维表示旋转
            self.pose_adjustment = nn.Parameter(torch.zeros((cameras.R.shape[0], 6), device=cameras.R.device))
            
        # 创建优化器
        if self.pose_adjustment is not None:
            self.optimizer = torch.optim.Adam(
                [self.pose_adjustment],
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = None

    def training_setup(self, module):
        cameras = module.datamodule.dataparser_outputs.train_set.cameras

        # initialize pose adjustment parameters
        if self.config.mode == "off":
            self.pose_adjustment = None
        else:
            self.pose_adjustment = nn.Parameter(torch.zeros((cameras.R.shape[0], 6), device=cameras.R.device))
        self.cameras = cameras

        optimizer = torch.optim.Adam(
            [self.pose_adjustment],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        ) if self.pose_adjustment is not None else None

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1000,  # 每1000步调整一次学习率
            gamma=0.1  # 学习率衰减系数
        ) if optimizer is not None else None

        return optimizer, scheduler
            
    def _exp_map_SO3xR3(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """计算SO3xR3的指数映射"""
        # 分离平移和旋转
        trans = tangent_vector[:, :3]
        rot = tangent_vector[:, 3:]
        
        # 计算旋转矩阵
        theta = torch.norm(rot, dim=1, keepdim=True)
        theta = torch.clamp(theta, min=1e-4)
        
        # 计算旋转矩阵的系数
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        one_minus_cos_theta = 1 - cos_theta
        
        # 构建旋转矩阵
        K = torch.zeros((rot.shape[0], 3, 3), device=rot.device)
        K[:, 0, 1] = -rot[:, 2]
        K[:, 0, 2] = rot[:, 1]
        K[:, 1, 0] = rot[:, 2]
        K[:, 1, 2] = -rot[:, 0]
        K[:, 2, 0] = -rot[:, 1]
        K[:, 2, 1] = rot[:, 0]
        
        K2 = torch.bmm(K, K)
        
        # 计算最终的旋转矩阵
        R = torch.eye(3, device=rot.device).unsqueeze(0) + \
            (sin_theta / theta) * K + \
            (one_minus_cos_theta / (theta * theta)) * K2
            
        # 构建变换矩阵
        transform = torch.zeros((rot.shape[0], 3, 4), device=rot.device)
        transform[:, :3, :3] = R
        transform[:, :3, 3] = trans
        
        return transform
        
    def _exp_map_SE3(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """计算SE3的指数映射"""
        # 分离平移和旋转
        trans = tangent_vector[:, :3]
        rot = tangent_vector[:, 3:]
        
        # 计算旋转角度
        theta = torch.norm(rot, dim=1, keepdim=True)
        theta = torch.clamp(theta, min=1e-4)
        
        # 计算旋转矩阵的系数
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        one_minus_cos_theta = 1 - cos_theta
        
        # 构建旋转矩阵
        K = torch.zeros((rot.shape[0], 3, 3), device=rot.device)
        K[:, 0, 1] = -rot[:, 2]
        K[:, 0, 2] = rot[:, 1]
        K[:, 1, 0] = rot[:, 2]
        K[:, 1, 2] = -rot[:, 0]
        K[:, 2, 0] = -rot[:, 1]
        K[:, 2, 1] = rot[:, 0]
        
        K2 = torch.bmm(K, K)
        
        # 计算最终的旋转矩阵
        R = torch.eye(3, device=rot.device).unsqueeze(0) + \
            (sin_theta / theta) * K + \
            (one_minus_cos_theta / (theta * theta)) * K2
            
        # 计算平移
        V = torch.eye(3, device=rot.device).unsqueeze(0) + \
            (one_minus_cos_theta / (theta * theta)) * K + \
            ((theta - sin_theta) / (theta * theta * theta)) * K2
            
        t = torch.bmm(V, trans.unsqueeze(-1)).squeeze(-1)
        
        # 构建变换矩阵
        transform = torch.zeros((rot.shape[0], 3, 4), device=rot.device)
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        
        return transform
        
    def get_optimized_cameras(self, camera_indices: Optional[torch.Tensor] = None) -> Cameras:
        """获取优化后的相机参数
        
        Args:
            camera_indices: 需要获取的相机索引，如果为None则返回所有相机
        """
        if self.config.mode == "off" or self.pose_adjustment is None:
            return self.cameras
            
        # 如果指定了相机索引，只获取这些相机的参数
        if camera_indices is not None:
            cameras = Cameras(
                R=self.cameras.R[camera_indices],
                T=self.cameras.T[camera_indices],
                fx=self.cameras.fx[camera_indices],
                fy=self.cameras.fy[camera_indices],
                cx=self.cameras.cx[camera_indices],
                cy=self.cameras.cy[camera_indices],
                width=self.cameras.width[camera_indices],
                height=self.cameras.height[camera_indices],
                appearance_id=self.cameras.appearance_id[camera_indices],
                normalized_appearance_id=self.cameras.normalized_appearance_id[camera_indices],
                distortion_params=self.cameras.distortion_params[camera_indices] if self.cameras.distortion_params is not None else None,
                camera_type=self.cameras.camera_type[camera_indices],
                time=self.cameras.time[camera_indices]
            )
            pose_adjustment = self.pose_adjustment[camera_indices]
        else:
            cameras = self.cameras
            pose_adjustment = self.pose_adjustment
            
        # 计算相机姿态调整
        if self.config.mode == "SO3xR3":
            adjustment = self._exp_map_SO3xR3(pose_adjustment)
        else:  # SE3
            adjustment = self._exp_map_SE3(pose_adjustment)
            
        # 应用调整到相机参数
        R = torch.bmm(adjustment[:, :3, :3], cameras.R)
        T = adjustment[:, :3, 3] + cameras.T
        
        # 创建新的相机对象
        optimized_cameras = Cameras(
            R=R,
            T=T,
            fx=cameras.fx,
            fy=cameras.fy,
            cx=cameras.cx,
            cy=cameras.cy,
            width=cameras.width,
            height=cameras.height,
            appearance_id=cameras.appearance_id,
            normalized_appearance_id=cameras.normalized_appearance_id,
            distortion_params=cameras.distortion_params,
            camera_type=cameras.camera_type,
            time=cameras.time
        )
        return optimized_cameras
    
        
    def get_loss_dict(self, loss_dict: dict) -> None:
        """添加正则化损失"""
        if self.config.mode != "off" and self.pose_adjustment is not None:
            loss_dict["camera_opt_regularizer"] = (
                self.pose_adjustment[:, :3].norm(dim=-1).mean() * self.config.trans_l2_penalty +
                self.pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
            )
            
    def get_state_dict(self) -> Dict[str, Any]:
        """获取优化器状态"""
        state_dict = {
            "params": {k: v.data.clone() for k, v in self.named_parameters()}
        }
        if self.optimizer is not None:
            state_dict["optimizer"] = self.optimizer.state_dict()
        return state_dict
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载优化器状态"""
        for k, v in state_dict["params"].items():
            self.get_parameter(k).data.copy_(v)
        if self.optimizer is not None and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"]) 


    def on_load_checkpoint(self, module, checkpoint):
        """从检查点加载相机优化器状态"""
        if "camera_optimizer" in checkpoint:
            state_dict = checkpoint["camera_optimizer"]
            self.load_state_dict(state_dict)
        else:
            raise KeyError("Checkpoint does not contain 'camera_optimizer' key")
        
    def on_save_checkpoint(self, module, checkpoint):
        """保存相机优化器状态到检查点"""
        checkpoint["camera_optimizer"] = self.get_state_dict()
        return checkpoint