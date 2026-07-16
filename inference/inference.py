import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from models.hwin_net import HWINNet
from utils.config import Config

@dataclass
class InferenceConfig:
    mc_dropout_samples: int = 10
    return_all_uncertainties: bool = True
    return_routing: bool = True
    return_intermediate: bool = False

class InferenceOutput:
    def __init__(self, prediction, routed, r0_vals, sigma2_aleat=None, sigma2_epi=None, sigma2_nonid=None, sigma2_total=None, mu_final=None, z_g=None, q_hat_raw=None):
        self.prediction = prediction
        self.routed = routed
        self.r0_vals = r0_vals
        self.sigma2_aleat = sigma2_aleat
        self.sigma2_epi = sigma2_epi
        self.sigma2_nonid = sigma2_nonid
        self.sigma2_total = sigma2_total
        self.mu_final = mu_final
        self.z_g = z_g
        self.q_hat_raw = q_hat_raw

def create_inference(model: HWINNet, config: InferenceConfig = None) -> 'InferenceEngine':
    if config is None:
        config = InferenceConfig()
    return InferenceEngine(model, config)

class InferenceEngine:
    def __init__(self, model: HWINNet, config: InferenceConfig):
        self.model = model
        self.config = config
        self.model.eval()
    
    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        M_O: torch.Tensor,
        a_idx: torch.Tensor,
    ) -> InferenceOutput:
        
        B = x.shape[0]
        device = x.device
        
        if self.config.mc_dropout_samples > 1 and self.config.return_all_uncertainties:
            return self._mc_dropout_forward(x, M_O, a_idx)
        else:
            return self._single_forward(x, M_O, a_idx)
    
    @torch.no_grad()
    def _single_forward(self, x, M_O, a_idx):
        outputs = self.model.forward(x, M_O, a_idx, training=False)
        
        return InferenceOutput(
            prediction=outputs['q_out'],
            routed=outputs['routed'],
            r0_vals=outputs['r0_vals'],
            sigma2_aleat=outputs.get('sigma2_aleat'),
            sigma2_epi=outputs.get('sigma2_epi'),
            sigma2_nonid=outputs.get('sigma2_nonid'),
            sigma2_total=outputs.get('sigma2_total'),
            mu_final=outputs.get('mu_final'),
            z_g=outputs.get('z_g'),
            q_hat_raw=outputs.get('q_hat'),
        )
    
    @torch.no_grad()
    def _mc_dropout_forward(self, x, M_O, a_idx):
        B = x.shape[0]
        mc_samples = self.config.mc_dropout_samples
        
        self.model.train()
        
        predictions = []
        routed_list = []
        r0_list = []
        sigma2_aleat_list = []
        sigma2_total_list = []
        
        for _ in range(mc_samples):
            outputs = self.model.forward(x, M_O, a_idx, training=True)
            predictions.append(outputs['q_out'])
            routed_list.append(outputs['routed'])
            r0_list.append(outputs['r0_vals'])
            if 'sigma2_aleat' in outputs:
                sigma2_aleat_list.append(outputs['sigma2_aleat'])
            if 'sigma2_total' in outputs:
                sigma2_total_list.append(outputs['sigma2_total'])
        
        self.model.eval()
        
        q_mc = torch.stack(predictions, dim=0)
        mean_pred = q_mc.mean(dim=0)
        var_epi = q_mc.var(dim=0)
        
        if routed_list:
            routed_mc = torch.stack(routed_list, dim=0)
            routed_final = (routed_mc.float().mean(dim=0) > 0.5).float()
        else:
            routed_final = None
        
        if r0_list:
            r0_mc = torch.stack(r0_list, dim=0)
            r0_mean = r0_mc.mean(dim=0)
        else:
            r0_mean = torch.zeros(B, device=x.device)
        
        sigma2_aleat = None
        if sigma2_aleat_list:
            sigma2_aleat = torch.stack(sigma2_aleat_list, dim=0).mean(dim=0)
        
        sigma2_total = None
        if sigma2_total_list:
            sigma2_mc = torch.stack(sigma2_total_list, dim=0)
            sigma2_total = sigma2_mc.mean(dim=0)
        
        return InferenceOutput(
            prediction=mean_pred,
            routed=routed_final if routed_final is not None else torch.ones(B, device=x.device),
            r0_vals=r0_mean,
            sigma2_aleat=sigma2_aleat,
            sigma2_epi=var_epi,
            sigma2_nonid=None,
            sigma2_total=sigma2_total,
        )
