import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

from .schema_encoder import SchemaEncoder, SchemaEncoderConfig
from .recovery_module import RecoveryModule, RecoveryConfig
from .manifold_retraction import ManifoldRetraction, RetractionConfig
from .query_head import QueryHead, QueryHeadConfig
from .identifiability_gate import IdentifiabilityGate, GateConfig
from .no_leakage import NoLeakageRegularizer
from .no_leakage import NoLeakageRegularizerConfig
from utils.config import Config


class IdentityEncoder(nn.Module):
    def __init__(self, output_dim: int, num_platforms: int):
        super().__init__()
        self.output_dim = output_dim
        self.num_platforms = num_platforms
        self.platform_embedding = nn.Embedding(num_platforms, output_dim)
        self.platform_embedding.weight.data.normal_(0, 0.01)
    
    def forward(self, x, M_O, a_idx, e_a=None):
        B = x.shape[0]
        device = x.device
        z_g = torch.zeros(B, self.output_dim, device=device)
        h_a = self.platform_embedding(a_idx)
        return z_g, h_a


class IdentityRecovery(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_platforms: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, latent_dim)
        self.projection.weight.data.normal_(0, 0.01)
        self.projection.bias.data.zero_()
    
    def forward(self, z_g, a_idx):
        return self.projection(z_g)
    
    def equivariance_loss(self, z_g, a_idx):
        return torch.tensor(0.0, device=z_g.device)


class IdentityRetraction(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
    
    def forward(self, mu_hat):
        return mu_hat
    
    def idempotence_loss(self, mu_hat):
        return torch.tensor(0.0, device=mu_hat.device)


class IdentityQueryHead(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, mu_final):
        q_hat = self.net(mu_final)
        return q_hat, None


class IdentityGate(nn.Module):
    def __init__(self, num_platforms: int = 5):
        super().__init__()
        self.num_platforms = num_platforms
    
    def forward(self, M_O, a_idx, mu_final, z_g, training=True):
        B = M_O.shape[0]
        device = M_O.device
        routed = torch.ones(B, device=device)
        r0_vals = torch.ones(B, device=device) * 999
        q_prior = torch.zeros(B, device=device)
        sigma2_prior = torch.ones(B, device=device)
        sigma2_nonid = torch.zeros(B, device=device)
        return routed, r0_vals, q_prior, sigma2_prior, sigma2_nonid
    
    def route(self, M_O, a_idx, q_hat, sigma2_aleat, sigma2_epi=None):
        B = M_O.shape[0]
        device = M_O.device
        routed = torch.ones(B, device=device)
        # Ensure sigma2_aleat is [B] shape - if scalar, expand to [B]
        if sigma2_aleat is None or sigma2_aleat.dim() == 0:
            sigma2_aleat = torch.ones(B, device=device)
        sigma2_total = sigma2_aleat
        return routed, q_hat, sigma2_total, sigma2_aleat, torch.zeros(B, device=device)


class IdentityNoLeakage(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z_g, a_idx, training=True):
        device = z_g.device
        return {
            'disc_loss': torch.tensor(0.0, device=device),
            'enc_loss': torch.tensor(0.0, device=device),
            'mi_estimate': torch.tensor(0.0, device=device),
            'noleak_loss': torch.tensor(0.0, device=device),
            'disc_acc': torch.tensor(0.0, device=device),
        }


class HWINNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if config.encoder.enabled:
            self.schema_encoder = SchemaEncoder(
                SchemaEncoderConfig(
                    encoder_type=config.encoder.encoder_type,
                    n_layers=config.encoder.n_layers,
                    hidden_dim=config.encoder.hidden_dim,
                    output_dim=config.encoder.output_dim,
                    platform_embedding_dim=config.encoder.platform_embedding_dim,
                    num_platforms=config.encoder.num_platforms,
                    share_platform_encoder=config.encoder.share_platform_encoder,
                    mask_mode=config.encoder.mask_mode,
                    dropout=config.encoder.dropout,
                    activation=config.encoder.activation,
                    norm_type=config.encoder.norm_type,
                    n_vars=config.data.num_variables,
                )
            )
        else:
            self.schema_encoder = IdentityEncoder(
                output_dim=config.encoder.output_dim,
                num_platforms=config.encoder.num_platforms,
            )

        recovery_obs_dim = config.encoder.output_dim
        self.recovery_decoder = nn.Linear(config.recovery.latent_dim, config.data.num_variables)
        self.recovery_decoder.weight.data.normal_(0, 0.01)
        self.recovery_decoder.bias.data.zero_()

        if config.recovery.enabled:
            self.recovery_module = RecoveryModule(
                RecoveryConfig(
                    recovery_type=config.recovery.recovery_type,
                    latent_dim=config.recovery.latent_dim,
                    hidden_dim=config.recovery.hidden_dim,
                    n_layers=config.recovery.n_layers,
                    intertwiner_type=config.recovery.intertwiner_type,
                    base_platform=config.recovery.base_platform,
                    tie_t_base=config.recovery.tie_t_base,
                    dropout=config.recovery.dropout,
                    activation=config.recovery.activation,
                    norm_type=config.recovery.norm_type,
                    equivariance_loss_weight=config.recovery.equivariance_loss_weight,
                    obs_dim=recovery_obs_dim,
                    num_platforms=config.encoder.num_platforms,
                )
            )
        else:
            self.recovery_module = IdentityRecovery(
                input_dim=recovery_obs_dim,
                latent_dim=config.recovery.latent_dim,
                num_platforms=config.encoder.num_platforms,
            )

        if config.retraction.enabled:
            self.manifold_retraction = ManifoldRetraction(RetractionConfig(
                retraction_type=config.retraction.retraction_type,
                latent_dim=config.retraction.latent_dim,
                manifold_basis_path=config.retraction.manifold_basis_path,
                pca_components=config.retraction.pca_components,
                vae_latent_dim=config.retraction.vae_latent_dim,
                vae_encoder_layers=config.retraction.vae_encoder_layers,
                vae_decoder_layers=config.retraction.vae_decoder_layers,
                max_iter=config.retraction.max_iter,
                tolerance=config.retraction.tolerance,
                idempotence_loss_weight=config.retraction.idempotence_loss_weight,
            ))
        else:
            self.manifold_retraction = IdentityRetraction(
                latent_dim=config.retraction.latent_dim,
            )

        if config.query_head.enabled:
            self.query_head = QueryHead(QueryHeadConfig(
                head_type=config.query_head.head_type,
                latent_dim=config.query_head.latent_dim,
                output_dim=config.query_head.output_dim,
                hidden_dim=config.query_head.hidden_dim,
                n_layers=config.query_head.n_layers,
                output_distribution=config.query_head.output_distribution,
                dropout=config.query_head.dropout,
                activation=config.query_head.activation,
                norm_type=config.query_head.norm_type,
            ))
        else:
            self.query_head = IdentityQueryHead(
                latent_dim=config.query_head.latent_dim,
                output_dim=config.query_head.output_dim,
                hidden_dim=config.query_head.hidden_dim,
            )

        if config.gate.enabled:
            self.identifiability_gate = IdentifiabilityGate(GateConfig(
                r0_method=config.gate.r0_method,
                r0_init=config.gate.r0_init,
                prior_predictive_type=config.gate.prior_predictive_type,
                prior_mean=config.gate.prior_mean,
                prior_var=config.gate.prior_var,
                hard_gate=config.gate.hard_gate,
                schemas=config.gate.schemas,
                num_platforms=config.encoder.num_platforms,
                r0_regressor_hidden=config.gate.r0_regressor_hidden,
            ))
        else:
            self.identifiability_gate = IdentityGate(
                num_platforms=config.encoder.num_platforms,
            )

        if config.no_leakage.enabled:
            self.no_leakage = NoLeakageRegularizer(
                NoLeakageRegularizerConfig(
                    mi_estimator=config.no_leakage.mi_estimator,
                    discriminator_type=config.no_leakage.discriminator_type,
                    discriminator_hidden=config.no_leakage.discriminator_hidden,
                    discriminator_layers=config.no_leakage.discriminator_layers,
                    discriminator_dropout=config.no_leakage.discriminator_dropout,
                    discriminator_activation=config.no_leakage.discriminator_activation,
                    discriminator_norm=config.no_leakage.discriminator_norm,
                    gradient_reversal=config.no_leakage.gradient_reversal,
                    grl_lambda_init=config.no_leakage.grl_lambda,
                    grl_lambda_schedule=config.no_leakage.grl_lambda_schedule,
                    grl_max_lambda=config.no_leakage.grl_max_lambda,
                    lambda_mi=config.no_leakage.lambda_mi,
                ),
                z_dim=config.encoder.output_dim,
                num_platforms=config.encoder.num_platforms,
            )
        else:
            self.no_leakage = IdentityNoLeakage()

    def forward(
        self,
        x: torch.Tensor,
        M_O: torch.Tensor,
        a_idx: torch.Tensor,
        e_a: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        training: bool = True,
        equivariance_warmup_factor: float = 1.0,
    ) -> Dict[str, Any]:
        z_g, h_a = self.schema_encoder(x, M_O, a_idx, e_a)
        mu_hat = self.recovery_module(z_g, a_idx)
        mu_final = self.manifold_retraction(mu_hat)
        q_hat, sigma2_aleat = self.query_head(mu_final)
        routed, r0_vals, q_prior, sigma2_prior, sigma2_nonid = self.identifiability_gate(
            M_O, a_idx, mu_final, z_g, training
        )
        q_hat_1d = q_hat.squeeze(-1) if q_hat.dim() > 1 else q_hat
        if sigma2_aleat is not None:
            sigma2_aleat_1d = sigma2_aleat.squeeze(-1) if sigma2_aleat.dim() > 1 else sigma2_aleat
        else:
            sigma2_aleat_1d = torch.tensor(0.0, device=q_hat.device)

        if training:
            gate_vals = routed
            sigma2_total = sigma2_aleat_1d + sigma2_nonid
            q_out = gate_vals * q_hat_1d + (1.0 - gate_vals) * q_prior
            sigma2_aleat_out = sigma2_aleat_1d * gate_vals + sigma2_prior * (1.0 - gate_vals)
            sigma2_nonid_out = sigma2_nonid
        else:
            routed, q_out, sigma2_total, sigma2_aleat_out, sigma2_nonid_out =                 self.identifiability_gate.route(
                    M_O, a_idx,
                    q_hat_1d,
                    sigma2_aleat_1d,
                    sigma2_epi=None
                )
        outputs = {
            'z_g': z_g,
            'h_a': h_a,
            'mu_hat': mu_hat,
            'mu_final': mu_final,
            'q_hat': q_hat,
            'q_out': q_out,
            'sigma2_aleat': sigma2_aleat_out,
            'sigma2_total': sigma2_total,
            'routed': routed,
            'r0_vals': r0_vals,
            'sigma2_nonid': sigma2_nonid_out,
            'q_prior': q_prior,
        }

        losses = {}

        if training and self.config.no_leakage.lambda_mi > 0:
            noleak_out = self.no_leakage(z_g, a_idx, training=True)
            outputs.update({k: v for k, v in noleak_out.items() if k != 'noleak_loss'})
            outputs['noleak_disc_loss'] = noleak_out.get('disc_loss', torch.tensor(0.0, device=z_g.device))
            outputs['noleak_enc_loss'] = noleak_out.get('enc_loss', torch.tensor(0.0, device=z_g.device))

        if training and self.config.loss.lambda_rec > 0:
            x_recon = self.recovery_decoder(mu_hat)
            rec_loss = F.mse_loss(x_recon * M_O, x * M_O)
            losses['rec_loss'] = self.config.loss.lambda_rec * rec_loss

        if training and self.config.loss.lambda_equiv > 0:
            equiv_loss = self.recovery_module.equivariance_loss(z_g, a_idx)
            losses['equiv_loss'] = self.config.loss.lambda_equiv * equiv_loss * equivariance_warmup_factor

        if training and self.config.retraction.idempotence_loss_weight > 0:
            idemp_loss = self.manifold_retraction.idempotence_loss(mu_hat)
            losses['idempotence_loss'] = self.config.retraction.idempotence_loss_weight * idemp_loss

        if y is not None:
            if self.config.loss.pred_loss_type == 'mse':
                pred_loss = F.mse_loss(q_out, y)
            elif self.config.loss.pred_loss_type == 'nll':
                if sigma2_total is not None:
                    pred_loss = 0.5 * (
                        torch.log(sigma2_total + 1e-6) + (q_out - y) ** 2 / (sigma2_total + 1e-6)
                    ).mean()
                else:
                    pred_loss = F.mse_loss(q_out, y)
            else:
                pred_loss = F.mse_loss(q_out, y)
            losses['pred_loss'] = pred_loss

        if training and self.config.loss.lambda_complex > 0:
            complex_loss = mu_final.pow(2).mean()
            losses['complex_loss'] = self.config.loss.lambda_complex * complex_loss

        loss_components = {
            'pred_loss': self.config.loss.lambda_pred,
            'rec_loss': self.config.loss.lambda_rec,
            'equiv_loss': 1.0,
            'complex_loss': 1.0,
            'idempotence_loss': 1.0,
        }

        total_main_loss = sum(
            losses.get(k, 0) * loss_components.get(k, 0)
            for k in loss_components
        )
        losses['total_loss'] = total_main_loss

        outputs['losses'] = losses
        return outputs

    def forward_inference(
        self,
        x: torch.Tensor,
        M_O: torch.Tensor,
        a_idx: torch.Tensor,
        e_a: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward(x, M_O, a_idx, e_a, y=None, training=False)


def create_hwin_net(config: Config) -> HWINNet:
    return HWINNet(config)
