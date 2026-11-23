"""Rigid SeFMol diffusion shared by the FT and PocketPref-DPO checkpoints."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import ShiftedSoftplus, compose_context
from models.segment_ops import segment_mean
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral


def sigmoid_beta_schedule(beta_start, beta_end, num_timesteps):
    grid = np.linspace(-6, 6, num_timesteps)
    return (1 / (np.exp(-grid) + 1)) * (beta_end - beta_start) + beta_start


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def posterior_log_variance_clipped(posterior_variance):
    variance = np.asarray(posterior_variance, dtype=np.float64).reshape(-1)
    if variance.size == 0:
        raise ValueError('posterior_variance must be non-empty')
    if variance.size == 1:
        clipped = np.maximum(variance, 1e-20)
    else:
        clipped = np.append(variance[1], variance[1:])
    return np.log(np.maximum(clipped, 1e-20))


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand):
    offset = segment_mean(protein_pos, batch_protein, dim=0)
    return protein_pos - offset[batch_protein], ligand_pos - offset[batch_ligand], offset


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def log_sample_categorical(logits, uniform=None):
    if uniform is None:
        uniform = torch.rand_like(logits)
    else:
        uniform = uniform.to(logits)
        if uniform.shape != logits.shape:
            raise ValueError(
                'categorical uniform noise must match the logits shape')
        if bool(((uniform < 0.0) | (uniform > 1.0)).any()):
            raise ValueError('categorical uniform noise must be in [0, 1]')
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# %%


# Time embedding


def initialize_diffusion_params(model, config, num_timesteps):
    if config.beta_schedule != 'sigmoid' or config.v_beta_schedule != 'cosine':
        raise ValueError('Expected the published sigmoid position and cosine atom-type schedules.')
    betas = sigmoid_beta_schedule(config.beta_start, config.beta_end, num_timesteps)
    alphas = 1.0 - betas

    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

    betas = to_torch_const(betas)
    alphas_cumprod = to_torch_const(alphas_cumprod)
    alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_var = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_mean_c0_coef = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_mean_ct_coef = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)
    # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    posterior_logvar = to_torch_const(
        posterior_log_variance_clipped(posterior_var))

    # atom type diffusion schedule in log space
    alphas_v = cosine_beta_schedule(num_timesteps, config.v_beta_s)
    log_alphas_v = np.log(alphas_v)
    log_alphas_cumprod_v = np.cumsum(log_alphas_v)
    log_alphas_v = to_torch_const(log_alphas_v)
    log_one_minus_alphas_v = log_1_min_a(log_alphas_v)
    log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
    log_one_minus_alphas_cumprod_v = log_1_min_a(log_alphas_cumprod_v)

    model['betas'] = betas
    model['Lt_history'] = torch.zeros(num_timesteps)
    model['Lt_count'] = torch.zeros(num_timesteps)
    model['alphas_cumprod'] = alphas_cumprod
    model['alphas_cumprod_prev'] = alphas_cumprod_prev
    model['sqrt_alphas_cumprod'] = sqrt_alphas_cumprod
    model['sqrt_one_minus_alphas_cumprod'] = sqrt_one_minus_alphas_cumprod
    model['sqrt_recip_alphas_cumprod'] = sqrt_recip_alphas_cumprod
    model['sqrt_recipm1_alphas_cumprod'] = sqrt_recipm1_alphas_cumprod
    model['posterior_mean_c0_coef'] = posterior_mean_c0_coef
    model['posterior_mean_ct_coef'] = posterior_mean_ct_coef.float()
    model['posterior_var'] = posterior_var
    model['posterior_logvar'] = posterior_logvar
    model['log_alphas_v'] = log_alphas_v
    model['log_alphas_cumprod_v'] = log_alphas_cumprod_v
    model['log_one_minus_alphas_v'] = log_one_minus_alphas_v
    model['log_one_minus_alphas_cumprod_v'] = log_one_minus_alphas_cumprod_v


def merge_node_property(ligand_v, p_feat, batch_size, batch_ligand):
    p_feat = p_feat.reshape(batch_size, -1)
    new_feature_dim = ligand_v.shape[-1] + p_feat.shape[-1]
    new_init_ligand_v = torch.zeros(ligand_v.shape[0], new_feature_dim).to(ligand_v.device)
    for i in range(batch_size):
        mask = (batch_ligand == i)
        ligand_masked = ligand_v[mask]
        p_feat_masked = p_feat[i].repeat(ligand_masked.shape[0], 1)
        new_init_ligand_v[mask] = torch.cat([ligand_masked, p_feat_masked], dim=-1)
    return new_init_ligand_v


class SeFMol(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, n_timesteps=1000):
        super().__init__()
        self.num_timesteps = int(n_timesteps)
        self.loss_v_weight = float(config.loss_v_weight)

        schedule = {}
        initialize_diffusion_params(schedule, config, self.num_timesteps)
        for name, value in schedule.items():
            if isinstance(value, nn.Parameter):
                setattr(self, name, value)
            else:
                self.register_buffer(name, value)

        self.hidden_dim = int(config.hidden_dim)
        self.num_classes = int(ligand_atom_feature_dim)

        if config.model_type != 'uni_o2' or config.model_mean_type != 'C0':
            raise ValueError('Expected a PocketPref-DPO rigid UniTransformer checkpoint.')
        if (config.center_pos_mode != 'protein' or config.time_emb_dim != 0
                or config.sample_time_method != 'symmetric'):
            raise ValueError('Expected the published centering, time embedding and timestep sampling.')

        if (config.cutoff_mode != 'knn' or config.ew_net_type != 'global'
                or config.edge_feat_dim != 4 or config.x2h_out_fc or config.sync_twoup):
            raise ValueError('Expected the published rigid attention settings.')

        self.condition_dim = int(config.condition_dim)
        self.node_indicator = bool(config.node_indicator)
        embedding_dim = self.hidden_dim - int(self.node_indicator)
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, embedding_dim)
        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + config.condition_hidden_dim, embedding_dim)
        self.refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            act_fn=config.act_fn,
            norm=config.norm,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            safe_knn=bool(config.get('safe_knn', False)),
        )
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )
        self.condition_embedding = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, config.condition_hidden_dim),
        )

    def forward(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, properties):
        batch_size = int(batch_protein.max().item()) + 1
        protein_h = self.protein_atom_emb(protein_v)
        ligand_feature = F.one_hot(ligand_v, self.num_classes).float()
        if properties is None:
            properties = ligand_feature.new_zeros((batch_size, self.condition_dim))
        property_feature = self.condition_embedding(properties.reshape(batch_size, -1).to(ligand_feature))
        ligand_feature = merge_node_property(ligand_feature, property_feature, batch_size, batch_ligand)
        ligand_h = self.ligand_atom_emb(ligand_feature)
        if self.node_indicator:
            protein_h = torch.cat((protein_h, protein_h.new_zeros((len(protein_h), 1))), dim=-1)
            ligand_h = torch.cat((ligand_h, ligand_h.new_ones((len(ligand_h), 1))), dim=-1)

        features, positions, batch, ligand_mask = compose_context(
            protein_h, ligand_h, protein_pos, ligand_pos, batch_protein, batch_ligand)
        outputs = self.refine_net(features, positions, ligand_mask, batch)
        return {
            'pred_ligand_pos': outputs['x'][ligand_mask],
            'pred_ligand_v': self.v_inference(outputs['h'][ligand_mask]),
        }

    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)      # alpha
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)  # beta

        log_probs = log_add_exp(        # z = x_t-1 * alpha_t + beta_t / K
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs


    def q_v_sample(self, log_v0, t, batch, uniform=None):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0, uniform=uniform)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def sample_time(self, num_graphs, device):
        half = torch.randint(0, self.num_timesteps, (num_graphs // 2 + 1,), device=device)
        return torch.cat((half, self.num_timesteps - half - 1))[:num_graphs]

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = segment_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v,
            batch_ligand, properties, time_step, position_noise, categorical_uniform):
        protein_pos, ligand_pos, _ = center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand)
        time_step = time_step.to(device=protein_pos.device, dtype=torch.long).view(-1)
        alpha = self.alphas_cumprod.index_select(0, time_step)[batch_ligand].unsqueeze(-1)
        noisy_pos = alpha.sqrt() * ligand_pos + (1.0 - alpha).sqrt() * position_noise.to(ligand_pos)
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        noisy_v, log_ligand_vt = self.q_v_sample(
            log_ligand_v0, time_step, batch_ligand, uniform=categorical_uniform)
        prediction = self(
            protein_pos, protein_v, batch_protein, noisy_pos, noisy_v, batch_ligand, properties)
        predicted_pos = prediction['pred_ligand_pos']
        loss_pos = segment_mean(((predicted_pos - ligand_pos) ** 2).sum(dim=-1), batch_ligand, dim=0)
        predicted_log_v0 = F.log_softmax(prediction['pred_ligand_v'], dim=-1)
        log_v_model = self.q_v_posterior(predicted_log_v0, log_ligand_vt, time_step, batch_ligand)
        log_v_true = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        loss_v = self.compute_v_Lt(log_v_model, log_ligand_v0, log_v_true, time_step, batch_ligand)
        return {'loss_diffusion_graph': loss_pos + self.loss_v_weight * loss_v}

    @torch.no_grad()
    def sample_diffusion(
            self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v,
            batch_ligand, properties, num_steps):
        if not 1 <= num_steps <= self.num_timesteps:
            raise ValueError(f'num_steps must be in [1, {self.num_timesteps}]')
        protein_pos, ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand)
        num_graphs = int(batch_protein.max().item()) + 1
        ligand_v = init_ligand_v
        for step in range(self.num_timesteps - 1, self.num_timesteps - num_steps - 1, -1):
            time_step = torch.full((num_graphs,), step, dtype=torch.long, device=protein_pos.device)
            prediction = self(
                protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, properties)
            log_v0 = F.log_softmax(prediction['pred_ligand_v'], dim=-1)
            posterior_mean = self.q_pos_posterior(
                prediction['pred_ligand_pos'], ligand_pos, time_step, batch_ligand)
            posterior_logvar = extract(self.posterior_logvar, time_step, batch_ligand)
            nonzero = (time_step != 0).to(ligand_pos.dtype)[batch_ligand].unsqueeze(-1)
            position_noise = torch.randn_like(ligand_pos)
            log_vt = index_to_log_onehot(ligand_v, self.num_classes)
            log_v_posterior = self.q_v_posterior(log_v0, log_vt, time_step, batch_ligand)
            ligand_pos = posterior_mean + nonzero * (0.5 * posterior_logvar).exp() * position_noise
            category_noise = torch.rand_like(log_v_posterior)
            ligand_v = log_sample_categorical(log_v_posterior, uniform=category_noise)

        return {'pos': ligand_pos + offset[batch_ligand], 'v': ligand_v}


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)
