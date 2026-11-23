"""Optional PAFlow atom-count predictor for pocket-conditioned sampling."""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist

from models.common import ShiftedSoftplus


class AtomCountPredictor(nn.Module):
    """PAFlow pocket-to-ligand-atom-count predictor."""

    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.BatchNorm1d(128),
            ShiftedSoftplus(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            ShiftedSoftplus(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            ShiftedSoftplus(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.fc(x)


class PAFlowAtomCountSampler:
    def __init__(self, ckpt_path, device):
        self.device = device
        self.model = AtomCountPredictor().to(device)
        # The original PAFlow file includes a pickled configuration.
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint.get('model', checkpoint))
        self.model.eval()

    @torch.no_grad()
    def sample(self, pocket, n_data):
        positions = pocket.protein_pos.detach().cpu().numpy()
        distances = np.sort(pdist(positions, metric='euclidean'))[::-1]
        pocket_size = float(np.median(distances[:10]))
        features = torch.tensor(
            [len(positions), pocket.volume, pocket.area, pocket_size],
            dtype=torch.float, device=self.device,
        ).unsqueeze(0)
        normalized = float(self.model(features).reshape(-1)[0].item())
        normalized = np.maximum(normalized + np.random.normal(0.0, 0.01, int(n_data)), 0.0)
        return np.clip(np.round(normalized * 103 + 3).astype(int), 3, 106).tolist()
