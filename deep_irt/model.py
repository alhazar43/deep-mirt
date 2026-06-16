"""
model.py  --  small ma-irt-style model for the anchoring-drift v0 experiment.

Architecture:
  Encoder: LSTM (hidden_dim=32) over (item_emb || response_emb) interactions
           -> linear projection to scalar theta per timestep.
  Item embeddings: nn.Embedding(num_items, emb_dim=8).
                   New items are appended as new rows; existing rows can be frozen.
  Decoder (GPCM head): from an item embedding, produce
           a  = softplus(linear(emb))        [discrimination, >0]
           b  = linear(emb) shape (K-1,)    [step thresholds, unconstrained output
                                              but sorted at eval time for stability]
           Given theta and item params, GPCM category probabilities for K categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPCMDecoder(nn.Module):
    """Maps item embedding -> (a, b) and computes GPCM log-probabilities."""

    def __init__(self, emb_dim: int, n_cats: int):
        super().__init__()
        self.n_cats = n_cats          # K
        # discrimination: scalar
        self.fc_a = nn.Linear(emb_dim, 1)
        # step thresholds: K-1 values
        self.fc_b = nn.Linear(emb_dim, n_cats - 1)

    def item_params(self, emb: torch.Tensor):
        """
        emb: (..., emb_dim)
        returns:
          a: (..., 1)        discrimination (softplus -> >0)
          b: (..., K-1)      step thresholds (raw; NOT sorted here so gradients flow)
        """
        a = F.softplus(self.fc_a(emb))          # >0
        b = self.fc_b(emb)                       # unconstrained
        return a, b

    def log_probs(self, theta: torch.Tensor, a: torch.Tensor,
                  b: torch.Tensor) -> torch.Tensor:
        """
        theta: (batch,) or (batch, 1)
        a:     (batch, 1)
        b:     (batch, K-1)
        returns log_probs: (batch, K)

        psi_0 = 0
        psi_k = sum_{c=1..k}  a * (theta - b[:,c-1])
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(1)           # (batch, 1)

        K = self.n_cats
        # differences: (batch, K-1)
        diff = a * (theta - b)                   # (batch, K-1)
        # cumulative sum -> psi_1 .. psi_{K-1}
        psi_pos = torch.cumsum(diff, dim=1)      # (batch, K-1)
        # prepend psi_0 = 0
        zeros = torch.zeros(theta.size(0), 1, device=theta.device)
        psi = torch.cat([zeros, psi_pos], dim=1)  # (batch, K)
        return F.log_softmax(psi, dim=1)


class AbilityModel(nn.Module):
    """
    Full model: item embeddings + LSTM encoder + GPCM decoder.

    num_items: total number of items in the embedding table (B, or B+E later).
    emb_dim:   item embedding dimension.
    hidden_dim: LSTM hidden state size.
    n_cats:    number of ordered response categories K.
    n_resp:    number of distinct response values (= K for 0-indexed responses).
    """

    def __init__(self, num_items: int, emb_dim: int = 8,
                 hidden_dim: int = 32, n_cats: int = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_cats = n_cats

        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.resp_emb = nn.Embedding(n_cats, emb_dim)    # one-hot-style learned

        lstm_input_dim = emb_dim + emb_dim               # item_emb || resp_emb
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)
        self.theta_proj = nn.Linear(hidden_dim, 1)       # hidden -> scalar theta

        self.decoder = GPCMDecoder(emb_dim, n_cats)

    def encode(self, item_ids: torch.Tensor,
               responses: torch.Tensor) -> torch.Tensor:
        """
        item_ids:  (batch, seq_len)  long
        responses: (batch, seq_len)  long  (0 .. K-1)
        returns theta: (batch, seq_len)  scalar ability at each timestep
        """
        ie = self.item_emb(item_ids)                          # (B, T, emb_dim)
        re = self.resp_emb(responses)                         # (B, T, emb_dim)
        x = torch.cat([ie, re], dim=-1)                       # (B, T, 2*emb_dim)
        h, _ = self.lstm(x)                                   # (B, T, hidden_dim)
        theta = self.theta_proj(h).squeeze(-1)                # (B, T)
        return theta

    def forward(self, item_ids: torch.Tensor,
                responses: torch.Tensor) -> torch.Tensor:
        """
        Compute GPCM NLL for each (timestep, response) pair.
        Returns mean NLL over the batch (scalar).

        At each timestep t we predict P(r_t | theta_{t-1}, item_t).
        We use theta at t-1 (shift by one): theta_0 = 0 (prior), then
        theta_{1..T-1} from the encoder output.
        """
        theta_seq = self.encode(item_ids, responses)       # (B, T)
        # shift: use theta from previous step to predict current response
        # prepend zeros as the "no-history" prior theta
        batch, T = theta_seq.shape
        zeros = torch.zeros(batch, 1, device=theta_seq.device)
        theta_in = torch.cat([zeros, theta_seq[:, :-1]], dim=1)  # (B, T)

        # item params for each position
        embs = self.item_emb(item_ids)                      # (B, T, emb_dim)
        # flatten for decoder
        embs_flat = embs.view(batch * T, self.emb_dim)
        a_flat, b_flat = self.decoder.item_params(embs_flat)
        theta_flat = theta_in.reshape(batch * T)

        log_p = self.decoder.log_probs(theta_flat, a_flat, b_flat)  # (BT, K)
        resp_flat = responses.reshape(batch * T)
        nll = F.nll_loss(log_p, resp_flat)
        return nll

    def get_final_theta(self, item_ids: torch.Tensor,
                        responses: torch.Tensor) -> torch.Tensor:
        """Returns the final-timestep theta for each sequence. Shape: (batch,)."""
        theta_seq = self.encode(item_ids, responses)     # (B, T)
        return theta_seq[:, -1]                          # (B,)
