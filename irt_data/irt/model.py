"""
K-factor Multidimensional IRT model in PyTorch.

Two modes:
  1. Pure latent: θ_agent · a_task + d_task  (standard MIRT)
  2. Feature-informed: latent vectors are initialized/regularized by metadata
     features, so new agents with known model/scaffold can get reasonable priors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import FeatureSchema


class MIRT(nn.Module):
    """
    Multidimensional IRT: P(correct | agent i, task j) = σ(θ_i · a_j + d_j)

    With optional feature side-information:
      θ_i = θ_embed_i + f_agent(features_i)
      a_j = a_embed_j + f_task(features_j)
    """

    def __init__(
        self,
        n_agents: int,
        n_tasks: int,
        k: int = 4,
        schema: FeatureSchema | None = None,
        embed_dim: int = 8,
        use_features: bool = True,
    ):
        super().__init__()
        self.k = k
        self.use_features = use_features and schema is not None

        # Core latent parameters
        self.theta = nn.Embedding(n_agents, k)
        self.a = nn.Embedding(n_tasks, k)
        self.d = nn.Embedding(n_tasks, 1)

        # Initialize
        nn.init.normal_(self.theta.weight, 0, 0.1)
        nn.init.normal_(self.a.weight, 0, 0.1)
        nn.init.zeros_(self.d.weight)

        # Feature projection networks
        if self.use_features:
            self._build_feature_nets(schema, embed_dim)

    def _build_feature_nets(self, schema: FeatureSchema, embed_dim: int):
        # Agent feature encoding
        self.agent_cat_embeddings = nn.ModuleDict()
        agent_feat_dim = 0
        for col in schema.categorical_agent:
            vocab = schema.vocab_sizes.get(col, 10)
            self.agent_cat_embeddings[col] = nn.Embedding(vocab + 1, embed_dim)
            agent_feat_dim += embed_dim

        n_cont = len(schema.continuous_agent) + len(schema.boolean_agent)
        agent_feat_dim += n_cont

        if agent_feat_dim > 0:
            self.agent_proj = nn.Sequential(
                nn.Linear(agent_feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.k),
            )
        else:
            self.agent_proj = None

        # Task feature encoding
        self.task_cat_embeddings = nn.ModuleDict()
        task_feat_dim = 0
        for col in schema.categorical_task:
            vocab = schema.vocab_sizes.get(col, 10)
            self.task_cat_embeddings[col] = nn.Embedding(vocab + 1, embed_dim)
            task_feat_dim += embed_dim

        n_task_cont = len(schema.continuous_task) + len(schema.boolean_task)
        task_feat_dim += n_task_cont

        if task_feat_dim > 0:
            self.task_proj = nn.Sequential(
                nn.Linear(task_feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.k),
            )
        else:
            self.task_proj = None

    def _encode_agent_features(self, agent_cat: torch.Tensor, agent_cont: torch.Tensor) -> torch.Tensor:
        parts = []
        for i, col in enumerate(self.agent_cat_embeddings):
            parts.append(self.agent_cat_embeddings[col](agent_cat[:, i]))
        if agent_cont.shape[1] > 0:
            parts.append(agent_cont)
        if not parts:
            return torch.zeros(agent_cat.shape[0], self.k, device=agent_cat.device)
        return self.agent_proj(torch.cat(parts, dim=-1))

    def _encode_task_features(self, task_cat: torch.Tensor, task_cont: torch.Tensor) -> torch.Tensor:
        parts = []
        for i, col in enumerate(self.task_cat_embeddings):
            parts.append(self.task_cat_embeddings[col](task_cat[:, i]))
        if task_cont.shape[1] > 0:
            parts.append(task_cont)
        if not parts:
            return torch.zeros(task_cat.shape[0], self.k, device=task_cat.device)
        return self.task_proj(torch.cat(parts, dim=-1))

    def forward(
        self,
        agent_idx: torch.Tensor,
        task_idx: torch.Tensor,
        agent_cat: torch.Tensor | None = None,
        agent_cont: torch.Tensor | None = None,
        task_cat: torch.Tensor | None = None,
        task_cont: torch.Tensor | None = None,
    ) -> torch.Tensor:
        theta = self.theta(agent_idx)
        a = self.a(task_idx)
        d = self.d(task_idx).squeeze(-1)

        if self.use_features and self.agent_proj is not None:
            theta = theta + self._encode_agent_features(
                agent_cat[agent_idx], agent_cont[agent_idx]
            )
        if self.use_features and self.task_proj is not None:
            a = a + self._encode_task_features(
                task_cat[task_idx], task_cont[task_idx]
            )

        logit = (theta * a).sum(dim=-1) + d
        return logit

    def predict_proba(self, agent_idx, task_idx, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(agent_idx, task_idx, **kwargs))
