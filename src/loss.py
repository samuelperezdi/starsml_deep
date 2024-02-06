import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    """contrastive loss for comparing catalog1 and catalog2 features with clip"""

    def __init__(self):
        super().__init__()

    def _get_normalized_logits(self, cat1_features: torch.Tensor, cat2_features: torch.Tensor, logit_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        # normalize features for stability
        norm_cat1_features = F.normalize(cat1_features, dim=-1, eps=1e-3)
        norm_cat2_features = F.normalize(cat2_features, dim=-1, eps=1e-3)
        
        # compute scaled dot product logits
        logits_cat1 = logit_scale * norm_cat1_features @ norm_cat2_features.T
        return logits_cat1, logits_cat1.T

    def forward(
        self, 
        cat1_features: torch.Tensor, 
        cat2_features: torch.Tensor, 
        logit_scale: float = 1.0, 
        output_dict: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # calculate logits
        logits_cat1, logits_cat2 = self._get_normalized_logits(cat1_features, cat2_features, logit_scale)
        
        # labels for contrastive learning
        labels = torch.arange(logits_cat1.size(0), device=cat1_features.device, dtype=torch.long)
        
        # compute symmetric contrastive loss
        loss = (F.cross_entropy(logits_cat1, labels) + F.cross_entropy(logits_cat2, labels)) / 2
        
        # optional dictionary output
        return {"contrastive_loss": loss} if output_dict else loss
