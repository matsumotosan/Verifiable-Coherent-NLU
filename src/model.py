import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP classification head used for state classification."""
    def __init__(self, hidden_dim, out_dim, dropout):
        super().__init__()
        
        self.model = nn.Seqeuntial(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.model(x)
        return out


class TieredModel(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        # Calculate embeddings
        out = self.embed(x)
        
        # Physical state classification (precondition & effect)
        precondition_pred = self.classify_preconditions(out)
        effect_pred = self.classify_effects(out)
        
        # Conflict detection
        conflict_pred = self.detect_conflict(out)
        
        # Story choice classification
        story_pred = self.classify_story(out)
        
        return story_pred, conflict_pred, precondition_pred, effect_pred
    
    def training_step(self, x):
        preds, losses = self.shared_step(x)

    def validation_step(self, x):
        preds, losses = self.shared_step(x)
        metrics = self.calculate_metrics(preds)

    def test_step(self, x):
        preds, losses = self.shared_step(x)
        metrics = self.calculate_metrics(preds)

    def shared_step(self, x):
        # Forward pass through tiered model (multi-tiered prediction)
        story_pred, conflict_pred, precondition_pred, effect_pred = self(x)
        
        # Calculate losses
        loss_story = 0
        loss_conflict = 0
        loss_precondition = 0
        loss_effect = 0
        loss_total = loss_story + loss_conflict + loss_precondition + loss_effect
        
        # Log in dict
        preds = {
            "story": story_pred,
            "conflict": conflict_pred,
            "precondition": precondition_pred,
            "effect": effect_pred
        }
        
        losses = {
            "story": loss_story,
            "conflict": loss_conflict,
            "precondition": loss_precondition,
            "effect": loss_effect,
            "total": loss_total
        }
        
        return preds, losses

    def embed(self, x):
        pass
    
    def classify_preconditions(self, x):
        pass
    
    def classify_effects(self, x):
        pass
    
    def detect_conflict(self, x):
        pass
    
    def classify_story(self, x):
        pass
    
    def on_train_epoch_end(self):
        pass

    def calculate_metrics(self, x):
        pass