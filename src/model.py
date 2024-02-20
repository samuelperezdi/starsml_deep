import torch
import torch.nn as nn
import pytorch_lightning as pl
from loss import CLIPLoss

class MLP(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dim, num_hidden, dropout=0.0):
        super().__init__()
        layers = [self._make_layer(input_dim, hidden_dim, dropout)]
        layers.extend([self._make_layer(hidden_dim, hidden_dim, dropout) for _ in range(num_hidden - 2)])
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout)])
        self.add_module('mlp', nn.Sequential(*layers))

    def _make_layer(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

class AstroCLR(pl.LightningModule):
    def __init__(
        self,
        catalog1_input_dim,
        catalog2_input_dim,
        emb_dim,
        num_hidden=4,
        head_depth=2,
        catalog1_encoder=None,
        catalog2_encoder=None,
        pretraining_head=None,
        dropout=0.0,
    ):
        """
        AstroCLR Model.
        """
        super().__init__()
        self.catalog1_encoder = catalog1_encoder if catalog1_encoder else MLP(catalog1_input_dim, emb_dim, num_hidden, dropout)
        self.catalog2_encoder = catalog2_encoder if catalog2_encoder else MLP(catalog2_input_dim, emb_dim, num_hidden, dropout)
        self.pretraining_head = pretraining_head if pretraining_head else MLP(emb_dim, emb_dim, head_depth, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, catalog1_input, catalog2_input):
        emb_catalog1 = self.catalog1_encoder(catalog1_input)
        emb_catalog2 = self.catalog2_encoder(catalog2_input)

        embeddings_catalog1 = self.pretraining_head(emb_catalog1)
        embeddings_catalog2 = self.pretraining_head(emb_catalog2)
        return embeddings_catalog1, embeddings_catalog2 

    def training_step(self, batch, batch_idx):
        xray_batch, optical_batch = batch
        embeddings_catalog1, embeddings_catalog2 = self(xray_batch, optical_batch)
        clip_loss = CLIPLoss()
        loss = clip_loss(embeddings_catalog1, embeddings_catalog2)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        xray_batch, optical_batch = batch
        embeddings_catalog1, embeddings_catalog2 = self(xray_batch, optical_batch)
        clip_loss = CLIPLoss()
        val_loss = clip_loss(embeddings_catalog1, embeddings_catalog2)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
    
    def get_embeddings(self, catalog1_input, catalog2_input):
        with torch.no_grad():
            emb_catalog1 = self.catalog1_encoder(catalog1_input)
            emb_catalog2 = self.catalog2_encoder(catalog2_input)
        return emb_catalog1, emb_catalog2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer
