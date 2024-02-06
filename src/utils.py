import random

import numpy as np
import torch
from tqdm.notebook import tqdm

def train_epoch(model, criterion, train_loader, optimizer, device, epoch):
    model.train()
    epoch_loss = 0.0

    for x in train_loader:

        x = x.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Get combined embeddings
        combined_emb = model(x)

        # Compute loss - note that the criterion and the way you calculate loss may need to be adapted 
        # based on how your labels or targets are defined
        loss = criterion(combined_emb)
        loss.backward()

        # Update model weights
        optimizer.step()

        # log progress
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader.dataset)


def train_epoch_old(model, criterion, train_loader, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for x in train_loader:
        x = x.to(device)

        # get embeddings
        emb_anchor, emb_positive = model(x)

        # compute loss
        loss = criterion(emb_anchor, emb_positive)
        loss.backward()

        # update model weights
        optimizer.step()

        # reset gradients
        optimizer.zero_grad()

        # log progress
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader.dataset)


def dataset_embeddings(model, loader, device):
    xray_embeddings = []
    optical_embeddings = []

    # Process each batch
    for xray_batch, optical_batch in tqdm(loader):
        xray_batch = xray_batch.to(device)
        optical_batch = optical_batch.to(device)

        with torch.no_grad():  # Ensure no gradient computations
            emb_xray, emb_optical = model.get_embeddings(xray_batch, optical_batch)

        xray_embeddings.append(emb_xray.cpu())
        optical_embeddings.append(emb_optical.cpu())

    # Concatenate all collected embeddings
    xray_embeddings = torch.cat(xray_embeddings).numpy()
    optical_embeddings = torch.cat(optical_embeddings).numpy()

    return xray_embeddings, optical_embeddings



def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
