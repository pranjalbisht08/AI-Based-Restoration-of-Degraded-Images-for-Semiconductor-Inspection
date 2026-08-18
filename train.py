import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import RestorationAutoencoder
from dataset import RestorationDataset


def main():
    # ---------- SETTINGS ----------
    EPOCHS = 80
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    MODEL_SAVE_PATH = "models/restoration_model.pth"

    # ---------- DEVICE ----------
    # Use GPU if available (much faster), otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- DATA ----------
    dataset = RestorationDataset(clean_dir="sample_images/clean")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training on {len(dataset)} images, {len(dataloader)} batches per epoch.")

    # ---------- MODEL, LOSS, OPTIMIZER ----------
    model = RestorationAutoencoder().to(device)
    criterion = nn.MSELoss()  # measures average pixel difference between output and clean image
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------- TRAINING LOOP ----------
    for epoch in range(1, EPOCHS + 1):
        model.train()  # tell PyTorch we're in training mode
        running_loss = 0.0

        for degraded_batch, clean_batch in dataloader:
            degraded_batch = degraded_batch.to(device)
            clean_batch = clean_batch.to(device)

            # 1. Forward pass: run degraded images through the model
            restored_batch = model(degraded_batch)

            # 2. Compute how wrong the output is vs the real clean image
            loss = criterion(restored_batch, clean_batch)

            # 3. Backward pass: compute gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch}/{EPOCHS}]  Loss: {avg_loss:.5f}")

    # ---------- SAVE MODEL ----------
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining complete. Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()