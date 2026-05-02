"""Training, evaluation, and full-pipeline orchestration."""

import os
from importlib.resources import as_file, files
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ultrasound_tumor_detection.data import BUSIDataset, download_busi_dataset
from ultrasound_tumor_detection.losses import bce_dice_loss, dice_score
from ultrasound_tumor_detection.model import UNet
from ultrasound_tumor_detection.visualization import visualize_predictions


EPOCHS = 10
MODEL_PATH = "models/unet_model.pth"
BUNDLED_MODEL_PATH = "assets/unet_model.pth"


def bundled_model_resource():
    return files("ultrasound_tumor_detection").joinpath(BUNDLED_MODEL_PATH)


def load_checkpoint(model, device, model_path=MODEL_PATH, allow_bundled=True):
    """Load a user checkpoint when present, otherwise fall back to the bundled model."""
    local_model_path = Path(model_path)

    if local_model_path.exists():
        checkpoint = torch.load(local_model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded saved model from {local_model_path}!")
        return True

    if not allow_bundled:
        return False

    bundled_model = bundled_model_resource()
    if bundled_model.is_file():
        with as_file(bundled_model) as checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded bundled pretrained model from {BUNDLED_MODEL_PATH}!")
        return True

    return False


def train_model(model, train_loader, optimizer, device, epochs=EPOCHS):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = bce_dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")


def evaluate_model(model, test_loader, device):
    model.eval()
    dice_total = 0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            dice_total += dice_score(preds, masks).item()

    return dice_total / len(test_loader)


def build_dataloaders(data_dir, batch_size=8, train_fraction=0.8):
    dataset = BUSIDataset(data_dir)
    train_size = int(train_fraction * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, train_loader, test_loader


def run_pipeline(
    data_dir=None,
    model_path=MODEL_PATH,
    epochs=None,
    batch_size=8,
    show_predictions=True,
):
    if data_dir is None:
        data_dir = download_busi_dataset()

    dataset, train_loader, test_loader = build_dataloaders(data_dir, batch_size=batch_size)
    print("Dataset size:", len(dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    should_train = not load_checkpoint(
        model,
        device,
        model_path=model_path,
        allow_bundled=os.fspath(model_path) == MODEL_PATH,
    )

    if should_train:
        print("No saved or bundled model found. Training from scratch...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if should_train:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        train_model(model, train_loader, optimizer, device, epochs=EPOCHS if epochs is None else epochs)
        torch.save(model.state_dict(), model_path)
        print("Model saved!")

    print("Evaluating... (This could take a while!)")
    average_dice = evaluate_model(model, test_loader, device)
    print("Average Dice Score:", average_dice)

    if show_predictions:
        imgs, masks = next(iter(test_loader))
        imgs = imgs.to(device)

        with torch.no_grad():
            preds = model(imgs)

        visualize_predictions(imgs.cpu(), masks.cpu(), preds.cpu())

    return average_dice


def main():
    run_pipeline()
