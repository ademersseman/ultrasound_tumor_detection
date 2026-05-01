import os
import kagglehub
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn


class BUSIDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.mask_paths = []
        skipped_orphans = 0
        skipped_corrupt = 0
        total_examples = 0

        for cls in ["benign", "malignant", "normal"]:
            folder = os.path.join(root_dir, cls)

            for file in os.listdir(folder):
                if "_mask" in file:
                    continue

                img_path = os.path.join(folder, file)
                mask_path = img_path.replace(".png", "_mask.png")

                if not os.path.exists(mask_path):
                    skipped_orphans +=1
                    continue

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if img is None or mask is None:
                    skipped_corrupt += 1
                    continue

                if img.shape != mask.shape:
                    skipped_corrupt += 1
                    continue
                
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                total_examples +=1 
        
        if len(self.image_paths) != total_examples or len(self.image_paths)!= len(self.mask_paths):
            raise ValueError(
                "BUSIDataset: Unexpected parsing error."
            )
        if len(self.image_paths)==0:
            raise ValueError(
                "BUSIDataset: No valid image-mask pairs found after filtering"
            )
        print(f'Orphans: {skipped_orphans} \n Corrupt: {skipped_corrupt}')
        print(f'Total: {total_examples}' )
                
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # should not happen after corruption handling in data load
        if img is None or mask is None: 
            raise RuntimeError(
                f"Failure to read index {idx}: "
            )
        
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(img, (256, 256))

        img = img / 255.0
        mask = mask / 255.0

        img = torch.tensor(img).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.enc1 = C(1, 64)
        self.enc2 = C(64, 128)
        self.enc3 = C(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            C(256, 512),
            nn.Dropout2d(0.3)
        )

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = C(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = C(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = C(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.shape}")

        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 input channel, got {x.shape[1]}")

        h, w = x.shape[2], x.shape[3]
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError("Height and width must be divisible by 8")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        if d3.shape[2:] != e3.shape[2:]:
            raise RuntimeError("Shape mismatch in skip connection (d3, e3)")
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            raise RuntimeError("Shape mismatch in skip connection (d2, e2)")
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            raise RuntimeError("Shape mismatch in skip connection (d1, e1)")
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


def bce_dice_loss(pred, target, smooth=1):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    if pred.numel() == 0:
        raise ValueError("Empty tensors passed to loss")

    bce = nn.BCEWithLogitsLoss()(pred, target)
    
    # Apply sigmoid here for dice since we need probabilities
    pred_prob = torch.sigmoid(pred)
    intersection = (pred_prob * target).sum()
    dice = 1 - (2 * intersection + smooth) / (pred_prob.sum() + target.sum() + smooth)
    
    return bce + dice


def dice_score(pred, target, smooth=1, threshold = 0.65):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    if pred.numel() == 0:
        raise ValueError("Empty tensors")

    pred = (torch.sigmoid(pred) > threshold).float()  # add sigmoid
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def main():
    path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
    data_dir = os.path.join(path, "Dataset_BUSI_with_GT")

    dataset = BUSIDataset(data_dir)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print("Dataset size:", len(dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    MODEL_PATH = "unet_model.pth"
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded saved model!")
        train_model = False
    else:
        print("No saved model found. Training from scratch...")
        train_model = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    if train_model:
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

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved!")

    model.eval()
    dice_total = 0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            dice_total += dice_score(preds, masks).item()

    print("Average Dice Score:", dice_total / len(test_loader))

    imgs, masks = next(iter(test_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        preds = model(imgs)

    imgs = imgs.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    for i in range(len(imgs)):
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(imgs[i].squeeze(), cmap='gray')


        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(masks[i].squeeze(), cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        # plt.imshow((preds[i].squeeze() > 0.65), cmap='gray')
        plt.imshow((torch.sigmoid(preds[i].squeeze()) > 0.65), cmap='gray')


        plt.show()

if __name__ == '__main__':
    main()
