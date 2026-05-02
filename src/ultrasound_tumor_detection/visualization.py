"""Prediction visualization helpers."""

import matplotlib.pyplot as plt
import torch


def visualize_predictions(imgs, masks, preds, threshold=0.65):
    for i in range(len(imgs)):
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(imgs[i].squeeze(), cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(masks[i].squeeze(), cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow((torch.sigmoid(preds[i].squeeze()) > threshold), cmap="gray")

        plt.show()
