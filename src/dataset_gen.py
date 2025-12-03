"""
Generate synthetic 50x50 images containing exactly one white pixel (255).
Saves train/val/test datasets in .npz format.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(n_samples: int,
                     size: int = 50,
                     normalize: bool = True,
                     seed: int | None = None):
    """
    Generate random images with one bright pixel and return images + labels.
    """
    if seed is not None:
        np.random.seed(seed)

    images = np.zeros((n_samples, size, size), dtype=np.uint8)
    labels = np.zeros((n_samples, 2), dtype=np.float32)

    for i in range(n_samples):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        images[i, y, x] = 255

        if normalize:
            labels[i, 0] = x / (size - 1)
            labels[i, 1] = y / (size - 1)
        else:
            labels[i, 0] = float(x)
            labels[i, 1] = float(y)

    return images, labels


def save_dataset(path: str, images, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, images=images, labels=labels)


def visualize_sample(image, label, normalize=True):
    size = image.shape[0]
    if normalize:
        x = int(round(label[0] * (size - 1)))
        y = int(round(label[1] * (size - 1)))
    else:
        x, y = int(label[0]), int(label[1])

    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.scatter([x], [y], c="red", marker="x")
    plt.title(f"GT coordinate: ({x}, {y})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    TRAIN = 20000
    VAL = 4000
    TEST = 4000

    train_imgs, train_labels = generate_dataset(TRAIN, seed=42)
    val_imgs, val_labels = generate_dataset(VAL, seed=123)
    test_imgs, test_labels = generate_dataset(TEST, seed=999)

    save_dataset("data/train.npz", train_imgs, train_labels)
    save_dataset("data/val.npz", val_imgs, val_labels)
    save_dataset("data/test.npz", test_imgs, test_labels)

    print("Datasets generated and saved to data/")
    visualize_sample(train_imgs[0], train_labels[0], normalize=True)
