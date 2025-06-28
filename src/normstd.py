import torch

def get_mean_std(loader):
    """Calculate mean and standard deviation for a dataset"""
    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (number of images in the batch)
        images = images.view(batch_samples, images.size(1), -1)  # reshape to (batch_size, channels, height*width)
        mean += images.mean(2).sum(0)  # sum over height*width
        std += images.std(2).sum(0)  # sum over height*width
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std

