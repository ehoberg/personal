import urllib.request
import tarfile
import os
import pickle
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def download_and_load_cifar10(url):
    """
    Download CIFAR-10 dataset from URL and load the images.
    
    Args:
        url: URL to the CIFAR-10 tar.gz file
    
    Returns:
        Dictionary with training and test images and labels
    """
    tar_path = url.split("/")[-1]
    
    if not os.path.exists(tar_path):
        print(f"Downloading CIFAR-10 dataset from {url}...")
        urllib.request.urlretrieve(url, tar_path)
    
    cifar10_path = ''
    with tarfile.open(tar_path, "r:gz") as tar:
        cifar10_path = tar.getnames()[0]  # Get the root directory name from the tar file
        if not os.path.exists(cifar10_path):
            tar.extractall(path=".")
    
    data = {"train_images": [], "train_labels": [], "test_images": [], "test_labels": []}
    
    for i in range(1, 6):
        batch_file = os.path.join(cifar10_path, f"data_batch_{i}")
        with open(batch_file, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            data["train_images"].append(batch[b"data"])
            data["train_labels"].extend(batch[b"labels"])

    # Concatenate and reshape the images to a more usefull format.  
    # Original data constist of a single row containing rgb data
    # that needs to be reshaped to reflect the actual size (32x32)
    data["train_images"] = np.concatenate(data["train_images"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data["train_labels"] = np.array(data["train_labels"])   

    
    # Load test batch
    test_file = os.path.join(cifar10_path, "test_batch")
    with open(test_file, "rb") as f:
        test_batch = pickle.load(f, encoding="bytes")
        data["test_images"] = test_batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        data["test_labels"] = np.array(test_batch[b"labels"])
    
    # Load metadata
    meta_file = os.path.join(cifar10_path, "batches.meta")
    with open(meta_file, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
        data["label_names"] = meta[b"label_names"]
    
   
    print(f"Training images shape: {data['train_images'].shape}")
    print(f"Test images shape: {data['test_images'].shape}")
    
    return data