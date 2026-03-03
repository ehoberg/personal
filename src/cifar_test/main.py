from tkinter import Image

from matplotlib import pyplot as plt
import numpy as np
from app import App
from utils import download_and_load_cifar10



def main():

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar10 = download_and_load_cifar10(url)
    
    print(f"\nLabel names: {[name.decode() for name in cifar10['label_names']]}")

    app = App(cifar10)
    app.show()


if __name__ == "__main__":
    main()