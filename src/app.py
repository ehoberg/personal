

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


class App():
    def __init__(self, dataset):
        # store dataset and prepare view state
        self.data = dataset
        self.label_names = [name.decode() for name in dataset.get("label_names", [])]
        self.images = dataset.get("train_images", np.zeros((0, 32, 32, 3), dtype=np.uint8))
        self.labels = dataset.get("train_labels", np.zeros((0,), dtype=int))
        self.index = 0

        # create main window but do not start loop until show() is called
        self.root = tk.Tk()
        self.root.title("CIFAR-10 Viewer")

        # canvas for image display
        # we'll scale the 32x32 image up for visibility
        self.canvas = tk.Canvas(self.root, width=32 * 4, height=32 * 4)
        self.canvas.pack()

        # navigation buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        prev_btn = tk.Button(btn_frame, text="<< Prev", command=self.prev_image)
        prev_btn.pack(side=tk.LEFT, padx=5)
        next_btn = tk.Button(btn_frame, text="Next >>", command=self.next_image)
        next_btn.pack(side=tk.LEFT, padx=5)

        # initially show first image
        self.show_image()

    def show_image(self):
        if len(self.images) == 0:
            return

        img_array = self.images[self.index]
        # convert to PIL image
        pil = Image.fromarray(img_array.astype("uint8"), "RGB")
        # resize for easier viewing
        resized = pil.resize((128, 128), Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        label_text = "Unknown"
        if self.labels is not None and self.index < len(self.labels):
            lbl = self.labels[self.index]
            if 0 <= lbl < len(self.label_names):
                label_text = self.label_names[lbl]
        self.root.title(f"CIFAR-10 Viewer - {label_text} ({self.index})")

    def prev_image(self):
        if len(self.images) == 0:
            return
        self.index = (self.index - 1) % len(self.images)
        self.show_image()

    def next_image(self):
        if len(self.images) == 0:
            return
        self.index = (self.index + 1) % len(self.images)
        self.show_image()

    def show(self):
        # start Tkinter main loop
        self.root.mainloop()


    