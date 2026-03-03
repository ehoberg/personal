import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class App():
    def __init__(self, dataset):

        self.dataset = dataset
        self.label_names = [name.decode() for name in dataset.get("label_names", [])]
        self.images = dataset.get(
            "train_images", np.zeros((0, 32, 32, 3), dtype=np.uint8)
        )
        self.labels = dataset.get("train_labels", np.zeros((0,), dtype=int))
        self.index = 0

        self.root = tk.Tk()
        self.root.title("Viewer")
        self.root.geometry("1400x700")

        # Create main horizontal split with PanedWindow
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Split the screen into a lef tand right frame
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=1)

        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)

        # Canvas for image display (4x larger than original for better visibility)
        self.canvas = tk.Canvas(left_frame, width=128, height=128, bg="gray")
        self.canvas.pack(pady=10, padx=10)

        # navigation buttons
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=5)
        prev_btn = tk.Button(btn_frame, text="<< Prev", command=self.prev_image)
        prev_btn.pack(side=tk.LEFT, padx=5)
        next_btn = tk.Button(btn_frame, text="Next >>", command=self.next_image)
        next_btn.pack(side=tk.LEFT, padx=5)

        self.setup_combo_box(left_frame)
        self.setup_group_action(left_frame)
        self.setup_histogram(right_frame)

        # initially show first image
        self.show_image()

    def setup_combo_box(self, parent):
        # dropdown for label options
        dropdown_frame = ttk.Frame(parent)
        dropdown_frame.pack(pady=10, fill=tk.X, padx=10)
        dd_label = ttk.Label(dropdown_frame, text="Choose label:")
        dd_label.pack(side=tk.LEFT)

        # prepend All option
        combo_values = ["All"] + self.label_names
        self.combobox = ttk.Combobox(
            dropdown_frame, values=combo_values, state="readonly"
        )
        self.combobox.current(0)  # default to All
        self.combobox.bind("<<ComboboxSelected>>", self.combobox_changed)
        self.combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def setup_group_action(self, parent):
        action_frame = ttk.Frame(parent)
        action_frame.pack(pady=10, fill=tk.X, padx=10)

        label = ttk.Label(action_frame, text="Group actions", font=(None, 14, "bold"))
        label.pack(pady=(10, 0))

        sum_btn = tk.Button(action_frame, text="Mean", command=self.mean_action)
        sum_btn.pack(side=tk.LEFT, padx=5)

        sum_btn = tk.Button(action_frame, text="median", command=self.median_action)
        sum_btn.pack(side=tk.LEFT, padx=5)

    def setup_histogram(self, parent):
        # label above graphs
        label = ttk.Label(parent, text="RGB distribution", font=(None, 14, "bold"))
        label.pack(pady=(10, 0))

        self.histogram, self.axs = plt.subplots(3, sharex=True, sharey=True)

        self.axs[0].set_title("Red")
        self.axs[1].set_title("Green")
        self.axs[2].set_title("Blue")

        canvas = FigureCanvasTkAgg(self.histogram, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def combobox_changed(self, event):
        selected = self.combobox.get()
        if selected == "All":
            self.images = self.dataset.get(
                "train_images", np.zeros((0, 32, 32, 3), dtype=np.uint8)
            )
            self.labels = self.dataset.get("train_labels", np.zeros((0,), dtype=int))
        else:
            label_index = self.label_names.index(selected)
            mask = (
                self.dataset.get("train_labels", np.zeros((0,), dtype=int))
                == label_index
            )
            self.images = self.dataset.get(
                "train_images", np.zeros((0, 32, 32, 3), dtype=np.uint8)
            )[mask]
            self.labels = self.dataset.get("train_labels", np.zeros((0,), dtype=int))[
                mask
            ]

        self.index = 0
        self.show_image()

    def show_image(self):
        if len(self.images) == 0:
            return

        img_array = self.images[self.index]

        self.update_image(img_array)
        self.update_histogram(img_array)

        label_text = "Unknown"
        if self.labels is not None and self.index < len(self.labels):
            lbl = self.labels[self.index]
            if 0 <= lbl < len(self.label_names):
                label_text = self.label_names[lbl]
        self.root.title(
            f"CIFAR-10 Viewer - {label_text} ({self.index}/{len(self.images)}. filter={self.combobox.get()})"
        )

    def update_histogram(self, img_array):

        for ax in self.axs:
            ax.clear()

        self.axs[0].hist(img_array[:, :, 0].ravel(), color="red")
        self.axs[1].hist(img_array[:, :, 1].ravel(), color="green")
        self.axs[2].hist(img_array[:, :, 2].ravel(), color="blue")

        self.histogram.canvas.draw()

    def update_image(self, img_array):
        pil = Image.fromarray(img_array.astype("uint8"), "RGB")
        resized = pil.resize((128, 128), Image.NEAREST)
        self.draw_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.draw_img)

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

    def mean_action(self):

        mean_img = np.mean(self.images, axis=0).astype(np.uint8)

        self.update_histogram(mean_img)
        self.update_image(mean_img)

    def median_action(self):

        median_img = np.median(self.images, axis=0).astype(np.uint8)

        self.update_histogram(median_img)
        self.update_image(median_img)

    def show(self):
        # start Tkinter main loop
        self.root.mainloop()
