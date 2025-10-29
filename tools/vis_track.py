import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image
import numpy as np

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]


def read_gt_file(file_path):
    columns = ["frame_id", "target_id", "x", "y", "w", "h", "conf1", "cls", "conf3"]
    data = pd.read_csv(file_path, header=None, names=columns)
    return data

def save_trajectories(data, background_image_path, save_path, dpi=500):

    background = Image.open(background_image_path)

    fig, ax = plt.subplots()
    ax.imshow(background)

    target_ids = data["target_id"].unique()

    colors = plt.cm.tab10(np.linspace(0, 1, 15))
    for i, target_id in enumerate(target_ids):
        target_data = data[data["target_id"] == target_id]
        x = target_data["x"] + target_data["w"] / 2
        y = target_data["y"] + target_data["h"] / 2
        ax.scatter(x, y, s=1, label=f"ID: {int(target_id)}", color = 'blue')

    ax.axis("off")

    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":
    gt_file_path = '....../dataset/ICPR/val_data/001/gt.txt'
    background_image_path = '....../dataset/ICPR/val_data/001/img1/000001.jpg'

    trajectory_save_path = '......'

    data = read_gt_file(gt_file_path)

    save_trajectories(data=data, background_image_path=background_image_path, save_path=trajectory_save_path)
