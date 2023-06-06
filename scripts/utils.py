import matplotlib.pyplot as plt
import cv2
from typing import List
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.engine.model import YOLO


def plot_results(results: List[Results], model: YOLO) -> None:
    """Takes results and the model and plots the results.

    Args:
        results (List[Results]): List of results.
        model (YOLO): Model which results will be plotted.
    """
    for result in results:
        img = result.orig_img

        res = model(img)
        res_plotted = res[0].plot()
        plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        plt.axis(False)
        plt.show()
