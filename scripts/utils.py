import cv2
import os
import wandb
import matplotlib.pyplot as plt
from typing import List
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.engine.model import YOLO
from math import ceil, sqrt


def plot_results(model: YOLO, image: str) -> None:
    """Takes results and the model and plots the results.

    Args:
        image (str): Path to image.
        model (YOLO): Model which results will be plotted.
    """
    results = model(image)
    for result in results:
        img = result.orig_img

        res = model(img)
        res_plotted = res[0].plot()
        plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        plt.axis(False)
        plt.show()


def plot_results_one_fig(model: YOLO, dir: str) -> None:
    """Plots detected bounding boxes on one figure for all images from the given directory.

    Args:
        model (YOLO): Model used for inference.
        dir (str): Directory with images.
    """
    # Get list of image names to plot
    images = os.listdir(dir)

    # Calculate the shape of plotting figure
    n = ceil(sqrt(len(images)))
    fig, ax = plt.subplots(nrows=n, ncols=n, figsize=(10, 10))

    # Plotting images with bounding boxes
    for i, img in enumerate(images):
        img_path = dir + rf"/{img}"  # Get path to the image
        res = model(img_path)  # Get the models predictions
        res_plotted = res[0].plot()

        ax[i // n, i % n].imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        ax[i // n, i % n].axis("off")

    # Turn off the axis for remaining subplots
    for j in range(i + 1, n * n):
        ax[j // n, j % n].axis("off")

    plt.show()


def load_model_run(run_id: int) -> YOLO:
    """Takes a run id and returns the best model version trained in that run.

    Args:
        run_id (int): id number of the run, used to get the path to the model.

    Returns:
        YOLO: Returns a loaded ultralytics YOLOv8 model.
    """
    path_to_trained_model = rf"runs\detect\train{run_id}\weights\best.pt"
    model = YOLO(path_to_trained_model)
    return model


def load_model_from_wandb(
    project: str,
    team: str = "czarna_magia",
    verbose: bool = False,
    model_name: str = "yolo:v0",
) -> YOLO:
    """Loads YOLO model from wandb.ai using artifacts.

    Args:
        project (str): Name of your project.
        team (str, optional): Name of team. Defaults to "czarna_magia".
        verbose (bool, optional): Prints information that the model was loaded. Defaults to False.

    Returns:
        YOLO: Ultralytics yolo model.
    """
    # Initialize wandb run
    run = wandb.init(project=project)

    # Load artifact
    model_artifact = run.use_artifact(f"{team}/{project}/{model_name}", type="model")
    artifact_dir = model_artifact.download()

    # Initalize YOLO model
    weights_path = f"{artifact_dir}/best.pt"  # Access best possible weights
    model = YOLO(weights_path)

    # Check if the model is loaded properly
    assert type(model) == YOLO

    if verbose:
        print(f"[INFO] {weights_path} loaded from wandb.ai")

    return model
