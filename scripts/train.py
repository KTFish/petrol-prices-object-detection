import torch
import wandb
from ultralytics import YOLO


# def train(model: YOLO, device, epochs: int, patience: int = 5, pretrained: bool = True):
#     return model.train(
#         data="configv2.yaml",  # Path to data file
#         epochs=epochs,
#         device=device,
#         # project="petrol-prices-object-detection",
#         # name=f"yolo-{device}-epochs-{epochs}",
#         patience=patience,  # Epochs to wait for no observable improvement for early stopping of training
#         verbose=False,
#         pretrained=pretrained,
#         plots=False,  # Don't save plots during train/val
#         seed=42,
#     )
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        results = config.model.train(
            data="configv2.yaml",  # Path to data file
            epochs=config.epochs,
            device=config.device,
            # project="petrol-prices-object-detection",
            # name=f"yolo-{device}-epochs-{epochs}",
            patience=config.patience,  # Epochs to wait for no observable improvement for early stopping of training
            verbose=False,
            pretrained=config.pretrained,
            plots=False,  # Don't save plots during train/val
            seed=42,
        )
    print(type(results))
    return results
