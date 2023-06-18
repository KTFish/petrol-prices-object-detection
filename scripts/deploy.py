import gradio as gr
import random
import pathlib
from typing import Tuple, Dict, List

title = "Petrol Prices Detection"
description = "Object Detection using YOLOv8 model."

train_images_paths = list(
    pathlib.Path(
        r"C:\Users\janek\notebooks\petrol-prices-object-detection\custom_data\roboflow\images\train"
    ).glob("*.jpg")
)
train_images_paths

example_list = [[str(path)] for path in random.sample(train_images_paths, k=3)]
example_list


demo = gr.Inference(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predicitons"),
        gr.Number(label="Prediction time"),
    ],
    examples=example_list,
    title=title,
    description=description,
)

demo.launch(share=True, debug=False)
