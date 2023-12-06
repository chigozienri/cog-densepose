# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import shutil
import tempfile

import cv2
from cog import BasePredictor, Input, Path

import cv2
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer as Visualizer,
)
from densepose.vis.extractor import DensePoseResultExtractor
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

def densepose(im, predictor):
    width, height = im.shape[1], im.shape[0]

    with torch.no_grad():
        outputs = predictor(im)["instances"]

    results = DensePoseResultExtractor()(outputs)
    # MagicAnimate uses the Viridis colormap for their training data
    cmap = cv2.COLORMAP_VIRIDIS
    # Visualizer outputs black for background, but we want the 0 value of
    # the colormap, so we initialize the array with that value
    arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
    out = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)
    return out

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(
            "/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        )
        cfg.MODEL.WEIGHTS = "/model_final_162be9.pkl"
        self.predictor = DefaultPredictor(cfg)

    def predict(
        self,
        input: Path = Input(description="Input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        tempdir = tempfile.mkdtemp()
        in_path = os.path.join(tempdir, "image.png")
        out_path = os.path.join(tempdir, "image_dp_segm.png")
        shutil.copy(input, in_path)
        im = cv2.imread(in_path)
        out = densepose(im, self.predictor)
        cv2.imwrite(out_path, out)
        return Path(out_path)
