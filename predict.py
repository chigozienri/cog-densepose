# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import mimetypes
import os
import shutil
import subprocess
import tempfile

import av
import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path
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
        input: Path = Input(description="Input image or video"),
    ) -> Path:
        """Run a single prediction on the model"""
        tempdir = tempfile.mkdtemp()
        # Check if input is image or video using the file mimetype
        mimetype, _ = mimetypes.guess_type(str(input))
        if mimetype and mimetype.startswith("image/"):
            # We have an image
            in_path = os.path.join(tempdir, "image.png")
            out_path = os.path.join(tempdir, "image_dp_segm.png")
            shutil.copy(input, in_path)
            im = cv2.imread(in_path)
            out = densepose(im, self.predictor)
            cv2.imwrite(out_path, out)
            return Path(out_path)
        elif mimetype and mimetype.startswith("video/"):
            # We have a video
            in_path = os.path.join(tempdir, "input_video.mp4")
            out_path = os.path.join(tempdir, "video_dp_segm.mp4")
            shutil.copy(input, in_path)
            container = av.open(in_path)
            stream = container.streams.video[0]
            with av.open(out_path, mode="w") as target_container:
                stream_out = target_container.add_stream("mpeg4", rate=25)
                stream_out.width = stream.width
                stream_out.height = stream.height
                for frame in container.decode(stream):
                    pil_image = frame.to_image()
                    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    out = densepose(cv2_image, self.predictor)
                    out_frame = av.VideoFrame.from_ndarray(out, format="bgr24")
                    for packet in stream_out.encode(out_frame):
                        target_container.mux(packet)
            # Re-encode the video to a widely compatible format using ffmpeg
            reencoded_out_path = out_path + ".reencoded.mp4"
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-i",
                    str(out_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "slow",
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    str(reencoded_out_path),
                ]
            )
            shutil.move(reencoded_out_path, out_path)
            return Path(out_path)
        else:
            raise ValueError("Input must be an image or video")
