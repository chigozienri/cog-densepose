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
    DensePoseResultsCustomContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsMplContourVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.extractor import DensePoseResultExtractor
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def densepose(im, predictor, visualizer, cmap=cv2.COLORMAP_VIRIDIS, base_image=None):
    width, height = im.shape[1], im.shape[0]

    with torch.no_grad():
        outputs = predictor(im)["instances"]

    results = DensePoseResultExtractor()(outputs)
    if base_image is not None:
        arr = np.array(base_image)
    else:
        # Visualizer outputs black for background, but we want the 0 value of
        # the colormap, so we initialize the array with that value
        arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
    out = visualizer(
        alpha=(0.5 if base_image is not None else 1.0), cmap=cmap
    ).visualize(arr, results)
    return out


model_original_urls = [
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x_legacy/164832157/model_final_d366fa.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x_legacy/164832182/model_final_10af0e.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_s1x/165712097/model_final_0ed407.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC1_s1x/173862049/model_final_289019.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC2_s1x/173861455/model_final_3abe14.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_WC1_s1x/173067973/model_final_b1e525.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_WC2_s1x/173859335/model_final_60fed4.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_WC1_s1x/171402969/model_final_9e47f0.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_WC2_s1x/173860702/model_final_5ea023.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC1_s1x/173858525/model_final_f359f3.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2_s1x/173294801/model_final_6e1ed1.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC1M_s1x/217144516/model_final_48a9d9.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC2M_s1x/216245640/model_final_d79ada.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_WC1M_s1x/216245703/model_final_61971e.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_WC2M_s1x/216245758/model_final_7bfb43.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_WC1M_s1x/216453687/model_final_0a7287.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_WC2M_s1x/216245682/model_final_e354d9.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC1M_s1x/216245771/model_final_0ebeb3.pkl",
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC2M_s1x/216245790/model_final_de6e7a.pkl",
    "https://dl.fbaipublicfiles.com/densepose/evolution/densepose_R_50_FPN_DL_WC1M_3x_Atop10P_CA/217578784/model_final_9fe1cc.pkl",
    "https://dl.fbaipublicfiles.com/densepose/evolution/densepose_R_50_FPN_DL_WC1M_3x_Atop10P_CA_B_uniform/256453729/model_final_241ff5.pkl",
    "https://dl.fbaipublicfiles.com/densepose/evolution/densepose_R_50_FPN_DL_WC1M_3x_Atop10P_CA_B_uv/256452095/model_final_d689e2.pkl",
    "https://dl.fbaipublicfiles.com/densepose/evolution/densepose_R_50_FPN_DL_WC1M_3x_Atop10P_CA_B_finesegm/256452819/model_final_cb4ac6.pkl",
    "https://dl.fbaipublicfiles.com/densepose/evolution/densepose_R_50_FPN_DL_WC1M_3x_Atop10P_CA_B_coarsesegm/256455697/model_final_a6a4bf.pkl",
]


def original_urls_to_modelmap(original_urls):
    modelmap = {}
    for url in original_urls:
        model = url.split("/")[-3]
        shortname = model.replace("densepose_rcnn_", "").replace("densepose_", "")
        modelmap[shortname] = {}
        modelmap[shortname]["name"] = model
        modelmap[shortname]["replicate_weights_url"] = "/".join(
            url.replace(
                "https://dl.fbaipublicfiles.com/",
                "https://weights.replicate.delivery/default/facebookresearch/",
            ).split("/")[:-1]
        )
        modelmap[shortname]["model_filename"] = url.split("/")[-1]
        modelmap[shortname]["config_filename"] = (
            ("evolution/" if "evolution" in url else "") + url.split("/")[-3] + ".yaml"
        )
    return modelmap


MODELMAP = original_urls_to_modelmap(model_original_urls)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.predictor = None
        self.model = None

    def predict(
        self,
        input: Path = Input(description="Input image or video"),
        model: str = Input(
            description="DensePose model. Read about the available models: https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_IUV.md",
            choices=list(MODELMAP.keys()),
            default="R_50_FPN_s1x",
        ),
        overlay: bool = Input(
            description="Overlay the segmentation on the input image", default=False
        ),
        visualizer: str = Input(
            description="Visualizer to use",
            choices=["MplContour", "CustomContour", "FineSegmentation", "U", "V"],
            default="FineSegmentation",
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if model != self.model:
            os.makedirs(model, exist_ok=True)
            subprocess.run(
                [
                    "pget",
                    MODELMAP[model]["replicate_weights_url"]
                    + "/"
                    + MODELMAP[model]["model_filename"],
                    f"{model}/{MODELMAP[model]['model_filename']}",
                ]
            )
            cfg = get_cfg()
            add_densepose_config(cfg)
            cfg.merge_from_file(
                f"/detectron2/projects/DensePose/configs/{MODELMAP[model]['config_filename']}"
            )
            cfg.MODEL.WEIGHTS = f"{model}/{MODELMAP[model]['model_filename']}"
            self.predictor = DefaultPredictor(cfg)
            self.model = model

        visualizer_map = {
            "MplContour": DensePoseResultsMplContourVisualizer,
            "CustomContour": DensePoseResultsCustomContourVisualizer,
            "FineSegmentation": DensePoseResultsFineSegmentationVisualizer,
            "U": DensePoseResultsUVisualizer,
            "V": DensePoseResultsVVisualizer,
        }
        tempdir = tempfile.mkdtemp()
        # Check if input is image or video using the file mimetype
        mimetype, _ = mimetypes.guess_type(str(input))
        if mimetype and mimetype.startswith("image/"):
            # We have an image
            in_path = os.path.join(tempdir, "image.png")
            out_path = os.path.join(tempdir, "image_dp_segm.png")
            shutil.copy(input, in_path)
            im = cv2.imread(in_path)
            out = densepose(
                im,
                self.predictor,
                visualizer_map[visualizer],
                base_image=(im if overlay else None),
            )
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
                    out = densepose(
                        cv2_image,
                        self.predictor,
                        visualizer_map[visualizer],
                        base_image=(cv2_image if overlay else None),
                    )
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
