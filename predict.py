# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import shutil
import subprocess
import tempfile

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        tempdir = tempfile.mkdtemp()
        in_path = os.path.join(tempdir, "image.png")
        out_path = os.path.join(tempdir, "image_dp_segm.png")
        shutil.copy(image, in_path)
        tempdir_contents = set(os.listdir(tempdir))
        subprocess.check_output(
            f"python /detectron2/projects/DensePose/apply_net.py show /detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml /model_final_162be9.pkl {in_path} dp_segm --output {out_path}",
            shell=True,
        )
        # Weird workaround because the output file is not named correctly, it has a .0001 inserted before the extension
        filename = list(set(os.listdir(tempdir)).difference(tempdir_contents))[0]
        out_path = os.path.join(tempdir, filename)
        return Path(out_path)
