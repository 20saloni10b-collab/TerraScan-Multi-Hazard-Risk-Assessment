"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
--------------------------------------------------------
Ye module batata hai ki model ne image ke KAUNSE hisse dekh ke apna
prediction diya — ek heatmap overlay ke through (red/yellow = zyada important
region, blue = kam important).

Kaam kaise karta hai (short version):
  1. Image branch ke last convolutional layer (ResNet ka layer4) ke
     activations aur gradients capture karte hain (forward + backward hook).
  2. Gradients ka spatial-average lekar har channel ka "importance weight"
     nikalte hain.
  3. Weighted-sum of activations -> ReLU -> normalize -> ek 2D heatmap.
  4. Heatmap ko original image size pe resize karke color-overlay karte hain.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.cm as cm


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, image_tensor, tabular_tensor, class_idx):
        """Returns a 2D numpy heatmap (values in [0,1]) for the given class."""
        self.model.zero_grad()
        output = self.model(image_tensor, tabular_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]           # (C, H, W)
        activations = self.activations[0]        # (C, H, W)
        weights = gradients.mean(dim=(1, 2))      # (C,) — global-average-pooled gradients

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def overlay_heatmap(pil_image, cam, alpha=0.45):
    """cam: 2D numpy array in [0,1]. Returns a PIL RGB image with heatmap overlay."""
    size = pil_image.size  # (W, H)
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(size, resample=Image.BICUBIC)
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0

    heatmap = cm.jet(cam_arr)[:, :, :3]           # (H, W, 3) in [0,1]
    heatmap = (heatmap * 255).astype(np.uint8)

    orig = np.array(pil_image.convert("RGB").resize(size)).astype(np.float32)
    blended = heatmap.astype(np.float32) * alpha + orig * (1 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)
