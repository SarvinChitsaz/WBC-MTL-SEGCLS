import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()

        _, cls_output = self.model(input_tensor)
        score = cls_output[:, target_class]

        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam[0]

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def remove_hooks(self):
        if self.forward_handle:
            self.forward_handle.remove()
        if self.backward_handle:
            self.backward_handle.remove()


def show_gradcam(model, dataset, idx_to_label, num_samples=4, alpha=0.3, device="cuda"):
    model.eval()

    indices = random.sample(range(len(dataset)), num_samples)
    images = torch.stack([dataset[i]["image"] for i in indices]).to(device)

    cam = GradCAM(model=model, target_layer=model.center)

    plt.figure(figsize=(8, 3 * num_samples), dpi=80)

    for i in range(num_samples):
        img_tensor = images[i].unsqueeze(0)

        img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        with torch.no_grad():
            _, cls_pred = model(img_tensor)
            pred_class = cls_pred.argmax(dim=1).item()
            pred_name = idx_to_label[pred_class]

        cam_map = cam.generate(img_tensor, target_class=pred_class)
        cam_map = cv2.resize(cam_map, (img.shape[1], img.shape[0]))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0

        overlay = (1 - alpha) * img + alpha * heatmap
        overlay = np.clip(overlay, 0, 1)

        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(img)
        plt.title(f"Original\nClass: {pred_name}")
        plt.axis("off")

        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    cam.remove_hooks()
