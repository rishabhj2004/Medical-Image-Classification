import torch
import cv2
import numpy as np

def generate_gradcam(model, img_tensor, meta_tensor):
    model.eval()
    gradients, activations = [], []

    def forward_hook(module, input, output): activations.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

    # RECURSIVE SEARCH: Finds the last Conv2d inside nested EfficientNet blocks
    target_layer = None
    for module in model.features.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise ValueError("No Conv2d layer found. Check model architecture.")

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    with torch.enable_grad():
        output = model(img_tensor, meta_tensor)
        pred_idx = output.argmax(dim=1)
        model.zero_grad()
        output[0, pred_idx].backward()

    fh.remove(); bh.remove()

    grad = gradients[0][0].detach().cpu().numpy()
    act = activations[0][0].detach().cpu().numpy()
    weights = np.mean(grad, axis=(1, 2))
    
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0) # ReLU
    cam = cv2.resize(cam, (224, 224))
    cam = cam / (cam.max() + 1e-8)
    return cam
