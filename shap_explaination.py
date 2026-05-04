import shap
import torch
import numpy as np

explainer = None  # cache

def generate_shap(mu, mm, img_tensor, mask, meta_tensor):
    global explainer

    mu.eval()
    mm.eval()

    masked_img = (img_tensor * mask).detach()
    meta_np = meta_tensor.detach().cpu().numpy().astype(np.float32)

    # ✅ Better background (realistic)
    background = np.vstack([
        meta_np,
        meta_np * 0.9,
        meta_np * 1.1,
        np.clip(meta_np + 0.1, 0, 1),
        np.clip(meta_np - 0.1, 0, 1)
    ]).astype(np.float32)

    # ✅ Wrapper
    def f(x):
        x_t = torch.tensor(x).float()

        batch_size = x_t.shape[0]

        # memory-safe expand
        img_batch = img_tensor.expand(batch_size, -1, -1, -1)
        masked_batch = masked_img.expand(batch_size, -1, -1, -1)

        with torch.no_grad():
            log_u = mu(img_batch, x_t)
            log_m = mm(masked_batch, x_t)
            logits = (log_u + log_m) / 2

        return logits.cpu().numpy()

    # ✅ Cache explainer
    if explainer is None:
        explainer = shap.KernelExplainer(f, background)

    shap_values = explainer.shap_values(
        meta_np,
        nsamples=50,   # faster
        l1_reg="num_features(5)"
    )

    preds = f(meta_np)
    pred_class = np.argmax(preds[0])

    # handle formats
    if isinstance(shap_values, list):
        vals = shap_values[pred_class][0]
    else:
        vals = shap_values[0, :, pred_class]

    return vals.tolist()
