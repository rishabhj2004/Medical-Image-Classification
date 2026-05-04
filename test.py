from model import load_models, predict
from unet_model import UNet

mu, mm = load_models()
unet = UNet()

pred, probs, mask = predict("test.jpg", 45, "male", mu, mm, unet)

print("Prediction:", pred)
print("Probs:", probs)
print("Mask shape:", mask.shape)
