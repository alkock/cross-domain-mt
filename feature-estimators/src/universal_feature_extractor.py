import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np


class UniversalFeatureExtractor:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name.lower()
        print(f"Initializing Model: {self.model_name.upper()}")
        self.model = self.init_model(self.model_name)
        self.preprocess = self.preprocessing()
        self.model.to(self.device)
        self.model.eval()

    def init_model(self, model_name):
        if model_name == "dinov2":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50(weights=weights)
            return nn.Sequential(*list(backbone.children())[:-1])
        elif model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            return nn.Sequential(*list(backbone.children())[:-1])
        elif model_name == "vgg16":
            weights = models.VGG16_Weights.DEFAULT
            backbone = models.vgg16(weights=weights)
            return nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def preprocessing(self):
        # All images are resized to 224x224 and normalized using ImageNet statistics
        # to align with the preprocessing of the pretrained backbone networks.
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def get_features(self, image_input):
        """
        Returns:
            Feature embedding: shape [D]
        """
        if isinstance(image_input, (str, os.PathLike)):
            img = Image.open(image_input)
        else:
            img = image_input

        # Grayscale images are replicated across R,G,B for backbone compatibility
        img = img.convert("RGB")
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.model_name == "dinov2":
                out = self.model.forward_features(img_t)
                cls = out["x_norm_clstoken"][0]
                return cls.detach().cpu().numpy()
            else:
                emb = self.model(img_t)
                return emb.detach().cpu().flatten().numpy()
