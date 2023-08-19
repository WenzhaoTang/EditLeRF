from typing import Optional

from torch import nn
from torchvision.transforms import transforms
import clip
import open_clip
import torch


class ClipEncoding(nn.Module):
    def __init__(self,
                 clip_model_type: str = 'ViT-B-32',
                 pretrained_type: str = 'laion2b_s34b_b79k',
                 device: Optional[str] = 'cuda'
                 ) -> None:
        super().__init__()
        self.device = device
        if '16' in clip_model_type:
            self.precision = 'fp16'
        else:
            self.precision = 'fp32'

        self.encoder, _, _ = open_clip.create_model_and_transforms(
            clip_model_type,  # e.g., ViT-B-16
            pretrained=pretrained_type,  # e.g., laion2b_s34b_b88k
            precision=self.precision,
        )
        self.encoder = self.encoder.to('cuda')
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.loss = torch.nn.CosineSimilarity()

    def encode_text(self, text: str) -> torch.Tensor:
        tokens = clip.tokenize([text]).to(self.device)
        text_features = self.encoder.encode_text(tokens).detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self.preprocess(image).to(self.device)
        image_features = self.encoder.encode_image(image)
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, rgb_outputs: torch.Tensor, text_features: torch.Tensor,
                rgb_src: Optional[torch.Tensor] = None, text_src: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        image_features = self.encode_image(rgb_outputs)
        if rgb_src is not None and text_src is not None:
            rgb_src = self.encode_image(rgb_src)
            edit_direction = (image_features - rgb_src)
            edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

            text_direction = (text_features - text_src)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)
            loss = 1 - self.loss(edit_direction, text_direction).mean()
        else:
            loss = 1 - self.loss(image_features, text_features).mean()

        return loss
