from torch import nn
from collections import OrderedDict
from typing import Any, Tuple
import torch
from segmentation_models_pytorch.base import modules as md
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from Pipeline.configs import *


class CutUnet(nn.Module):
    def __init__(self, num_classes: int, encoder: str, target_size: Tuple[int, int]) -> None:
        super().__init__()
        """
        Lungs DX pathology segmentation nn model class.
        :param num_classes: number of model output classes.
        :param encoder: segmentation model encoder name.
        :param target_size: the size of the input images expected by the model.
        """
        u = smp.Unet(
            encoder,
            in_channels=3,
            classes=num_classes,
            aux_params=dict(pooling='avg', classes=num_classes),
            encoder_weights=None
        )
        self.encoder = u.encoder
        self.center = u.decoder.center
        self.blocks = u.decoder.blocks[:2]
        in_channels = [512, 256, 128, 64, 32, 16]
        self.stage_indexes = [2, 3, 5]
        self.depth = 5
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels[2], num_classes, kernel_size=(1, 1), padding=(0, 0)),
            nn.Upsample(size=(target_size[0], target_size[1]), mode="bicubic", align_corners=True)
        )
        self.classification_head = u.classification_head

    def forward_decoder(self, *features) -> torch.Tensor:
        """
        Method for getting outputs from decoder.
        :param features: tensor containing image under study.
        :returns torch tensor outputs.
        """
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        head = features[0]
        skips = features[1:]
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for getting predictions for single image.
        :param x: tensor containing image under study.
        :returns list of torch tensors containing class and segmentation mask predictions.
        """
        features = self.encoder(x)
        decoder_output = self.forward_decoder(*features)
        masks = self.segmentation_head(decoder_output)
        masks = torch.sigmoid(masks)
        labels = torch.amax(masks, (2, 3))
        return labels, masks

    def load_weights(self, weights) -> None:
        """
        Method for loading trained weights into model.
        :param weights: trained weights.
        """
        new_weights = OrderedDict()
        for key, val in weights.items():
            if key.startswith("model"):
                new_weights[key[len("model."):]] = val

        new_weights.pop("criterion.cls_loss.bce.pos_weight", None)
        self.load_state_dict(new_weights, strict=True)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x



class SMPModel(nn.Module):
    def __init__(self, model_architecture: Any, encoder: str, num_classes: int, target_size: Tuple[int, int]) -> None:
        """
        Lungs segmentation nn model class.
        :parameter model_architecture: list of parameters indicating which nn weights to load.
        :param encoder: pathology segmentation model encoder.
        :param num_classes: the device on which the models working in the pipeline will be launched.
        """
        super().__init__()
        self.model = model_architecture(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            aux_params = dict(pooling='avg', classes=num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for making predictions for single image.
        :param x: tensor containing image under study.
        :returns torch tensors containing predicted mask and class predicted class labels.
        """
        masks, labels = self.model(x)
        masks = torch.sigmoid(masks)
        labels = torch.amax(masks, (2, 3))
        return labels, masks

    def load_weights(self, weights):
        """
        Method for weights loading.
        :param weights: state dictionary with weights.
        """
        new_weights = OrderedDict()
        for key, val in weights.items():
            if key.startswith("model"):
                new_weights[key[len("model."):]] = val

        new_weights.pop("criterion.cls_loss.bce.pos_weight", None)
        self.load_state_dict(new_weights, strict=False)


class DeeplabV3Plus(SMPModel):
    def __init__(self, encoder: str, num_classes: int, target_size: Tuple[int, int]):
        super().__init__(
            model_architecture=smp.DeepLabV3Plus, encoder=encoder, num_classes=num_classes, target_size=target_size
        )


class UnetPlusPlus(SMPModel):
    def __init__(self, encoder: str, num_classes: int, target_size: Tuple[int, int]):
        super().__init__(
            model_architecture=smp.UnetPlusPlus, encoder=encoder, num_classes=num_classes, target_size=target_size
        )


class MANet(SMPModel):
    def __init__(self, encoder: str, num_classes: int, target_size: Tuple[int, int]):
        super().__init__(
            model_architecture=smp.MAnet, encoder=encoder, num_classes=num_classes, target_size=target_size
        )


class Unet(SMPModel):
    def __init__(self, encoder: str, num_classes: int, target_size: Tuple[int, int]):
        super().__init__(
            model_architecture=smp.Unet, encoder=encoder, num_classes=num_classes, target_size=target_size
        )

class DeeplabV3(SMPModel):
    def __init__(self, encoder: str, num_classes: int, target_size: Tuple[int, int]):
        super().__init__(
            model_architecture=smp.DeepLabV3, encoder=encoder, num_classes=num_classes, target_size=target_size
        )


class PAN(SMPModel):
    def __init__(self, encoder: str, num_classes: int, target_size: Tuple[int, int]):
        super().__init__(
            model_architecture=smp.PAN, encoder=encoder, num_classes=num_classes, target_size=target_size
        )


class FPN(SMPModel):
    def __init__(self, encoder: str, num_classes: int, target_size: Tuple[int, int]):
        super().__init__(
            model_architecture=smp.FPN, encoder=encoder, num_classes=num_classes, target_size=target_size
        )


class PSPNet(SMPModel):
    def __init__(self, encoder: str, num_classes: int, target_size: Tuple[int, int]):
        super().__init__(
            model_architecture=smp.PSPNet, encoder=encoder, num_classes=num_classes, target_size=target_size
        )


class LinkNet(SMPModel):
    def __init__(self, encoder: str, num_classes: int, target_size: Tuple[int, int]):
        super().__init__(
            model_architecture=smp.Linknet, encoder=encoder, num_classes=num_classes, target_size=target_size
        )