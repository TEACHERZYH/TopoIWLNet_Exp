"""TopoIWL-Net model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1) -> None:
        pad = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int = 4) -> None:
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            ConvBNAct(in_ch, hidden, kernel=1),
            ConvBNAct(hidden, hidden, kernel=3, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_res:
            return x + y
        return y


class LightweightEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, channels: tuple[int, int, int, int] = (24, 40, 80, 160)) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = ConvBNAct(in_channels, c1, stride=2)
        self.stage1 = nn.Sequential(InvertedResidual(c1, c1, 1), InvertedResidual(c1, c1, 1))
        self.stage2 = nn.Sequential(InvertedResidual(c1, c2, 2), InvertedResidual(c2, c2, 1))
        self.stage3 = nn.Sequential(InvertedResidual(c2, c3, 2), InvertedResidual(c3, c3, 1))
        self.stage4 = nn.Sequential(InvertedResidual(c3, c4, 2), InvertedResidual(c4, c4, 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class TorchvisionMobileNetV3Encoder(nn.Module):
    """MobileNetV3 encoder using torchvision ImageNet weights."""

    def __init__(
        self,
        variant: str = "large",
        in_channels: int = 3,
        pretrained: bool = False,
        weights: str | None = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import (
                MobileNet_V3_Large_Weights,
                MobileNet_V3_Small_Weights,
                mobilenet_v3_large,
                mobilenet_v3_small,
            )
        except ImportError as exc:
            raise ImportError("torchvision is required for MobileNetV3 pretrained encoders") from exc

        variant = variant.lower()
        if variant == "large":
            weight_enum = self._resolve_weights(MobileNet_V3_Large_Weights, pretrained, weights)
            backbone = mobilenet_v3_large(weights=weight_enum)
            self.return_indices = (1, 3, 6, 12)
            self.out_channels = (16, 24, 40, 112)
        elif variant == "small":
            weight_enum = self._resolve_weights(MobileNet_V3_Small_Weights, pretrained, weights)
            backbone = mobilenet_v3_small(weights=weight_enum)
            self.return_indices = (0, 1, 3, 8)
            self.out_channels = (16, 16, 24, 48)
        else:
            raise ValueError(f"Unsupported MobileNetV3 variant: {variant}")

        self.features = backbone.features
        if in_channels != 3:
            self._adapt_first_conv(in_channels)
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def _resolve_weights(self, weight_cls: object, pretrained: bool, weights: str | None) -> object | None:
        if not pretrained:
            return None
        if weights in (None, "", "DEFAULT", "default"):
            return weight_cls.DEFAULT
        return weight_cls[weights]

    def _adapt_first_conv(self, in_channels: int) -> None:
        first_conv = self.features[0][0]
        if not isinstance(first_conv, nn.Conv2d):
            raise TypeError("Unexpected MobileNetV3 first layer; cannot adapt input channels")
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
            padding_mode=first_conv.padding_mode,
        )
        with torch.no_grad():
            old_weight = first_conv.weight
            if in_channels == 1:
                new_weight = old_weight.mean(dim=1, keepdim=True)
            else:
                base = old_weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                new_weight = base * (3.0 / float(in_channels))
            new_conv.weight.copy_(new_weight)
            if first_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)
        self.features[0][0] = new_conv

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.return_indices:
                outputs.append(x)
            if len(outputs) == 4:
                break
        return outputs


class TimmFeatureEncoder(nn.Module):
    """Generic timm feature encoder for stronger SOTA-oriented variants."""

    def __init__(
        self,
        model_name: str,
        in_channels: int = 3,
        pretrained: bool = False,
        freeze: bool = False,
        out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for transformer/SOTA encoders; install it with `pip install timm`.") from exc

        self.features = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels,
        )
        self.out_channels = tuple(int(ch) for ch in self.features.feature_info.channels())
        if len(self.out_channels) != 4:
            raise ValueError(f"Expected four feature stages from {model_name}, got {self.out_channels}")
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.features(x))


class FusionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_ch, out_ch, kernel=1)
        self.refine = nn.Sequential(
            ConvBNAct(out_ch, out_ch, kernel=3, groups=out_ch),
            ConvBNAct(out_ch, out_ch, kernel=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.proj(x))


class TopoIWLNet(nn.Module):
    """Lightweight network for mask, boundary, and distance prediction."""

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 64,
        encoder_name: str = "lightweight",
        pretrained: bool = False,
        pretrained_weights: str | None = None,
        freeze_encoder: bool = False,
        highres_refine: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        encoder_name = encoder_name.lower()
        if encoder_name == "lightweight":
            enc_channels = (24, 40, 80, 160)
            self.encoder = LightweightEncoder(in_channels, enc_channels)
        elif encoder_name == "mobilenet_v3_large":
            self.encoder = TorchvisionMobileNetV3Encoder(
                variant="large",
                in_channels=in_channels,
                pretrained=pretrained,
                weights=pretrained_weights,
                freeze=freeze_encoder,
            )
            enc_channels = self.encoder.out_channels
        elif encoder_name == "mobilenet_v3_small":
            self.encoder = TorchvisionMobileNetV3Encoder(
                variant="small",
                in_channels=in_channels,
                pretrained=pretrained,
                weights=pretrained_weights,
                freeze=freeze_encoder,
            )
            enc_channels = self.encoder.out_channels
        elif encoder_name.startswith("timm:"):
            self.encoder = TimmFeatureEncoder(
                model_name=encoder_name.split(":", 1)[1],
                in_channels=in_channels,
                pretrained=pretrained,
                freeze=freeze_encoder,
            )
            enc_channels = self.encoder.out_channels
        elif encoder_name.startswith(("mit_", "swin_", "convnext_", "coat_", "pvt_")):
            self.encoder = TimmFeatureEncoder(
                model_name=encoder_name,
                in_channels=in_channels,
                pretrained=pretrained,
                freeze=freeze_encoder,
            )
            enc_channels = self.encoder.out_channels
        else:
            raise ValueError(f"Unsupported encoder_name: {encoder_name}")
        self.lateral4 = FusionBlock(enc_channels[3], width)
        self.lateral3 = FusionBlock(enc_channels[2] + width, width)
        self.lateral2 = FusionBlock(enc_channels[1] + width, width)
        self.lateral1 = FusionBlock(enc_channels[0] + width, width)
        self.highres_refine = highres_refine
        if highres_refine:
            stem_width = max(width // 2, 32)
            self.image_stem = nn.Sequential(
                ConvBNAct(in_channels, stem_width, kernel=3, stride=2),
                ConvBNAct(stem_width, width, kernel=3),
            )
            self.highres_fusion = FusionBlock(width * 2, width)

        self.mask_head = self._make_head(width, 1)
        self.boundary_head = self._make_head(width, 1)
        self.distance_head = self._make_head(width, 1)

    def _make_head(self, width: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            ConvBNAct(width, width, kernel=3),
            nn.Conv2d(width, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)
        p4 = self.lateral4(f4)
        p3 = self.lateral3(torch.cat([f3, F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        p2 = self.lateral2(torch.cat([f2, F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        p1 = self.lateral1(torch.cat([f1, F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        if self.highres_refine:
            hr = self.image_stem(x)
            p1 = F.interpolate(p1, size=hr.shape[-2:], mode="bilinear", align_corners=False)
            p1 = self.highres_fusion(torch.cat([p1, hr], dim=1))

        mask = F.interpolate(self.mask_head(p1), size=input_size, mode="bilinear", align_corners=False)
        boundary = F.interpolate(self.boundary_head(p1), size=input_size, mode="bilinear", align_corners=False)
        distance = F.interpolate(self.distance_head(p1), size=input_size, mode="bilinear", align_corners=False)
        return {"mask": mask, "boundary": boundary, "distance": distance}


def as_waterline_output(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Wrap a mask-only baseline output in the project prediction contract."""

    return {"mask": mask, "boundary": mask, "distance": torch.zeros_like(mask)}


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            ConvBNAct(in_ch, out_ch, kernel=3),
            ConvBNAct(out_ch, out_ch, kernel=3),
        )


class UNetBaseline(nn.Module):
    """Compact U-Net baseline for water mask segmentation."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return as_waterline_output(self.head(d1))


class MobileNetV3UNetBaseline(nn.Module):
    """MobileNetV3 encoder-decoder baseline with only mask supervision."""

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 64,
        variant: str = "large",
        pretrained: bool = False,
        pretrained_weights: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = TorchvisionMobileNetV3Encoder(
            variant=variant,
            in_channels=in_channels,
            pretrained=pretrained,
            weights=pretrained_weights,
            freeze=freeze_encoder,
        )
        enc_channels = self.encoder.out_channels
        self.lateral4 = FusionBlock(enc_channels[3], width)
        self.lateral3 = FusionBlock(enc_channels[2] + width, width)
        self.lateral2 = FusionBlock(enc_channels[1] + width, width)
        self.lateral1 = FusionBlock(enc_channels[0] + width, width)
        self.head = nn.Sequential(ConvBNAct(width, width, kernel=3), nn.Conv2d(width, 1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)
        p4 = self.lateral4(f4)
        p3 = self.lateral3(torch.cat([f3, F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        p2 = self.lateral2(torch.cat([f2, F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        p1 = self.lateral1(torch.cat([f1, F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        mask = F.interpolate(self.head(p1), size=input_size, mode="bilinear", align_corners=False)
        return as_waterline_output(mask)


class DeepLabV3MobileNetBaseline(nn.Module):
    """Torchvision DeepLabV3-MobileNetV3 baseline with a single mask head."""

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
        pretrained_weights: str | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("DeepLabV3MobileNetBaseline currently supports 3-channel inputs only")
        try:
            from torchvision.models import MobileNet_V3_Large_Weights
            from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        except ImportError as exc:
            raise ImportError("torchvision is required for DeepLabV3-MobileNetV3") from exc
        weights_backbone = None
        if pretrained:
            if pretrained_weights in (None, "", "DEFAULT", "default"):
                weights_backbone = MobileNet_V3_Large_Weights.DEFAULT
            else:
                weights_backbone = MobileNet_V3_Large_Weights[pretrained_weights]
        self.model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=weights_backbone, num_classes=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return as_waterline_output(self.model(x)["out"])


class DeepLabV3ResNet50Baseline(nn.Module):
    """Torchvision DeepLabV3-ResNet50 baseline with a single mask head."""

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
        pretrained_weights: str | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("DeepLabV3ResNet50Baseline currently supports 3-channel inputs only")
        try:
            from torchvision.models import ResNet50_Weights
            from torchvision.models.segmentation import deeplabv3_resnet50
        except ImportError as exc:
            raise ImportError("torchvision is required for DeepLabV3-ResNet50") from exc
        weights_backbone = None
        if pretrained:
            if pretrained_weights in (None, "", "DEFAULT", "default"):
                weights_backbone = ResNet50_Weights.DEFAULT
            else:
                weights_backbone = ResNet50_Weights[pretrained_weights]
        self.model = deeplabv3_resnet50(weights=None, weights_backbone=weights_backbone, num_classes=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return as_waterline_output(self.model(x)["out"])


def build_model(model_cfg: dict[str, object]) -> nn.Module:
    """Build a model variant from the config dictionary."""

    architecture = str(model_cfg.get("architecture", "topoiwl")).lower()
    if architecture in {"unet", "u-net"}:
        return UNetBaseline(
            in_channels=int(model_cfg.get("in_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", model_cfg.get("width", 32))),
        )
    if architecture in {"mobilenetv3_unet", "mobilenet_v3_unet", "mbv3_unet"}:
        encoder_name = str(model_cfg.get("encoder", "mobilenet_v3_large")).lower()
        variant = "small" if "small" in encoder_name else "large"
        return MobileNetV3UNetBaseline(
            in_channels=int(model_cfg.get("in_channels", 3)),
            width=int(model_cfg.get("width", 64)),
            variant=variant,
            pretrained=bool(model_cfg.get("pretrained", False)),
            pretrained_weights=model_cfg.get("pretrained_weights") or None,
            freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
        )
    if architecture in {"deeplabv3_mobilenet", "deeplabv3_mobilenetv3", "deeplabv3"}:
        return DeepLabV3MobileNetBaseline(
            in_channels=int(model_cfg.get("in_channels", 3)),
            pretrained=bool(model_cfg.get("pretrained", False)),
            pretrained_weights=model_cfg.get("pretrained_weights") or None,
        )
    if architecture in {"deeplabv3_resnet50", "deeplabv3_resnet"}:
        return DeepLabV3ResNet50Baseline(
            in_channels=int(model_cfg.get("in_channels", 3)),
            pretrained=bool(model_cfg.get("pretrained", False)),
            pretrained_weights=model_cfg.get("pretrained_weights") or None,
        )

    return TopoIWLNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        width=int(model_cfg.get("width", 64)),
        encoder_name=str(model_cfg.get("encoder", "lightweight")),
        pretrained=bool(model_cfg.get("pretrained", False)),
        pretrained_weights=model_cfg.get("pretrained_weights") or None,
        freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
        highres_refine=bool(model_cfg.get("highres_refine", False)),
    )
