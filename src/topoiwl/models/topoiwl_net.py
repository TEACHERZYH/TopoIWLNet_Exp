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
    ) -> None:
        super().__init__()
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
        else:
            raise ValueError(f"Unsupported encoder_name: {encoder_name}")
        self.lateral4 = FusionBlock(enc_channels[3], width)
        self.lateral3 = FusionBlock(enc_channels[2] + width, width)
        self.lateral2 = FusionBlock(enc_channels[1] + width, width)
        self.lateral1 = FusionBlock(enc_channels[0] + width, width)

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

        mask = F.interpolate(self.mask_head(p1), size=input_size, mode="bilinear", align_corners=False)
        boundary = F.interpolate(self.boundary_head(p1), size=input_size, mode="bilinear", align_corners=False)
        distance = F.interpolate(self.distance_head(p1), size=input_size, mode="bilinear", align_corners=False)
        return {"mask": mask, "boundary": boundary, "distance": distance}


def build_model(model_cfg: dict[str, object]) -> TopoIWLNet:
    """Build a TopoIWL-Net variant from the config dictionary."""

    return TopoIWLNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        width=int(model_cfg.get("width", 64)),
        encoder_name=str(model_cfg.get("encoder", "lightweight")),
        pretrained=bool(model_cfg.get("pretrained", False)),
        pretrained_weights=model_cfg.get("pretrained_weights") or None,
        freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
    )
