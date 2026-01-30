#!/usr/bin/env python
"""
Export trained model to ONNX format.
"""

import argparse
import logging
from pathlib import Path

import torch

from uterus_scope.models.unified import UterusScopeModel
from uterus_scope.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: str | None,
    output_path: str,
    input_size: int = 224,
    opset_version: int = 14,
):
    """Export model to ONNX format."""
    
    logger.info("Loading model...")
    
    if checkpoint_path and Path(checkpoint_path).exists():
        model = UterusScopeModel.from_pretrained(checkpoint_path, device='cpu')
    else:
        logger.warning("No checkpoint provided, using untrained model")
        model = UterusScopeModel(pretrained=True)
    
    model.eval()
    model.cpu()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, input_size, input_size)
    
    logger.info(f"Exporting to ONNX: {output_path}")
    
    # Export
    torch.onnx.export(
        model.backbone,  # Export just backbone for simplicity
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'features': {0: 'batch_size'},
        },
    )
    
    logger.info("Export complete!")
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", "-o", type=str, default="model.onnx", help="Output path")
    parser.add_argument("--size", type=int, default=224, help="Input size")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_size=args.size,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
