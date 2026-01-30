#!/usr/bin/env python
"""
Generate synthetic ultrasound dataset for testing.
"""

import argparse
import logging
from pathlib import Path

from uterus_scope.data.synthetic import generate_synthetic_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ultrasound data")
    parser.add_argument("--output", "-o", type=str, default="./data/synthetic", help="Output directory")
    parser.add_argument("--count", "-n", type=int, default=100, help="Number of samples")
    parser.add_argument("--videos", action="store_true", help="Generate video sequences")
    parser.add_argument("--frames", type=int, default=30, help="Frames per video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"Generating {args.count} synthetic ultrasound samples...")
    print(f"Output directory: {args.output}")
    
    metadata = generate_synthetic_dataset(
        output_dir=args.output,
        num_samples=args.count,
        include_videos=args.videos,
        video_frames=args.frames,
        seed=args.seed,
    )
    
    print(f"\nGeneration complete!")
    print(f"  - Images: {len(metadata)}")
    print(f"  - Masks: {len(metadata)}")
    if args.videos:
        print(f"  - Videos: {len(metadata)}")
    
    # Print statistics
    thicknesses = [m.endometrial_thickness_mm for m in metadata]
    vasc_counts = [sum(1 for m in metadata if m.vascularity_type == t) for t in range(4)]
    fibrosis_count = sum(1 for m in metadata if m.fibrosis_score > 0.2)
    
    print(f"\nDataset Statistics:")
    print(f"  - Thickness range: {min(thicknesses):.1f} - {max(thicknesses):.1f} mm")
    print(f"  - Vascularity distribution: Type0={vasc_counts[0]}, Type1={vasc_counts[1]}, Type2={vasc_counts[2]}, Type3={vasc_counts[3]}")
    print(f"  - Samples with fibrosis: {fibrosis_count}")


if __name__ == "__main__":
    main()
