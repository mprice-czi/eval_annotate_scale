#!/usr/bin/env python3
"""
Legacy Intelligent Passage Preprocessing (Orchestrator Version)

This script provides backward compatibility by orchestrating the two modular scripts:
1. segment_passages.py (Stage 1: Segmentation)  
2. generate_marginal_pairs.py (Stage 2: Pair Generation)

Zero code duplication - just calls the existing scripts with proper arguments.

Usage:
    bazel run //scripts:intelligent_preprocessing -- \
        --config configs/preprocessing_config.yaml \
        --output data/outputs/marginal_pairs.json \
        --max-passages 50 --target-pairs 25 --resume
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("intelligent_preprocessing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_script(script_name: str, args: list) -> int:
    """Run a script with arguments and return exit code."""
    try:
        # Determine if we're running under Bazel or direct Python
        if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
            # Running under Bazel - use bazel run
            bazel_target = f"//scripts:{script_name.replace('.py', '')}"
            cmd = ["bazel", "run", bazel_target, "--"] + args
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, check=True, cwd=os.environ["BUILD_WORKSPACE_DIRECTORY"]
            )
        else:
            # Running directly - use python
            script_path = Path(__file__).parent / script_name
            cmd = [sys.executable, str(script_path)] + args
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)

        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Script {script_name} failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        logger.error(f"Error running {script_name}: {e}")
        return 1


def main():
    """Main orchestration function - calls modular scripts in sequence."""
    parser = argparse.ArgumentParser(
        description="Legacy intelligent preprocessing (orchestrator)"
    )
    parser.add_argument(
        "--config", required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for marginal pairs JSON"
    )
    parser.add_argument(
        "--max-passages", type=int, help="Maximum number of passages to process"
    )
    parser.add_argument(
        "--target-pairs", type=int, help="Target number of marginal pairs to generate"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous run"
    )
    parser.add_argument(
        "--clear-csv", default="data/CLEAR.csv", help="Path to CLEAR.csv file"
    )

    args = parser.parse_args()

    logger.info(
        "🚀 Starting legacy intelligent preprocessing (zero code duplication orchestrator)"
    )

    # Create temporary file for intermediate segmented passages
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_segmented_passages.json", delete=False
    ) as temp_file:
        temp_segments_path = temp_file.name

    try:
        # Stage 1: Run segment_passages.py
        logger.info("📝 Stage 1: Running segment_passages.py")

        stage1_args = [
            "--config",
            args.config,
            "--output",
            temp_segments_path,
            "--clear-csv",
            args.clear_csv,
        ]

        if args.max_passages:
            stage1_args.extend(["--max-passages", str(args.max_passages)])

        if args.resume:
            stage1_args.append("--resume")

        exit_code = run_script("segment_passages.py", stage1_args)
        if exit_code != 0:
            logger.error("Stage 1 (segmentation) failed")
            return exit_code

        logger.info("✅ Stage 1 complete")

        # Stage 2: Run generate_marginal_pairs.py
        logger.info("🔗 Stage 2: Running generate_marginal_pairs.py")

        stage2_args = [
            "--input",
            temp_segments_path,
            "--config",
            args.config,
            "--output",
            args.output,
        ]

        if args.target_pairs:
            stage2_args.extend(["--target-pairs", str(args.target_pairs)])

        if args.resume:
            stage2_args.append("--resume")

        exit_code = run_script("generate_marginal_pairs.py", stage2_args)
        if exit_code != 0:
            logger.error("Stage 2 (pair generation) failed")
            return exit_code

        logger.info("✅ Stage 2 complete")

        logger.info(
            f"""
🎉 Legacy preprocessing completed successfully!

This orchestrator script called:
1. segment_passages.py (with all improvements: flexible time ranges, overlap, vocab assessment)
2. generate_marginal_pairs.py (with caching and resume capability)

Final output: {args.output}
"""
        )
        return 0

    finally:
        # Clean up temporary file
        if os.path.exists(temp_segments_path):
            os.unlink(temp_segments_path)
            logger.debug(f"Cleaned up temporary file: {temp_segments_path}")


if __name__ == "__main__":
    sys.exit(main())
