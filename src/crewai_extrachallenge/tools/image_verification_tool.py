from crewai.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field
import os
import glob


class ImageVerificationInput(BaseModel):
    """Input schema for ImageVerificationTool."""
    action: str = Field(..., description="Action to perform: 'list_available' or 'verify_exists'")
    image_path: str = Field(None, description="Specific image path to verify (only for verify_exists action)")


class ImageVerificationTool(BaseTool):
    name: str = "Image Verification Tool"
    description: str = (
        "Verifies which images are available in the reports/images directory. "
        "Use 'list_available' to get all available images, or 'verify_exists' to check a specific image."
    )
    args_schema: Type[BaseModel] = ImageVerificationInput

    def _run(self, action: str, image_path: str = None) -> str:
        """Verify available images in the reports directory."""

        try:
            images_dir = os.path.join(os.getcwd(), 'reports', 'images')

            if action == "list_available":
                return self._list_available_images(images_dir)
            elif action == "verify_exists":
                return self._verify_image_exists(images_dir, image_path)
            else:
                return "Error: Action must be 'list_available' or 'verify_exists'"

        except Exception as e:
            return f"Error verifying images: {str(e)}"

    def _list_available_images(self, images_dir: str) -> str:
        """List all available images with their types and paths."""

        if not os.path.exists(images_dir):
            return "No images directory found. No charts have been generated yet."

        # Get all PNG files in the images directory
        image_files = glob.glob(os.path.join(images_dir, "*.png"))

        if not image_files:
            return """## No Images Available for Report

**Warning: No images found in reports/images directory.**

This means previous agents did not successfully generate any visualizations.
The report should:
1. Note that visualizations were not available
2. Focus on text-based analysis and findings
3. Include recommendations for data visualization in future analyses

**Do NOT reference any image files in the markdown report.**"""

        result = "## Available Images for Report\n\n"
        result += "| Image Type | Filename | Markdown Reference |\n"
        result += "|------------|----------|-------------------|\n"

        for image_path in sorted(image_files):
            filename = os.path.basename(image_path)
            relative_path = f"./images/{filename}"

            # Determine image type from filename
            image_type = self._determine_image_type(filename)

            result += f"| {image_type} | {filename} | `![{image_type}]({relative_path})` |\n"

        result += f"\n**Total images available: {len(image_files)}**\n"
        result += "\n### Usage Instructions:\n"
        result += "1. Copy the exact markdown reference from the table above\n"
        result += "2. Paste it directly into your report content\n"
        result += "3. Do NOT modify the image paths or filenames\n"
        result += "4. Do NOT use wildcard (*) syntax\n\n"

        # Add information about expected vs actual images
        expected_images = [
            "correlation_heatmap", "fraud_comparison", "feature_importance",
            "scatter", "histogram", "time_series", "distribution", "box_plot"
        ]

        found_types = set()
        for filename in [os.path.basename(f) for f in image_files]:
            for expected in expected_images:
                if expected in filename.lower():
                    found_types.add(expected)

        missing_types = set(expected_images) - found_types
        if missing_types:
            result += f"### Missing Expected Images:\n"
            result += f"The following chart types were expected but not found: {', '.join(sorted(missing_types))}\n"
            result += "Note: This indicates some agents may not have successfully completed their visualization tasks.\n\n"

        return result

    def _verify_image_exists(self, images_dir: str, image_path: str) -> str:
        """Verify if a specific image exists."""

        if not image_path:
            return "Error: No image path provided for verification."

        # Handle both full paths and relative paths
        if image_path.startswith('./images/'):
            filename = image_path.replace('./images/', '')
            full_path = os.path.join(images_dir, filename)
        else:
            full_path = os.path.join(images_dir, image_path)

        if os.path.exists(full_path):
            filename = os.path.basename(full_path)
            image_type = self._determine_image_type(filename)
            return f"✅ Image exists: {filename} ({image_type}). Use: ![{image_type}](./images/{filename})"
        else:
            return f"❌ Image not found: {image_path}. Check available images with 'list_available' action."

    def _determine_image_type(self, filename: str) -> str:
        """Determine the type of chart from filename."""

        filename_lower = filename.lower()

        if 'correlation' in filename_lower or 'heatmap' in filename_lower:
            return "Correlation Heatmap"
        elif 'fraud_comparison' in filename_lower or 'comparison' in filename_lower:
            return "Fraud vs Legitimate Comparison"
        elif 'feature_importance' in filename_lower or 'importance' in filename_lower:
            return "Feature Importance"
        elif 'scatter' in filename_lower:
            return "Feature Scatter Plot"
        elif 'histogram' in filename_lower:
            return "Distribution Histogram"
        elif 'time_series' in filename_lower or 'temporal' in filename_lower:
            return "Time Series"
        elif 'distribution' in filename_lower:
            return "Distribution Plot"
        elif 'box_plot' in filename_lower or 'boxplot' in filename_lower:
            return "Box Plot"
        else:
            return "Chart"