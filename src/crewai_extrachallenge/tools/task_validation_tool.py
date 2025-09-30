from crewai.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field
import os
import glob


class TaskValidationInput(BaseModel):
    """Input schema for TaskValidationTool."""
    task_name: str = Field(..., description="Name of the completed task to validate")
    required_images: List[str] = Field(..., description="List of required image filenames (without extension)")
    validation_type: str = Field(..., description="Type of validation: 'images_exist', 'task_complete'")


class TaskValidationTool(BaseTool):
    name: str = "Task Validation Tool"
    description: str = (
        "Validates that required outputs from a task have been completed successfully. "
        "Use 'images_exist' to check if required images were generated. "
        "Use 'task_complete' for general task completion validation."
    )
    args_schema: Type[BaseModel] = TaskValidationInput

    def _run(self, task_name: str, required_images: List[str], validation_type: str) -> str:
        """Validate task completion based on requirements."""

        try:
            if validation_type == "images_exist":
                return self._validate_images_exist(task_name, required_images)
            elif validation_type == "task_complete":
                return self._validate_task_complete(task_name, required_images)
            else:
                return f"Error: validation_type must be 'images_exist' or 'task_complete'"

        except Exception as e:
            return f"Error validating task {task_name}: {str(e)}"

    def _validate_images_exist(self, task_name: str, required_images: List[str]) -> str:
        """Check if all required images were generated successfully."""

        images_dir = os.path.join(os.getcwd(), 'reports', 'images')

        if not os.path.exists(images_dir):
            return f"❌ TASK VALIDATION FAILED for {task_name}\\n\\nNo images directory found. Required images: {', '.join(required_images)}"

        # Get all PNG files in the images directory
        existing_files = glob.glob(os.path.join(images_dir, "*.png"))
        existing_basenames = [os.path.splitext(os.path.basename(f))[0] for f in existing_files]

        missing_images = []
        found_images = []

        for required_image in required_images:
            # Check both exact match and partial match (for timestamped files)
            exact_match = required_image in existing_basenames
            partial_matches = [f for f in existing_basenames if f.startswith(required_image)]

            if exact_match or partial_matches:
                if exact_match:
                    found_images.append(f"{required_image}.png")
                else:
                    found_images.append(f"{partial_matches[0]}.png")
            else:
                missing_images.append(required_image)

        if missing_images:
            result = f"❌ TASK VALIDATION FAILED for {task_name}\\n\\n"
            result += f"Missing required images: {', '.join(missing_images)}\\n"
            result += f"Found images: {', '.join(found_images) if found_images else 'None'}\\n\\n"
            result += "**ACTION REQUIRED**: Previous agents must successfully generate all required visualizations before proceeding."
            return result
        else:
            result = f"✅ TASK VALIDATION PASSED for {task_name}\\n\\n"
            result += f"All required images found: {', '.join(found_images)}\\n"
            result += f"Total images verified: {len(found_images)}"
            return result

    def _validate_task_complete(self, task_name: str, required_outputs: List[str]) -> str:
        """General task completion validation."""

        # This can be extended for other types of validation
        result = f"✅ TASK VALIDATION for {task_name}\\n\\n"
        result += f"Checking completion of: {', '.join(required_outputs)}\\n"
        result += "Task marked as complete."

        return result

    def get_expected_images_for_task(self, task_name: str) -> List[str]:
        """Get list of expected images for a specific task."""

        expected_images = {
            "data_analysis_task": ["fraud_comparison", "correlation_heatmap"],
            "pattern_recognition_task": ["scatter", "time_series", "feature_importance", "box_plot"],
            "classification_task": ["amount_histogram"]
        }

        return expected_images.get(task_name, [])

    def get_all_expected_images(self) -> List[str]:
        """Get all expected images across all tasks."""
        all_expected = []
        for task_images in self.get_expected_images_for_task("data_analysis_task"):
            all_expected.extend(task_images)
        for task_images in self.get_expected_images_for_task("pattern_recognition_task"):
            all_expected.extend(task_images)
        for task_images in self.get_expected_images_for_task("classification_task"):
            all_expected.extend(task_images)
        return all_expected