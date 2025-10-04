import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, Any, Callable, Optional, Tuple
from contextlib import contextmanager

# Add parent directory to import CrewAI classes
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from src.crewai_extrachallenge.crew import CrewaiExtrachallenge
from src.crewai_extrachallenge.config.database_config import DatabaseConfig


class StreamlitCrewRunner:
    """Wrapper class for running CrewAI fraud detection with Streamlit integration."""

    def __init__(self, dataset_path: str, use_database: bool = False, db_conversion_result: Optional[Dict[str, Any]] = None):
        """
        Initialize the crew runner.

        Args:
            dataset_path: Path to CSV dataset
            use_database: Whether to use database mode
            db_conversion_result: Database conversion result (if converted)
        """
        self.dataset_path = dataset_path
        self.use_database = use_database
        self.db_conversion_result = db_conversion_result
        self.current_agent = None
        self.progress = 0
        self.status = "Initializing..."
        self.is_running = False
        self.results = {}
        self.error = None

    def run_analysis(self, progress_callback: Optional[Callable] = None,
                    status_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the complete CrewAI fraud detection analysis.

        Args:
            progress_callback: Function to call with progress updates (0-100)
            status_callback: Function to call with status updates

        Returns:
            Dict with analysis results, report path, and image paths
        """
        try:
            self.is_running = True
            self.error = None

            # Ensure we're running from the correct directory (project root)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            original_cwd = os.getcwd()
            print(f"StreamlitCrewRunner: Original working directory: {original_cwd}")
            print(f"StreamlitCrewRunner: Changing to project root: {project_root}")
            os.chdir(project_root)

            # Set environment variable for dataset path
            os.environ['DATASET_PATH'] = self.dataset_path

            # Configure database mode if enabled
            if self.use_database and self.db_conversion_result:
                os.environ['USE_DATABASE'] = 'true'
                db_path = self.db_conversion_result.get('db_path', DatabaseConfig.DB_PATH)
                os.environ['DB_PATH'] = db_path
                print(f"StreamlitCrewRunner: Using database mode with {db_path}")
            else:
                os.environ['USE_DATABASE'] = 'false'
                print(f"StreamlitCrewRunner: Using CSV mode")

            # Initialize callbacks
            self.progress_callback = progress_callback or (lambda x: None)
            self.status_callback = status_callback or (lambda x: None)

            # Phase 1: Initialize CrewAI (10%)
            mode_label = "Database" if self.use_database else "CSV"
            self._update_progress(10, f"Initializing CrewAI system ({mode_label} mode)...")
            crew_instance = CrewaiExtrachallenge()

            # Phase 2: Data Analysis (30%)
            self._update_progress(30, "Running Data Analysis Agent...")
            self.current_agent = "Data Analyst"

            # Phase 3: Pattern Recognition (60%)
            self._update_progress(60, "Running Pattern Recognition Agent...")
            self.current_agent = "Pattern Recognition Agent"

            # Phase 4: Classification (80%)
            self._update_progress(80, "Running Classification Agent...")
            self.current_agent = "Classification Agent"

            # Phase 5: Report Generation (90%)
            self._update_progress(90, "Generating Final Report...")
            self.current_agent = "Reporting Analyst"

            # Execute the crew with proper inputs
            inputs = {
                'dataset_path': self.dataset_path,
                'current_year': str(datetime.now().year)
            }
            print(f"StreamlitCrewRunner: Executing crew with inputs: {inputs}")
            result = crew_instance.crew().kickoff(inputs=inputs)

            # Phase 6: Complete (100%)
            self._update_progress(100, "Analysis Complete!")
            self.current_agent = None

            # Collect results
            results = self._collect_results(result)
            results['mode'] = 'database' if self.use_database else 'csv'
            if self.use_database and self.db_conversion_result:
                results['db_info'] = {
                    'db_path': self.db_conversion_result.get('db_path'),
                    'table_name': self.db_conversion_result.get('table_name'),
                    'row_count': self.db_conversion_result.get('row_count'),
                    'db_size_mb': self.db_conversion_result.get('db_size_mb')
                }
            self.results = results

            return results

        except Exception as e:
            self.error = str(e)
            self._update_progress(0, f"Error: {str(e)}")
            raise e
        finally:
            self.is_running = False
            # Restore original working directory
            if 'original_cwd' in locals():
                os.chdir(original_cwd)

    def run_analysis_async(self, progress_callback: Optional[Callable] = None,
                          status_callback: Optional[Callable] = None) -> threading.Thread:
        """Run analysis in a separate thread to prevent UI blocking."""

        def run_thread():
            try:
                self.run_analysis(progress_callback, status_callback)
            except Exception as e:
                self.error = str(e)

        thread = threading.Thread(target=run_thread)
        thread.daemon = True
        thread.start()
        return thread

    def _update_progress(self, progress: int, status: str):
        """Update progress and status with callbacks."""
        self.progress = progress
        self.status = status

        if hasattr(self, 'progress_callback'):
            self.progress_callback(progress)
        if hasattr(self, 'status_callback'):
            self.status_callback(status)

    def _collect_results(self, crew_result: Any) -> Dict[str, Any]:
        """Collect and organize results from CrewAI execution."""

        results = {
            'crew_result': crew_result,
            'report_path': None,
            'image_paths': [],
            'dataset_path': self.dataset_path,
            'execution_time': time.time(),
            'images_found': 0
        }

        # Check for generated report
        report_path = 'reports/fraud_detection_report.md'
        if os.path.exists(report_path):
            results['report_path'] = report_path
            with open(report_path, 'r') as f:
                results['report_content'] = f.read()

        # Check for generated images
        images_dir = 'reports/images'
        if os.path.exists(images_dir):
            image_files = []
            for file in os.listdir(images_dir):
                if file.endswith('.png'):
                    full_path = os.path.join(images_dir, file)
                    image_files.append({
                        'filename': file,
                        'path': full_path,
                        'relative_path': f'./images/{file}',
                        'size_kb': round(os.path.getsize(full_path) / 1024, 1)
                    })

            results['image_paths'] = image_files
            results['images_found'] = len(image_files)

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the analysis."""
        return {
            'is_running': self.is_running,
            'progress': self.progress,
            'status': self.status,
            'current_agent': self.current_agent,
            'error': self.error,
            'results': self.results
        }


class ProgressTracker:
    """Helper class for tracking progress with different granularity levels."""

    def __init__(self):
        self.phases = {
            'initialization': {'weight': 10, 'name': 'Initializing System'},
            'data_analysis': {'weight': 20, 'name': 'Data Analysis'},
            'pattern_recognition': {'weight': 30, 'name': 'Pattern Recognition'},
            'classification': {'weight': 20, 'name': 'Classification Analysis'},
            'report_generation': {'weight': 20, 'name': 'Report Generation'}
        }

        self.current_phase = None
        self.phase_progress = 0

    def start_phase(self, phase_name: str):
        """Start a new phase."""
        if phase_name in self.phases:
            self.current_phase = phase_name
            self.phase_progress = 0

    def update_phase_progress(self, progress: int):
        """Update progress within current phase (0-100)."""
        self.phase_progress = max(0, min(100, progress))

    def get_overall_progress(self) -> int:
        """Calculate overall progress across all phases."""
        if not self.current_phase:
            return 0

        completed_weight = 0
        current_weight = 0

        for phase, config in self.phases.items():
            if phase == self.current_phase:
                current_weight = config['weight'] * (self.phase_progress / 100)
                break
            else:
                completed_weight += config['weight']

        return int(completed_weight + current_weight)

    def get_phase_status(self) -> str:
        """Get human-readable status for current phase."""
        if not self.current_phase:
            return "Ready to start"

        phase_name = self.phases[self.current_phase]['name']
        return f"{phase_name} ({self.phase_progress}%)"


def validate_crew_environment() -> Tuple[bool, str]:
    """Validate that CrewAI environment is properly set up."""
    try:
        # Check if we can import CrewAI
        from src.crewai_extrachallenge.crew import CrewaiExtrachallenge

        # Check for required environment variables
        if not os.getenv('OPENAI_API_KEY') and not os.getenv('OLLAMA_BASE_URL'):
            return False, "❌ Missing API configuration. Set OPENAI_API_KEY or ensure Ollama is running"

        # Check if we can create crew instance
        crew = CrewaiExtrachallenge()

        return True, "✅ CrewAI environment ready"

    except ImportError as e:
        return False, f"❌ CrewAI import failed: {str(e)}"
    except Exception as e:
        return False, f"❌ CrewAI setup error: {str(e)}"


def estimate_analysis_time(dataset_info: Dict[str, Any], use_database: bool = False) -> str:
    """
    Estimate analysis time based on dataset characteristics.

    Args:
        dataset_info: Dataset information dictionary
        use_database: Whether using database mode

    Returns:
        Estimated time string
    """
    rows = dataset_info.get('rows', 0)

    if use_database:
        # Database mode is faster
        if rows < 1000:
            return "~1-2 minutes"
        elif rows < 10000:
            return "~2-3 minutes"
        elif rows < 100000:
            return "~3-5 minutes"
        else:
            return "~5-8 minutes (database optimized)"
    else:
        # CSV mode
        if rows < 1000:
            return "~2-3 minutes"
        elif rows < 10000:
            return "~3-5 minutes"
        elif rows < 100000:
            return "~5-8 minutes"
        else:
            return "~8-15 minutes"