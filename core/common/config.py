import os

class Config:
    # Debug mode - set to True to raise errors instead of catching them
    DEBUG = os.environ.get('VISIONMIND_DEBUG', 'False').lower() in ('true', '1', 'yes')
    
    # API settings
    API_BASE_URL = "http://localhost:8000/api/v1"
    
    # UI settings
    APP_NAME = "VisionMind"
    VERSION = "1.0.1"
    
    # Theme colors (matching the React frontend)
    PRIMARY_COLOR = "#2563eb"  # blue-600
    BG_DARK = "#030712"        # gray-950
    BG_MAIN = "#1a1a1a"        # dark-grey
    BORDER_COLOR = "#333333"   # neutral-border
    
    # Paths
    # Current file is at core/common/config.py
    # We need to go up 3 levels to reach the project root (VisionMind)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RESOURCE_DIR = os.path.join(ROOT_DIR, "resources")
    QSS_DIR = os.path.join(RESOURCE_DIR, "qss")
    
    # Service Paths
    PROJECTS_DIR = os.path.join(ROOT_DIR, "projects")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    BACKEND_DIR = os.path.join(ROOT_DIR, "core", "backend")
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
    YOLO_DIR = os.path.join(WEIGHTS_DIR, "yolo")
    OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
    
    # Generation Paths (legacy, generation is now a plugin)
    GENERATION_BACKEND_DIR = os.path.join(ROOT_DIR, "plugins", "generation")
