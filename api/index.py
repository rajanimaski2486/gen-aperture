import sys
import os

# Resolve backend path and insert it into sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

# Import the FastAPI application instance
from app.main import app
