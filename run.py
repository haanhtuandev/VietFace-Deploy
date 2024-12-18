import os
import sys
from pathlib import Path
import torch


# Fix import path
project_root = Path(__file__).resolve().parent  # Correctly calculate project root

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    print(f"Added to Python path: {project_root}")

if __name__ == "__main__":
    # Get main.py path
    main_path = project_root/ 'app' / 'main.py'  # Path relative to project root
    
    # Verify main.py exists
    if not main_path.exists():
        print(f"Error: Could not find main.py at {main_path}")
        sys.exit(1)
    
    print(f"Starting Streamlit app from: {main_path}")
    
    # Run streamlit
    os.system(f'streamlit run "{str(main_path)}"')
