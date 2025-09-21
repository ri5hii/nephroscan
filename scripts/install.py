#!/usr/bin/env python3
"""
NephroScan v3 Installation Script
=================================

Installs all required dependencies for the NephroScan application.
This script sets up the complete environment including PyQt5, PyTorch, 
and all machine learning dependencies.

Usage:
    python scripts/install.py
    
Author: NephroScan Team
Version: 3.0.0
"""

import subprocess
import sys
import os
from pathlib import Path

# Get project root directory
project_root = Path(__file__).parent.parent
requirements_file = project_root / "requirements.txt"

def run_command(command, description):
    """Run a command and return success status"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main installation function"""
    print("🚀 NephroScan v3 GUI Installation")
    print("=" * 40)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return 1
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(os.path.dirname(current_dir), "requirements.txt")
    
    print(f"📁 Working directory: {current_dir}")
    
    # Check if requirements file exists
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file not found: {requirements_file}")
        return 1

    print(f"📋 Requirements file: {requirements_file}")
    
    # Install requirements
    install_command = f"{sys.executable} -m pip install -r {requirements_file}"
    
    if not run_command(install_command, "Installing requirements"):
        print("💡 Try running with --user flag if permission denied")
        install_command_user = f"{sys.executable} -m pip install --user -r {requirements_file}"
        if not run_command(install_command_user, "Installing requirements (user mode)"):
            return 1
    
    # Verify critical imports
    print("\n🔍 Verifying installation...")
    
    critical_packages = [
        ("PyQt5", "from PyQt5.QtWidgets import QApplication"),
        ("ultralytics", "import ultralytics"),
        ("torch", "import torch"),
        ("matplotlib", "import matplotlib.pyplot"),
        ("cv2", "import cv2"),
        ("numpy", "import numpy"),
        ("pandas", "import pandas"),
        ("seaborn", "import seaborn"),
        ("sklearn", "import sklearn"),
        ("yaml", "import yaml"),
        ("reportlab", "import reportlab")
    ]
    
    failed_imports = []
    
    for package_name, import_statement in critical_packages:
        try:
            exec(import_statement)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name}")
            failed_imports.append(package_name)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("💡 Try reinstalling these packages manually")
        return 1
    
    print("\n🎉 Installation completed successfully!")
    print("🚀 You can now run the GUI with: python launch.py")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)