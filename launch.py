#!/usr/bin/env python3
"""
NephroScan Application Launcher
===============================

Main entry point for the NephroScan kidney stone detection application.
This script initializes and launches the GUI application.

Usage:
    python launch.py
    
Author: NephroScan Team
Version: 3.0.0
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QIcon
    
    # Import NephroScan modules
    sys.path.insert(0, str(project_root / "src"))
    from gui.main_window import KidneyStoneDetectionGUI
    from utils.config import Config
    
    # Initialize configuration
    config = Config()
    app_info = config.get_app_info()
    
    print("🚀 NephroScan v3 - Kidney Stone Detection System")
    print("=" * 55)
    print(f"📁 Project root: {project_root}")
    print(f"📂 Data directory: {config.data_dir}")
    print(f"📊 App version: {app_info.get('version', '3.0.0')}")
    print(f"⚙️  Configuration: Loaded from model_config_v3.yaml")
    print(f"🏥 Training: Medical-optimized parameters ({config.training_config['epochs']} epochs)")
    print(f"🔍 Checking dependencies...")
    
    # Verify critical dependencies
    dependencies = {
        "PyQt5": "✅ PyQt5 available",
        "ultralytics": "✅ Ultralytics available", 
        "matplotlib": "✅ Matplotlib available",
        "torch": "✅ PyTorch available"
    }
    
    for dep, msg in dependencies.items():
        try:
            __import__(dep)
            print(msg)
        except ImportError:
            print(f"❌ {dep} not found. Install with: pip install {dep}")
            sys.exit(1)
    
    print("🎯 Launching GUI...")
    
    def main():
        """Main application entry point"""
        app = QApplication(sys.argv)
        app.setApplicationName("NephroScan")
        app.setApplicationVersion("3.0.0")
        app.setOrganizationName("NephroScan Team")
        
        # Set application style
        try:
            app.setStyle('Fusion')  # Use Fusion style for consistency
        except:
            pass  # Fallback to default style
        
        # Create and show main window
        try:
            window = KidneyStoneDetectionGUI()
            window.show()
            
            print("✨ GUI launched successfully!")
            print("💡 Use the Upload tab to select your data")
            print("📊 Use the Analysis tab for training/testing insights")
            print("🖼️  Use the Results tab for inference results")
            
            return app.exec_()
            
        except Exception as e:
            print(f"❌ Failed to launch GUI: {e}")
            QMessageBox.critical(None, "Launch Error", 
                               f"Failed to launch NephroScan GUI:\\n{str(e)}")
            return 1
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Please install required dependencies with:")
    print("   python scripts/install.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)