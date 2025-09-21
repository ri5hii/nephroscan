#!/usr/bin/env python3
"""
NephroScan Main Application
=========================

Main entry point for the NephroScan kidney stone detection application.
"""

import sys
import os
import matplotlib
matplotlib.use('Qt5Agg')  # Set backend before PyQt5 import

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QMetaType
from PyQt5.QtGui import QTextCursor

# Register meta types for thread-safe communication (if available)
try:
    from PyQt5.QtCore import qRegisterMetaType
    qRegisterMetaType('QTextCursor')
except (ImportError, AttributeError):
    # Skip meta type registration if not available
    pass

# Import our GUI
from gui.main_window import KidneyStoneDetectionGUI


def main():
    """Main application entry point"""
    # Create Qt Application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("NephroScan")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Medical AI Solutions")
    
    # Create and show main window
    main_window = KidneyStoneDetectionGUI()
    main_window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
