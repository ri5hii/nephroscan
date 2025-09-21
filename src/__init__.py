"""
NephroScan - Kidney Stone Detection System
==========================================

A comprehensive medical imaging system for kidney stone detection and analysis
using YOLOv8 deep learning models with advanced visualization and reporting.

Author: NephroScan Team
Version: 3.0.0
"""

__version__ = "3.0.0"
__author__ = "NephroScan Team"
__email__ = "support@nephroscan.ai"

from backend.v3_model import V3ModelBackend
from gui.main_window import KidneyStoneDetectionGUI

__all__ = ["V3ModelBackend", "KidneyStoneDetectionGUI"]