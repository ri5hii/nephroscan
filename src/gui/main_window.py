"""
Main GUI Window for NephroScan
==============================

PyQt5-based graphical user interface for kidney stone detection and analysis.
"""

import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QGroupBox, QRadioButton, QButtonGroup, QScrollArea, QSplitter,
    QFrame, QGridLayout, QMessageBox, QLineEdit, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject
from PyQt5.QtGui import QPixmap, QFont, QIcon, QTextCursor
import threading
import time
from datetime import datetime

# Register QTextCursor as a meta type for Qt signals
# Register meta types for thread-safe communication (if available)
try:
    from PyQt5.QtCore import qRegisterMetaType
    qRegisterMetaType('QTextCursor')
except (ImportError, AttributeError):
    # Skip meta type registration if not available
    pass

# Import our custom modules
from backend.v3_model import V3ModelBackend
from utils.config import config
from gui.widgets.report_generator import PDFReportGenerator


class MetricsWidget(QWidget):
    """Widget for displaying comprehensive metrics"""
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Model Performance Metrics")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Metrics frame
        self.metrics_frame = QFrame()
        self.metrics_frame.setFrameStyle(QFrame.StyledPanel)
        self.metrics_layout = QGridLayout(self.metrics_frame)
        
        # Initialize empty metrics
        self.metric_labels = {}
        self.clear_metrics()
        
        layout.addWidget(self.metrics_frame)
        
    def clear_metrics(self):
        """Clear all metrics"""
        for i in reversed(range(self.metrics_layout.count())): 
            self.metrics_layout.itemAt(i).widget().setParent(None)
        
        placeholder = QLabel("No metrics available")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-style: italic;")
        self.metrics_layout.addWidget(placeholder, 0, 0, 1, 2)
        
    def update_metrics(self, metrics: dict):
        """Update metrics display"""
        # Clear existing
        for i in reversed(range(self.metrics_layout.count())): 
            self.metrics_layout.itemAt(i).widget().setParent(None)
        
        row = 0
        for key, value in metrics.items():
            # Format key
            display_key = key.replace('_', ' ').title()
            
            # Format value
            if isinstance(value, float):
                if key in ['precision', 'recall', 'map50', 'map75', 'map', 'f1_score']:
                    display_value = f"{value:.3f}"
                else:
                    display_value = f"{value:.4f}"
            else:
                display_value = str(value)
            
            # Create labels
            key_label = QLabel(f"{display_key}:")
            key_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            
            value_label = QLabel(display_value)
            value_label.setFont(QFont("Arial", 10))
            
            # Color coding for performance metrics
            if key in ['precision', 'recall', 'f1_score', 'map50']:
                if isinstance(value, float):
                    if value >= 0.8:
                        value_label.setStyleSheet("color: green; font-weight: bold;")
                    elif value >= 0.6:
                        value_label.setStyleSheet("color: orange; font-weight: bold;")
                    else:
                        value_label.setStyleSheet("color: red; font-weight: bold;")
            
            self.metrics_layout.addWidget(key_label, row, 0)
            self.metrics_layout.addWidget(value_label, row, 1)
            row += 1


class PlotWidget(QWidget):
    """Widget for displaying matplotlib plots"""
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.plots = {}
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Scroll area for plots
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)
        
        # Initial message
        self.show_no_plots_message()
        
    def show_no_plots_message(self):
        """Show message when no plots available"""
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().setParent(None)
            
        message = QLabel("No analysis plots available\nRun training or testing to generate plots")
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet("color: gray; font-style: italic; font-size: 14px;")
        self.scroll_layout.addWidget(message)
        
    def update_plots(self, plots: dict):
        """Update plots display"""
        # Clear existing plots
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().setParent(None)
        
        if not plots:
            self.show_no_plots_message()
            return
            
        self.plots = plots
        
        for plot_name, figure in plots.items():
            # Create group box for each plot
            group_box = QGroupBox(plot_name.replace('_', ' ').title())
            group_layout = QVBoxLayout(group_box)
            
            # Create canvas
            canvas = FigureCanvas(figure)
            canvas.setFixedHeight(400)  # Fixed height for consistency
            group_layout.addWidget(canvas)
            
            self.scroll_layout.addWidget(group_box)
        
        # Add stretch to push plots to top
        self.scroll_layout.addStretch()
    
    def display_image(self, image_path):
        """Display an image from file path"""
        # Clear existing content
        self.clear_plot()
        
        try:
            # Create QLabel to display image
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            
            # Scale image to fit while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            
            # Add to scroll layout
            self.scroll_layout.addWidget(image_label)
            self.scroll_layout.addStretch()
            
        except Exception as e:
            self.show_message(f"Error loading image: {str(e)}")
    
    def clear_plot(self):
        """Clear the plot widget"""
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def show_message(self, message):
        """Show a message in the plot widget"""
        # Clear existing content
        self.clear_plot()
        
        message_label = QLabel(message)
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setStyleSheet("color: gray; font-style: italic; font-size: 14px;")
        self.scroll_layout.addWidget(message_label)
        self.scroll_layout.addStretch()


class WorkerThread(QThread):
    """Worker thread for running YOLO operations without blocking the GUI"""
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, operation_type, data_path, model_path=None):
        super().__init__()
        self.operation_type = operation_type
        self.data_path = data_path
        self.model_path = model_path
        
    def run(self):
        # This is now handled by the V3ModelBackend directly
        # Just emit a finished signal for compatibility
        self.finished.emit({"status": "completed", "operation": self.operation_type})
    
    def emit_progress(self, message):
        self.progress_update.emit(message)


class ProgressSignals(QObject):
    """Signals for thread-safe GUI updates"""
    progress_updated = pyqtSignal(str, int)  # message, percentage
    log_updated = pyqtSignal(str)  # message
    metrics_updated = pyqtSignal(dict)  # metrics dictionary
    completion_callback = pyqtSignal(str, dict)  # operation_type, result_data
    
class KidneyStoneDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.gui_config["window_title"])
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize signals for thread-safe communication
        self.signals = ProgressSignals()
        self.signals.progress_updated.connect(self._update_progress_impl)
        self.signals.log_updated.connect(self._update_logs_impl)
        self.signals.metrics_updated.connect(self._update_metrics_impl)
        self.signals.completion_callback.connect(self._on_operation_completed_impl)
        
        # Initialize v3 backend
        self.backend = V3ModelBackend(
            progress_callback=self.update_progress,
            log_callback=self.update_logs,
            metrics_callback=self.update_metrics,
            plot_callback=self.update_plots,
            completion_callback=self.on_operation_completed
        )
        
        # Initialize variables
        self.current_results = {}
        self.selected_data_path = str(config.dataset_path)  # Default to configured data folder
        self.worker_thread = None
        
        self.setup_ui()
        
        # Initialize training runs dropdown
        self.refresh_training_runs()
        
        # Timer for checking inference results
        self.result_timer = QTimer()
        self.result_timer.timeout.connect(self.check_inference_results)
        
    def update_progress(self, message: str, percentage: int = None):
        """Update progress from backend with optional percentage"""
        # Emit signal for thread-safe GUI update
        if percentage is not None:
            self.signals.progress_updated.emit(message, percentage)
        else:
            self.signals.progress_updated.emit(message, -1)
        
    def _update_progress_impl(self, message: str, percentage: int):
        """Thread-safe implementation of progress update"""
        self.progress_label.setText(message)
        
        # Update progress bar if percentage is provided
        if percentage >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percentage)
            self.progress_bar.setVisible(True)
        
        # Hide progress bar when operation completes
        if "completed" in message.lower() or "failed" in message.lower():
            self.progress_bar.setVisible(False)
            
            # If training completed, refresh the training runs dropdown
            if "training completed" in message.lower():
                self.refresh_training_runs()
                self.log_text.append("Training runs list refreshed")
                
        # Force GUI update
        QApplication.processEvents()
            
    def update_logs(self, message: str):
        """Update logs from backend"""
        # Emit signal for thread-safe GUI update
        self.signals.log_updated.emit(message)
        
    def _update_logs_impl(self, message: str):
        """Thread-safe implementation of log update"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
        
    def update_metrics(self, metrics: dict):
        """Update metrics from backend"""
        # Emit signal for thread-safe GUI update
        self.signals.metrics_updated.emit(metrics)
        
    def _update_metrics_impl(self, metrics: dict):
        """Thread-safe implementation of metrics update"""
        if hasattr(self, 'metrics_widget'):
            self.metrics_widget.update_metrics(metrics)
            
    def update_plots(self, plots: dict):
        """Update plots from backend"""
        if hasattr(self, 'plots_widget'):
            self.plots_widget.update_plots(plots)
    
    def on_operation_completed(self, operation_type, result_data):
        """Handle completion of training, testing, or inference operations"""
        # Emit signal for thread-safe GUI update
        self.signals.completion_callback.emit(operation_type, result_data)
        
    def _on_operation_completed_impl(self, operation_type, result_data):
        """Thread-safe implementation of operation completion"""
        if result_data.get("status") == "success":
            if operation_type == "training":
                self.on_training_finished(result_data)
            elif operation_type == "testing":
                self.on_testing_finished(result_data)
            elif operation_type == "inference":
                self.on_inference_finished(result_data)
        else:
            # Handle error
            error_msg = result_data.get("error", "Unknown error")
            self.on_operation_error(f"{operation_type.capitalize()} failed: {error_msg}")
        
    def setup_ui(self):
        """Setup the main UI with three tabs"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.setup_upload_tab()
        self.setup_analysis_tab()
        self.setup_result_tab()
        
    def setup_upload_tab(self):
        """Setup the upload tab with file selection and operation options"""
        upload_tab = QWidget()
        self.tab_widget.addTab(upload_tab, "Upload")
        
        layout = QVBoxLayout(upload_tab)
        
        # File selection section
        file_group = QGroupBox("Data Selection")
        file_layout = QVBoxLayout(file_group)
        
        # Data type selection
        self.data_type_group = QButtonGroup()
        self.dataset_radio = QRadioButton("Dataset (for training/testing)")
        self.image_radio = QRadioButton("Single Image (for inference)")
        self.dataset_radio.setChecked(True)
        
        self.data_type_group.addButton(self.dataset_radio)
        self.data_type_group.addButton(self.image_radio)
        
        file_layout.addWidget(self.dataset_radio)
        file_layout.addWidget(self.image_radio)
        
        # File selection
        file_select_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_files)
        
        file_select_layout.addWidget(QLabel("Selected Path:"))
        file_select_layout.addWidget(self.file_path_label, 1)
        file_select_layout.addWidget(self.browse_button)
        file_layout.addLayout(file_select_layout)
        
        layout.addWidget(file_group)
        
        # Operation selection
        operation_group = QGroupBox("Operations")
        operation_layout = QVBoxLayout(operation_group)
        
        # Buttons for different operations
        button_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        
        self.test_button = QPushButton("Test Model")
        self.test_button.clicked.connect(self.start_testing)
        self.test_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        
        self.inference_button = QPushButton("Run Inference")
        self.inference_button.clicked.connect(self.start_inference)
        self.inference_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 10px; }")
        
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.inference_button)
        
        operation_layout.addLayout(button_layout)
        
        # Connect radio buttons to update button states
        self.dataset_radio.toggled.connect(self.update_button_states)
        self.image_radio.toggled.connect(self.update_button_states)
        
        layout.addWidget(operation_group)
        
        # Status section
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_group)
        
        # Update initial button states
        self.update_button_states()
        
    def setup_analysis_tab(self):
        """Setup the analysis tab with comprehensive metrics and plots"""
        analysis_tab = QWidget()
        self.tab_widget.addTab(analysis_tab, "Analysis")
        
        layout = QVBoxLayout(analysis_tab)
        
        # Title
        title = QLabel("Comprehensive Analysis Dashboard")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Progress section
        progress_group = QGroupBox("Operation Status")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Metrics and Logs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_widget = MetricsWidget()
        metrics_layout.addWidget(self.metrics_widget)
        left_layout.addWidget(metrics_group)
        
        # Logs
        logs_group = QGroupBox("Process Logs")
        logs_layout = QVBoxLayout(logs_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_text.setStyleSheet("QTextEdit { background-color: #f0f0f0; font-family: 'Courier New'; }")
        logs_layout.addWidget(self.log_text)
        
        clear_logs_button = QPushButton("Clear Logs")
        clear_logs_button.clicked.connect(self.clear_logs)
        logs_layout.addWidget(clear_logs_button)
        left_layout.addWidget(logs_group)
        
        # Right panel: Plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Plot selection controls
        plot_controls_group = QGroupBox("Plot Selection")
        plot_controls_layout = QHBoxLayout(plot_controls_group)
        
        # Training run dropdown
        plot_controls_layout.addWidget(QLabel("Training Run:"))
        self.training_run_combo = QComboBox()
        self.training_run_combo.currentTextChanged.connect(self.on_training_run_changed)
        plot_controls_layout.addWidget(self.training_run_combo)
        
        # Plot type dropdown
        plot_controls_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Training Results",
            "Confusion Matrix", 
            "Confusion Matrix (Normalized)",
            "Box F1 Curve",
            "Box PR Curve", 
            "Box Precision Curve",
            "Box Recall Curve",
            "Training Batches",
            "Validation Batches"
        ])
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)
        plot_controls_layout.addWidget(self.plot_type_combo)
        
        # Refresh button
        self.refresh_plots_button = QPushButton("Refresh")
        self.refresh_plots_button.clicked.connect(self.refresh_training_runs)
        plot_controls_layout.addWidget(self.refresh_plots_button)
        
        right_layout.addWidget(plot_controls_group)
        
        plots_group = QGroupBox("Analysis Plots")
        plots_layout = QVBoxLayout(plots_group)
        self.plots_widget = PlotWidget()
        plots_layout.addWidget(self.plots_widget)
        right_layout.addWidget(plots_group)
        
        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 800])  # Give more space to plots
        
        layout.addWidget(main_splitter)
        
    def setup_result_tab(self):
        """Setup the result tab with output display and report generation"""
        result_tab = QWidget()
        self.tab_widget.addTab(result_tab, "Results")
        
        layout = QVBoxLayout(result_tab)
        
        # Results display
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout(results_group)
        
        # Image display
        image_layout = QHBoxLayout()
        
        # Original image
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout(original_group)
        self.original_image_label = QLabel("No image loaded")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(300, 300)
        self.original_image_label.setStyleSheet("QLabel { border: 2px solid gray; }")
        original_layout.addWidget(self.original_image_label)
        image_layout.addWidget(original_group)
        
        # Processed image
        processed_group = QGroupBox("Processed Image (with detections)")
        processed_layout = QVBoxLayout(processed_group)
        self.processed_image_label = QLabel("No processed image")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setMinimumSize(300, 300)
        self.processed_image_label.setStyleSheet("QLabel { border: 2px solid gray; }")
        processed_layout.addWidget(self.processed_image_label)
        image_layout.addWidget(processed_group)
        
        results_layout.addLayout(image_layout)
        
        # Detection details
        details_group = QGroupBox("Detection Details")
        details_layout = QGridLayout(details_group)
        
        # Add detection summary at the top
        self.detection_summary_label = QLabel("Analysis Summary: No analysis performed yet")
        self.detection_summary_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; padding: 5px; }")
        details_layout.addWidget(self.detection_summary_label, 0, 0, 1, 2)  # Span across both columns
        
        self.stone_existence_label = QLabel("Stone Detected: N/A")
        self.stone_count_label = QLabel("Number of Stones: N/A")
        self.stone_location_label = QLabel("Individual Detections: N/A")
        self.stone_size_label = QLabel("Details: N/A")
        self.confidence_label = QLabel("Total Count: N/A")
        
        details_layout.addWidget(self.stone_existence_label, 1, 0)
        details_layout.addWidget(self.stone_count_label, 1, 1)
        details_layout.addWidget(self.stone_location_label, 2, 0)
        details_layout.addWidget(self.stone_size_label, 2, 1)
        details_layout.addWidget(self.confidence_label, 3, 0)
        
        results_layout.addWidget(details_group)
        layout.addWidget(results_group)
        
        # Report generation
        report_group = QGroupBox("Report Generation")
        report_layout = QVBoxLayout(report_group)
        
        self.generate_report_button = QPushButton("Generate PDF Report")
        self.generate_report_button.clicked.connect(self.generate_report)
        self.generate_report_button.setEnabled(False)
        self.generate_report_button.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 10px; }")
        
        report_layout.addWidget(self.generate_report_button)
        layout.addWidget(report_group)
        
    def update_button_states(self):
        """Update button states based on selected data type"""
        is_dataset = self.dataset_radio.isChecked()
        
        self.train_button.setEnabled(is_dataset)
        self.test_button.setEnabled(is_dataset)
        self.inference_button.setEnabled(not is_dataset)
        
    def browse_files(self):
        """Open file dialog to select dataset or image"""
        if self.dataset_radio.isChecked():
            # Browse for dataset folder
            folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
            if folder_path:
                self.selected_data_path = folder_path
                self.file_path_label.setText(folder_path)
        else:
            # Browse for single image
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", 
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
            )
            if file_path:
                self.selected_data_path = file_path
                self.file_path_label.setText(file_path)
    
    def start_training(self):
        """Start model training with v3 backend"""
        if not self.selected_data_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset first!")
            return
        
        # Check if training is already running
        if hasattr(self.backend, 'is_running') and self.backend.is_running:
            QMessageBox.warning(self, "Warning", "Training is already in progress! Please wait for it to complete.")
            return
            
        self.log_text.append("Starting training process...")
        self.set_operation_state(True, "Training in progress...")
        
        # Switch to analysis tab
        self.tab_widget.setCurrentIndex(1)
        
        # Start training with v3 backend using enhanced config
        from utils.config import config
        training_config = config.training_config
        
        epochs = training_config.get("epochs", 150)  # Use medical-optimized default
        batch_size = training_config.get("batch_size", 16)  # Use medical-optimized default
        learning_rate = training_config.get("learning_rate", 0.01)
        img_size = training_config.get("img_size", 640)
        
        self.log_text.append(f"Training with medical-optimized parameters:")
        self.log_text.append(f"  - Epochs: {epochs}")
        self.log_text.append(f"  - Batch Size: {batch_size}")
        self.log_text.append(f"  - Learning Rate: {learning_rate}")
        self.log_text.append(f"  - Image Size: {img_size}")
        
        try:
            # Use optimized configuration file
            config_path = "/home/rishi/Desktop/nephroscan/models/yolov8_kidney_stone_v3/configs/model_config_v3.yaml"
            
            self.log_text.append(f"Using configuration file: {config_path}")
            
            self.backend.train_model(
                self.selected_data_path, 
                epochs=epochs, 
                batch_size=batch_size,
                learning_rate=learning_rate,
                img_size=img_size,
                config_path=config_path
            )
        except Exception as e:
            self.log_text.append(f"Training failed with error: {str(e)}")
            self.set_operation_state(False, "Training failed")
            QMessageBox.critical(self, "Training Error", f"Training failed: {str(e)}")
            return
    
    def start_testing(self):
        """Start model testing with v3 backend"""
        if not self.selected_data_path:
            QMessageBox.warning(self, "Warning", "Please select a dataset first!")
            return
            
        self.log_text.append("Starting testing process...")
        self.set_operation_state(True, "Testing in progress...")
        
        # Switch to analysis tab
        self.tab_widget.setCurrentIndex(1)
        
        # Start testing with v3 backend
        try:
            self.backend.test_model(self.selected_data_path)
        except Exception as e:
            self.log_text.append(f"Testing failed with error: {str(e)}")
            self.set_operation_state(False, "Testing failed")
            QMessageBox.critical(self, "Testing Error", f"Testing failed: {str(e)}")
            return
    
    def start_inference(self):
        """Start inference on single image with v3 backend"""
        if not self.selected_data_path:
            QMessageBox.warning(self, "Warning", "Please select an image first!")
            return
            
        self.log_text.append("Starting inference...")
        self.set_operation_state(True, "Running inference...")
        
        # Switch to analysis tab temporarily
        self.tab_widget.setCurrentIndex(1)
        
        # Start inference with v3 backend
        self.backend.run_inference(self.selected_data_path)
        
        # Start timer to check for results
        self.result_timer.start(1000)  # Check every second
    
    def check_inference_results(self):
        """Check if inference results are ready"""
        if not self.backend.is_running:
            self.result_timer.stop()
            results = self.backend.get_inference_results()
            if results:
                self.set_operation_state(False, "Inference completed")
                self.update_result_display(results)
                # Switch to results tab
                self.tab_widget.setCurrentIndex(2)
                # Note: Success popup will be shown by completion callback, not here
            else:
                self.set_operation_state(False, "Inference failed")
                QMessageBox.warning(self, "Error", "Inference failed!")
    
    def set_operation_state(self, in_progress, status_text):
        """Set the UI state for ongoing operations"""
        if in_progress:
            self.progress_bar.setVisible(True)
            # For training operations, use deterministic progress (0-100)
            if "training" in status_text.lower():
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
            else:
                # For other operations, use indeterminate progress
                self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setVisible(False)
        
        self.progress_label.setText(status_text)
        
        # Disable operation buttons during processing
        self.train_button.setEnabled(not in_progress and self.dataset_radio.isChecked())
        self.test_button.setEnabled(not in_progress and self.dataset_radio.isChecked())
        self.inference_button.setEnabled(not in_progress and self.image_radio.isChecked())
    
    def clear_logs(self):
        """Clear the log text"""
        self.log_text.clear()
    
    def update_result_display(self, result):
        """Update the result tab with detection results"""
        self.current_results = result
        
        # Load and display images
        if 'original_image_path' in result:
            original_pixmap = QPixmap(result['original_image_path'])
            if not original_pixmap.isNull():
                scaled_original = original_pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.original_image_label.setPixmap(scaled_original)
        
        if 'annotated_image_path' in result:
            processed_pixmap = QPixmap(result['annotated_image_path'])
            if not processed_pixmap.isNull():
                scaled_processed = processed_pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.processed_image_label.setPixmap(scaled_processed)
        
        # Update detection details
        stone_detected = result.get('stone_detected', False)
        num_stones = result.get('number_of_stones', 0)
        
        # Update detection summary
        detections = result.get('detections', [])
        if stone_detected and detections:
            # Calculate average confidence for summary
            confidence_values = []
            size_info = []
            
            for detection in detections:
                conf = detection.get('confidence', 0)
                if isinstance(conf, (int, float)):
                    confidence_values.append(conf)
                
                # Check for millimeter size data
                size_mm_data = detection.get('size_mm', {})
                if isinstance(size_mm_data, dict) and 'diameter_mm' in size_mm_data:
                    diameter = size_mm_data.get('diameter_mm', 'N/A')
                    category = size_mm_data.get('size_category', '')
                    size_info.append(f"{diameter}mm ({category})")
            
            avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
            
            if size_info:
                size_summary = f"Sizes: {', '.join(size_info[:2])}" + ("..." if len(size_info) > 2 else "")
                summary_text = f"Analysis Complete: {num_stones} stone{'s' if num_stones != 1 else ''} detected with {avg_confidence:.1%} avg confidence. {size_summary}"
            else:
                summary_text = f"Analysis Complete: {num_stones} stone{'s' if num_stones != 1 else ''} detected with {avg_confidence:.1%} average confidence"
        else:
            summary_text = "Analysis Complete: No kidney stones detected in the scanned image"
        
        self.detection_summary_label.setText(summary_text)
        
        self.stone_existence_label.setText(f"Stone Detected: {'Yes' if stone_detected else 'No'}")
        self.stone_count_label.setText(f"Number of Stones: {num_stones}")
        
        detections = result.get('detections', [])
        if detections:
            stone_details = []
            
            for i, detection in enumerate(detections, 1):
                # Get confidence and location
                confidence = detection.get('confidence', 'N/A')
                location = detection.get('location', 'N/A')
                
                # Format confidence to show maximum 4 decimal places
                if isinstance(confidence, (int, float)):
                    confidence_display = f"{confidence:.4f}"
                else:
                    confidence_display = str(confidence)
                
                # Use millimeter size if available, fallback to pixel size
                size_mm_data = detection.get('size_mm', {})
                if isinstance(size_mm_data, dict) and 'width_mm' in size_mm_data:
                    diameter_mm = size_mm_data.get('diameter_mm', 'N/A')
                    category = size_mm_data.get('size_category', '').replace(' (', ' (').replace(')', ')')
                    
                    # Concise format with location: Stone N: Diameter mm, Category, Location, Confidence
                    stone_detail = f"Stone {i}: {diameter_mm}mm diameter, {category}, {location}, {confidence_display} confidence"
                else:
                    # Fallback to pixel size with location
                    pixel_size = detection.get('size', 'N/A')
                    stone_detail = f"Stone {i}: {pixel_size}, {location}, {confidence_display} confidence"
                
                stone_details.append(stone_detail)
            
            # Join with newlines for one stone per line
            combined_text = '\n'.join(stone_details)
            
            # Display detailed detection information
            self.stone_location_label.setText("Individual Detections:")
            self.stone_size_label.setText(combined_text)
            self.confidence_label.setText(f"Total: {len(detections)} stone{'s' if len(detections) != 1 else ''} detected")
        else:
            self.stone_location_label.setText("Individual Detections:")
            self.stone_size_label.setText("No kidney stones detected")
            self.confidence_label.setText("Total: 0 stones detected")
        
        # Enable report generation
        self.generate_report_button.setEnabled(True)
    
    def log_message(self, message):
        """Add message to log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_logs(self):
        """Clear the log text"""
        self.log_text.clear()
    
    def on_training_finished(self, result):
        """Handle training completion"""
        self.set_operation_state(False, "Training completed")
        self.log_message("Training completed successfully!")
        self.current_results = result
        
        # Update metrics if provided
        if 'metrics' in result:
            self._update_metrics_impl(result['metrics'])
        
        # Refresh training runs dropdown to include the new training
        self.refresh_training_runs()
        
        QMessageBox.information(self, "Success", "Model training completed successfully!")
    
    def on_testing_finished(self, result):
        """Handle testing completion"""
        self.set_operation_state(False, "Testing completed")
        self.log_message("Testing completed successfully!")
        self.current_results = result
        QMessageBox.information(self, "Success", "Model testing completed successfully!")
    
    def on_inference_finished(self, result):
        """Handle inference completion"""
        self.set_operation_state(False, "Inference completed")
        self.log_message("Inference completed successfully!")
        self.current_results = result
        
        # Update result tab
        self.update_result_display(result)
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)
        
        QMessageBox.information(self, "Success", "Inference completed successfully!")
    
    def on_operation_error(self, error_message):
        """Handle operation errors"""
        self.set_operation_state(False, "Operation failed")
        self.log_message(f"Error: {error_message}")
        QMessageBox.critical(self, "Error", f"Operation failed: {error_message}")
    
    def generate_report(self):
        """Generate PDF report"""
        if not self.current_results:
            QMessageBox.warning(self, "Warning", "No results available for report generation!")
            return
        
        # Get save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "kidney_stone_report.pdf", 
            "PDF Files (*.pdf)"
        )
        
        if file_path:
            try:
                report_generator = PDFReportGenerator()
                report_generator.generate_report(self.current_results, file_path)
                QMessageBox.information(self, "Success", f"Report saved successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to generate report: {str(e)}")

    def refresh_training_runs(self):
        """Refresh the list of available training runs"""
        self.training_run_combo.clear()
        
        try:
            training_cycles_dir = Path("/home/rishi/Desktop/nephroscan/output/training_cycles")
            if training_cycles_dir.exists():
                # Get all training directories, sorted by creation time (newest first)
                training_dirs = [d for d in training_cycles_dir.iterdir() if d.is_dir()]
                training_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                for training_dir in training_dirs:
                    # Check if it has results
                    if (training_dir / "results.csv").exists():
                        self.training_run_combo.addItem(training_dir.name, str(training_dir))
                
                if self.training_run_combo.count() == 0:
                    self.training_run_combo.addItem("No training runs found", None)
            else:
                self.training_run_combo.addItem("Training cycles directory not found", None)
                
        except Exception as e:
            self.training_run_combo.addItem(f"Error loading runs: {str(e)}", None)
    
    def on_training_run_changed(self, run_name):
        """Handle training run selection change"""
        if run_name and run_name != "No training runs found" and not run_name.startswith("Error"):
            self.load_training_metrics()
            self.on_plot_type_changed(self.plot_type_combo.currentText())
    
    def on_plot_type_changed(self, plot_type):
        """Handle plot type selection change"""
        current_data = self.training_run_combo.currentData()
        if not current_data:
            return
            
        training_dir = Path(current_data)
        if not training_dir.exists():
            return
            
        try:
            # Map plot types to file names
            plot_files = {
                "Training Results": "results.png",
                "Confusion Matrix": "confusion_matrix.png",
                "Confusion Matrix (Normalized)": "confusion_matrix_normalized.png", 
                "Box F1 Curve": "BoxF1_curve.png",
                "Box PR Curve": "BoxPR_curve.png",
                "Box Precision Curve": "BoxP_curve.png",
                "Box Recall Curve": "BoxR_curve.png",
                "Training Batches": "train_batch0.jpg",
                "Validation Batches": "val_batch0_pred.jpg"
            }
            
            plot_file = plot_files.get(plot_type)
            if plot_file:
                plot_path = training_dir / plot_file
                if plot_path.exists():
                    self.plots_widget.display_image(str(plot_path))
                else:
                    self.plots_widget.clear_plot()
                    self.plots_widget.show_message(f"Plot file not found: {plot_file}")
            
        except Exception as e:
            self.plots_widget.clear_plot()
            self.plots_widget.show_message(f"Error loading plot: {str(e)}")
    
    def load_training_metrics(self):
        """Load and display training metrics from selected run"""
        current_data = self.training_run_combo.currentData()
        if not current_data:
            return
            
        training_dir = Path(current_data)
        results_file = training_dir / "results.csv"
        
        if not results_file.exists():
            return
            
        try:
            # Load results CSV
            import pandas as pd
            df = pd.read_csv(results_file)
            
            if len(df) > 0:
                # Get final epoch metrics
                final_metrics = df.iloc[-1].to_dict()
                
                # Format metrics for display
                display_metrics = {}
                
                # Training metrics
                if 'train/box_loss' in final_metrics:
                    display_metrics['Final Train Box Loss'] = f"{final_metrics['train/box_loss']:.4f}"
                if 'train/cls_loss' in final_metrics:
                    display_metrics['Final Train Class Loss'] = f"{final_metrics['train/cls_loss']:.4f}"
                if 'train/dfl_loss' in final_metrics:
                    display_metrics['Final Train DFL Loss'] = f"{final_metrics['train/dfl_loss']:.4f}"
                
                # Validation metrics
                if 'val/box_loss' in final_metrics:
                    display_metrics['Final Val Box Loss'] = f"{final_metrics['val/box_loss']:.4f}"
                if 'val/cls_loss' in final_metrics:
                    display_metrics['Final Val Class Loss'] = f"{final_metrics['val/cls_loss']:.4f}"
                if 'val/dfl_loss' in final_metrics:
                    display_metrics['Final Val DFL Loss'] = f"{final_metrics['val/dfl_loss']:.4f}"
                
                # Performance metrics
                if 'metrics/precision(B)' in final_metrics:
                    display_metrics['Precision'] = f"{final_metrics['metrics/precision(B)']:.4f}"
                if 'metrics/recall(B)' in final_metrics:
                    display_metrics['Recall'] = f"{final_metrics['metrics/recall(B)']:.4f}"
                if 'metrics/mAP50(B)' in final_metrics:
                    display_metrics['mAP@0.5'] = f"{final_metrics['metrics/mAP50(B)']:.4f}"
                if 'metrics/mAP50-95(B)' in final_metrics:
                    display_metrics['mAP@0.5:0.95'] = f"{final_metrics['metrics/mAP50-95(B)']:.4f}"
                
                # F1 Score calculation
                if 'metrics/precision(B)' in final_metrics and 'metrics/recall(B)' in final_metrics:
                    precision = final_metrics['metrics/precision(B)']
                    recall = final_metrics['metrics/recall(B)']
                    if precision + recall > 0:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                        display_metrics['F1 Score'] = f"{f1_score:.4f}"
                
                # Training info
                display_metrics['Total Epochs'] = f"{len(df)}"
                display_metrics['Training Directory'] = training_dir.name
                
                # Update metrics widget
                self.metrics_widget.update_metrics(display_metrics)
                
        except Exception as e:
            self.metrics_widget.clear_metrics()
            self.log_text.append(f"Error loading metrics: {str(e)}")


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = KidneyStoneDetectionGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()