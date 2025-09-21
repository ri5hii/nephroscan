import os
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, Frame, PageTemplate, BaseDocTemplate
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from PIL import Image as PILImage
import json


class PDFReportGenerator:
    """Generate comprehensive PDF reports for kidney stone detection results"""
    
    def __init__(self):
        """Initialize PDF report generator with custom styles."""
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=12,
            textColor=colors.darkred
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=8,
            textColor=colors.darkgreen
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.black,
            alignment=TA_CENTER
        ))
        
        # Table text style
        self.styles.add(ParagraphStyle(
            name='TableText',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.black,
            alignment=TA_CENTER,
            wordWrap='CJK'
        ))
    
    def generate_report(self, results_data, output_path):
        """Generate a comprehensive PDF report"""
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story (content)
        story = []
        
        # Add title and executive summary combined (no separate title page)
        story.extend(self.create_title_and_executive_summary(results_data))
        
        # Add images section first (side-by-side comparison)
        story.extend(self.create_images_section(results_data))
        
        # Add detection results section (textual summary + data table)
        story.extend(self.create_detection_results(results_data))
        
        # Add technical details
        story.extend(self.create_technical_details(results_data))
        
        # Add metrics (if available)
        if 'metrics' in results_data:
            story.extend(self.create_metrics_section(results_data))
        
        # Add appendix
        story.extend(self.create_appendix(results_data))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def create_title_and_executive_summary(self, results_data):
        """Create the title and executive summary section combined"""
        story = []
        
        # Title
        title = Paragraph("Kidney Stone Detection Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 30))
        
        # Report details table
        report_data = [
            ['Report Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Analysis Type:', self.get_analysis_type(results_data)],
            ['Status:', results_data.get('status', 'Unknown')],
            ['Model Used:', os.path.basename(results_data.get('model_path', 'Default'))],
        ]
        
        if 'timestamp' in results_data:
            report_data.append(['Analysis Date:', results_data['timestamp'][:19]])
        
        table = Table(report_data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 30))
        
        # Executive Summary section
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Determine if this is inference results
        if 'detections' in results_data:
            detections = results_data['detections']
            num_stones = len(detections)
            
            if num_stones > 0:
                # Calculate average confidence (handle string confidence values)
                confidence_values = []
                for d in detections:
                    conf = d.get('confidence', 0)
                    try:
                        # Convert string confidence to float
                        if isinstance(conf, str):
                            conf = float(conf)
                        confidence_values.append(conf)
                    except (ValueError, TypeError):
                        confidence_values.append(0.0)
                
                avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
                
                # Check for size data (use correct field name)
                has_size_data = any('size_mm' in d and isinstance(d['size_mm'], dict) for d in detections)
                size_note = " Comprehensive size analysis in millimeters is included." if has_size_data else ""
                
                summary_text = f"""
                <b>Analysis Result:</b> {num_stones} kidney stone(s) detected with an average confidence of {avg_confidence:.1%}.{size_note}<br/><br/>
                <b>Recommendation:</b> Medical consultation recommended for further evaluation and treatment planning.
                """
            else:
                summary_text = """
                <b>Analysis Result:</b> No kidney stones detected in the analyzed image.<br/><br/>
                <b>Recommendation:</b> If symptoms persist, consult with a healthcare provider for additional imaging or evaluation.
                """
        else:
            # Training/Testing summary
            if 'epochs' in results_data:
                summary_text = f"""
                <b>Training Completed:</b> Model trained for {results_data.get('epochs', 'N/A')} epochs.<br/>
                <b>Dataset:</b> {results_data.get('dataset_path', 'N/A')}<br/>
                <b>Model Saved:</b> {os.path.basename(results_data.get('model_path', 'N/A'))}
                """
            else:
                summary_text = f"""
                <b>Testing Completed:</b> Model evaluation performed.<br/>
                <b>Test Images:</b> {results_data.get('num_test_images', 'N/A')}<br/>
                <b>Dataset:</b> {results_data.get('dataset_path', 'N/A')}
                """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Add disclaimer
        disclaimer = Paragraph(
            "<b>Disclaimer:</b> This report is generated by an automated kidney stone detection system. "
            "Results should be reviewed by qualified medical professionals before making any clinical decisions.",
            self.styles['Normal']
        )
        story.append(disclaimer)
        story.append(Spacer(1, 20))
        
        return story
    
    def create_detection_results(self, results_data):
        """Create enhanced detection results section with textual summary and comprehensive data table"""
        story = []
        
        if 'detections' not in results_data:
            return story
        
        story.append(Paragraph("Detection Results & Analysis", self.styles['SectionHeader']))
        
        detections = results_data['detections']
        
        if len(detections) == 0:
            story.append(Paragraph("No kidney stones detected in the analyzed image.", self.styles['Normal']))
            story.append(Paragraph(
                "The AI analysis did not identify any objects with sufficient confidence to classify as kidney stones. "
                "This could indicate either the absence of stones or stones that are too small, unclear, or obscured to detect.",
                self.styles['Normal']
            ))
        else:
            # Create textual summary
            story.extend(self.create_stones_summary(detections))
            
            # Add detailed data table
            story.append(Paragraph("Detailed Detection Data", self.styles['SubSection']))
            story.extend(self.create_comprehensive_detection_table(detections))
        
        story.append(Spacer(1, 20))
        return story
    
    def create_stones_summary(self, detections):
        """Create comprehensive textual summary of detected stones"""
        story = []
        
        num_stones = len(detections)
        story.append(Paragraph(f"<b>Summary:</b> {num_stones} kidney stone{'s' if num_stones != 1 else ''} detected", self.styles['SubSection']))
        
        # Analyze stone characteristics (handle string confidence values)
        def get_confidence_float(detection):
            """Convert confidence to float, handling string values"""
            conf = detection.get('confidence', 0)
            try:
                if isinstance(conf, str):
                    return float(conf)
                return float(conf)
            except (ValueError, TypeError):
                return 0.0
        
        high_conf_stones = [d for d in detections if get_confidence_float(d) >= 0.7]
        medium_conf_stones = [d for d in detections if 0.4 <= get_confidence_float(d) < 0.7]
        low_conf_stones = [d for d in detections if get_confidence_float(d) < 0.4]
        
        # Confidence analysis
        confidence_summary = []
        if high_conf_stones:
            confidence_summary.append(f"{len(high_conf_stones)} with high confidence (≥70%)")
        if medium_conf_stones:
            confidence_summary.append(f"{len(medium_conf_stones)} with medium confidence (40-70%)")
        if low_conf_stones:
            confidence_summary.append(f"{len(low_conf_stones)} with low confidence (<40%)")
        
        if confidence_summary:
            story.append(Paragraph(
                f"<b>Confidence Distribution:</b> {', '.join(confidence_summary)}.",
                self.styles['Normal']
            ))
        
        # Size analysis (if millimeter data available, use correct field name)
        stones_with_size = [d for d in detections if 'size_mm' in d and isinstance(d['size_mm'], dict)]
        if stones_with_size:
            size_categories = {}
            clinical_notes = set()
            
            for detection in stones_with_size:
                size_data = detection['size_mm']  # Use correct field name
                category = size_data.get('size_category', 'Unknown')
                clinical = size_data.get('clinical_significance', '')
                
                size_categories[category] = size_categories.get(category, 0) + 1
                if clinical:
                    clinical_notes.add(clinical)
            
            # Size category summary
            if size_categories:
                size_text = []
                for category, count in size_categories.items():
                    size_text.append(f"{count} {category.lower()}")
                
                story.append(Paragraph(
                    f"<b>Size Analysis:</b> {', '.join(size_text)}.",
                    self.styles['Normal']
                ))
            
            # Clinical significance
            if clinical_notes:
                story.append(Paragraph(
                    f"<b>Clinical Considerations:</b>",
                    self.styles['Normal']
                ))
                for note in clinical_notes:
                    story.append(Paragraph(f"• {note}", self.styles['Normal']))
        
        # Location analysis
        if detections:
            # Calculate average confidence safely
            confidence_values = []
            for d in detections:
                conf = d.get('confidence', 0)
                try:
                    if isinstance(conf, str):
                        conf = float(conf)
                    confidence_values.append(float(conf))
                except (ValueError, TypeError):
                    confidence_values.append(0.0)
            
            avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
            story.append(Paragraph(
                f"<b>Detection Quality:</b> Average confidence score of {avg_confidence:.1%} indicates "
                f"{'high' if avg_confidence >= 0.7 else 'moderate' if avg_confidence >= 0.4 else 'low'} detection reliability.",
                self.styles['Normal']
            ))
        
        story.append(Spacer(1, 15))
        return story
    
    def create_comprehensive_detection_table(self, detections):
        """Create comprehensive detection data table with all available information"""
        story = []
        
        # Determine available data columns (use correct field name)
        has_size_mm = any('size_mm' in d and isinstance(d['size_mm'], dict) for d in detections)
        
        if has_size_mm:
            # Enhanced table with millimeter measurements (no pixel size column)
            table_data = [[
                'Stone #', 'Confidence', 'Millimeter Size\n(W x H, Diameter)', 
                'Size Category', 'Location\n(x, y)', 'Clinical Significance'
            ]]
            
            for i, detection in enumerate(detections, 1):
                # Handle confidence conversion from string to float
                confidence = detection.get('confidence', 0)
                try:
                    if isinstance(confidence, str):
                        confidence = float(confidence)
                    confidence_display = f"{confidence:.1%}"
                except (ValueError, TypeError):
                    confidence_display = str(confidence)
                
                # Handle width/height conversion
                try:
                    width = float(detection.get('width', 0))
                    height = float(detection.get('height', 0))
                    pixel_size_text = f"{width:.0f} x {height:.0f}"  # Use plain 'x'
                except (ValueError, TypeError):
                    pixel_size_text = f"{detection.get('width', 'N/A')} x {detection.get('height', 'N/A')}"
                
                # Handle center location
                center = detection.get('center', [0, 0])
                try:
                    if isinstance(center, list) and len(center) >= 2:
                        location_text = f"({center[0]:.0f}, {center[1]:.0f})"
                    else:
                        location_text = str(center)
                except (ValueError, TypeError):
                    location_text = str(center)
                
                # Size data (using the correct field name from detection data)
                size_mm = detection.get('size_mm', {})  # This is the correct field name
                if isinstance(size_mm, dict):
                    width_mm = size_mm.get('width_mm', 'N/A')
                    height_mm = size_mm.get('height_mm', 'N/A')
                    diameter_mm = size_mm.get('diameter_mm', 'N/A')
                    category = size_mm.get('size_category', 'Unknown')
                    clinical = size_mm.get('clinical_significance', 'N/A')
                    
                    # Use plain text characters and format for table bounds
                    if width_mm != 'N/A':
                        mm_size_text = f"{width_mm} x {height_mm}\nD: {diameter_mm}mm"
                    else:
                        mm_size_text = 'N/A'
                    
                    # Truncate category to fit in table
                    if len(str(category)) > 15:
                        category = str(category)[:15] + "..."
                    
                    # Truncate clinical significance to fit in table
                    if len(str(clinical)) > 35:
                        clinical = str(clinical)[:35] + "..."
                else:
                    mm_size_text = 'N/A'
                    category = 'Unknown'
                    clinical = 'Size calculation failed'
                
                # Create Paragraph objects for better text wrapping
                category_para = Paragraph(str(category), self.styles['TableText']) if len(str(category)) > 15 else str(category)
                clinical_para = Paragraph(str(clinical), self.styles['TableText']) if len(str(clinical)) > 30 else str(clinical)
                
                table_data.append([
                    str(i),
                    confidence_display,
                    mm_size_text,
                    category_para,
                    location_text,
                    clinical_para
                ])
            
            # Adjusted column widths without pixel size column
            col_widths = [0.5*inch, 0.8*inch, 1.4*inch, 1.2*inch, 0.8*inch, 2.3*inch]
            
        else:
            # Basic table without millimeter data
            table_data = [['Stone #', 'Confidence', 'Size (pixels)', 'Location (x, y)', 'Dimensions (W x H)']]
            
            for i, detection in enumerate(detections, 1):
                bbox = detection.get('bbox', [0, 0, 0, 0])
                center = detection.get('center', [0, 0])
                size = detection.get('size', 0)
                width = detection.get('width', 0)
                height = detection.get('height', 0)
                confidence = detection.get('confidence', 0)
                
                # Handle confidence conversion
                try:
                    if isinstance(confidence, str):
                        confidence = float(confidence)
                    confidence_display = f"{confidence:.1%}"
                except (ValueError, TypeError):
                    confidence_display = str(confidence)
                
                # Handle numeric conversions safely
                try:
                    size_display = f"{float(size):.0f}"
                except (ValueError, TypeError):
                    size_display = str(size)
                
                try:
                    width_display = f"{float(width):.0f}"
                    height_display = f"{float(height):.0f}"
                    dimensions_display = f"{width_display} x {height_display}"  # Use plain 'x' instead of ×
                except (ValueError, TypeError):
                    dimensions_display = f"{width} x {height}"  # Use plain 'x' instead of ×
                
                try:
                    if isinstance(center, list) and len(center) >= 2:
                        location_display = f"({center[0]:.0f}, {center[1]:.0f})"
                    else:
                        location_display = str(center)
                except (ValueError, TypeError):
                    location_display = str(center)
                
                table_data.append([
                    str(i),
                    confidence_display,
                    size_display,
                    location_display,
                    dimensions_display
                ])
            
            col_widths = [0.7*inch, 1*inch, 1.2*inch, 1.5*inch, 1.5*inch]
        
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightblue]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),  # Enable word wrapping
            ('ROWHEIGHT', (0, 1), (-1, -1), None),  # Allow auto row height
        ]))
        
        story.append(table)
        return story
    
    def create_technical_details(self, results_data):
        """Create technical details section"""
        story = []
        
        story.append(Paragraph("Technical Details", self.styles['SectionHeader']))
        
        # Analysis parameters
        tech_data = []
        
        if 'confidence_threshold' in results_data:
            tech_data.append(['Confidence Threshold:', f"{results_data['confidence_threshold']:.2f}"])
        
        if 'model_path' in results_data:
            model_name = os.path.basename(results_data['model_path'])
            tech_data.append(['Model Used:', model_name])
        
        if 'original_image_path' in results_data:
            image_name = os.path.basename(results_data['original_image_path'])
            tech_data.append(['Input Image:', image_name])
        
        tech_data.append(['Analysis Timestamp:', results_data.get('timestamp', 'N/A')[:19]])
        tech_data.append(['Processing Status:', results_data.get('status', 'Unknown')])
        
        if tech_data:
            table = Table(tech_data, colWidths=[2*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
        
        story.append(Spacer(1, 20))
        return story
    
    def create_images_section(self, results_data):
        """Create images section with side-by-side original and inference result images"""
        story = []
        
        story.append(Paragraph("Image Analysis", self.styles['SectionHeader']))
        
        # Check if we have both images for side-by-side comparison
        original_path = results_data.get('original_image_path', '')
        output_path = results_data.get('annotated_image_path', '') or results_data.get('output_image_path', '')
        
        has_original = original_path and os.path.exists(original_path)
        has_output = output_path and os.path.exists(output_path)
        
        if has_original and has_output:
            # Create side-by-side image comparison
            story.extend(self.create_side_by_side_images(original_path, output_path))
        else:
            # Fallback to individual images if both aren't available
            if has_original:
                story.append(Paragraph("Original Image", self.styles['SubSection']))
                try:
                    img = self.resize_image_for_pdf(original_path, max_width=4*inch, max_height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
                except Exception as e:
                    story.append(Paragraph(f"Error loading original image: {str(e)}", self.styles['Normal']))
            
            if has_output:
                story.append(Paragraph("Inference Result Image", self.styles['SubSection']))
                try:
                    img = self.resize_image_for_pdf(output_path, max_width=4*inch, max_height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
                except Exception as e:
                    story.append(Paragraph(f"Error loading inference result image: {str(e)}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def create_side_by_side_images(self, original_path, output_path):
        """Create side-by-side image comparison table"""
        story = []
        
        try:
            # Resize both images to same dimensions for comparison
            max_img_width = 3.5 * inch
            max_img_height = 2.5 * inch
            
            original_img = self.resize_image_for_pdf(original_path, max_img_width, max_img_height)
            output_img = self.resize_image_for_pdf(output_path, max_img_width, max_img_height)
            
            # Create headers
            headers = [
                Paragraph("<b>Original Image</b>", self.styles['MetricValue']),
                Paragraph("<b>Inference Result</b>", self.styles['MetricValue'])
            ]
            
            # Create image comparison table
            image_table_data = [
                headers,
                [original_img, output_img]
            ]
            
            image_table = Table(image_table_data, colWidths=[3.7*inch, 3.7*inch])
            image_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            story.append(image_table)
            story.append(Spacer(1, 15))
            
        except Exception as e:
            story.append(Paragraph(f"Error creating side-by-side comparison: {str(e)}", self.styles['Normal']))
            story.append(Spacer(1, 10))
        
        return story
    
    def create_metrics_section(self, results_data):
        """Create metrics section for training/testing results"""
        story = []
        
        metrics = results_data.get('metrics', {})
        if not metrics:
            return story
        
        story.append(Paragraph("Performance Metrics", self.styles['SectionHeader']))
        
        # Create metrics table
        metrics_data = [['Metric', 'Value']]
        
        for key, value in metrics.items():
            if key != 'note':
                # Format the key to be more readable
                formatted_key = key.replace('_', ' ').replace('final ', '').title()
                formatted_value = str(value) if value != 'N/A' else 'N/A'
                
                if isinstance(value, (int, float)) and value != 'N/A':
                    if key in ['precision', 'recall', 'map50', 'map50_95']:
                        formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = f"{value:.4f}"
                
                metrics_data.append([formatted_key, formatted_value])
        
        if 'note' in metrics:
            story.append(Paragraph(f"Note: {metrics['note']}", self.styles['Normal']))
            story.append(Spacer(1, 10))
        
        if len(metrics_data) > 1:
            table = Table(metrics_data, colWidths=[2.5*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightblue])
            ]))
            
            story.append(table)
        
        story.append(Spacer(1, 20))
        return story
    
    def create_appendix(self, results_data):
        """Create appendix with raw data"""
        story = []
        
        story.append(Paragraph("Appendix - Raw Data", self.styles['SectionHeader']))
        
        # Add raw results as formatted text
        raw_data = json.dumps(results_data, indent=2, default=str)
        
        # Truncate if too long
        if len(raw_data) > 2000:
            raw_data = raw_data[:2000] + "\n... (truncated)"
        
        story.append(Paragraph("Raw Analysis Results:", self.styles['SubSection']))
        story.append(Paragraph(f"<font name='Courier'>{raw_data}</font>", self.styles['Normal']))
        
        return story
    
    def resize_image_for_pdf(self, image_path, max_width=4*inch, max_height=3*inch):
        """Resize image to fit in PDF while maintaining aspect ratio"""
        try:
            # Open image to get dimensions
            with PILImage.open(image_path) as pil_img:
                orig_width, orig_height = pil_img.size
            
            # Calculate aspect ratio
            aspect_ratio = orig_width / orig_height
            
            # Determine new dimensions
            if orig_width > orig_height:
                new_width = min(max_width, orig_width)
                new_height = new_width / aspect_ratio
                if new_height > max_height:
                    new_height = max_height
                    new_width = new_height * aspect_ratio
            else:
                new_height = min(max_height, orig_height)
                new_width = new_height * aspect_ratio
                if new_width > max_width:
                    new_width = max_width
                    new_height = new_width / aspect_ratio
            
            # Create ReportLab Image object
            img = Image(image_path, width=new_width, height=new_height)
            return img
            
        except Exception as e:
            # Return error paragraph if image can't be loaded
            return Paragraph(f"Error loading image: {str(e)}", self.styles['Normal'])
    
    def get_analysis_type(self, results_data):
        """Determine the type of analysis from results data"""
        if 'detections' in results_data:
            return "Kidney Stone Detection (Inference)"
        elif 'epochs' in results_data:
            return "Model Training"
        elif 'num_test_images' in results_data:
            return "Model Testing"
        else:
            return "Unknown Analysis Type"


# Utility function for testing
def generate_sample_report():
    """Generate a sample report for testing"""
    sample_results = {
        'status': 'completed',
        'original_image_path': 'sample_image.jpg',
        'output_image_path': 'sample_output.jpg',
        'model_path': 'kidney_stone_model.pt',
        'confidence_threshold': 0.25,
        'timestamp': datetime.now().isoformat(),
        'detections': [
            {
                'bbox': [100, 150, 200, 250],
                'center': [150, 200],
                'size': 10000,
                'width': 100,
                'height': 100,
                'confidence': 0.85,
                'class': 'kidney_stone'
            }
        ],
        'summary': {
            'stones_detected': True,
            'num_stones': 1,
            'avg_confidence': 0.85
        }
    }
    
    generator = PDFReportGenerator()
    output_path = "sample_kidney_stone_report.pdf"
    generator.generate_report(sample_results, output_path)
    print(f"Sample report generated: {output_path}")


if __name__ == "__main__":
    generate_sample_report()