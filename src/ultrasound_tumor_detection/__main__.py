import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QFileDialog, QProgressBar, QScrollArea, QTabWidget, QGridLayout
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from ultrasound_tumor_detection import UNet
from ultrasound_tumor_detection.pipeline import MODEL_PATH, load_checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageAnalysisTab(QWidget):
    def __init__(self, model, device, image_data, tab_index, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.image_data = image_data
        self.tab_index = tab_index
        self.current_pred_raw = None
        
        # Panning variables
        self.panning = False
        self.press_x = None
        self.press_y = None
        self.orig_xlim = None
        self.orig_ylim = None
        
        self.initUI()
    
    def initUI(self):
        layout = QHBoxLayout()
        
        # Left side - controls
        left_layout = QVBoxLayout()
        
        title = QLabel(f"Image {self.tab_index}: {self.image_data['name']}")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        left_layout.addWidget(title)
                
        # Confidence info
        self.conf_label = QLabel("Confidence: N/A")
        conf_font = QFont()
        conf_font.setPointSize(11)
        self.conf_label.setFont(conf_font)
        left_layout.addWidget(self.conf_label)

        self.reset_btn = QPushButton("Reset Zoom")
        self.reset_btn.clicked.connect(self.reset_zoom)
        left_layout.addWidget(self.reset_btn)
        
        # Confidence bar
        self.conf_bar = QProgressBar()
        self.conf_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        left_layout.addWidget(self.conf_bar)
        
        # Tumor detected label
        self.tumor_label = QLabel("Tumor Status: N/A")
        tumor_font = QFont()
        tumor_font.setPointSize(10)
        self.tumor_label.setFont(tumor_font)
        left_layout.addWidget(self.tumor_label)
        
        # Tumor pixels info
        self.pixels_label = QLabel("Tumor Pixels: N/A")
        pixels_font = QFont()
        pixels_font.setPointSize(10)
        pixels_font.setBold(True)
        self.pixels_label.setFont(pixels_font)
        self.pixels_label.setStyleSheet("color: #FF6B6B;")
        left_layout.addWidget(self.pixels_label)
        
        # Statistics
        self.stats_label = QLabel("Statistics:\nCoverage: N/A")
        left_layout.addWidget(self.stats_label)
        
        left_layout.addStretch()
        
        # Right side - original image and heatmap stacked vertically
        right_layout = QVBoxLayout()

        # Original image
        orig_title = QLabel("Raw Image")
        orig_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(orig_title)

        self.orig_canvas = FigureCanvas(Figure(figsize=(3.5, 3.5), dpi=100))
        self.orig_canvas.setStyleSheet("background-color: #292929;")
        self.orig_canvas.figure.patch.set_facecolor('#292929')
        self.orig_ax = self.orig_canvas.figure.subplots()
        self.orig_ax.set_facecolor('#292929')
        self.orig_ax.axis('off')
        right_layout.addWidget(self.orig_canvas)

        # Heatmap below
        heatmap_title = QLabel("Heatmap")
        heatmap_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(heatmap_title)

        self.heatmap_canvas = FigureCanvas(Figure(figsize=(3.5, 3.5), dpi=100))
        self.heatmap_canvas.setStyleSheet("background-color: #292929;")
        self.heatmap_canvas.figure.patch.set_facecolor('#292929')
        self.heatmap_ax = self.heatmap_canvas.figure.subplots()
        self.heatmap_ax.axis('off')
        self.heatmap_ax.set_facecolor('#292929')
        right_layout.addWidget(self.heatmap_canvas, alignment=Qt.AlignHCenter)

        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 2)
        self.setLayout(layout)

        self.display_image()
        self.predict()
        
        # Zoom selectors
        self.selector = RectangleSelector(
            self.orig_ax,
            self.on_select,
            useblit=True,
            button=[1],
            spancoords='pixels',
            interactive=True,
        )
        self.heatmap_selector = RectangleSelector(
            self.heatmap_ax,
            self.on_select,
            useblit=True,
            button=[1],
            spancoords='pixels',
            interactive=True,
        )
        
        # Connect panning events to both canvases
        self.orig_canvas.mpl_connect('button_press_event', self.on_press)
        self.orig_canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.orig_canvas.mpl_connect('button_release_event', self.on_release)
        
        self.heatmap_canvas.mpl_connect('button_press_event', self.on_press)
        self.heatmap_canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.heatmap_canvas.mpl_connect('button_release_event', self.on_release)
    
    def display_image(self):
        self.orig_ax.clear()
        self.orig_ax.set_facecolor('#292929')
        self.orig_canvas.figure.patch.set_facecolor('#292929')
        self.orig_ax.imshow(self.image_data['image'], cmap='gray', origin='upper')
        self.orig_ax.axis('off')
        self.orig_canvas.draw()

    def display_heatmap(self):
        if self.current_pred_raw is None:
            return

        self.heatmap_ax.clear()
        self.heatmap_ax.set_facecolor('#292929')
        self.heatmap_canvas.figure.patch.set_facecolor('#292929')

        im = self.heatmap_ax.imshow(self.current_pred_raw, cmap='hot', vmin=0, vmax=1, origin='upper')
        self.heatmap_ax.axis('off')

        cbar = self.heatmap_canvas.figure.colorbar(im, ax=self.heatmap_ax, fraction=0.046, pad=0.04)
        cbar.outline.set_edgecolor('#292929')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')

        self.heatmap_canvas.draw()

    def predict(self):
        if self.image_data['image'] is None:
            return
        
        # Preprocess
        img_tensor = torch.tensor(self.image_data['image'] / 255.0).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # Predict (keep raw predictions for heatmap)
        with torch.no_grad():
            pred_raw = torch.sigmoid(self.model(img_tensor)) 
        
        self.current_pred_raw = pred_raw.squeeze().cpu().numpy()
                        
        # Calculate confidence metrics
        self.update_confidence_metrics()
        
        # Display heatmap
        self.display_heatmap()
    
    def update_confidence_metrics(self):
        if self.current_pred_raw is None:
            return
        
        # Max confidence
        max_conf = np.max(self.current_pred_raw) * 100

        # Tumor area calculations
        tumor_pixels = np.sum(self.current_pred_raw > 0.65)
        total_pixels = self.current_pred_raw.shape[0] * self.current_pred_raw.shape[1]
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        
        # Update UI
        
        # Tumor status
        if tumor_pixels > 1000:
            self.tumor_label.setText(f"⚠ Tumor Detected ({tumor_percentage:.1f}%)")
            self.tumor_label.setStyleSheet("color: red; font-weight: bold;")
            self.conf_label.setText(f"Confidence: {min(tumor_pixels/10, 100):.1f}%")
            self.conf_bar.setValue(int(min(tumor_pixels/10, 100)))
        else:
            self.tumor_label.setText("✓ No Tumor Detected")
            self.tumor_label.setStyleSheet("color: green; font-weight: bold;")
            self.conf_label.setText(f"Confidence: {(100 - tumor_pixels/10):.1f}%")
            self.conf_bar.setValue(int(100 - tumor_pixels/10))

        # Tumor pixels display
        self.pixels_label.setText(f"Tumor Pixels: {int(tumor_pixels)} / {total_pixels}")
        
        # Statistics
        self.stats_label.setText(
            f"Statistics:\n"
            f"Coverage: {tumor_percentage:.2f}%\n"
            f"Max Confidence: {max_conf:.1f}%"
        )
    
    def on_select(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return

        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.zoom_region(x1, y1, x2, y2)

        # Hide the zoom box after selection
        if hasattr(self, "selector"):
            self.selector.set_visible(False)
        if hasattr(self, "heatmap_selector"):
            self.heatmap_selector.set_visible(False)

        # Disable selectors after zooming to enable panning
        if hasattr(self, "selector"):
            self.selector.set_active(False)
        if hasattr(self, "heatmap_selector"):
            self.heatmap_selector.set_active(False)

        self.orig_canvas.draw_idle()
        self.heatmap_canvas.draw_idle()
    
    def zoom_region(self, x1, y1, x2, y2):
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))

        self.orig_ax.set_xlim(xmin, xmax)
        self.orig_ax.set_ylim(ymax, ymin)
        self.heatmap_ax.set_xlim(xmin, xmax)
        self.heatmap_ax.set_ylim(ymax, ymin)

        self.orig_canvas.draw()
        self.heatmap_canvas.draw()

    def on_press(self, event):
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            # Don't pan if selectors are active (during zoom selection)
            if (hasattr(self, "selector") and self.selector.get_active()) or \
               (hasattr(self, "heatmap_selector") and self.heatmap_selector.get_active()):
                return
            
            # Check if zoomed in (not at full limits)
            if self.is_zoomed():
                self.panning = True
                self.press_x = event.xdata
                self.press_y = event.ydata
                self.orig_xlim = self.orig_ax.get_xlim()
                self.orig_ylim = self.orig_ax.get_ylim()

    def on_motion(self, event):
        if self.panning and event.xdata is not None and event.ydata is not None:
            dx = self.press_x - event.xdata
            dy = self.press_y - event.ydata
            
            new_xlim = (self.orig_xlim[0] + dx, self.orig_xlim[1] + dx)
            new_ylim = (self.orig_ylim[0] + dy, self.orig_ylim[1] + dy)
            
            self.orig_ax.set_xlim(new_xlim)
            self.orig_ax.set_ylim(new_ylim)
            self.heatmap_ax.set_xlim(new_xlim)
            self.heatmap_ax.set_ylim(new_ylim)
            
            self.orig_canvas.draw_idle()
            self.heatmap_canvas.draw_idle()

    def on_release(self, event):
        if event.button == 1:
            self.panning = False
            self.press_x = None
            self.press_y = None
            self.orig_xlim = None
            self.orig_ylim = None

    def is_zoomed(self):
        """Check if the view is zoomed in."""
        h, w = self.image_data['image'].shape
        xlim = self.orig_ax.get_xlim()
        ylim = self.orig_ax.get_ylim()
        return not (xlim == (0, w) and ylim == (h, 0))

    def reset_zoom(self):
        if self.image_data["image"] is None:
            return

        h, w = self.image_data["image"].shape
        self.orig_ax.set_xlim(0, w)
        self.orig_ax.set_ylim(h, 0)

        if self.current_pred_raw is not None:
            ph, pw = self.current_pred_raw.shape
            self.heatmap_ax.set_xlim(0, pw)
            self.heatmap_ax.set_ylim(ph, 0)

        # Re-enable selectors for zooming
        if hasattr(self, "selector"):
            self.selector.set_active(True)
        if hasattr(self, "heatmap_selector"):
            self.heatmap_selector.set_active(True)
        
        self.panning = False
        self.orig_canvas.draw()
        self.heatmap_canvas.draw()

class UltrasoundGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load model
        self.device = device
        self.model = UNet().to(self.device)
        
        if not load_checkpoint(self.model, self.device, model_path=MODEL_PATH):
            print("Warning: Model file not found. Using untrained model.")
        
        self.model.eval()
        self.images = []  # Store multiple images
        
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Ultrasound Tumor Detection - Multi-Image Analysis")
        self.setGeometry(100, 100, 1800, 1000)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout()
        
        # Top controls
        control_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)
        
        self.load_multiple_btn = QPushButton("Load Multiple Images")
        self.load_multiple_btn.clicked.connect(self.load_multiple_images)
        control_layout.addWidget(self.load_multiple_btn)
                
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        control_layout.addWidget(self.clear_btn)
        
        self.image_count_label = QLabel("Images loaded: 0")
        image_count_font = QFont()
        image_count_font.setPointSize(11)
        image_count_font.setBold(True)
        self.image_count_label.setFont(image_count_font)
        control_layout.addWidget(self.image_count_label)
        
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Tab widget for multiple images with close buttons
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        layout.addWidget(self.tabs)
        
        main_widget.setLayout(layout)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp)")
        
        if file_path:
            self.add_image(file_path)
    
    def load_multiple_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Multiple Images", "", "Image Files (*.png *.jpg *.bmp)")
        
        for file_path in file_paths:
            self.add_image(file_path)
    
    def add_image(self, file_path):
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            
            file_name = os.path.basename(file_path)
            
            image_data = {
                'image': img,
                'name': file_name
            }
            
            self.images.append(image_data)
            
            # Create new tab
            tab = ImageAnalysisTab(self.model, self.device, image_data, len(self.images))
            self.tabs.addTab(tab, file_name)
            
            # Update count
            self.image_count_label.setText(f"Images loaded: {len(self.images)}")
            
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def close_tab(self, index):
        self.tabs.removeTab(index)
        if index < len(self.images):
            self.images.pop(index)
        
        self.image_count_label.setText(f"Images loaded: {len(self.images)}")
        
    def clear_all(self):
        self.tabs.clear()
        self.images = []
        self.image_count_label.setText("Images loaded: 0")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = UltrasoundGUI()
    gui.show()
    sys.exit(app.exec_())
