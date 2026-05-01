import sys
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

# Import model and device from main
from main import UNet, device, MODEL_PATH

class ImageAnalysisTab(QWidget):
    def __init__(self, model, device, image_data, tab_index, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.image_data = image_data
        self.tab_index = tab_index
        self.current_pred_raw = None
        
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
        orig_font = QFont()
        orig_font.setBold(True)
        orig_title.setFont(orig_font)
        orig_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(orig_title)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(280, 280)
        right_layout.addWidget(self.img_label)

        # Heatmap below
        heatmap_title = QLabel("Heatmap")
        heatmap_font = QFont()
        heatmap_font.setBold(True)
        heatmap_title.setFont(heatmap_font)
        heatmap_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(heatmap_title)

        fig = Figure(figsize=(3.5, 3.5), dpi=100)
        fig.patch.set_facecolor('#292929')
        self.heatmap_canvas = FigureCanvas(fig)
        self.heatmap_canvas.setStyleSheet("background-color: #292929;")
        right_layout.addWidget(self.heatmap_canvas)

        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 2.5)        
        self.setLayout(layout)
        
        # Display original image
        self.display_image(self.image_data['image'], self.img_label)

        self.predict()
    
    def predict(self):
        if self.image_data['image'] is None:
            return
        
        # Preprocess
        img_tensor = torch.tensor(self.image_data['image'] / 255.0).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # Predict (keep raw predictions for heatmap)
        with torch.no_grad():
            pred_raw = self.model(img_tensor)
        
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
    
    def display_image(self, cv_image, label):
        h, w = cv_image.shape
        bytes_per_line = w
        q_img = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaledToWidth(280, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
    
    def display_heatmap(self):
        if self.current_pred_raw is None:
            return
        
        bg_color = '#292929'

        fig = self.heatmap_canvas.figure
        fig.clear()
        fig.patch.set_facecolor(bg_color)

        # Create main axis manually positioned
        ax = fig.add_axes([0.25, 0.1, 0.5, 0.8])  # [left, bottom, width, height] [0.1, 0.1, 0.65, 0.8]
        ax.set_facecolor(bg_color)

        im = ax.imshow(self.current_pred_raw, cmap='hot', vmin=0, vmax=1)
        ax.axis('off')

        # Create separate colorbar axis
        cax = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        fig.colorbar(im, cax=cax, label='Confidence')

        self.heatmap_canvas.draw()

class UltrasoundGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load model
        self.device = device
        self.model = UNet().to(self.device)
        
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print("Model loaded successfully!")
        else:
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