import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import os
import threading
from skimage import restoration, measure, filters, morphology, segmentation
from scipy import ndimage, signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

class AdvancedImageRestorationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chương trình Khôi phục Ảnh AI Nâng cao")
        self.root.geometry("1600x1000")
        
        # Biến lưu trữ ảnh
        self.original_image = None
        self.processed_image = None
        self.current_image_path = None
        self.processing_steps = []
        
        # Thiết lập giao diện
        self.setup_ui()
        
    def setup_ui(self):
        # Frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame điều khiển
        control_frame = ttk.LabelFrame(main_frame, text="Điều khiển", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Nút chọn ảnh
        ttk.Button(control_frame, text="Chọn Ảnh", command=self.load_image).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(control_frame, text="Lưu Ảnh", command=self.save_image).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(control_frame, text="So sánh Chi tiết", command=self.detailed_comparison).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(control_frame, text="Xử lý Hàng loạt", command=self.batch_process).grid(row=0, column=3, padx=(0, 10))
        
        # Frame tùy chọn xử lý nâng cao
        processing_frame = ttk.LabelFrame(main_frame, text="Xử lý AI Nâng cao", padding="10")
        processing_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Preset modes
        ttk.Label(processing_frame, text="Chế độ:").grid(row=0, column=0, sticky=tk.W)
        self.preset_mode = tk.StringVar(value="Tự động")
        mode_combo = ttk.Combobox(processing_frame, textvariable=self.preset_mode, width=15,
                                 values=["Tự động", "Ảnh cũ", "Tài liệu", "Ảnh chân dung", "Tùy chỉnh"])
        mode_combo.grid(row=0, column=1, sticky=tk.W)
        mode_combo.bind('<<ComboboxSelected>>', self.on_preset_change)
        
        # AI Khử nhiễu
        ttk.Label(processing_frame, text="AI Khử nhiễu:").grid(row=1, column=0, sticky=tk.W)
        self.ai_denoise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(processing_frame, variable=self.ai_denoise_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(processing_frame, text="Cường độ:").grid(row=1, column=2, padx=(20, 5))
        self.ai_denoise_strength = tk.DoubleVar(value=0.8)
        ttk.Scale(processing_frame, from_=0.1, to=2.0, variable=self.ai_denoise_strength, 
                 orient=tk.HORIZONTAL, length=120).grid(row=1, column=3)
        
        # Khử mờ nâng cao
        ttk.Label(processing_frame, text="Khử mờ AI:").grid(row=2, column=0, sticky=tk.W)
        self.ai_deblur_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(processing_frame, variable=self.ai_deblur_var).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(processing_frame, text="Kernel:").grid(row=2, column=2, padx=(20, 5))
        self.deblur_kernel = tk.StringVar(value="Adaptive")
        ttk.Combobox(processing_frame, textvariable=self.deblur_kernel, width=10,
                    values=["Adaptive", "Gaussian", "Motion", "Defocus"]).grid(row=2, column=3)
        
        # Super Resolution AI
        ttk.Label(processing_frame, text="Super Resolution:").grid(row=3, column=0, sticky=tk.W)
        self.super_res_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(processing_frame, variable=self.super_res_var).grid(row=3, column=1, sticky=tk.W)
        
        ttk.Label(processing_frame, text="Tỉ lệ:").grid(row=3, column=2, padx=(20, 5))
        self.scale_factor = tk.StringVar(value="2x")
        ttk.Combobox(processing_frame, textvariable=self.scale_factor, width=10,
                    values=["2x", "3x", "4x"]).grid(row=3, column=3)
        
        # Khôi phục màu sắc
        ttk.Label(processing_frame, text="Khôi phục màu:").grid(row=4, column=0, sticky=tk.W)
        self.color_restore_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(processing_frame, variable=self.color_restore_var).grid(row=4, column=1, sticky=tk.W)
        
        ttk.Label(processing_frame, text="Cường độ:").grid(row=4, column=2, padx=(20, 5))
        self.color_strength = tk.DoubleVar(value=1.2)
        ttk.Scale(processing_frame, from_=0.5, to=2.0, variable=self.color_strength, 
                 orient=tk.HORIZONTAL, length=120).grid(row=4, column=3)
        
        # Khôi phục cấu trúc
        ttk.Label(processing_frame, text="Khôi phục cấu trúc:").grid(row=5, column=0, sticky=tk.W)
        self.structure_restore_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(processing_frame, variable=self.structure_restore_var).grid(row=5, column=1, sticky=tk.W)
        
        ttk.Label(processing_frame, text="Độ mạnh:").grid(row=5, column=2, padx=(20, 5))
        self.structure_strength = tk.DoubleVar(value=0.6)
        ttk.Scale(processing_frame, from_=0.1, to=1.0, variable=self.structure_strength, 
                 orient=tk.HORIZONTAL, length=120).grid(row=5, column=3)
        
        # Nút xử lý
        process_btn = ttk.Button(processing_frame, text="🚀 Xử lý AI", command=self.process_image)
        process_btn.grid(row=6, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        reset_btn = ttk.Button(processing_frame, text="🔄 Reset", command=self.reset_parameters)
        reset_btn.grid(row=6, column=2, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Progress bar với thông tin
        self.progress = ttk.Progressbar(processing_frame, mode='determinate')
        self.progress.grid(row=7, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_label = ttk.Label(processing_frame, text="Sẵn sàng")
        self.progress_label.grid(row=8, column=0, columnspan=4)
        
        # Frame hiển thị ảnh
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Ảnh gốc
        original_frame = ttk.LabelFrame(image_frame, text="Ảnh Gốc", padding="5")
        original_frame.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.original_canvas = tk.Canvas(original_frame, width=750, height=500, bg='white')
        self.original_canvas.grid(row=0, column=0)
        
        # Ảnh xử lý
        processed_frame = ttk.LabelFrame(image_frame, text="Ảnh Sau Xử lý AI", padding="5")
        processed_frame.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.processed_canvas = tk.Canvas(processed_frame, width=750, height=500, bg='white')
        self.processed_canvas.grid(row=0, column=0)
        
        # Thông tin ảnh
        info_frame = ttk.LabelFrame(main_frame, text="Thông tin Xử lý", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=6, width=100)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        # Cấu hình grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)
        
    def on_preset_change(self, event=None):
        """Thay đổi tham số theo preset"""
        mode = self.preset_mode.get()
        
        if mode == "Ảnh cũ":
            self.ai_denoise_strength.set(1.2)
            self.color_strength.set(1.5)
            self.structure_strength.set(0.8)
            self.deblur_kernel.set("Gaussian")
            
        elif mode == "Tài liệu":
            self.ai_denoise_strength.set(0.6)
            self.color_strength.set(1.0)
            self.structure_strength.set(0.9)
            self.deblur_kernel.set("Adaptive")
            
        elif mode == "Ảnh chân dung":
            self.ai_denoise_strength.set(0.4)
            self.color_strength.set(1.3)
            self.structure_strength.set(0.5)
            self.deblur_kernel.set("Defocus")
            
        elif mode == "Tự động":
            self.ai_denoise_strength.set(0.8)
            self.color_strength.set(1.2)
            self.structure_strength.set(0.6)
            self.deblur_kernel.set("Adaptive")
    
    def load_image(self):
        """Tải ảnh từ file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.original_image = cv2.imread(file_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.display_image(self.original_image, self.original_canvas)
                
                # Phân tích ảnh tự động
                self.analyze_image()
                
                messagebox.showinfo("Thành công", "Đã tải ảnh thành công!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")
    
    def analyze_image(self):
        """Phân tích ảnh và đưa ra khuyến nghị"""
        if self.original_image is None:
            return
            
        h, w = self.original_image.shape[:2]
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        
        # Tính toán các chỉ số chất lượng
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_score = np.std(gray)
        contrast_score = gray.std()
        brightness_score = gray.mean()
        
        info = f"=== PHÂN TÍCH ẢNH ===\n"
        info += f"Kích thước: {w}x{h}\n"
        info += f"Độ mờ: {blur_score:.2f} {'(Mờ)' if blur_score < 100 else '(Sắc nét)'}\n"
        info += f"Nhiễu: {noise_score:.2f} {'(Nhiều nhiễu)' if noise_score > 50 else '(Ít nhiễu)'}\n"
        info += f"Tương phản: {contrast_score:.2f} {'(Thấp)' if contrast_score < 40 else '(Tốt)'}\n"
        info += f"Độ sáng: {brightness_score:.2f} {'(Tối)' if brightness_score < 100 else '(Sáng)' if brightness_score > 180 else '(Vừa)'}\n"
        
        # Khuyến nghị
        info += "\n=== KHUYẾN NGHỊ ===\n"
        if blur_score < 100:
            info += "• Cần khử mờ mạnh\n"
        if noise_score > 50:
            info += "• Cần khử nhiễu cao\n"
        if contrast_score < 40:
            info += "• Cần tăng tương phản\n"
        if brightness_score < 100:
            info += "• Cần tăng độ sáng\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
    
    def display_image(self, image, canvas):
        """Hiển thị ảnh trên canvas"""
        if image is None:
            return
            
        # Resize ảnh để fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 750, 500
        
        h, w = image.shape[:2]
        scale = min(canvas_width/w, canvas_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        if len(image.shape) == 3:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            pil_image = Image.fromarray(resized)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            pil_image = Image.fromarray(resized).convert('RGB')
        
        photo = ImageTk.PhotoImage(pil_image)
        
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=photo)
        canvas.image = photo
    
    def advanced_ai_denoise(self, image):
        """Khử nhiễu AI tiên tiến"""
        strength = self.ai_denoise_strength.get()
        
        # Chuyển đổi sang float32
        img_float = image.astype(np.float32) / 255.0
        
        # Áp dụng BM3D-like denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 
                                                   h=strength*10, 
                                                   hColor=strength*10, 
                                                   templateWindowSize=7, 
                                                   searchWindowSize=21)
        
        # Wavelet denoising
        from skimage.restoration import denoise_wavelet
        denoised_wavelet = denoise_wavelet(img_float, method='BayesShrink', 
                                          mode='soft', rescale_sigma=True)
        denoised_wavelet = (denoised_wavelet * 255).astype(np.uint8)
        
        # Kết hợp hai phương pháp
        result = cv2.addWeighted(denoised, 0.6, denoised_wavelet, 0.4, 0)
        
        # Edge-preserving filter
        result = cv2.edgePreservingFilter(result, flags=2, sigma_s=50, sigma_r=0.4)
        
        return result
    
    def ai_deblur_advanced(self, image):
        """Khử mờ AI nâng cao"""
        kernel_type = self.deblur_kernel.get()
        
        # Tạo kernel dựa trên loại blur
        if kernel_type == "Gaussian":
            kernel = cv2.getGaussianKernel(15, 3)
            kernel = kernel @ kernel.T
        elif kernel_type == "Motion":
            kernel = np.zeros((15, 15))
            kernel[7, :] = 1
            kernel = kernel / 15
        elif kernel_type == "Defocus":
            kernel = np.zeros((15, 15))
            y, x = np.ogrid[-7:8, -7:8]
            mask = x**2 + y**2 <= 7**2
            kernel[mask] = 1
            kernel = kernel / np.sum(kernel)
        else:  # Adaptive
            # Ước tính kernel từ ảnh
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            kernel = restoration.unsupervised_wiener(gray, gray)[1]
        
        # Wiener deconvolution
        deblurred_channels = []
        for i in range(3):
            channel = image[:, :, i]
            deblurred = restoration.wiener(channel, kernel, balance=0.1)
            deblurred_channels.append(deblurred)
        
        deblurred = np.stack(deblurred_channels, axis=2)
        deblurred = np.clip(deblurred * 255, 0, 255).astype(np.uint8)
        
        # Unsharp masking thông minh
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Kết hợp kết quả
        result = cv2.addWeighted(deblurred, 0.7, unsharp, 0.3, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def ai_super_resolution(self, image):
        """Super Resolution AI tiên tiến"""
        scale = int(self.scale_factor.get()[0])
        
        h, w = image.shape[:2]
        
        # EDSR-inspired super resolution
        # Bước 1: Upscale bằng LANCZOS4
        new_h, new_w = h * scale, w * scale
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Bước 2: Residual enhancement
        # Tạo high-frequency features
        blur = cv2.GaussianBlur(upscaled, (5, 5), 1.0)
        high_freq = cv2.subtract(upscaled, blur)
        
        # Tăng cường high-frequency
        enhanced_hf = cv2.addWeighted(high_freq, 2.0, high_freq, 0, 0)
        
        # Kết hợp lại
        result = cv2.add(blur, enhanced_hf)
        
        # Bước 3: Edge enhancement
        # Sobel edge detection
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        
        # Tăng cường edges
        edges_normalized = (edges / edges.max() * 255).astype(np.uint8)
        edges_colored = cv2.applyColorMap(edges_normalized, cv2.COLORMAP_JET)
        
        # Blend với ảnh gốc
        result = cv2.addWeighted(result, 0.9, edges_colored, 0.1, 0)
        
        # Bước 4: Adaptive sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.2
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.addWeighted(result, 0.8, sharpened, 0.2, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def ai_color_restoration(self, image):
        """Khôi phục màu sắc AI"""
        strength = self.color_strength.get()
        
        # Chuyển sang LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Tăng cường kênh L (lightness)
        l = cv2.equalizeHist(l)
        
        # Tăng cường kênh A và B (chrominance)
        a = cv2.addWeighted(a, strength, a, 0, 0)
        b = cv2.addWeighted(b, strength, b, 0, 0)
        
        # Kết hợp lại
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Điều chỉnh saturation
        hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.addWeighted(s, strength, s, 0, 0)
        enhanced_hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def ai_structure_restoration(self, image):
        """Khôi phục cấu trúc AI"""
        strength = self.structure_strength.get()
        
        # Morphological operations để khôi phục cấu trúc
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Opening để loại bỏ noise nhỏ
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Closing để lấp đầy gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Gradient morphology để tăng cường edges
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        
        # Kết hợp với ảnh gốc
        result = cv2.addWeighted(closed, 1-strength, gradient, strength, 0)
        
        # Tophat để tăng cường chi tiết sáng
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        result = cv2.add(result, tophat)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def reset_parameters(self):
        """Reset tất cả tham số về mặc định"""
        self.ai_denoise_strength.set(0.8)
        self.color_strength.set(1.2)
        self.structure_strength.set(0.6)
        self.deblur_kernel.set("Adaptive")
        self.scale_factor.set("2x")
        self.preset_mode.set("Tự động")
    
    def process_image(self):
        """Xử lý ảnh theo các tùy chọn được chọn"""
        if self.original_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        
        # Bắt đầu processing
        self.progress.configure(mode='determinate')
        self.progress['value'] = 0
        
        # Chạy xử lý trong thread riêng
        thread = threading.Thread(target=self._process_image_thread)
        thread.start()
    
    def _process_image_thread(self):
        """Xử lý ảnh trong thread riêng"""
        try:
            result = self.original_image.copy()
            total_steps = 5
            current_step = 0
            
            # Bước 1: AI Khử nhiễu
            if self.ai_denoise_var.get():
                self.root.after(0, lambda: self.update_progress(current_step/total_steps*100, "Đang khử nhiễu AI..."))
                result = self.advanced_ai_denoise(result)
                current_step += 1
            
            # Bước 2: AI Khử mờ
            if self.ai_deblur_var.get():
                self.root.after(0, lambda: self.update_progress(current_step/total_steps*100, "Đang khử mờ AI..."))
                result = self.ai_deblur_advanced(result)
                current_step += 1
            
            # Bước 3: Super Resolution
            if self.super_res_var.get():
                self.root.after(0, lambda: self.update_progress(current_step/total_steps*100, "Đang nâng độ phân giải..."))
                result = self.ai_super_resolution(result)
                current_step += 1
            
            # Bước 4: Khôi phục màu sắc
            if self.color_restore_var.get():
                self.root.after(0, lambda: self.update_progress(current_step/total_steps*100, "Đang khôi phục màu sắc..."))
                result = self.ai_color_restoration(result)
                current_step += 1
            
            # Bước 5: Khôi phục cấu trúc
            if self.structure_restore_var.get():
                self.root.after(0, lambda: self.update_progress(current_step/total_steps*100, "Đang khôi phục cấu trúc..."))
                result = self.ai_structure_restoration(result)
                current_step += 1
            
            self.processed_image = result
            
            # Hoàn thành
            self.root.after(0, lambda: self.update_progress(100, "Hoàn thành!"))
            self.root.after(0, self._update_ui_after_processing)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi xử lý ảnh: {str(e)}"))
            self.root.after(0, lambda: self.update_progress(0, "Lỗi xử lý"))
    
    def update_progress(self, value, text):
        """Cập nhật progress bar"""
        self.progress['value'] = value
        self.progress_label.config(text=text)
        self.root.update_idletasks()
    
    def _update_ui_after_processing(self):
        """Cập nhật UI sau khi xử lý xong"""
        self.display_image(self.processed_image, self.processed_canvas)
        self.add_processing_info()
        messagebox.showinfo("Thành công", "Xử lý ảnh hoàn tất!")
    
    def add_processing_info(self):
        """Thêm thông tin về quá trình xử lý"""
        info = "\n=== KẾT QUẢ XỬ LÝ ===\n"
        
        if self.ai_denoise_var.get():
            info += f"✓ Khử nhiễu AI (Cường độ: {self.ai_denoise_strength.get():.1f})\n"
        
        if self.ai_deblur_var.get():
            info += f"✓ Khử mờ AI (Kernel: {self.deblur_kernel.get()})\n"
        
        if self.super_res_var.get():
            info += f"✓ Super Resolution (Tỉ lệ: {self.scale_factor.get()})\n"
        
        if self.color_restore_var.get():
            info += f"✓ Khôi phục màu sắc (Cường độ: {self.color_strength.get():.1f})\n"
        
        if self.structure_restore_var.get():
            info += f"✓ Khôi phục cấu trúc (Độ mạnh: {self.structure_strength.get():.1f})\n"
        
        # Tính toán chỉ số chất lượng
        if self.original_image is not None and self.processed_image is not None:
            psnr, ssim = self.calculate_quality_metrics()
            info += f"\n=== CHỈ SỐ CHẤT LƯỢNG ===\n"
            info += f"PSNR: {psnr:.2f} dB\n"
            info += f"SSIM: {ssim:.4f}\n"
        
        self.info_text.insert(tk.END, info)
        self.info_text.see(tk.END)
    
    def calculate_quality_metrics(self):
        """Tính toán PSNR và SSIM"""
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        
        # Resize processed image to match original if needed
        if self.processed_image.shape != self.original_image.shape:
            h, w = self.original_image.shape[:2]
            processed_resized = cv2.resize(self.processed_image, (w, h))
        else:
            processed_resized = self.processed_image
        
        # Convert to grayscale for SSIM calculation
        orig_gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        proc_gray = cv2.cvtColor(processed_resized, cv2.COLOR_RGB2GRAY)
        
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(self.original_image, processed_resized)
        
        # Calculate SSIM
        ssim = structural_similarity(orig_gray, proc_gray)
        
        return psnr, ssim
    
    def save_image(self):
        """Lưu ảnh đã xử lý"""
        if self.processed_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh được xử lý!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Lưu ảnh",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Convert RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, bgr_image)
                messagebox.showinfo("Thành công", f"Đã lưu ảnh tại: {file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {str(e)}")
    
    def detailed_comparison(self):
        """Hiển thị so sánh chi tiết"""
        if self.original_image is None or self.processed_image is None:
            messagebox.showwarning("Cảnh báo", "Cần có ảnh gốc và ảnh đã xử lý!")
            return
        
        # Tạo cửa sổ so sánh mới
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("So sánh Chi tiết")
        comparison_window.geometry("1400x800")
        
        # Frame cho histogram
        hist_frame = ttk.LabelFrame(comparison_window, text="Histogram So sánh", padding="10")
        hist_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tạo matplotlib figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        
        # Hiển thị ảnh gốc
        ax1.imshow(self.original_image)
        ax1.set_title('Ảnh Gốc')
        ax1.axis('off')
        
        # Hiển thị ảnh đã xử lý
        processed_display = self.processed_image
        if self.processed_image.shape != self.original_image.shape:
            h, w = self.original_image.shape[:2]
            processed_display = cv2.resize(self.processed_image, (w, h))
        
        ax2.imshow(processed_display)
        ax2.set_title('Ảnh Sau Xử lý')
        ax2.axis('off')
        
        # Histogram ảnh gốc
        for i, color in enumerate(['red', 'green', 'blue']):
            hist = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
            ax3.plot(hist, color=color, alpha=0.7)
        ax3.set_title('Histogram Ảnh Gốc')
        ax3.set_xlabel('Pixel Value')
        ax3.set_ylabel('Frequency')
        
        # Histogram ảnh đã xử lý
        for i, color in enumerate(['red', 'green', 'blue']):
            hist = cv2.calcHist([processed_display], [i], None, [256], [0, 256])
            ax4.plot(hist, color=color, alpha=0.7)
        ax4.set_title('Histogram Ảnh Sau Xử lý')
        ax4.set_xlabel('Pixel Value')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Nhúng matplotlib vào tkinter
        canvas = FigureCanvasTkinter(fig, hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Thêm thông tin chi tiết
        detail_frame = ttk.LabelFrame(comparison_window, text="Thông tin Chi tiết", padding="10")
        detail_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        detail_text = tk.Text(detail_frame, height=8, width=100)
        detail_text.pack(fill=tk.BOTH, expand=True)
        
        # Tính toán chi tiết
        psnr, ssim = self.calculate_quality_metrics()
        
        detail_info = f"=== PHÂN TÍCH CHI TIẾT ===\n"
        detail_info += f"PSNR: {psnr:.2f} dB (>30dB = Tốt, >40dB = Rất tốt)\n"
        detail_info += f"SSIM: {ssim:.4f} (>0.8 = Tốt, >0.9 = Rất tốt)\n"
        
        # Phân tích histogram
        orig_mean = np.mean(self.original_image)
        proc_mean = np.mean(processed_display)
        orig_std = np.std(self.original_image)
        proc_std = np.std(processed_display)
        
        detail_info += f"\n=== THỐNG KÊ PIXEL ===\n"
        detail_info += f"Độ sáng trung bình: {orig_mean:.1f} → {proc_mean:.1f}\n"
        detail_info += f"Độ lệch chuẩn: {orig_std:.1f} → {proc_std:.1f}\n"
        detail_info += f"Thay đổi tương phản: {((proc_std - orig_std) / orig_std * 100):.1f}%\n"
        
        detail_text.insert(tk.END, detail_info)
    
    def batch_process(self):
        """Xử lý hàng loạt ảnh"""
        input_folder = filedialog.askdirectory(title="Chọn thư mục ảnh đầu vào")
        if not input_folder:
            return
        
        output_folder = filedialog.askdirectory(title="Chọn thư mục đầu ra")
        if not output_folder:
            return
        
        # Tìm tất cả ảnh trong thư mục
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        image_files = []
        
        for file in os.listdir(input_folder):
            if file.lower().endswith(image_extensions):
                image_files.append(file)
        
        if not image_files:
            messagebox.showwarning("Cảnh báo", "Không tìm thấy ảnh nào trong thư mục!")
            return
        
        # Tạo cửa sổ progress cho batch processing
        batch_window = tk.Toplevel(self.root)
        batch_window.title("Xử lý Hàng loạt")
        batch_window.geometry("600x300")
        
        ttk.Label(batch_window, text=f"Đang xử lý {len(image_files)} ảnh...").pack(pady=10)
        
        batch_progress = ttk.Progressbar(batch_window, mode='determinate', length=500)
        batch_progress.pack(pady=10)
        
        batch_label = ttk.Label(batch_window, text="Chuẩn bị...")
        batch_label.pack(pady=5)
        
        batch_text = tk.Text(batch_window, height=10, width=70)
        batch_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Chạy batch processing trong thread riêng
        def batch_process_thread():
            try:
                total_files = len(image_files)
                processed_count = 0
                
                for i, filename in enumerate(image_files):
                    input_path = os.path.join(input_folder, filename)
                    output_path = os.path.join(output_folder, f"restored_{filename}")
                    
                    # Cập nhật progress
                    progress = (i / total_files) * 100
                    batch_window.after(0, lambda p=progress, f=filename: [
                        batch_progress.config(value=p),
                        batch_label.config(text=f"Đang xử lý: {f}")
                    ])
                    
                    try:
                        # Load và xử lý ảnh
                        image = cv2.imread(input_path)
                        if image is None:
                            continue
                        
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Áp dụng các bước xử lý
                        result = image_rgb.copy()
                        
                        if self.ai_denoise_var.get():
                            result = self.advanced_ai_denoise(result)
                        
                        if self.ai_deblur_var.get():
                            result = self.ai_deblur_advanced(result)
                        
                        if self.super_res_var.get():
                            result = self.ai_super_resolution(result)
                        
                        if self.color_restore_var.get():
                            result = self.ai_color_restoration(result)
                        
                        if self.structure_restore_var.get():
                            result = self.ai_structure_restoration(result)
                        
                        # Lưu kết quả
                        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(output_path, result_bgr)
                        
                        processed_count += 1
                        batch_window.after(0, lambda f=filename: batch_text.insert(tk.END, f"✓ Hoàn thành: {f}\n"))
                        
                    except Exception as e:
                        batch_window.after(0, lambda f=filename, err=str(e): batch_text.insert(tk.END, f"✗ Lỗi {f}: {err}\n"))
                
                # Hoàn thành
                batch_window.after(0, lambda: [
                    batch_progress.config(value=100),
                    batch_label.config(text=f"Hoàn thành! Đã xử lý {processed_count}/{total_files} ảnh"),
                    batch_text.insert(tk.END, f"\n=== HOÀN THÀNH ===\nĐã xử lý: {processed_count}/{total_files} ảnh\n")
                ])
                
            except Exception as e:
                batch_window.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi xử lý hàng loạt: {str(e)}"))
        
        thread = threading.Thread(target=batch_process_thread)
        thread.start()
    
    def enhance_with_ai_models(self, image):
        """Tích hợp các mô hình AI tiên tiến (placeholder for future implementation)"""
        # Đây là nơi có thể tích hợp các mô hình AI như:
        # - Real-ESRGAN
        # - SwinIR
        # - BSRGAN
        # - CodeFormer (cho ảnh chân dung)
        
        # Hiện tại sử dụng các kỹ thuật xử lý ảnh truyền thống
        # Trong tương lai có thể load pre-trained models
        
        pass
    
    def create_processing_pipeline(self):
        """Tạo pipeline xử lý tùy chỉnh"""
        pipeline = []
        
        if self.ai_denoise_var.get():
            pipeline.append(("AI Denoise", self.advanced_ai_denoise))
        
        if self.ai_deblur_var.get():
            pipeline.append(("AI Deblur", self.ai_deblur_advanced))
        
        if self.super_res_var.get():
            pipeline.append(("Super Resolution", self.ai_super_resolution))
        
        if self.color_restore_var.get():
            pipeline.append(("Color Restoration", self.ai_color_restoration))
        
        if self.structure_restore_var.get():
            pipeline.append(("Structure Restoration", self.ai_structure_restoration))
        
        return pipeline
    
    def export_processing_report(self):
        """Xuất báo cáo xử lý"""
        if self.original_image is None or self.processed_image is None:
            messagebox.showwarning("Cảnh báo", "Cần có ảnh gốc và ảnh đã xử lý!")
            return
        
        report_path = filedialog.asksaveasfilename(
            title="Lưu báo cáo",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if report_path:
            try:
                psnr, ssim = self.calculate_quality_metrics()
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("=== BÁO CÁO XỬ LÝ ẢNH AI ===\n")
                    f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Ảnh gốc: {self.current_image_path}\n")
                    f.write(f"Kích thước gốc: {self.original_image.shape}\n")
                    f.write(f"Kích thước sau xử lý: {self.processed_image.shape}\n\n")
                    
                    f.write("=== CÁC BƯỚC XỬ LÝ ===\n")
                    if self.ai_denoise_var.get():
                        f.write(f"- AI Khử nhiễu (Cường độ: {self.ai_denoise_strength.get()})\n")
                    if self.ai_deblur_var.get():
                        f.write(f"- AI Khử mờ (Kernel: {self.deblur_kernel.get()})\n")
                    if self.super_res_var.get():
                        f.write(f"- Super Resolution (Tỉ lệ: {self.scale_factor.get()})\n")
                    if self.color_restore_var.get():
                        f.write(f"- Khôi phục màu sắc (Cường độ: {self.color_strength.get()})\n")
                    if self.structure_restore_var.get():
                        f.write(f"- Khôi phục cấu trúc (Độ mạnh: {self.structure_strength.get()})\n")
                    
                    f.write(f"\n=== CHỈ SỐ CHẤT LƯỢNG ===\n")
                    f.write(f"PSNR: {psnr:.2f} dB\n")
                    f.write(f"SSIM: {ssim:.4f}\n")
                    
                    f.write(f"\n=== ĐÁNH GIÁ ===\n")
                    if psnr > 30:
                        f.write("- Chất lượng PSNR: Tốt\n")
                    elif psnr > 25:
                        f.write("- Chất lượng PSNR: Trung bình\n")
                    else:
                        f.write("- Chất lượng PSNR: Cần cải thiện\n")
                    
                    if ssim > 0.8:
                        f.write("- Chất lượng SSIM: Tốt\n")
                    elif ssim > 0.6:
                        f.write("- Chất lượng SSIM: Trung bình\n")
                    else:
                        f.write("- Chất lượng SSIM: Cần cải thiện\n")
                
                messagebox.showinfo("Thành công", f"Đã xuất báo cáo tại: {report_path}")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể xuất báo cáo: {str(e)}")

# Thêm import cần thiết
from datetime import datetime

def main():
    """Hàm main để chạy ứng dụng"""
    root = tk.Tk()
    app = AdvancedImageRestorationApp(root)
    
    # Thêm menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Mở ảnh", command=app.load_image)
    file_menu.add_command(label="Lưu ảnh", command=app.save_image)
    file_menu.add_separator()
    file_menu.add_command(label="Xử lý hàng loạt", command=app.batch_process)
    file_menu.add_separator()
    file_menu.add_command(label="Thoát", command=root.quit)
    
    # Tools menu
    tools_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Công cụ", menu=tools_menu)
    tools_menu.add_command(label="So sánh chi tiết", command=app.detailed_comparison)
    tools_menu.add_command(label="Xuất báo cáo", command=app.export_processing_report)
    tools_menu.add_command(label="Reset tham số", command=app.reset_parameters)
    
    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Trợ giúp", menu=help_menu)
    help_menu.add_command(label="Hướng dẫn", command=lambda: messagebox.showinfo(
        "Hướng dẫn", 
        "1. Chọn ảnh từ menu File\n"
        "2. Chọn chế độ xử lý phù hợp\n"
        "3. Điều chỉnh các tham số\n"
        "4. Nhấn 'Xử lý AI' để bắt đầu\n"
        "5. So sánh và lưu kết quả\n\n"
        "Chế độ:\n"
        "- Tự động: Phù hợp với hầu hết ảnh\n"
        "- Ảnh cũ: Tối ưu cho ảnh scan cũ\n"
        "- Tài liệu: Tối ưu cho văn bản\n"
        "- Ảnh chân dung: Tối ưu cho ảnh người"
    ))
    help_menu.add_command(label="Về chương trình", command=lambda: messagebox.showinfo(
        "Về chương trình", 
        "Chương trình Khôi phục Ảnh AI Nâng cao\n"
        "Phiên bản: 1.0\n"
        "Tác giả: AI Assistant\n\n"
        "Tính năng:\n"
        "✓ Khử nhiễu AI\n"
        "✓ Khử mờ thông minh\n"
        "✓ Super Resolution\n"
        "✓ Khôi phục màu sắc\n"
        "✓ Khôi phục cấu trúc\n"
        "✓ Xử lý hàng loạt\n"
        "✓ Đánh giá chất lượng PSNR/SSIM"
    ))
    
    root.mainloop()

if __name__ == "__main__":
    main()