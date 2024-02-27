import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageFilter, ImageOps
from tkinter import ttk
from tkinter.font import Font
from tkinterdnd2 import DND_FILES, TkinterDnD
from functools import partial
import os
import sys

# Function to get the resource path for the image
# This function is necessary because when the script is converted to an executable, the path to the resource changes
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Main application class
# This class contains the GUI and the image processing methods
# The GUI is created using the Tkinter library
# The image processing methods are implemented using the OpenCV and NumPy libraries
class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FilterX - Image Processor App") # Set window title
        self.root.state("zoomed")  # Maximize window
        self.root.config(bg="skyblue") # Set window background color

        # Variables
        self.image = None # Variable to store the uploaded image
        self.processed_images = []  # List to store processed images

        # Set font styles
        labelfont = Font(family="Lexend", size=24, weight="bold")
        buttonfont = Font(family="Lexend", size=10, weight="bold")

        # Set default font styles for all widgets
        root.option_add("*Button*Font", buttonfont)
        root.option_add("*Label*Font", labelfont)

        # Frames
        self.left_frame  =  tk.Frame(root,  width=350,  height=750,  bg='skyblue')
        self.left_frame.place(x=10,  y=10,  relx=0.01,  rely=0.01)

        self.right_frame = tk.Frame(root, width=1125, height=750, bg='lightgreen')
        self.right_frame.place(x=370, y=10, relx=0.01, rely=0.01, anchor=tk.NW)

        # Labels
        tk.Label(self.left_frame,  text="Original Image", relief=tk.GROOVE,  width=350).place(relx=0.5,  anchor=tk.N)

        # Placeholders
        original_image  =  ImageTk.PhotoImage(Image.open(resource_path("placeholder.jpg")))

        self.plimage = tk.Label(self.left_frame, width=350, height=280)
        self.plimage.place(rely=0.06, relwidth=1)
        self.plimage.config(image=original_image)
        self.plimage.image = original_image

        # Tool and Filter Bars
        tool_bar  =  tk.Frame(self.left_frame,  width=175,  height=360,  bg='silver')
        tool_bar.place(x=0,  rely=0.52)

        filter_bar  =  tk.Frame(self.left_frame,  width=175,  height=360,  bg='silver')
        filter_bar.place(x=175,  rely=0.52)

        # Labels for Tool and Filter Bars
        tk.Label(tool_bar, text="Tools", relief=tk.RIDGE, width=150, height=1, bg='gold', fg='white').place(in_=tool_bar, relx=0.5, anchor=tk.N)
        tk.Label(filter_bar, text="Filters", relief=tk.RIDGE, width=150, height=1, bg='gold', fg='white').place(in_=filter_bar, relx=0.5, anchor=tk.N)

        # Tool buttons
        tk.Button(self.left_frame, text="Upload Image", relief=tk.GROOVE  , command=self.upload_image_from_dialog, width=150, height=3).place(in_=self.left_frame, relx=0.5, rely=0.52, anchor=tk.S)
        tk.Button(tool_bar, text="Change Component", relief=tk.GROOVE  , command=self.change_component, width=150, height=3).place(in_=tool_bar, relx=0.5, rely=0.20, anchor=tk.CENTER)
        tk.Button(tool_bar, text="Invert Colors", relief=tk.GROOVE  , command=self.invert_colors, width=150, height=3).place(in_=tool_bar, relx=0.5, rely=0.35, anchor=tk.CENTER)
        tk.Button(tool_bar, text="Split RGB", relief=tk.GROOVE  , command=self.split_rgb, width=150, height=3).place(in_=tool_bar, relx=0.5, rely=0.50, anchor=tk.CENTER)
        tk.Button(tool_bar, text="Merge Images", relief=tk.GROOVE  , command=self.merge_images_with_animation, width=150, height=3).place(in_=tool_bar, relx=0.5, rely=0.65, anchor=tk.CENTER)
        tk.Button(tool_bar, text="Embed Watermark", relief=tk.GROOVE  , command=self.embed_watermark, width=150, height=3).place(in_=tool_bar, relx=0.5, rely=0.80, anchor=tk.CENTER)
        tk.Button(tool_bar, text="Extract Watermark", relief=tk.GROOVE  , command=self.extract_watermark, width=150, height=3).place(in_=tool_bar, relx=0.5, rely=0.95, anchor=tk.CENTER)

        filter_options = [
                "Blurring Filter",
                "Sharpening Filter",
                "Median Filter",
                "Erosion Filter",
                "Dilation Filter",
                "Sobel Filter"
        ]

        # Filter buttons
        tk.Button(filter_bar, text="Blurring Filter", relief=tk.GROOVE  , command=partial(self.Filter, filter_options[0]), width=150, height=3).place(in_=filter_bar, relx=0.5, rely=0.20, anchor=tk.CENTER)
        tk.Button(filter_bar, text="Sharpening Filter", relief=tk.GROOVE  , command=partial(self.Filter, filter_options[1]), width=150, height=3).place(in_=filter_bar, relx=0.5, rely=0.35, anchor=tk.CENTER)
        tk.Button(filter_bar, text="Median Filter", relief=tk.GROOVE  , command=partial(self.Filter, filter_options[2]), width=150, height=3).place(in_=filter_bar, relx=0.5, rely=0.50, anchor=tk.CENTER)
        tk.Button(filter_bar, text="Erosion Filter", relief=tk.GROOVE  , command=partial(self.Filter, filter_options[3]), width=150, height=3).place(in_=filter_bar, relx=0.5, rely=0.65, anchor=tk.CENTER)
        tk.Button(filter_bar, text="Dilation Filter", relief=tk.GROOVE  , command=partial(self.Filter, filter_options[4]), width=150, height=3).place(in_=filter_bar, relx=0.5, rely=0.80, anchor=tk.CENTER)
        tk.Button(filter_bar, text="Sobel Filter", relief=tk.GROOVE  , command=partial(self.Filter, filter_options[5]), width=150, height=3).place(in_=filter_bar, relx=0.5, rely=0.95, anchor=tk.CENTER)      

        # Drag and Drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.upload_image_from_drop)
        

    def upload_image_from_drop(self, event):
        """
        Upload image from drag and drop event
        # Arguments
            event: event object
        # Returns
            None
        """
        file_path = event.data
        self.load_image(file_path)

    def upload_image_from_dialog(self):
        """
        Upload image from file dialog
        # Arguments
            None
        # Returns
            None
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        """
        Load image from file path
        # Arguments
            file_path: str, path to the image file
        # Returns
            None
        """
        file_path = file_path.strip('{}')

        image = cv2.imread(file_path)
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        # Resize the image to fit in the label
        max_width = 350
        max_height = 280
        height, width, _ = image.shape
        ratio = min(max_width / width, max_height / height)

        # Check if resizing is necessary
        if width > max_width or height > max_height:
            # Resize the image proportionally
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = cv2.resize(image, (new_width, new_height))

        # Convert the OpenCV image to a format compatible with Tkinter
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.clear_processed_images()
        # Update the label with the new image
        self.plimage.config(image=image)
        self.plimage.image = image

    def clear_processed_images(self):
        """
        Clear the processed images from the right frame
        # Arguments
            None
        # Returns
            None
        """
        for img_label, caption_label in self.processed_images:
            img_label.grid_forget()  # Remove image label from the grid
            if caption_label:
                caption_label.grid_forget()  # Remove caption label from the grid
                caption_label.destroy()  # Destroy the caption label widget
            img_label.destroy()
        self.processed_images.clear()  # Clear the list of processed images
    
    def display_images(self, images, captions=None):
        """
        Display images in the right frame
        # Arguments
            images: list, list of images to display
            captions: list, list of captions for the images 
        # Returns
            None
        """
        num_images = len(images)

        # Calculate the width and height of the right_frame
        frame_width = self.right_frame.winfo_width()
        frame_height = self.right_frame.winfo_height()

        # Calculate the number of rows and columns based on the number of images
        if num_images <= 3:
            num_rows = 1
            num_cols = num_images
        else:
            num_cols = 3
            num_rows = (num_images + num_cols - 1) // num_cols

        # Resize images only if they exceed the frame dimensions
        resized_images = []
        for image in images:
            height, width = image.shape[:2]
            if width > frame_width / num_cols or height > frame_height / num_rows:
                ratio = min(frame_width / (width * num_cols), frame_height / (height * num_rows))
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            resized_images.append(image)

        # Display resized images
        for i, (image, caption) in enumerate(zip(resized_images, captions or [])):
            img = Image.fromarray(image)
            img = ImageTk.PhotoImage(img)

            img_label = tk.Label(self.right_frame, image=img, relief=tk.GROOVE, anchor=tk.CENTER, justify=tk.CENTER, bg=self.right_frame.cget('background'))
            img_label.image = img

            # Calculate grid row and column
            row_idx = i // num_cols
            col_idx = i % num_cols

            # Grid placement
            img_label.grid(row=row_idx, column=col_idx, padx=5, pady=5)

            # Add caption label below the image
            if caption:
                caption_label = tk.Label(self.right_frame, text=caption, anchor=tk.CENTER, justify=tk.CENTER, bg=self.right_frame.cget('background'), font=("Lexend", 18))
                caption_label.grid(row=row_idx + 1, column=col_idx, padx=5, pady=2)

                

            self.processed_images.append((img_label, caption_label))

        # Clear existing images and captions
        for label, caption_label in self.processed_images:
            label.grid_forget()
            if caption_label:
                caption_label.grid_forget()

        # Place each image in the center of the frame
        for i, (img_label, caption_label) in enumerate(self.processed_images):
            row_idx = i // num_cols
            col_idx = i % num_cols
            x_offset = col_idx * (frame_width // num_cols)
            y_offset = row_idx * (frame_height // num_rows)

            # Calculate the x offset to center the image horizontally
            x_offset += (frame_width // num_cols - img_label.winfo_reqwidth()) // 2
            # Calculate the y offset to center the image vertically
            y_offset += (frame_height // num_rows - img_label.winfo_reqheight()) // 2 - ((frame_height // num_rows - img_label.winfo_reqheight()) // 2 if num_rows > 1 else 0)

            img_label.place(x=x_offset, y=y_offset)

            if caption_label:
                # Adjust the x and y offsets for the caption label
                caption_width = caption_label.winfo_reqwidth()
                caption_height = caption_label.winfo_reqheight()
                caption_x_offset = col_idx * (frame_width // num_cols) + (frame_width // num_cols - caption_width) // 2
                caption_y_offset = row_idx * (frame_height // num_rows) + frame_height // num_rows - (frame_height // num_rows - img_label.winfo_reqheight()) // 2 - (caption_height // 2 if num_rows > 1 else 0) + 5
                caption_label.place(x=caption_x_offset, y=caption_y_offset)

    # Image processing methods
    # These methods are called when the corresponding buttons are clicked
    def invert_colors(self):
        if self.image is not None:
            processed_image = cv2.bitwise_not(self.image)
            self.clear_processed_images()  # Clear existing processed images
            # self.display_image(self.image)  # Original image
            self.display_images([processed_image], ["Inverted Colors"])
        else:
            messagebox.showerror("Error", "No image uploaded.")

    def split_rgb(self):
        if self.image is not None:
            b, g, r = cv2.split(self.image)
            # Create an empty image with zeros for the green and red channels
            empty_channel = np.zeros_like(b)
            green_image = cv2.merge((empty_channel, g, empty_channel))
            red_image = cv2.merge((r, empty_channel, empty_channel))
            blue_image = cv2.merge((empty_channel, empty_channel, b))

            self.clear_processed_images()  # Clear existing processed images
            # self.display_image(self.image)  # Original image
            display_images = [red_image, green_image, blue_image]
            self.display_images(display_images, ["Red Channel", "Green Channel", "Blue Channel"])
        else:
            messagebox.showerror("Error", "No image uploaded.")

    def embed_watermark(self):
        if self.image is not None:
            file_path = filedialog.askopenfilename()
            if file_path:
                watermark = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                # Convert watermark to binary
                _, watermark_binary = cv2.threshold(watermark, 127, 255, cv2.THRESH_BINARY)

                # self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

                # Get the size of the input image and the watermark
                input_height, input_width, _ = self.image.shape
                watermark_height, watermark_width = watermark_binary.shape

                # Check if the watermark is larger than the input image
                # If so, resize the watermark to match the input image
                # Otherwise, tile the watermark to match the input image
                if watermark_height > input_height or watermark_width > input_width:
                    watermark_binary = watermark_binary[:input_height, :input_width]
                else:
                    watermark_binary = np.tile(watermark_binary, (input_height // watermark_height + 1, input_width // watermark_width + 1))[:input_height, :input_width]

                embedded_images = [watermark_binary]
                
                # Embed watermark into blue channel for each bit plane
                for bit_plane in range(8):
                    # Extract the bit plane from the blue channel
                    blue_channel = self.image[:, :, 2]

                    # Embed the watermark into the bit plane of the blue channel
                    # If the watermark pixel is 255, set the bit to 1 (logical OR) 
                    # Otherwise, set the bit to 0 (logical AND)
                    new_blue_channel = np.where(watermark_binary == 255, blue_channel | (1 << bit_plane), blue_channel & ~(1 << bit_plane))

                    # Merge the new blue channel with the original green and red channels
                    embedded_image = self.image.copy()
                    embedded_image[:, :, 2] = new_blue_channel

                    embedded_images.append(embedded_image)
                
                # Save the embedded images
                cv2.imwrite("embedded_image.png", embedded_images[1])
                
                caption_labels = ["Watermark"] + [f"Bit Plane {i+1}" for i in range(8)]
                self.clear_processed_images()  # Clear existing processed images
                self.display_images(embedded_images, caption_labels) 
        else:
            messagebox.showerror("Error", "No image uploaded.")
    
    def extract_watermark(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image_with_watermark = cv2.imread(file_path)

            # Create a copy of the image to preserve the original
            image_without_watermark = image_with_watermark.copy()

            # Extract the watermark from the 1st bit plane of the blue channel
            blue_channel = image_without_watermark[:, :, 2]
            watermark_bit_plane = (blue_channel >> 0) & 1

            # Restore the original pixel values for the 1st bit plane of the blue channel using XOR
            blue_channel ^= watermark_bit_plane << 0 # Toggle the 1st bit plane

            # Display the extracted images
            extracted_images = [image_with_watermark, watermark_bit_plane * 255, image_without_watermark]
            caption_labels = ["Image with watermark", "Watermark", "Image without watermark"]
            self.clear_processed_images()
            self.display_images(extracted_images, caption_labels)
        else:
            messagebox.showerror("Error", "No image selected.")


    def change_component(self):
        if self.image is not None:
            component = self.askComboValue("Enter component to change (R, G, or B):", ("R", "G", "B"))
            if component not in ["R", "G", "B"]:
                messagebox.showerror("Error", "Invalid component. Please enter 'R', 'G', or 'B'.")
                return
            constant = simpledialog.askinteger("Change Component", "Enter constant value:")
            if constant is None:
                return
            # Apply change to the specified component
            b, g, r = cv2.split(self.image)
            if component == "R":
                r += constant
            elif component == "G":
                g += constant
            elif component == "B":
                b += constant
            processed_image = cv2.merge((b, g, r))
            self.clear_processed_images()  # Clear existing processed images
            self.display_images([processed_image], [f"{component} channel changed by {constant}"])  # Processed image
        else:
            messagebox.showerror("Error", "No image uploaded.")
    
    def merge_images_with_animation(self):
        if self.image is not None:
            file_path = filedialog.askopenfilename()
            if file_path:
                second_image = cv2.imread(file_path)
                second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)
                second_image = cv2.resize(second_image, (self.image.shape[1], self.image.shape[0]))  # Resize second image to match dimensions
                step_size = 0.1  # Change this value to adjust the step size for alpha
                for alpha in np.arange(0, 1 + step_size, step_size):
                    processed_image = cv2.addWeighted(self.image, alpha, second_image, 1 - alpha, 0)
                    self.clear_processed_images()  # Clear existing processed images
                    self.display_images([second_image, processed_image], ["Second Image", "Merged Image"])
                    
                    self.root.update()  # Update the GUI to display the new image
                    self.root.after(1000)  # Delay between each frame (milliseconds)
        else:
            messagebox.showerror("Error", "No image uploaded.")


    def askComboValue(self, text, values):
        top = tk.Toplevel() # use Toplevel() instead of Tk()
        tk.Label(top, text=text).pack()
        box_value = tk.StringVar()
        combo = ttk.Combobox(top, width=27, textvariable=box_value)
        combo['values'] = values
        combo.pack()
        combo.bind('<<ComboboxSelected>>', lambda _: top.destroy())
        top.grab_set()
        top.wait_window(top)  # wait for itself destroyed, so like a modal dialog
        return box_value.get()
    
    def Filter(self, selected_filter):
        if self.image is not None:
            processed_image = None
            if selected_filter == "Blurring Filter":
                kernel_size = simpledialog.askinteger("Blurring Filter", "Enter kernel size (odd number):", initialvalue=3)
                if kernel_size is None or kernel_size % 2 == 0:
                    messagebox.showerror("Error", "Invalid kernel size. Please enter an odd number.")
                    return
                processed_image = self.blur_filter(kernel_size)
                processed_image_wo_cv2 = self.blur_filter_wo_cv2(kernel_size)
            elif selected_filter == "Sharpening Filter":
                processed_image = self.sharpen_filter()
                processed_image_wo_cv2 = self.sharpen_filter_wo_cv2()
            elif selected_filter == "Median Filter":
                kernel_size = simpledialog.askinteger("Median Filter", "Enter kernel size (odd number):", initialvalue=3)
                if kernel_size is None or kernel_size % 2 == 0:
                    messagebox.showerror("Error", "Invalid kernel size. Please enter an odd number.")
                    return
                processed_image = self.median_filter(kernel_size)
                processed_image_wo_cv2 = self.median_filter_wo_cv2(kernel_size)
            elif selected_filter == "Erosion Filter":
                processed_image = self.erosion_filter()
                processed_image_wo_cv2 = self.erosion_filter_wo_cv2()
            elif selected_filter == "Dilation Filter":
                processed_image = self.dilation_filter()
                processed_image_wo_cv2 = self.dilation_filter_wo_cv2()
            elif selected_filter == "Sobel Filter":
                processed_image = self.sobel_filter()
                processed_image_wo_cv2 = self.sobel_filter_wo_cv2()
            
            self.clear_processed_images()  # Clear existing processed images
            self.display_images([processed_image, processed_image_wo_cv2, processed_image - processed_image_wo_cv2], [f"{selected_filter} cv2", f"{selected_filter} no cv2", "Difference"])  # Original image
        else:
            messagebox.showerror("Error", "No image uploaded.")

    def blur_filter(self, kernel_size):
        # Apply the blurring filter
        return cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 5)

    def blur_filter_wo_cv2(self, kernel_size):
        # Apply the blurring filter
        kernel = self.generate_gaussian_kernel(kernel_size, 5)
        return self.apply_filter(kernel).clip(0, 255).astype(np.uint8)
    
    def generate_gaussian_kernel(self, kernel_size, sigma=1.0):
        # Generate a Gaussian kernel
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2) / (2 * sigma ** 2)),
            (kernel_size, kernel_size))
        return kernel / np.sum(kernel)  # Normalize the kernel
    
    def sharpen_filter(self):

        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], np.float32)

        return cv2.filter2D(self.image, -1, kernel)
    
    def sharpen_filter_wo_cv2(self):

        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], np.float32)

        return self.apply_filter(kernel).clip(0, 255).astype(np.uint8)

    def median_filter(self, kernel_size):

        return cv2.medianBlur(self.image, kernel_size)
    
    def median_filter_wo_cv2(self, kernel_size):

        return self.apply_filter(kernel_size, method='median').astype(np.uint8)

    def erosion_filter(self):

        kernel = np.ones((5, 5), np.uint8)

        processed_image = cv2.erode(self.image, kernel, iterations=1)
        
        return processed_image
    
    def erosion_filter_wo_cv2(self):

        kernel = np.ones((5, 5), np.uint8)

        processed_image = self.apply_filter(kernel, method='erosion')

        return processed_image.astype(np.uint8)
    
    def dilation_filter(self):

        kernel = np.ones((5, 5), np.uint8)

        processed_image = cv2.dilate(self.image, kernel, iterations=1)

        return processed_image
    
    def dilation_filter_wo_cv2(self):

        kernel = np.ones((5, 5), np.uint8)

        processed_image = self.apply_filter(kernel, method='dilation')

        return processed_image.astype(np.uint8)
    
    def sobel_filter(self):
        # Apply Sobel filter in x-direction
        sobel_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=3)
        # Apply Sobel filter in y-direction
        sobel_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=3)
        # Combine the results to get the final image
        processed_image = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
        return processed_image
    
    def sobel_filter_wo_cv2(self):

        sobel_x_kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
        sobel_y_kernel = sobel_x_kernel.T

        sobel_x = self.apply_filter(sobel_x_kernel)
        sobel_y = self.apply_filter(sobel_y_kernel)
        return np.sqrt(np.square(sobel_x) + np.square(sobel_y)).astype(np.uint8)
    
    def apply_filter(self, kernel_or_size, method='convolution'):
        # Apply the specified filter method
        # If kernel_or_size is an integer, use it as the kernel size
        # If kernel_or_size is a 2D array, use it as the kernel
        # window_size is the size of the window around the pixel
        if isinstance(kernel_or_size, int):
            kernel_size = kernel_or_size
            window_size = kernel_size // 2
        else:
            kernel = kernel_or_size
            window_size = kernel.shape[0] // 2
        
        # Pad the image with zeros to handle the borders of the image 
        # it is necessary to avoid the out of bounds error
        padded_image = np.pad(self.image, ((window_size, window_size), (window_size, window_size), (0, 0)), mode='constant', constant_values=0)
        
        # Initialize processed image
        processed_image = np.zeros_like(self.image, dtype=np.float32)
        
        # Apply the filter based on the specified method
        for i in range(window_size, padded_image.shape[0] - window_size):
            for j in range(window_size, padded_image.shape[1] - window_size):
                for k in range(padded_image.shape[2]):
                    # +1 is added to the window size to include the center pixel
                    window = padded_image[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1, k]
                    if method == 'convolution':
                        processed_image[i - window_size, j - window_size, k] = np.sum(window * kernel)
                    elif method == 'median':
                        processed_image[i - window_size, j - window_size, k] = np.median(window)
                    elif method == 'dilation':
                        processed_image[i - window_size, j - window_size, k] = np.max(window)
                    elif method == 'erosion':
                        processed_image[i - window_size, j - window_size, k] = np.min(window)
        
        return processed_image

# Main function
# This function creates the main window and runs the application
if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
