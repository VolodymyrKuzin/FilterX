# FilterX - Image Processor App

**Description:**

FilterX is an image processing application built using Python and Tkinter. It provides a user-friendly interface for uploading images, applying various filters and tools, and visualizing the processed images. This application utilizes the OpenCV and NumPy libraries for image processing tasks.

**Features:**

- **Upload Image:** Allows users to upload images from their local filesystem via file dialog or drag and drop functionality.

- **Tools:**
  - **Change Component:** Modify the intensity of a specific color channel (R, G, or B).
  - **Invert Colors:** Invert the colors of the uploaded image.
  - **Split RGB:** Split the uploaded image into its red, green, and blue channels.
  - **Merge Images:** Merge the uploaded image with another image using alpha blending.
  - **Embed Watermark:** Embed a watermark into the uploaded image.
  - **Extract Watermark:** Extract a watermark from an image.

- **Filters:**
  - **Blurring Filter:** Apply Gaussian blur to the uploaded image.
  - **Sharpening Filter:** Enhance the edges and details of the uploaded image.
  - **Median Filter:** Apply median filtering to reduce noise in the image.
  - **Erosion Filter:** Perform erosion operation on the image.
  - **Dilation Filter:** Perform dilation operation on the image.
  - **Sobel Filter:** Apply Sobel edge detection to highlight edges in the image.

**Usage:**

1. Run the script to launch the application.
2. Upload an image using the "Upload Image" button.
3. Select a tool or filter from the provided options.
4. View the processed images in the right frame.
5. Experiment with different tools and filters to achieve desired results.

**Requirements:**

- Python 3.x
- Tkinter
- OpenCV
- NumPy
- PIL (Python Imaging Library)

**Note:** Ensure all the required libraries are installed before running the application.

**Author:**
[Volodymyr "KHVZIX" Kuzin]

**Contact:**
- Email: volHkuz@gmail.com
