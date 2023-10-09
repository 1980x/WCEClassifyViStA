import os
import cv2

# Source directory and target directory paths
source_directory = '../datasets/Auto-WCEBleedGen Challenge Test Dataset/Test Dataset 1'
target_directory = '../datasets/Auto-WCEBleedGen Challenge Test Dataset/inpainted_test_dataset_1'

# Ensure the target directory exists, create if not
os.makedirs(target_directory, exist_ok=True)

# Loop through all files in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Consider only image files
        # Read the original image
        image_path = os.path.join(source_directory, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, binary_mask = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

        # Perform morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Inpaint the masked area using neighboring pixel interpolation
        inpainted_image = cv2.inpaint(image, binary_mask_cleaned, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Save the inpainted image to the target directory with the same filename
        output_path = os.path.join(target_directory, filename)
        cv2.imwrite(output_path, inpainted_image)

        print(f'Processed and saved: {output_path}')

print('Processing and saving completed.')

