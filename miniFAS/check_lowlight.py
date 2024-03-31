import cv2

def is_low_light(image, threshold=50):
    # Load the image

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average pixel intensity
    average_intensity = cv2.mean(gray_image)[0]

    # Check if average intensity is below the threshold
    if average_intensity < threshold:
        return True
    else:
        return False

# Example usage
# image_path = '/home/linhhima/FAS/miniFAS/test_image_llie1llie1.jpeg'
# threshold_value = 70
# if is_low_light(image_path, threshold_value):
#     print("The image has low light.")
# else:
#     print("The image does not have low light.")

# import os

# def is_image_file(filename):
#     """Check if a file has an image extension."""
#     image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
#     return any(filename.lower().endswith(ext) for ext in image_extensions)

# arr = []
# def process_images_in_folder(folder_path):
#     """Process all image files in a folder."""
#     for iii, filename in enumerate(os.listdir(folder_path)):
#         file_path = os.path.join(folder_path, filename)
#         # print(file_path)
#         image = cv2.imread(file_path)
#         if image is None:
#             print("None")
            
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         average_intensity = cv2.mean(gray_image)[0]
#         arr.append(average_intensity)
#         # if is_low_light(file_path):
#         #     print(f"The image {filename} has low light.")
#         # else:
#         #     print(f"The image {filename} does not have low light.")
#         # if iii//100 == 0:
#         #     print('Processing ', iii, '/', len())
# # Example usage
# folder_path = '/home/linhhima/FAS/miniFAS/datasets/Test_Part2/'
# process_images_in_folder(folder_path)
# print(sum(arr)/len(arr))
