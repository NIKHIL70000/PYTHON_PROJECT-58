import cv2
import matplotlib.pyplot as plt


def convert_image_to_sketch(image_path):
    # Read the image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to the image
    blurred_image = cv2.GaussianBlur(gray_image, (13, 13), 0)

    # Use Canny edge detection to create the sketch
    sketch_image = cv2.Canny(blurred_image, 30, 150)

    return original_image, sketch_image


def display_images(original_image, sketch_image):
    # Display the original image
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    # Display the sketch image
    plt.subplot(132)
    plt.imshow(sketch_image, cmap='gray')
    plt.title('Sketch Image')

    # Display the side-by-side comparison
    plt.subplot(133)
    comparison_image = cv2.hconcat(
        [cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(sketch_image, cv2.COLOR_GRAY2RGB)])
    plt.imshow(comparison_image)
    plt.title('Side-by-Side Comparison')

    plt.show()


def save_sketch_image(sketch_image, output_path):
    cv2.imwrite(output_path, sketch_image)


if __name__ == '__main__':
    input_image_path = 'NotsoShoujo.png'  # Replace this with the path to your input image
    output_image_path = 'output_sketch.jpg'  # Replace this with the path to save the sketch image

    original_image, sketch_image = convert_image_to_sketch(input_image_path)
    display_images(original_image, sketch_image)
    save_sketch_image(sketch_image, output_image_path)
