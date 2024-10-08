import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

def image_to_height_map(image_path, scale_factor=0.1):
    # Load the image and convert it to grayscale
    image = Image.open(image_path).convert('L')
    # Get the image data as a numpy array
    image_array = np.array(image)
    # Invert the image (bright pixels become high points)
    image_array = 255 - image_array
    # Scale the image to the desired size
    scaled_image_array = image_array * scale_factor
    return scaled_image_array

def print_image_in_3d(image_path):
    # Convert the image to a height map
    height_map = image_to_height_map(image_path)
    # Create x and y coordinates
    x = np.linspace(0, height_map.shape[1], height_map.shape[1])
    y = np.linspace(0, height_map.shape[0], height_map.shape[0])
    X, Y = np.meshgrid(x, y)
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    ax.plot_surface(X, Y, height_map, cmap='gray', rstride=1, cstride=1)
    # Show the plot
    plt.show()

def main():
   # print("Enter the path to the image file:")
   # image_path = input().strip()
    #print_image_in_3d("/home/rafael/Desktop/nossa.jpg")
    #print_image_in_3d("/home/rafael/Desktop/nossa2.jpg")
    #print_image_in_3d("/home/rafael/Desktop/nossa3.jpg")
    print_image_in_3d("/home/rafael/Desktop/ela.jpg")

if __name__ == "__main__":
    main()