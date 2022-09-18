import cv2
import rembg
import numpy as np 

def read_image(image_path):
    image = cv2.imread(image_path)
    return image

def show_image(image):
    
    cv2.imshow('My Image', image)
    while True:
        k = cv2.waitKey(0) & 0xFF
        print(k)
        if k == 27:
            cv2.destroyAllWindows()
            break
       
       
# {
# Driver Code starts
if __name__ == "__main__":
    # write your code here
    image_path = "TEST IMAGES/1.jpg"
    
    image = read_image(image_path)
    
    print(image)
    show_image(image)

# } Driver Code ends