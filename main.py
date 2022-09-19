import cv2
from rembg import remove
import numpy as np 

def read_image(image_path):
    image = cv2.imread(image_path)
    return image


def show_image(image, image_window = "Temp"):
    cv2.imshow(image_window, image)


def select_roi(image):
    # Select ROI
    r = cv2.selectROI("Image", image)
    
    # Crop image
    cropped_image = image[int(r[1]):int(r[1]+r[3]),
                        int(r[0]):int(r[0]+r[2])]
    # return the cropeed image
    return cropped_image, (r[0], r[1])


def remove_bg(roi):
    output_img = remove(roi)
    cv2.imwrite('ROI_fg.jpg',output_img)
    return output_img


def draw_outline(roi_fg):
    temp_image = roi_fg.copy()
    
    gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (15,15), 0)
    
    _ , thresholded = cv2.threshold(blur, 10, 180,cv2.THRESH_BINARY)
    
    edges = cv2.Canny(thresholded, threshold1= 0, threshold2=20)
    
    contours, hierarchy= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(temp_image, contours, -1, (0,255,0),15)
    
    cv2.imwrite('ROI_fg.jpg',temp_image)
 
    return temp_image


def place_outline_on_input_img(input_img, outline_fg, roi_coord):
    x,y = roi_coord
    
    _outline = outline_fg.copy()
    
    h1, w1 = _outline.shape[:2]
    
    roi = input_img[y: y+h1, x:x + w1]
    
    gray= cv2.cvtColor(_outline, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 250, cv2.THRESH_BINARY_INV)
    
    
    img_bg = cv2.bitwise_and(roi, roi, mask = mask)
    
    dst = cv2.add(img_bg, _outline)

    input_img[y: y+h1, x:x + w1] = dst

# {
# Driver Code starts
if __name__ == "__main__":
    # We'll create a main window where the original image
    # is displayed
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 700, 400)
    
    image_path = "TEST IMAGES/1.jpg"
        
    image = read_image(image_path)
    
    while True:
            
        roi, roi_coord = select_roi(image)
        
        roi_fg = remove_bg(roi)
        
        # reading the image to convert to 3 channels
        roi_fg = cv2.imread("ROI_fg.jpg")

        # Draw outline on the roi_fg image
        outline_fg = draw_outline(roi_fg)
        
        place_outline_on_input_img(image, outline_fg, roi_coord)
        
        show_image(image, "Image")
        
        # as long as the user presses "q" the window will be displaying
        k = cv2.waitKey(0) & 0xFF
        
        if k == ord('q'):        
            cv2.destroyAllWindows()
            break
        
        elif k == ord('c'):
            image = read_image(image_path)
            

    

# } Driver Code ends