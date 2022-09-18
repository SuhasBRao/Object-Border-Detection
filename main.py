import cv2
from rembg import remove
import numpy as np 

def read_image(image_path):
    img = cv2.imread(image_path)
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
    return cropped_image, r

def remove_bg(roi):
    output_img = remove(roi)
    cv2.imwrite('ROI_fg.jpg',output_img)
    return output_img

def draw_outline(roi_fg):
    temp_image = roi_fg.copy()
    
    
    gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    # cv2.imshow("blur", blur)
    
    _ , thresholded = cv2.threshold(blur, 10, 180,cv2.THRESH_BINARY)
    
    
    edges1 = cv2.Canny(thresholded, threshold1= 50, threshold2=250)
    cv2.imshow("New edge", edges1)
    
    # edges = cv2.Canny(blur,250,255)
    # cv2.imshow("edges", edges)

    contours, hierarchy= cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(temp_image, contours, -10, (0,255,0),3)

    show_image(temp_image, "Outline")
    

# {
# Driver Code starts
if __name__ == "__main__":
    # We'll create a main window where the original image
    # is displayed
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 700, 400)
        
    # cv2.namedWindow("FG", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("FG",400, 400)
    
    while True:
        
        image_path = "TEST IMAGES/2.jpg"
        
        image = read_image(image_path)
        
        roi, roi_coord = select_roi(image)
        
        roi_fg = remove_bg(roi)
        
        # # Draw outline on the roi_fg image
        
        draw_outline(roi_fg)
        

        
        
        # as long as the user presses "q" the window will be displaying
        if cv2.waitKey(0) & 0xFF == ord('q'):        
            cv2.destroyAllWindows()
            break
    
    

# } Driver Code ends