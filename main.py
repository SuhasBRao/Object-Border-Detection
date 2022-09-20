import cv2
from rembg import remove
import numpy as np
import os

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

def remove_bg_using_cmd(cmd):
    os.system(cmd)
    pass

def read_transparent_img(img_path):
    src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    bgr = src[:,:,:3] # Channels 0..2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)

    bgr = cv2.cvtColor(gray, cv2.COLOR_BGRA2BGR)
    alpha = src[:,:,3] # Channel 3
    result = np.dstack([bgr, alpha])
    return result
    

def draw_outline(roi_fg):
    temp_image = roi_fg.copy()
    # gray = cv2.cvtColor(temp_image, cv2.COLOR_BGRA2GRAY)
    blur = cv2.GaussianBlur(temp_image, (29,29), 0)
    # cv2.imwrite("blur.jpg", blur)
    
    _ , thresholded = cv2.threshold(blur, 250 , 50,cv2.THRESH_BINARY )
    
    # cv2.imwrite("thresh.jpg", thresholded)
    edges = cv2.Canny(thresholded, 30, 60)
    # cv2.imwrite("edge.jpg", edges)
    
    contours, hierarchy= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(temp_image, contours, -1, (0,255,0),9)
    
    # cv2.imwrite('ROI_fg.jpg',temp_image)
 
    return temp_image


def place_outline_on_input_img(input_img, outline_fg, roi_coord):
    x,y = roi_coord
    
    _outline = outline_fg.copy()
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2BGRA)
    
    h1, w1 = _outline.shape[:2]
    
    roi = input_img[y: y+h1, x:x + w1]
    
    gray = cv2.cvtColor(_outline, cv2.COLOR_BGRA2GRAY)
    _, mask = cv2.threshold(gray, 25, 250, cv2.THRESH_BINARY_INV)
    
    
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
    
    image_path = "TEST IMAGES/2.jpg"
        
    image = read_image(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    h,w,c = image.shape
    font_scale = (w * h) / (1000 * 1000)
    
    while True:
        
        cv2.putText(image, "Select ROI", (int(w*0.04), int(h*0.07)), 
                    cv2.FONT_HERSHEY_PLAIN, font_scale, (255,0,0), 5)
        
        roi, roi_coord = select_roi(image)
        cv2.imwrite("roi.jpg", roi)
        
        # remove_bg(roi)
        remove_bg_using_cmd("rembg i roi.jpg TEMP.jpg")
        
        # roi_fg = cv2.imread("ROI_fg.jpg")
        trans = read_transparent_img("TEMP.jpg")
        # show_image(trans)
        
        # Draw outline on the roi_fg image
        # outline_fg = draw_outline(roi_fg)
        outline_fg = draw_outline(trans)
        
        
        place_outline_on_input_img(image, outline_fg, roi_coord)
        cv2.putText(image, "Press Q->Exit or C->Clear", (int(w*0.04), int(h*0.95)), 
                    cv2.FONT_HERSHEY_PLAIN, font_scale, (0,0,0), 5)
        
        show_image(image, "Image")
        
        # as long as the user presses "q" the window will be displaying
        k = cv2.waitKey(0) & 0xFF
        
        if k == ord('q'):        
            cv2.destroyAllWindows()
            break
        
        elif k == ord('c'):
            image = read_image(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            show_image(image, "Image")
            
    

# } Driver Code ends