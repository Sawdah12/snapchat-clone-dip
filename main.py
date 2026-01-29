import cv2
from filters import *

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
captured_img, display_frame, mode = None, None, "camera"

print("--- CONTROLS ---")
print("BANK 1 (Smoothing): B=Mean | 6=Gaussian | M=Median")
print("BANK 2 (Edges): S=Sobel | L=Laplacian | K=Canny | Q=Sharpen")
print("BANK 3 (Color): 1=LowBright | 2=HighContrast | 3=Invert | 4=Sepia | 5=Gray")
print("BANK 4 (Boost): Z=Boost Red | X=Boost Green | V=Boost Blue")
print("Accessories: H=Hat | G=Glasses")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    if mode == "camera":
        display_frame = frame.copy()
        cv2.putText(display_frame, "LIVE: C to Capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Advanced SNAP Project", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        captured_img = frame.copy()
        display_frame = captured_img.copy()
        mode = "filter"

    elif mode == "filter":
        # BANK 1 & 2: Smoothing/Edges
        if key == ord('b'): display_frame = apply_blur_bg(captured_img)
        elif key == ord('6'): display_frame = apply_gaussian(captured_img)
        elif key == ord('m'): display_frame = apply_median(captured_img)
        elif key == ord('s'): display_frame = apply_sobel(captured_img)
        elif key == ord('l'): display_frame = apply_laplacian(captured_img)
        elif key == ord('k'): display_frame = apply_canny(captured_img)
        elif key == ord('q'): display_frame = enhance_pixels(captured_img)
        
        # BANK 3: Color Transformation
        elif key == ord('1'): display_frame = filter_bright_low_con(captured_img)
        elif key == ord('2'): display_frame = filter_dark_high_con(captured_img)
        elif key == ord('3'): display_frame = filter_invert(captured_img)
        elif key == ord('4'): display_frame = filter_sepia(captured_img)
        elif key == ord('5'): display_frame = filter_grayscale(captured_img)
        
        # BANK 4: Channel Management
        elif key == ord('z'): display_frame = boost_channel(captured_img, 2) # RED
        elif key == ord('x'): display_frame = boost_channel(captured_img, 1) # GREEN
        elif key == ord('v'): display_frame = boost_channel(captured_img, 0) # BLUE

        # Accessories
        if key == ord('h') or key == ord('g'):
            gray_cap = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_cap, 1.1, 2)
            temp = captured_img.copy()
            
            for (x, y, w, h) in faces:
                if key == ord('h'): 
                    # Hat: Positioned above the head
                    temp = overlay_transparent(temp, "accessories/hat.png", int(x-w*0.2), y-int(h*0.45), int(w*1.4), int(h*0.7))
                
                elif key == ord('g'): 
                    # 1. Scaling: Wider to cover face
                    new_w = int(w * 1.15) 
                    
                    # 2. Aspect Ratio: Prevents "squashed" look
                    fixed_h = int(new_w / 2.5) 
                    
                    # 3. Translation: Centers on eyes
                    new_x = x - int(w * 0.07)
                    new_y = y + int(h / 3.2) 
                    
                    temp = overlay_transparent(temp, "accessories/sunglasses.png", new_x, new_y, new_w, fixed_h)
            
            display_frame = temp

    if key == ord('r'): 
        mode = "camera"
    elif key == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()