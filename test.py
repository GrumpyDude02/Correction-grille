import cv2
import numpy as np
import imutils
import convert_pdf as PDF

def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def detect_lines(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    return lines, edges

def non_max_suppression(points, threshold=10):
    filtered_points = []
    for pt in points:
        close = any(np.linalg.norm(np.array(pt) - np.array(fp)) < threshold for fp in filtered_points)
        if not close:
            filtered_points.append(pt)
    return filtered_points

if __name__ == "__main__":
    file = PDF.PDFFile("C:\\Users\\Youssef Bakouch\\Desktop\\cas erreurs.pdf")
    images = file.get_cv_images()
    
    image_index = 1
    original_image = images[image_index]
    resized_image = imutils.resize(original_image, width=original_image.shape[1] // 3)
    start = resized_image.shape[1] // 2
    roi = resized_image[:, start:]
    
    contrast_roi = enhance_contrast(roi)
    
    inverted_img = cv2.bitwise_not(contrast_roi)
    
    k_size = contrast_roi.shape[1] // 80
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_size))
    
    horizontal_lines = cv2.morphologyEx(inverted_img, cv2.MORPH_ERODE, horizontal_kernel, iterations=8)
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_DILATE, horizontal_kernel, iterations=14)
    
    vertical_lines = cv2.morphologyEx(inverted_img, cv2.MORPH_ERODE, vertical_kernel, iterations=8)
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_DILATE, vertical_kernel, iterations=8)
    
    combined_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    corners = cv2.goodFeaturesToTrack(combined_lines, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners = np.int8(corners) if corners is not None else []
    
    filtered_corners = non_max_suppression([tuple(c.ravel()) for c in corners])
    
    for corner in filtered_corners:
        x, y = corner
        cv2.circle(roi, (x, y), 5, (0, 0, 255), -1)
    
    lines, edges = detect_lines(combined_lines)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("Edges", edges)
    cv2.imshow("Detected Lines", roi)
    cv2.imshow("Combined Lines", combined_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()