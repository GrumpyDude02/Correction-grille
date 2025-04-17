import cv2
import numpy as np

# Function to calculate the angle between two points
def calculate_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
    return angle

# Load image
img = cv2.imread("temp/2.png")

# Detect QR code
qr_detector = cv2.QRCodeDetector()
data, points, _ = qr_detector.detectAndDecode(img)

if points is not None:
    print("QR Code corners:")
    print(points)  # Prints the corner points

    # Calculate the angle of rotation based on the first two corner points
    # We will assume that the QR code should be upright, and calculate the angle based on the top-left and top-right corners
    angle = calculate_angle(points[0][0], points[0][1])
    print(f"Calculated rotation angle: {angle} degrees")

    # Rotate the image based on the calculated angle
    # Get the center of the image
    center = (img.shape[1] // 2, img.shape[0] // 2)
    
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

    # Visualize corners on the rotated image
    for point in points[0]:
        cv2.circle(rotated_img, tuple(map(int, point)), 10, (0, 0, 255), -1)

    # Show the rotated image with corners
    cv2.imshow("Rotated QR Code with corners", rotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("QR code not found.")
