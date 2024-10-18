from flask import Flask, request, jsonify
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route('/detect-corners', methods=['POST'])
def detect_corners():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Read the image in memory
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)
    file_bytes = np.frombuffer(in_memory_file.read(), dtype=np.uint8)

    # Decode the image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Process the image to detect rooms and their corners
    rooms = process_image(img)

    # Return the coordinates of the detected corners
    return jsonify({'corners': rooms})

def order_corners(corners):
    # Sort the corners based on their x + y values
    sum_corners = corners.sum(axis=1)
    diff_corners = np.diff(corners, axis=1).flatten()

    # Top-left point will have the smallest sum
    top_left = corners[np.argmin(sum_corners)]
    # Bottom-right point will have the largest sum
    bottom_right = corners[np.argmax(sum_corners)]
    # Top-right point will have the smallest difference
    top_right = corners[np.argmin(diff_corners)]
    # Bottom-left point will have the largest difference
    bottom_left = corners[np.argmax(diff_corners)]

    ordered = np.array([top_left, top_right, bottom_right, bottom_left])
    return ordered

def process_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding for cleaner contour detection
    _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    # Dilate the image slightly to ensure room separation
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rooms = []
    output_image = img.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:  # Adjust this threshold based on room size
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # Epsilon adjustment for better precision

        if len(approx) == 4:  # We are only interested in quadrilaterals
            corners = approx.reshape(4, 2)
            ordered_corners = order_corners(corners)
            flattened_corners = ordered_corners.flatten().tolist()
            rooms.append(flattened_corners)

            # Draw contours for visualization
            cv2.drawContours(output_image, [ordered_corners.astype(int)], -1, (0, 255, 0), 2)

    # Save the output image for debugging purposes
    cv2.imwrite('detected_rooms_output.png', output_image)

    return rooms

if __name__ == '__main__':
    app.run(debug=True)
