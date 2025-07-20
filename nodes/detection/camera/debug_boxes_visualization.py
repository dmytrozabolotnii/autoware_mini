#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_boxes():
    print("Starting box visualization...")
    # Create a blank image (black background)
    img_height = 2000
    img_width = 2000
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Define the boxes to draw
    # Format: [x1, y1, x2, y2]
    head_pose_2d_boxes = [
        [1102, 1145, 1171, 1337],  # 2D box 0
        [1550, 1079, 1684, 1399],  # 2D box 1
        [945, 1179, 998, 1327]     # 2D box 2
    ]

    projected_3d_boxes = [
        [1460, 1273, 1644, 1583],  # Projected 3D->2D box 0
        [1040, 1229, 1146, 1397],  # Projected 3D->2D box 1
        [888, 1224, 967, 1373]     # Projected 3D->2D box 2
    ]

    print(f"Processing {len(head_pose_2d_boxes)} 2D boxes and {len(projected_3d_boxes)} 3D projected boxes")

    # Draw 2D boxes from head pose detections (in red)
    for i, box in enumerate(head_pose_2d_boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red
        cv2.putText(img, f"2D box {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print(f"Added 2D box {i}: {box}")

    # Draw projected 3D boxes (in green)
    for i, box in enumerate(projected_3d_boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green
        cv2.putText(img, f"3D->2D box {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"Added 3D->2D box {i}: {box}")

    # Calculate IoUs between 2D boxes and projected 3D boxes
    iou_matrix = np.zeros((len(head_pose_2d_boxes), len(projected_3d_boxes)))

    for i, box_2d in enumerate(head_pose_2d_boxes):
        for j, box_3d in enumerate(projected_3d_boxes):
            iou = calculate_iou(box_2d, box_3d)
            iou_matrix[i, j] = iou
            print(f"IoU between 2D box {i} and projected 3D box {j}: {iou:.4f}")

            # Calculate the center point of the 2D box for displaying IoU
            mid_x = (box_2d[0] + box_2d[2]) // 2
            mid_y = (box_2d[1] + box_2d[3]) // 2

            # Draw a line connecting the centers of matching boxes
            if iou > 0.05:  # Only draw lines for boxes with some overlap
                box3d_mid_x = (box_3d[0] + box_3d[2]) // 2
                box3d_mid_y = (box_3d[1] + box_3d[3]) // 2
                cv2.line(img, (mid_x, mid_y), (box3d_mid_x, box3d_mid_y), (255, 255, 0), 2)

                # Display IoU value near the middle of the line
                text_x = (mid_x + box3d_mid_x) // 2
                text_y = (mid_y + box3d_mid_y) // 2
                cv2.putText(img, f"IoU: {iou:.4f}", (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Find best matches based on IoU
    best_matches = []

    # Find best match for each 2D box
    for i in range(len(head_pose_2d_boxes)):
        best_iou = 0
        best_match = -1
        for j in range(len(projected_3d_boxes)):
            if iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_match = j

        if best_iou > 0.05:  # Threshold for a valid match
            best_matches.append((i, best_match, best_iou))
            print(f"Best match for 2D box {i}: 3D box {best_match} with IoU {best_iou:.4f}")

    # Display the image
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Box Comparison")
    plt.axis('off')
    plt.savefig("/tmp/box_visualization.png")
    plt.close()

    print(f"Visualization saved to /tmp/box_visualization.png")

    # Return the best matches for further analysis
    return best_matches

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

if __name__ == "__main__":
    draw_boxes()
