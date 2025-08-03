import cv2
import numpy as np

def preprocess_image(image, yolo_input_resolution):
    """Converts image to a suitable format for YOLO model

    :param image: input image
    :param yolo_input_resolution: size of the yolo input
    :return: preprocessed image
    """
    # Resize to match YOLO input dimensions
    out_img = cv2.resize(image, yolo_input_resolution, interpolation=cv2.INTER_LINEAR)
    # Normalize to [0,1]
    out_img = out_img.astype(np.float32) / 255.0
    # HWC to CHW
    out_img = np.transpose(out_img,[2,0,1])
    # CHW to NCHW
    out_img = np.expand_dims(out_img,axis = 0)
    # Convert the image to row-major order, also known as "C order":
    out_img = np.array(out_img, dtype = np.float32, order = 'C')

    return out_img

def non_maximum_supression_boxes(boxes, box_confidences, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
    confidence scores and return an array with the indexes of the bounding boxes we want to
    keep (and display later).

    Keyword arguments:
    :param boxes: a NumPy array containing N bounding-box coordinates that survived filtering,
    with shape (N,4); 4 for x,y,height,width coordinates of the boxes
    :param box_confidences: a Numpy array containing the corresponding confidences with shape N
    :param nms_threshold: IoU threshold, float value between 0 and 1
    """
    x_coord = boxes[:, 0]
    y_coord = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep_idxs = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep_idxs.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        # Compute the Intersection over Union (IoU) score:
        iou = intersection / union

        # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
        # candidates to a minimum. In this step, we keep only those elements whose overlap
        # with the current bounding box is lower than the threshold:
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep_idxs = np.array(keep_idxs)
    return keep_idxs

def convert_and_scale_boxes(box, original_img_size, yolo_input_resolution, xy_center=False):
    """Convert yolo output of x_1 y_1 w h to x_1 y_1 x_2 y_2 and scale the boxes based on the original image size

    :param box: a NumPy array containing yolo predicted box
    :param original_img_size: size of the original input image
    :param yolo_input_resolution: size of the yolo input
    :param xy_center: whether the x and y coordinate are the top-left corner or centerpoint
    """

    x_scale = original_img_size[1] / yolo_input_resolution[0]
    y_scale = original_img_size[0] / yolo_input_resolution[1]

    if xy_center:
        hw = box[:, 2] / 2
        hh = box[:, 3] / 2
        x1 = box[:, 0] - hw
        y1 = box[:, 1] - hh
        x2 = box[:, 0] + hw
        y2 = box[:, 1] + hh
    else:
        x1 = box[:, 0]
        y1 = box[:, 1]
        x2 = box[:, 0] + box[:, 2]
        y2 = box[:, 1] + box[:, 3]

    x1, x2 = x1*x_scale, x2*x_scale
    y1, y2 = y1*y_scale, y2*y_scale

    return np.rint(np.array([x1, y1, x2, y2]).T).astype(int)