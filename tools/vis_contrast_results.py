import cv2

def load_boxes(txt_file, pic_number):
    """
    frame_id,target_id,x,y,w,h,conf1,cls,conf3
    """
    boxes = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 9:
                continue
            frame_id = parts[0]
            if int(frame_id) != pic_number:
                continue
            target_id = parts[1]
            try:
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except ValueError:
                continue
            cls = parts[7]
            # if int(cls) != 2:
            #     continue
            box = {
                'frame_id': frame_id,
                'target_id': target_id,
                'box': [x, y, w, h],
                'cls': cls
            }
            boxes.append(box)
    return boxes

def compute_iou(box1, box2):
    """
    box: [x, y, w, h], (x, y) is the coordinate of the upper left corner
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.6):
    TP = []
    FP = []
    FN = []
    
    gt_matched = [False] * len(gt_boxes)
    
    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            iou = compute_iou(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            TP.append(pred)
            gt_matched[best_gt_idx] = True
        else:
            FP.append(pred)
            
    for matched, gt in zip(gt_matched, gt_boxes):
        if not matched:
            FN.append(gt)
    
    return TP, FP, FN

def draw_boxes(image, boxes, color, label):
    for box in boxes:
        x, y, w, h = map(int, box['box'])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def main():
    # path
    pred_file = '....../viso_002/002.txt'
    gt_file = '....../dataset/ICPR/val_data/002/gt.txt'

    image_file = '....../dataset/ICPR/val_data/002/img1/000100.jpg'
    output_file = '....../viso_002/100.jpg'
    pic_number = 100
    
    image = cv2.imread(image_file)
    if image is None:
        print("Unable to load image, please check the file path.")
        return
    
    pred_boxes = load_boxes(pred_file, pic_number)
    gt_boxes = load_boxes(gt_file, pic_number)
    
    TP, FP, FN = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5) 
    
    # TP - green; FP - red; FN - yellow
    draw_boxes(image, TP, (0, 255, 0), "TP")
    draw_boxes(image, FP, (0, 0, 255), "FP")
    draw_boxes(image, FN, (0, 255, 255), "FN")

    cv2.imwrite(output_file, image)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
