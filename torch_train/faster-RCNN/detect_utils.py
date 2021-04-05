import torchvision.transforms as transforms
import cv2
import numpy as np
import numpy

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# Settings
COLORS = np.random.uniform(0,255, size = (len(coco_names),3))

transforms = transforms.Compose([transforms.ToTensor(),])


def predict(image, model, device, detection_threshold):
    image = transforms(image).to(device)
    image = image.unsqueeze(0)
    outputs = model(image)

    # print the results individually
    print(f"BOXES: {outputs[0]['boxes']}")
    print(f"LABELS: {outputs[0]['labels']}")
    print(f"SCORES: {outputs[0]['scores']}")

    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes = pred_boxes[pred_scores >= detection_threshold].astype(np.int32)

    return boxes, pred_classes, outputs[0]['labels']


def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image