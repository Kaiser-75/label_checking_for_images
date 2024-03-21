import cv2
from IPython.display import Image, display

class ObjectDetector:
    def __init__(self, class_names):
        self.class_names = class_names

    def detect_and_annotate(self, image_path, annotation_path):
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        with open(annotation_path, 'r') as f:
            annotations = f.readlines()

        for annotation in annotations:
            class_id, center_x, center_y, bbox_width, bbox_height = map(float, annotation.strip().split())

            x = int((center_x - bbox_width / 2) * width)
            y = int((center_y - bbox_height / 2) * height)
            w = int(bbox_width * width)
            h = int(bbox_height * height)

            color = (255, 0, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            class_name = self.class_names[int(class_id)]
            label = f"{class_name}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        temp_annotated_image_path = 'temp_annotated_image.jpg'
        cv2.imwrite(temp_annotated_image_path, image)

        display(Image(filename=temp_annotated_image_path))

        import os
        os.remove(temp_annotated_image_path)


class_names = ['bus', 'car', 'rickshaw', 'CNG', 'motorbike', 'truck', 'pickup', 'minivan', 'suv', 'van', 'bicycle', 'auto rickshaw', 'human hauler', 'wheelbarrow', 'minibus', 'ambulance', 'taxi', 'army vehicle', 'scooter', 'policecar', 'garbagevan']
detector = ObjectDetector(class_names)

#SOURCES
image_path = '/content/drive/MyDrive/images/train/232.jpg'
annotation_path = '/content/drive/MyDrive/labels/train/232.txt'

#ANNOTATE
detector.detect_and_annotate(image_path, annotation_path)
