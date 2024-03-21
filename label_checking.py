import os
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

            color = (0, 0, 255)  
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            class_name = self.class_names[int(class_id)]
            label = f"{class_name}"
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image


class_names = ['bus', 'car', 'rickshaw', 'CNG', 'motorbike', 'truck', 'pickup', 'minivan', 'suv', 'van', 'bicycle', 'auto rickshaw', 'human hauler', 'wheelbarrow', 'minibus', 'ambulance', 'taxi', 'army vehicle', 'scooter', 'policecar', 'garbagevan']
detector = ObjectDetector(class_names)
#SOURCES
image_folder = '/content/drive/MyDrive/Thesis_dataset/images/train'
annotation_folder = '/content/drive/MyDrive/Thesis_dataset/extra/missing_labels'
output_folder = '/content/drive/MyDrive/Thesis_dataset/extra/missing_annotation'
os.makedirs(output_folder, exist_ok=True)

#ITERATE IN THE FOLDER
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_file)
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(annotation_folder, annotation_file)

        if os.path.exists(annotation_path):
            annotated_image = detector.detect_and_annotate(image_path, annotation_path)
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, annotated_image)
            # print(f"Annotated image saved: {output_path}")
        else:
            print(f"No annotation found for {image_file}")
