import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
import time

CHECKPOINT_PATH = os.path.join( "weights", "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "output"
TIME_FILE = "times.txt"

# Make output folder if it does not exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Loop over all image files in folder
with open(TIME_FILE, "w") as f:
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith(".png"):
            # Load image
            start_time = time.time()
            IMAGE_PATH = os.path.join(IMAGE_FOLDER, filename)
            image_bgr = cv2.imread(IMAGE_PATH)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Generate segmentation mask and time it
            sam_result = mask_generator.generate(image_rgb)
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000
            print(sam_result[0].keys())
            print(elapsed_time_ms)
            # Annotate image with mask
            mask_annotator = sv.MaskAnnotator()
            detections = sv.Detections.from_sam(sam_result=sam_result)
            annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

            # Save output image and time to file
            output_filename = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(output_filename, annotated_image)
            f.write(f"{filename}: {elapsed_time_ms:.2f} ms\n")
