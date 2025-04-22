import os
import numpy as np
import torch
from PIL import Image
import cv2
import open3d as o3d
import supervision as sv
import math
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util import box_ops
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from Grounded_Segment_Anything.segment_anything.segment_anything import sam_model_registry, SamPredictor
from transformers import BlipProcessor, BlipForConditionalGeneration


CONFIG_PATH = "./Grounded_Segment_Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./models/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "./models/sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#TEXT_PROMPT = "ear"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
VIEWS_DIR = "./render_views"
OUTPUT_DIR = "./segmented_views"
os.makedirs(OUTPUT_DIR, exist_ok=True)
view_list = []

groundingdino_model = load_model(CONFIG_PATH, CHECKPOINT_PATH).to(DEVICE)
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(DEVICE)
sam_predictor = SamPredictor(sam)

def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(DEVICE), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()
  

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def extract_segmented_object(image, mask):
    """Extracts only the masked object from the image (black background)."""
    # Ensure mask is binary (0 or 1)
    binary_mask = (mask > 0).astype(np.uint8)

    # Apply the mask to each channel
    segmented = cv2.bitwise_and(image, image, mask=binary_mask)

    return segmented

def box_to_pixel(box, image_shape):
    h, w = image_shape[:2]
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return np.array([x1, y1, x2, y2])

def get_masks_only(boxes, image_source, image_rgb):
    box = boxes[0].cpu().numpy()
    box_pixel = box_to_pixel(box, image_source.shape)

    sam_predictor.set_image(image_rgb)
    masks, scores, _ = sam_predictor.predict(
        box=box_pixel,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]

    return (best_mask.astype(np.uint8)) * 255

def render_views(mesh, mesh_name):
    angles = [0, 45, 90, 135, 180]
    
    # Create offscreen renderer
    width, height = 640, 480  # You can change resolution
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Set up material for rendering
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "default"

    # Setup scene for the renderer
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])  # White background
    scene.add_geometry(mesh_name, mesh, material)

    # Set up the camera properties
    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent().max()
    scene.camera.look_at(center, center + [0, 0, 1], [0, 1, 0])
    scene.camera.set_projection(60.0, width / height, 0.1, 1000.0)

    # Render views from different angles
    for i, angle in enumerate(angles):
        radians = np.deg2rad(angle)
        x = math.sin(radians)
        z = math.cos(radians)
        front = [x, 0, z]
        
        # Set the camera to the new angle
        scene.camera.set_front(front)

        img = renderer.render_to_image()
        
        VIEWS_DIR_OBJ = os.path.join(VIEWS_DIR, mesh_name)
        filename = f"{VIEWS_DIR_OBJ}/{mesh_name}_view_{i}.png"
        o3d.io.write_image(filename, img)
        print(f"Saved: {filename}")

    print(f"All views rendered and saved in {VIEWS_DIR_OBJ}")

def render_file_view(obj_file, mesh_name):
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh.compute_vertex_normals()
    render_views(mesh, mesh_name)

def render_default_objects():
    dataset_classes = [
        o3d.data.BunnyMesh,
        o3d.data.KnotMesh,
        o3d.data.ArmadilloMesh,
    ]
    for DatasetClass in dataset_classes:
        dataset = DatasetClass()
        mesh = o3d.io.read_triangle_mesh(dataset.path)
        mesh.compute_vertex_normals()

        mesh_name = DatasetClass.__name__
        render_views(mesh, mesh_name)


def caption_object(obj_file):
    mesh_name = os.path.splitext(os.path.basename(obj_file))[0]
    #render_file_view(obj_file)
    print("Rendering complete! Check the output directory for results.")
    #VLM to identify object
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
    image_path = os.path.join(VIEWS_DIR, mesh_name, "view_0.png")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)
    print(f"Caption: {text}")
    return text, mesh_name

def segment_and_save_views(seg_prompt, mesh_name):
    """Segment all views and save results as images."""
    VIEWS_DIR_OBJ = os.path.join(VIEWS_DIR, mesh_name)
    view_files = sorted([f for f in os.listdir(VIEWS_DIR_OBJ) if f.endswith(('.png', '.jpg'))])
    
    if not view_files:
        print(f"No images found in {VIEWS_DIR_OBJ}")
        return
    
    print(f"Found {len(view_files)} views to process")
    
    for view_file in view_files:
        print(f"Processing {view_file}...")
        view_path = os.path.join(VIEWS_DIR_OBJ, view_file)
        
        try:
            # Load and prepare image
            image_source, image = load_image(view_path)
            image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

            # Get boxes from GroundingDINO
            boxes, logits, _ = predict(
                model=groundingdino_model,
                image=image,
                caption=seg_prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE
            )
            
            if len(boxes) == 0:
                print(f"No objects detected in {view_file}")
                continue
            

            # Save results
            base_name = os.path.splitext(view_file)[0]

            # Save annotation with boxes
            annotated = annotate(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=[seg_prompt]*len(boxes)
            )
        
            segmented_frame_masks = segment(image_source, sam_predictor, boxes=boxes)
            annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated)
            masked = get_masks_only(boxes, image_source, image_rgb)
            OUTPUT_DIR_OBJ = os.path.join(OUTPUT_DIR, mesh_name)
            if not os.path.exists(OUTPUT_DIR_OBJ):
                os.makedirs(OUTPUT_DIR_OBJ)
            cv2.imwrite(os.path.join(OUTPUT_DIR_OBJ, f"{base_name}_annotated.png"), annotated)
            cv2.imwrite(os.path.join(OUTPUT_DIR_OBJ, f"{base_name}_highlighted.png"), annotated_frame_with_mask)
            highlighted_on_original = extract_segmented_object(image_source, masked)
            cv2.imwrite(os.path.join(OUTPUT_DIR_OBJ, f"{base_name}_masked_overlay.png"), highlighted_on_original)

            view_list.append(highlighted_on_original)
            print(f"Saved results for {view_file}")
        
        except Exception as e:
            print(f"Error processing {view_file}: {str(e)}")

    return view_list

if __name__ == "__main__":
    highlighted_on_original = segment_and_save_views()

    print("Segmentation complete! Check the output directory for results.")