import gradio as gr
from PIL import Image, ImageEnhance
import os
import random
import spaces
from datetime import datetime
import tempfile
import torch
import numpy as np
from typing import List, Optional, Union, Tuple

from OmniGen import OmniGenPipeline

# Global temp directory for consistent file management
TEMP_DIR = os.path.join(tempfile.gettempdir(), "omnigen_temp")
os.makedirs(TEMP_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimize prompts with photography-specific vocabulary for more natural results
PHOTOGRAPHY_STYLES = {
    "natural": "35mm film photography, natural lighting, slight film grain, realistic skin texture, candid moment",
    "portrait": "professional portrait photography, soft natural lighting, shallow depth of field, bokeh background",
    "documentary": "documentary style photography, authentic moment, photojournalism, natural lighting, unposed",
    "street": "street photography, authentic urban setting, natural lighting, candid moment, realistic",
    "editorial": "editorial photography, fashion magazine style, professional lighting, authentic look"
}

CAMERA_DETAILS = {
    "vintage": "shot on Leica, 35mm film, slight grain, Kodak Portra 400 film",
    "modern": "shot on Sony Alpha, natural colors, sharp details, professional clarity",
    "cinema": "cinematic framing, anamorphic lens, golden hour lighting, movie still quality"
}

LIGHTING_OPTIONS = {
    "natural": "natural sunlight, golden hour glow, authentic shadows",
    "soft": "soft diffused lighting, gentle shadows, flattering illumination",
    "dramatic": "dramatic side lighting, light and shadow play, atmospheric"
}

# Enhanced prompt engineering function
def enhance_prompt_with_photography_terms(text, style="natural", camera="modern", lighting="natural"):
    """Make prompt more photorealistic with professional photography terminology"""
    photo_base = f"{text}, {PHOTOGRAPHY_STYLES[style]}, {CAMERA_DETAILS[camera]}, {LIGHTING_OPTIONS[lighting]}"
    photo_technical = "professional photography, true-to-life colors, authentic moment, photorealistic, not AI generated"
    return f"{photo_base}, {photo_technical}"

def enhance_prompt_with_groq(text):
    """Enhance the prompt using Groq API (placeholder for actual implementation)"""
    try:
        # Placeholder for API integration
        enhanced = f"{text}, professional photography, natural lighting, authentic moment, realistic detail"
        return enhanced
    except Exception as e:
        print(f"Warning: Prompt enhancement failed: {e}")
        return text

def save_image_to_temp(img, prefix="temp"):
    """Save an image to a temporary file and return the path"""
    if img is None:
        return None
    
    # Create a unique filename
    temp_path = os.path.join(TEMP_DIR, f"{prefix}_{random.randint(0, 1000000)}.png")
    img.save(temp_path)
    return temp_path

def clean_temp_files(file_paths):
    """Clean up temporary files"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Warning: Failed to clean up temp file {path}: {e}")

@spaces.GPU(duration=240)
def generate_background(text, height, width, guidance_scale, inference_steps, seed, separate_cfg_infer, offload_model, randomize_seed, photo_style):
    """Generate a background image based on the prompt"""
    if randomize_seed:
        seed = random.randint(0, 10000000)
    
    # Enhanced background prompt with photography terms
    style, camera, lighting = "natural", "modern", "natural"
    if photo_style == "Portrait":
        style, lighting = "portrait", "soft"
    elif photo_style == "Documentary":
        style, camera = "documentary", "vintage"
    elif photo_style == "Cinematic":
        style, camera, lighting = "editorial", "cinema", "dramatic"
    
    # Add background-specific keywords to the prompt with photorealistic enhancement
    background_prompt = f"{text}, detailed environment scene, establishing shot, location photography"
    background_prompt = enhance_prompt_with_photography_terms(background_prompt, style, camera, lighting)
    background_prompt += ", no people, no subjects, authentic environment only"
    
    try:
        output = pipe(
            prompt=background_prompt,
            input_images=None,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=1.0,  # Default since there's no input image
            num_inference_steps=inference_steps,
            separate_cfg_infer=separate_cfg_infer,
            use_kv_cache=True,
            offload_kv_cache=True,
            offload_model=offload_model,
            use_input_image_size_as_output=False,
            seed=seed,
            max_input_image_size=max(height, width),
        )
        
        background_img = output[0]
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(background_img)
        background_img = enhancer.enhance(1.2)  # Reduced from 1.5 for more natural look
        
        return background_img
    except Exception as e:
        print(f"Error generating background: {e}")
        return None

@spaces.GPU(duration=180)
def generate_multi_subject_image(
    text: str, 
    subject_images: List[str], 
    background_img_path: str,
    height: int, 
    width: int, 
    guidance_scale: float, 
    img_guidance_scale: float,
    inference_steps: int, 
    seed: int, 
    separate_cfg_infer: bool, 
    offload_model: bool, 
    use_input_image_size_as_output: bool,
    max_input_image_size: int, 
    randomize_seed: bool,
    subject_focus: int = 2,
    custom_outfit: str = "",
    photo_style: str = "Natural"
) -> Optional[Image.Image]:
    """Generate image with multiple subjects integrated into background with optional outfit change"""
    
    if randomize_seed:
        seed = random.randint(0, 10000000)
    
    # Ensure we have valid paths
    if not subject_images or not background_img_path:
        print("Error: Missing subject images or background image path")
        return None
    
    # Debug image paths
    print(f"Number of subject images: {len(subject_images)}")
    for i, img_path in enumerate(subject_images):
        print(f"Subject image {i+1} path: {img_path}")
        print(f"Subject image {i+1} exists: {os.path.exists(img_path)}")
    
    print(f"Background image path: {background_img_path}")
    print(f"Background image exists: {os.path.exists(background_img_path)}")
    
    # Create reference tags for each subject image
    subject_references = []
    for i in range(len(subject_images)):
        subject_references.append(f"<img><|image_{i+1}|></img>")
    
    # Background is always the last image
    background_index = len(subject_images) + 1
    background_reference = f"<img><|image_{background_index}|></img>"
    
    # Set photography style parameters based on selected style
    style, camera, lighting = "natural", "modern", "natural"
    if photo_style == "Portrait":
        style, lighting = "portrait", "soft"
    elif photo_style == "Documentary":
        style, camera = "documentary", "vintage"
    elif photo_style == "Cinematic":
        style, camera, lighting = "editorial", "cinema", "dramatic"
    
    # Define balance keywords based on focus level
    if subject_focus <= 1:
        # Environmental focus, subjects are just part of the scene
        composition_style = "wide-angle environmental shot, subjects as elements of the larger scene, authentic candid moment"
        subject_focus_terms = "subtle natural presence in frame, part of the environment, mid-distance framing"
        technical_terms = f"wide aperture f/8-f/11, expansive depth of field, {CAMERA_DETAILS[camera]}" 
    elif subject_focus == 2:
        # Balanced - equal focus on subjects and environment
        composition_style = "balanced environmental portrait, natural interaction with surroundings, medium-wide shot"
        subject_focus_terms = "natural presence in setting, authentic interaction with environment, genuine expression"
        technical_terms = f"standard aperture f/5.6-f/8, good depth of field, {CAMERA_DETAILS[camera]}"
    elif subject_focus == 3:
        # Moderate subject focus but still natural
        composition_style = "classic environmental portrait, medium shot, natural positioning and posture"
        subject_focus_terms = "clearly visible subject features, naturally positioned in setting, authentic moment"
        technical_terms = f"medium aperture f/4-f/5.6, moderate depth of field, {CAMERA_DETAILS[camera]}"
    elif subject_focus == 4:
        # Strong subject focus while maintaining some environment
        composition_style = "intimate environmental portrait, medium-close framing, natural pose and expression"
        subject_focus_terms = "prominent subject features, authentic expression, contextual environment"
        technical_terms = f"wide aperture f/2.8-f/4, shallow depth of field, {CAMERA_DETAILS[camera]}"
    else:  # level 5
        # Subject-dominant
        composition_style = "intimate portrait, tight framing, natural expression and authentic detail"
        subject_focus_terms = "lifelike subject detail, authentic expression, subtle background context"
        technical_terms = f"very wide aperture f/1.4-f/2.8, extremely shallow depth of field, {CAMERA_DETAILS[camera]}"
    
    # Outfit customization logic
    outfit_description = ""
    if custom_outfit and custom_outfit.strip():
        outfit_description = f", wearing {custom_outfit.strip()}"
        # Adjust composition for outfit visibility based on focus level
        if subject_focus >= 3:
            outfit_description += ", outfit clearly visible while maintaining natural look"
        else:
            outfit_description += ", outfit visible as part of the natural scene"
    
    # Build the final blend prompt with photography terminology
    if len(subject_images) == 1:
        # Single subject case
        base_prompt = f"A natural {PHOTOGRAPHY_STYLES[style]} where the person from {subject_references[0]} is in the authentic environment from {background_reference}{outfit_description}. {text}"
        photo_prompt = f", {composition_style}, {subject_focus_terms}, {technical_terms}, {LIGHTING_OPTIONS[lighting]}"
        natural_terms = ", perfect natural integration, authentic human features, realistic skin texture, natural positioning, candid unposed moment, photojournalistic quality"
        blend_prompt = f"{base_prompt}{photo_prompt}{natural_terms}"
    else:
        # Multiple subjects case
        people_references = ", ".join([f"person {i+1} from {ref}" for i, ref in enumerate(subject_references)])
        base_prompt = f"A natural {PHOTOGRAPHY_STYLES[style]} in the authentic environment from {background_reference} with {people_references}{outfit_description}. {text}"
        photo_prompt = f", {composition_style}, {subject_focus_terms}, {technical_terms}, {LIGHTING_OPTIONS[lighting]}"
        natural_terms = ", perfect natural integration, authentic human interaction, realistic features, natural group positioning, candid unposed moment, photojournalistic quality"
        blend_prompt = f"{base_prompt}{photo_prompt}{natural_terms}"
    
    # Combine all images for OmniGen (subjects first, then background)
    input_images = subject_images + [background_img_path]
    
    print(f"Running final generation with prompt: {blend_prompt}")
    print(f"Input images: {input_images}")
    print(f"img_guidance_scale: {img_guidance_scale}")
    
    try:
        output = pipe(
            prompt=blend_prompt,
            input_images=input_images,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            num_inference_steps=inference_steps,
            separate_cfg_infer=separate_cfg_infer,
            use_kv_cache=True,
            offload_kv_cache=True,
            offload_model=offload_model,
            use_input_image_size_as_output=use_input_image_size_as_output,
            seed=seed,
            max_input_image_size=max_input_image_size,
        )
        
        if not output or len(output) == 0:
            print("Error: No output generated")
            return None
            
        final_img = output[0]
        
        # Save intermediate results for debugging
        debug_path = os.path.join(TEMP_DIR, f"debug_final_{random.randint(0, 1000000)}.png")
        final_img.save(debug_path)
        print(f"Saved debug image to: {debug_path}")
        
        # More natural post-processing: Slightly enhance sharpness for final image
        enhancer = ImageEnhance.Sharpness(final_img)
        final_img = enhancer.enhance(1.2)  # Reduced from 1.5 for more natural look
        
        return final_img
    except Exception as e:
        print(f"Error generating final image: {e}")
        return None

# Enhanced pipeline that supports multiple subjects and outfit customization
@spaces.GPU(duration=180)
def multi_subject_generation_pipeline(
    text, 
    subject_images, 
    height, 
    width, 
    guidance_scale, 
    img_guidance_scale, 
    inference_steps,
    seed, 
    separate_cfg_infer, 
    offload_model, 
    use_input_image_size_as_output, 
    max_input_image_size, 
    randomize_seed, 
    save_images, 
    use_enhanced_prompt,
    subject_focus,
    custom_outfit,
    photo_style,
    progress=gr.Progress()
):
    """Complete pipeline that handles both background generation and multi-subject blending with outfit customization"""
    
    # Step 0: Validate input
    if not subject_images or len(subject_images) == 0:
        yield None, None, "Error: No subject images provided"
        return
    
    temp_files = []  # Keep track of temporary files to clean up later
    
    try:
        # Step 1: Enhance the prompt if requested
        if use_enhanced_prompt:
            progress(0.05, "Enhancing prompt...")
            enhanced_text = enhance_prompt_with_groq(text)
            if enhanced_text:
                text = enhanced_text
                print(f"Enhanced prompt: {text}")
        
        # Step 2: Generate background based on prompt
        progress(0.1, "Generating photorealistic background...")
        background_img = generate_background(
            text=text,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            inference_steps=inference_steps,
            seed=seed,
            separate_cfg_infer=separate_cfg_infer,
            offload_model=offload_model,
            randomize_seed=randomize_seed,
            photo_style=photo_style
        )
        
        if background_img is None:
            yield None, None, "Error: Failed to generate background"
            return
        
        # Save background image to a temporary file
        background_img_path = save_image_to_temp(background_img, "background")
        temp_files.append(background_img_path)
        
        # Yield the background image immediately
        yield None, background_img, "Photorealistic background generated successfully"
        
        # Step 3: Generate final image with proper subject and background reference
        progress(0.5, f"Creating natural integration of {len(subject_images)} subject(s)...")
        
        final_img = generate_multi_subject_image(
            text=text,
            subject_images=subject_images,  # Already filepaths from gradio
            background_img_path=background_img_path,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            inference_steps=inference_steps,
            seed=seed,
            separate_cfg_infer=separate_cfg_infer,
            offload_model=offload_model,
            use_input_image_size_as_output=use_input_image_size_as_output,
            max_input_image_size=max_input_image_size,
            randomize_seed=randomize_seed,
            subject_focus=subject_focus,
            custom_outfit=custom_outfit,
            photo_style=photo_style
        )
        
        if final_img is None:
            yield None, background_img, "Error: Failed to generate final image"
            return
        
        # Save final results if requested
        if save_images:
            timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            background_img.save(os.path.join(OUTPUT_DIR, f'{timestamp}_background.png'))
            final_img.save(os.path.join(OUTPUT_DIR, f'{timestamp}_final.png'))
            message = "Complete! Images saved to outputs folder."
        else:
            message = "Complete! Natural photorealistic integration achieved."
        
        # Return both images
        progress(1.0, message)
        yield final_img, background_img, message
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        yield None, background_img if 'background_img' in locals() else None, f"Error: {str(e)}"
    finally:
        # Clean up temporary files
        clean_temp_files(temp_files)

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Enhanced OmniGen: Natural Photorealistic Subject Integration")
        gr.Markdown("Creates authentic-looking photographs with subjects naturally integrated into realistic environments.")
    
    status_message = gr.Textbox(label="Status", interactive=False)
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Enter your scene description (what are the subjects doing, where, when, etc.)", 
                placeholder="A person enjoying a sunny day in a coffee shop, casual atmosphere...",
                lines=3
            )
            
            photo_style = gr.Radio(
                label="Photography Style", 
                choices=["Natural", "Portrait", "Documentary", "Cinematic"],
                value="Natural"
            )
            
            use_enhanced_prompt = gr.Checkbox(label="Use AI to enhance prompt (requires internet)", value=True)
            
            with gr.Row():
                # File upload for multiple subject images
                image_input = gr.Files(
                    label="Subject Images (Upload one or more)", 
                    file_types=["image"], 
                    file_count="multiple"
                )
            
            custom_outfit = gr.Textbox(
                label="Change Outfit (Optional)", 
                placeholder="e.g., red summer dress, or blue business suit with tie",
                lines=1
            )
            
            subject_focus = gr.Slider(
                label="Subject Focus", 
                minimum=0, 
                maximum=5, 
                value=2, 
                step=1
            )
            
            with gr.Row():
                gr.Markdown("""
                **Focus Guide:**  
                0-1: Environmental (subjects blend with scene)  
                2-3: Balanced (natural integration)  
                4-5: Subject-centered (portrait style)
                """)
            
            with gr.Row():
                height_input = gr.Slider(
                    label="Height", minimum=512, maximum=2048, value=1024, step=64
                )
                width_input = gr.Slider(
                    label="Width", minimum=512, maximum=2048, value=1024, step=64
                )
            
            with gr.Accordion("Advanced Settings", open=False):
                guidance_scale_input = gr.Slider(
                    label="Guidance Scale", minimum=1.0, maximum=5.0, value=3, step=0.1
                )
                
                img_guidance_scale_input = gr.Slider(
                    label="Image Guidance Scale", minimum=1.0, maximum=4.0, value=2.8, step=0.1,
                    info="Higher values preserve subject identity better"
                )
                
                num_inference_steps = gr.Slider(
                    label="Inference Steps", minimum=25, maximum=100, value=50, step=5
                )
                                
                with gr.Row():
                    seed_input = gr.Slider(
                        label="Seed", minimum=0, maximum=2147483647, value=42, step=1
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                
                max_input_image_size = gr.Slider(
                    label="Max Input Image Size", minimum=512, maximum=2048, value=1024, step=64
                )
                
                separate_cfg_infer = gr.Checkbox(
                    label="Separate CFG Inference", 
                    info="Reduces memory cost with separate inference process", 
                    value=True,
                )
                offload_model = gr.Checkbox(
                    label="Offload Model", 
                    info="Reduces memory usage but slows generation", 
                    value=False,
                )
                use_input_image_size_as_output = gr.Checkbox(
                    label="Use Input Image Size as Output", 
                    info="Match output size to input image size", 
                    value=False,
                )

            generate_button = gr.Button("Generate Photorealistic Image", variant="primary")
            save_images = gr.Checkbox(label="Save all generated images", value=False)

        with gr.Column():
            with gr.Tab("Final Result"):
                output_image = gr.Image(label="Photorealistic Blended Image")
            with gr.Tab("Background Only"):
                background_image = gr.Image(label="Generated Environment")
            with gr.Tab("Tips for Best Results"):
                gr.Markdown("""
                ## Tips for Natural-Looking Results:
                
                ### 1. Prompt Crafting
                - **Be descriptive but natural**: "Person relaxing at a beach cafe" works better than "Person at beach"
                - **Include natural activities**: Describe what the subject is doing (reading, drinking coffee, walking)
                - **Mention time of day**: Morning, afternoon, sunset helps create authentic lighting
                
                ### 2. Photography Style
                - **Natural**: Best for everyday candid shots
                - **Portrait**: Professional-looking shots with more focus on the subject
                - **Documentary**: Authentic journalistic style with film-like quality
                - **Cinematic**: Movie-like scenes with dramatic lighting and framing
                
                ### 3. Outfit Changes
                - **Be specific but simple**: "Blue jeans and white t-shirt" works better than complex descriptions
                - **Match the environment**: Suggest outfits that would make sense in the scene
                - **Works best with focus levels 3-5**: When the subject is more prominent in the image
                
                ### 4. Subject Focus
                - For the most natural integration, use level 1-3
                - Higher levels (4-5) create more portrait-like results
                - Lower levels (0-1) make the subject part of the broader scene
                """)

    # Connect the button to the pipeline
    generate_button.click(
        multi_subject_generation_pipeline,
        inputs=[
            prompt_input,
            image_input,
            height_input,
            width_input,
            guidance_scale_input,
            img_guidance_scale_input,
            num_inference_steps,
            seed_input,
            separate_cfg_infer,
            offload_model,
            use_input_image_size_as_output,
            max_input_image_size,
            randomize_seed,
            save_images,
            use_enhanced_prompt,
            subject_focus,
            custom_outfit,
            photo_style,
        ],
        outputs=[output_image, background_image, status_message],
    )

if __name__ == "__main__":
    demo.launch(share=True)  # Set share=True for public access
