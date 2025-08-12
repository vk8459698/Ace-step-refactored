import os
import torch
from acestep.pipeline_ace_step import ACEStepPipeline
from loguru import logger

def compare_dits(prompt, lyrics, audio_duration, infer_step, output_dir="comparison_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Initializing pipeline for ORIGINAL pre-trained DIT...")
    original_pipeline = ACEStepPipeline(
        # Assuming default checkpoint_dir will download the original model
        # You can specify a different checkpoint_dir if your original model is elsewhere
        load_original_pretrained_dit=True,
        dtype="float32", # Use float32 for CPU comparison if no GPU
        device_id=0, # This will fallback to CPU if no CUDA
    )
    original_pipeline.load_checkpoint()
    logger.info("Generating audio with ORIGINAL pre-trained DIT...")
    original_audio_paths = original_pipeline(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=audio_duration,
        infer_step=infer_step,
        save_path=os.path.join(output_dir, "original_dit.wav"),
    )
    logger.info(f"Original DIT audio saved to: {original_audio_paths[0]}")

    logger.info("Initializing pipeline for NEWLY TRAINED DIT (placeholder)...")
    # For your newly trained DIT, you would need to:
    # 1. Train your model and save its weights/config.
    # 2. Update the checkpoint_dir below to point to your trained model's directory.
    #    If your trained model's path is different from the default ACE-Step checkpoint structure,
    #    you might need to adjust `get_checkpoint_path` or how `ACEStepTransformer2DModel` is loaded.
    new_pipeline = ACEStepPipeline(
        # Set load_original_pretrained_dit=False to initialize from config (for from-scratch training)
        load_original_pretrained_dit=False,
        # IMPORTANT: Replace 'path/to/your/trained/dit' with the actual path to your trained model
        # You might need to make sure this path contains a valid config.json for ACEStepTransformer2DModel
        # and other necessary files if you trained it independently.
        checkpoint_dir="path/to/your/trained/dit_model", # <--- UPDATE THIS PATH
        dtype="float32",
        device_id=0,
    )
    # new_pipeline.load_checkpoint() # Uncomment and call this after your model is trained and path is set
    logger.info("Please train your DIT model and update 'checkpoint_dir' in this script to enable generation with the newly trained DIT.")
    logger.info("Skipping audio generation for newly trained DIT as placeholder.")
    new_audio_paths = []

    logger.info("\n--- Comparison Instructions ---")
    logger.info("1. Listen to the generated audio files:")
    logger.info(f"   - Original DIT: {os.path.abspath(original_audio_paths[0])}")
    if new_audio_paths:
        logger.info(f"   - Newly Trained DIT: {os.path.abspath(new_audio_paths[0])}")
    else:
        logger.info("   - Newly Trained DIT audio will be generated here once you update the script with your trained model path.")
    logger.info("\n2. Qualitatively compare them based on:")
    logger.info("   - Musicality (coherence, pleasing sound)")
    logger.info("   - Prompt Adherence (how well it matches the text/lyrics)")
    logger.info("   - Lyric Synchronization/Clarity (if applicable)")
    logger.info("   - Absence of Artifacts/Noise")
    logger.info("   - Overall Quality and Aesthetic Appeal")
    logger.info("\n3. Document your observations and insights.")

if __name__ == "__main__":
    test_prompt = "A lively jazz piece with a saxophone solo."
    test_lyrics = "Smooth rhythms, a gentle tune, under the silver moon."
    test_duration = 30.0
    test_infer_step = 60

    compare_dits(test_prompt, test_lyrics, test_duration, test_infer_step)
