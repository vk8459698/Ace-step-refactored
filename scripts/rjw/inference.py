import os
import pandas as pd
from acestep.pipeline_ace_step import ACEStepPipeline
import tqdm
import itertools


def main():
    USER = os.environ.get("USER")
    PARENT_DIR = f"/home/{USER}/soundverse/ACE-step-lowvram"
    checkpoint_dir = f"/home/{USER}/.cache/ace-step/checkpoints/"

    style_prompts_path = os.path.join(PARENT_DIR, "scripts/rjw/test_data/styles.csv")
    lyrics_path = os.path.join(PARENT_DIR, "scripts/rjw/test_data/lyrics.txt")

    style_prompts = (
        pd.read_csv(filepath_or_buffer=style_prompts_path, header=None)
        .values.flatten()
        .tolist()
    )
    lyrics = (
        pd.read_csv(filepath_or_buffer=lyrics_path, header=None)
        .values.flatten()
        .tolist()
    )

    # ===== On T4: =====
    # BF16 = False
    # TORCH_COMPILE = True
    # CPU_OFFLOAD = True
    # OVERLAPPED_DECODE = True  # maybe try false?

    # ===== On L4: ===== (make parameters)
    BF16 = True
    TORCH_COMPILE = True
    CPU_OFFLOAD = False
    OVERLAPPED_DECODE = False  # maybe try false?
    MANUAL_SEEDS = 296772417648513
    AUDIO_DURATION = 43
    OUTPUT_PATH = "/mnt/disks/"
    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_dir,
        dtype="bfloat16" if BF16 else "float32",
        torch_compile=TORCH_COMPILE,
        cpu_offload=CPU_OFFLOAD,
        overlapped_decode=OVERLAPPED_DECODE,
    )
    pairs = list(itertools.product(style_prompts, lyrics))

    for _prompt, _lyrics in tqdm.tqdm(pairs):
        output_paths, input_params_json = model_demo(
            format="wav",
            audio_duration=AUDIO_DURATION,  #: float = 60.0,
            prompt=_prompt,
            lyrics=_lyrics,  #     lyrics: str = None,
            infer_step=80,
            scheduler_type="heun",
            # scheduler_type = "pingpong",
            # scheduler_type = "euler", # "euler" is recommended, heun will take more time
            cfg_type="apg",
            omega_scale=10.0,  # (this is "granularity" in the GUI, defaults to 10)
            manual_seeds=MANUAL_SEEDS,  # None,
            guidance_scale=15.0,  # When guidance_scale_lyric > 1 and guidance_scale_text > 1, the guidance scale will not be applied.
            guidance_interval=0.5,
            guidance_interval_decay=0.0,
            min_guidance_scale=3.0,
            use_erg_tag=True,
            use_erg_lyric=True,
            use_erg_diffusion=True,
            # use_erg_tag: bool = True,
            # use_erg_lyric: bool = True,
            # use_erg_diffusion: bool = True,
            # guidance_scale_text: float = 0.0,
            # guidance_scale_lyric: float = 0.0,
            # guidance_scale_text = 7,
            # guidance_scale_lyric = 2,
            # audio2audio_enable = True,
            # ref_audio_strength = 0.10, # 0.18 was good on Gradio site...
            # ref_audio_input = ASH_TEST_AUDIO_PATH,
        )
        print("Output paths:", output_paths)
        # print("Input parameters JSON:", input_params_json)
        print("Inference completed successfully.")


if __name__ == "__main__":
    # This is just for testing purposes, to run the inference script directly
    # and see the output paths.
    main()
