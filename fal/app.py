from pathlib import Path
from typing import List, Literal, Tuple

import fal
from fal.container import ContainerImage
from fal.exceptions import FieldException
from fal.toolkit import Audio, File, FAL_PERSISTENT_DIR

from pydantic import BaseModel, Field


class Input(BaseModel):
    text_input: List[str] = Field(
        description="The text to transform into speech.",
        default=[
            'Speaker 1: Hey, remember "See You Again"',
            'Speaker 2: Yeah… from Furious 7, right? That song always hits deep."',
        ],
        ui={"important": True},
    )
    speaker_names: List[str] = Field(
        description="speaker names in order",
        default=["Alice", "Bob"],
        ui={"important": True},
    )
    # voices: List[
    #     Literal[
    #         "en-Alice_woman",
    #         "en-Alice_woman_bgm",
    #         "en-Carter_man",
    #         "en-Frank_man",
    #         "en-Maya_woman",
    #         "in-Samuel_man",
    #         "zh-Anchen_man_bgm",
    #         "zh-Bowen_man",
    #         "zh-Xinran_woman",
    #     ]
    # ] = Field(
    #     description="'Key' of voices to the 'Value' of speaker names",
    #     default=["en-Alice_woman", "en-Frank_man"],
    #     ui={"important": True},
    # )
    cfg_scale: float = Field(
        description="CFG (Classifier-Free Guidance) scale for generation", default=1.3
    )


class Output(BaseModel):
    audio: File = Field(description="The generated audio file.")


def download_model(repo_id):
    import os
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        print(f"Checking for pre-downloaded model '{repo_id}'")
        snapshot_download(
            repo_id=repo_id,
            local_files_only=True,
        )
        print(f"Found pre-downloaded model '{repo_id}'")
    except LocalEntryNotFoundError:
        print(f"Downloading model '{repo_id}'")
        snapshot_download(
            repo_id=repo_id,
        )
        print(f"Downloaded model '{repo_id}'")
    display_tree(
        Path(os.environ["HF_HOME"])
        / "hub"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
    )


class VibeVoiceApp(
    fal.App,
    name="ms-vibevoice",
    image=ContainerImage.from_dockerfile("./Dockerfile"),
    kind="container",
    keep_alive=300,
    startup_timeout=600,
):
    machine_type = "GPU-A100"

    def setup(self):
        import os
        from pathlib import Path

        if False:
            # if on a small machine and a prior download was incomplete
            # activate this if-block to force redownload the model files
            import shutil

            repo_id = "WestZhang/VibeVoice-Large-pt"
            shutil.rmtree(
                Path(os.environ["HF_HOME"])
                / "hub"
                / f"models--{repo_id.replace('/', '--')}"
            )
            print(f"Removed prior {repo_id} download...")

        # https://github.com/microsoft/VibeVoice
        # https://huggingface.co/WestZhang/VibeVoice-Large-pt
        download_model("WestZhang/VibeVoice-Large-pt")
        download_model("Qwen/Qwen2.5-7B")

        assert os.environ["HF_HOME"].startswith(str(FAL_PERSISTENT_DIR))
        self.model_path = (
            Path(os.environ["HF_HOME"])
            / "hub/models--WestZhang--VibeVoice-Large-pt/snapshots/0b68ee6da8ca6bca98484758d06cbe9c33f49e7b"
        )
        assert self.model_path.exists()

        # clone of https://github.com/microsoft/VibeVoice/tree/main/demo
        voices_dir = Path("/repo/VibeVoice/demo/voices")
        self.voice_presets = {}
        wav_files = [
            f
            for f in os.listdir(voices_dir)
            if f.lower().endswith(".wav")
            and os.path.isfile(os.path.join(voices_dir, f))
        ]

        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path

        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))

        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path
            for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

        if False:
            # enable to run this warmup during debug/development
            # helps you know if the processors works without deploying
            self.text_to_audio(Input())

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        # First try exact match
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        # Try partial matching (case insensitive)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if (
                preset_name.lower() in speaker_lower
                or speaker_lower in preset_name.lower()
            ):
                return path

        # Default to first voice if no match found
        default_voice = list(self.voice_presets.values())[0]
        print(
            f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}"
        )
        return default_voice

    @fal.endpoint("/")
    def text_to_audio(self, input: Input) -> Output:
        import os
        import time
        import torch
        import sys

        sys.path.insert(0, "/repo/VibeVoice")
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        # hey gemini fill in the implementaiton here
        txt_content = input.text_input
        scripts, speaker_numbers = parse_txt_script(txt_content)
        if not scripts:
            raise FieldException(
                field="text_input",
                message="Error: No valid speaker scripts found in the text input",
            )

        print(f"Found {len(scripts)} speaker segments:")
        for i, (script, speaker_num) in enumerate(zip(scripts, speaker_numbers)):
            print(f"  {i + 1}. Speaker {speaker_num}")
            print(f"     Text preview: {script[:100]}...")

        # Map speaker numbers to provided speaker names
        speaker_name_mapping = {}
        speaker_names_list = (
            input.speaker_names
            if isinstance(input.speaker_names, list)
            else [input.speaker_names]
        )
        for i, name in enumerate(speaker_names_list, 1):
            speaker_name_mapping[str(i)] = name

            print("\nSpeaker mapping:")
            for speaker_num in set(speaker_numbers):
                mapped_name = speaker_name_mapping.get(
                    speaker_num, f"Speaker {speaker_num}"
                )
                print(f"  Speaker {speaker_num} -> {mapped_name}")

            # Map speakers to voice files using the provided speaker names
            voice_samples = []
            actual_speakers = []

            # Get unique speaker numbers in order of first appearance
            unique_speaker_numbers = []
            seen = set()
            for speaker_num in speaker_numbers:
                if speaker_num not in seen:
                    unique_speaker_numbers.append(speaker_num)
                    seen.add(speaker_num)

            for speaker_num in unique_speaker_numbers:
                speaker_name = speaker_name_mapping.get(
                    speaker_num, f"Speaker {speaker_num}"
                )
                voice_path = self.get_voice_path(speaker_name)
                voice_samples.append(voice_path)
                actual_speakers.append(speaker_name)
                print(
                    f"Speaker {speaker_num} ('{speaker_name}') -> Voice: {os.path.basename(voice_path)}"
                )

            # Prepare data for model
            full_script = "\n".join(scripts)

            # Load processor
            print(f"Loading processor & model from {self.model_path}")
            try:
                processor = VibeVoiceProcessor.from_pretrained(self.model_path)
            except OSError:
                display_tree(self.model_path)

            # Load model
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation="flash_attention_2",  # we only test flash_attention_2
            )

            model.eval()
            model.set_ddpm_inference_steps(num_steps=10)

            if hasattr(model.model, "language_model"):
                print(
                    f"Language model attention: {model.model.language_model.config._attn_implementation}"
                )

            # Prepare inputs for the model
            inputs = processor(
                text=[full_script],  # Wrap in list for batch processing
                voice_samples=[voice_samples],  # Wrap in list for batch processing
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            print(f"Starting generation with cfg_scale: {input.cfg_scale}")

            # Generate audio
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=input.cfg_scale,
                tokenizer=processor.tokenizer,
                # generation_config={'do_sample': False, 'temperature': 0.95, 'top_p': 0.95, 'top_k': 0},
                generation_config={"do_sample": False},
                verbose=True,
            )
            generation_time = time.time() - start_time
            print(f"Generation time: {generation_time:.2f} seconds")

            # Calculate audio duration and additional metrics
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                # Assuming 24kHz sample rate (common for speech synthesis)
                sample_rate = 24000
                audio_samples = (
                    outputs.speech_outputs[0].shape[-1]
                    if len(outputs.speech_outputs[0].shape) > 0
                    else len(outputs.speech_outputs[0])
                )
                audio_duration = audio_samples / sample_rate
                rtf = (
                    generation_time / audio_duration
                    if audio_duration > 0
                    else float("inf")
                )

                print(f"Generated audio duration: {audio_duration:.2f} seconds")
                print(f"RTF (Real Time Factor): {rtf:.2f}x")
            else:
                print("No audio output generated")

            # Calculate token metrics
            input_tokens = inputs["input_ids"].shape[1]  # Number of input tokens
            output_tokens = outputs.sequences.shape[
                1
            ]  # Total tokens (input + generated)
            generated_tokens = output_tokens - input_tokens

            print(f"Prefilling tokens: {input_tokens}")
            print(f"Generated tokens: {generated_tokens}")
            print(f"Total tokens: {output_tokens}")

            # Save output
            output_id = int(time.time() * 1e6)
            output_dir = FAL_PERSISTENT_DIR / "vibevoice-output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{output_id}_generated.wav")

            processor.save_audio(
                outputs.speech_outputs[0],  # First (and only) batch item
                output_path=output_path,
            )
            print(f"Saved output to {output_path}")

            # Print summary
            print("\n" + "=" * 50)
            print("GENERATION SUMMARY")
            print("=" * 50)
            print(f"Output file: {output_path}")
            print(f"Speaker names: {input.speaker_names}")
            print(f"Number of unique speakers: {len(set(speaker_numbers))}")
            print(f"Number of segments: {len(scripts)}")
            print(f"Prefilling tokens: {input_tokens}")
            print(f"Generated tokens: {generated_tokens}")
            print(f"Total tokens: {output_tokens}")
            print(f"Generation time: {generation_time:.2f} seconds")
            print(f"Audio duration: {audio_duration:.2f} seconds")
            print(f"RTF (Real Time Factor): {rtf:.2f}x")

            print("=" * 50)

        return Output(audio=Audio.from_path(output_path))


def parse_txt_script(text_content_lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Parse txt script content and extract speakers and their text
    Fixed pattern: Speaker 1, Speaker 2, Speaker 3, Speaker 4
    Returns: (scripts, speaker_numbers)
    """
    import re

    lines = text_content_lines
    scripts = []
    speaker_numbers = []

    # Pattern to match "Speaker X:" format where X is a number
    speaker_pattern = r"^Speaker\s+(\d+):\s*(.*)$"

    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            # If we have accumulated text from previous speaker, save it
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)

            # Start new speaker
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            # Continue text for current speaker
            if current_text:
                current_text += " " + line
            else:
                current_text = line

    # Don't forget the last speaker
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)

    return scripts, speaker_numbers


def display_tree(directory: str | Path, max_depth: int = 2):
    """
    Displays a tree of files and directories for a given path up to a specified depth.

    Args:
        directory (str | Path): The starting directory path.
        max_depth (int): The maximum depth to display. Defaults to 2.
    """
    from pathlib import Path

    base_path = Path(directory)
    if not base_path.is_dir():
        print(f"Error: '{base_path}' is not a valid directory.")
        return

    print(f"{base_path.name}/")
    _walk_directory(base_path, max_depth=max_depth)


def _walk_directory(
    path: Path, prefix: str = "", max_depth: int = 2, current_depth: int = 0
):
    """Helper function to recursively walk the directory."""
    if current_depth >= max_depth:
        return

    # Get directory contents and sort them
    try:
        contents = sorted(list(path.iterdir()))
    except PermissionError:
        print(f"{prefix}└── [Permission Denied]")
        return

    pointers = ["├── "] * (len(contents) - 1) + ["└── "]

    for pointer, item in zip(pointers, contents):
        print(f"{prefix}{pointer}{item.name}")
        if item.is_dir():
            # Determine the prefix for the next level
            extension = "│   " if pointer == "├── " else "    "
            _walk_directory(
                item,
                prefix=prefix + extension,
                max_depth=max_depth,
                current_depth=current_depth + 1,
            )
