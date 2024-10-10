import os
from pathlib import Path
import torch
import torchaudio
import demucs
from pydub import AudioSegment
import subprocess


from demucs import pretrained
from demucs.apply import apply_model
from mir_eval import separation
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.utils import download_asset
from torchaudio.transforms import Fade

from typing import Tuple, List, Dict
import math

# **Note:** The un-preprocessed Data Directory should be of the architecture:
# **directory**/
# ├── class_x
# │     ├── xxx.wav
# │     ├── xxy.mp3
# │     └── ...
# │     └── xxz.flac
# └── class_y
#       ├── 123.mp3
#       ├── nsdf3.aac
#       └── ...
#       └── asd932_.aiff

# *Build a Function which takes input as an audio file:
# 1. Normalizes the audio to a certain threshold and returns the normalized wave tensor,
# 2. Removes it's non-vocal Section and returns the wave tensor of the vocal section,
# 3. Splits it into equal shape tensors,
# 4. and then saves the splitted wave tensors into .wav file format into a new Directory (Dataset).


device = "cuda" if torch.cuda.is_available() else "cpu"

def convert_to_wav(input_file_path: str) -> str:
    """Convert non-wav files to .wav file type and saves them in the same directory as the input_file.

    Args:
        input_file_path (str): File Path of the audio to be preprocessed.

    Returns:
        str: File Path of the output .wav file type, if the input_file_path is not of .wav file type, else returns the same input path.
    """
    
    output_file_path = fr"{".".join(input_file_path.split(".")[:-1])}.wav"
    
    if os.path.basename(input_file_path).split(".")[-1] != "wav":
        sound = AudioSegment.from_file(input_file_path)
        sound.export(output_file_path, format="wav")
    
    return output_file_path


def normalize(wave_tensor: str, dB: int=80, device: torch.device=device) -> torch.Tensor:
    """Takes in the path of the audio file and normalizes it to a certain dB.

    Args:
        input_file (_type_): Takes in the path of the audio file.
        dB (int): The Decibal at which the audio should be normalized.

    Returns:
        tuple(Tensor, int): Returns a tuple containing Wave Tensor and Sample Rate.
    """
    transform = torchaudio.transforms.AmplitudeToDB("power", dB).to(device)
    # return torchaudio.transforms.AmplitudeToDB("power", dB).__call__(wave_tensor.to(device)).to(device)
    return transform(wave_tensor.to(device)).cpu()


def remove_non_vocals(wave_tensor: torch.Tensor, device: torch.device=device) -> torch.Tensor:
    """_summary_

    Args:
        wave_tensor (torch.Tensor): Input wave tensor

    Returns:
        Tensor: Returns the Vocal section of the audio.
    """
    # Load Pretrained Demucs Model
    # bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = pretrained.get_model("htdemucs")
    
    wave_tensor = wave_tensor.to(device)
    
    # Apply the model to Seperate the sources, which are in the form [Drums, Bass, Others, Vocals]
    *_, vocal_tensor = apply_model(model, wave_tensor.unsqueeze(0), progress=True, device=device).to(device).squeeze(0)
    return vocal_tensor.cpu()

bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model().to(device)
sample_rate = bundle.sample_rate

def separate_sources(mix: torch.Tensor,
                     model=model,
                     segment=10.0,
                     overlap=0.1,
                     device: torch.device=device):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        model (torch.nn.Module): Model to separate the tracks.
        mix (torch.Tensor): The Wave Tensor having 3 no. of dimensions. [batch, channels, no. of frames]
        segment (int): segment length in seconds
        device (torch.device): if provided, device on which to execute the computation.
    """
    batch, channels, length = mix.shape
    
    chunk_len = int(sample_rate*segment*(1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = sample_rate*overlap
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")
    
    final = torch.zeros(batch, len(model.sources), channels, length, device=device)
    
    while start < length-overlap_frames:
        chunk = mix[:, :, start:end]
        
        with torch.inference_mode():
            out = model.forward(chunk.to(device))
        out = fade(out)
        final[:, :, :, start:end] += out
        
        if start==0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end>length:
            fade.fade_out_len = 0
    
    return final[:, 3, :, :][0]

def split_into_equal_tensors(wave_tensor: torch.Tensor, window_size: int, hop_length: int, device: torch.device=device) -> torch.Tensor:
    """Takes in a Single Wave Tensor and splits it into equal shape of length window_size.

    Args:
        wave_tensor (torch.Tensor): Tensor to be splitted
        window_size (int): Length of the splits
        hop_length (int): Stride of the window. Determines the distance between the current and next split.

    Returns:
        Tensor: Returns a Tensor with an added Dimension. [Num of chunks, Channels, Num of Frames]
    """
    wave_tensor = wave_tensor.to(device)
    num_frames = wave_tensor.shape[-1]
    
    num_chunks = math.floor(((num_frames - window_size) / hop_length) + 1)
    start_idx=0
    end_idx=window_size
    chunk_tensor = []
    for _ in range(num_chunks):
        chunk_tensor.append(wave_tensor[:, start_idx:end_idx].unsqueeze(0))
        
        start_idx += hop_length
        end_idx += hop_length
    chunk_tensor = torch.cat(chunk_tensor, dim=0).cpu()
    return chunk_tensor


def save_tensors_to_directory(unsqueezed_tensor: torch.Tensor, sr: int, input_path: str, output_dir: str, class_name: str):
    r"""Saves each Audio Tensor from the unsueezed_tensor in output_dir\classname as .wav file type.

    Args:
        unsqueezed_tensor (torch.Tensor): Chunks of wave tensor with dimension 3. [Num of chunks, Channels, Num of Frames]
        sr (int): Sample Rate.
        input_path (str): Input File path of the Audio.
        output_dir (str): Path of the Directory in which the processed audio sample is going to be saved.
        class_name (str): Name of the class (label) this audio file belongs to.
    """
    class_path = os.path.join(output_dir, class_name)
    os.makedirs(fr"{class_path}", exist_ok=True)
    
    file_name = os.path.join(class_path, Path(input_path).stem)
    
    for i, wave_tensor in enumerate(unsqueezed_tensor):
        torchaudio.save(uri=fr"{file_name}_{i}.wav",
                        src=wave_tensor,
                        sample_rate=sr,
                        format="wav")
    print(f"Saved chunks of {Path(input_path).stem} in {class_path}")
    
    return None


if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Demucs Version: {demucs.__version__}")
    print(f"Cuda Avalilable: {torch.cuda.is_available()}")