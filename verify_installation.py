
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os

# Define local paths
model_path = "./Qwen3-TTS-12Hz-1.7B-CustomVoice"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {model_path} on {device}...")

try:
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        attn_implementation="eager" if device == "cpu" else "flash_attention_2",
    )
    
    print("Model loaded successfully.")
    
    text = "Olá, este é um teste do Qwen TTS rodando localmente no seu computador."
    
    print(f"Generating audio for text: '{text}'")
    
    wavs, sr = model.generate_custom_voice(
        text=text,
        language="Portuguese",
        speaker="Vivian", # Using a default speaker
    )
    
    output_file = "output_custom_voice.wav"
    sf.write(output_file, wavs[0], sr)
    
    print(f"Audio generated and saved to {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")
