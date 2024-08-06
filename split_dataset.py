import wave
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def split_wav_file(input_file, output_dir, num_chunks=20, target_framerate=44100):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the input WAV file
    with wave.open(input_file, 'rb') as wav:
        # Get properties of the input file
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()
        
        if framerate != target_framerate:
            raise ValueError(f"Input file sample rate is {framerate} Hz, but expected {target_framerate} Hz.")
        
        # Calculate the number of frames per chunk
        frames_per_chunk = n_frames // num_chunks
        
        # Read all frames from the input file
        frames = wav.readframes(n_frames)
        
        # Split and write chunks
        for i in range(num_chunks):
            chunk_frames = frames[i * frames_per_chunk * sampwidth * n_channels : (i + 1) * frames_per_chunk * sampwidth * n_channels]
            output_file = os.path.join(output_dir, f'chunk_{i+1}.wav')
            
            with wave.open(output_file, 'wb') as chunk_wav:
                chunk_wav.setnchannels(n_channels)
                chunk_wav.setsampwidth(sampwidth)
                chunk_wav.setframerate(target_framerate)
                chunk_wav.writeframes(chunk_frames)
                
            print(f'Created {output_file}')

def process_directory(input_dir, output_dir, num_chunks=20, target_framerate=44100):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_dir, filename)
            split_wav_file(input_file, output_dir, num_chunks, target_framerate)

if __name__ == "__main__":
    input_dir = os.getenv('INPUT_DIR')
    output_dir = os.getenv('OUTPUT_DIR', '/home/marek/Desktop/code/melgan-neurips/data/wavs/')
    
    if not input_dir:
        print("Please set the INPUT_DIR environment variable.")
        sys.exit(1)
    
    process_directory(input_dir, output_dir)

