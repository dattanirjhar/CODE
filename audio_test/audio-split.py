from pydub import AudioSegment
import os

def split_audio_single_file(input_file, output_dir, segment_length_ms):
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the audio file
    try:
        audio = AudioSegment.from_file(input_file)  # Supports various formats (wav, mp3, etc.)
        
        # Get the filename without extension for naming segments
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Split into 5-second segments
        for i in range(0, len(audio), segment_length_ms):
            segment = audio[i:i + segment_length_ms]
            segment_name = f"{file_name}_part{i//segment_length_ms + 1}.wav"
            segment_path = os.path.join(output_dir, segment_name)
            segment.export(segment_path, format="wav")
            print(f"Saved: {segment_path}")
            
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# Example usage
input_audio = "/home/nirjhar/Desktop/shreya/shreya.wav"  # Replace with your audio file path
output_directory = "/home/nirjhar/CODE/audio_test/dataset_5s/shreya"  # Replace with your desired output directory
split_audio_single_file(input_audio, output_directory, segment_length_ms=5000)