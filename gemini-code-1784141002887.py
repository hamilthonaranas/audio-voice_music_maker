import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
import scipy.signal as sig
from functools import partial

# Helper constants
SEMITONES_IN_OCTAVE = 12

def get_scale_degrees(scale: str):
    """
    Returns the valid pitch classes (0-11 semitones) for a given scale.
    Example: 'C:maj' -> [0, 2, 4, 5, 7, 9, 11]
    """
    degrees = librosa.key_to_degrees(scale)
    # Append the root note transposed up 1 octave to assist wrapping/rounding calculations
    return np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))

def closest_pitch_from_scale(f0_value, scale_degrees):
    """
    Maps a single F0 frequency (Hz) to the closest allowed note in the scale.
    """
    if np.isnan(f0_value) or f0_value <= 0:
        return 0.0 # PyWorld treats 0.0 as unvoiced/silent
    
    # Convert Hz to MIDI note number (float)
    midi_note = librosa.hz_to_midi(f0_value)
    
    # Extract pitch class (where C is 0, C# is 1... up to 11)
    pitch_class = midi_note % SEMITONES_IN_OCTAVE
    
    # Find the closest pitch class allowed in our scale
    closest_degree_idx = np.argmin(np.abs(scale_degrees - pitch_class))
    degree_difference = pitch_class - scale_degrees[closest_degree_idx]
    
    # Snap the original MIDI note to the scale by subtracting the difference
    corrected_midi = midi_note - degree_difference
    
    # Convert back to Hz
    return float(librosa.midi_to_hz(corrected_midi))

def autotune_vocals(audio_path, output_path, scale="C:maj", correction_strength=1.0):
    """
    Auto-tunes a vocal file to a target musical scale using PyWorld.
    
    Parameters:
        audio_path (str): Path to input vocal WAV (must be mono, sample rate >= 16kHz recommended)
        output_path (str): Path to write corrected WAV
        scale (str): Key and scale format (e.g., 'C:maj', 'A:min', 'F#:maj')
        correction_strength (float): 0.0 (no tune) to 1.0 (hard snap)
    """
    print(f"Loading {audio_path}...")
    # PyWorld expects double precision (float64) audio
    y, sr = librosa.load(audio_path, sr=None, mono=True, dtype=np.float64)
    
    print("Analyzing vocal features (Pitch, Spectrum, Aperiodicity)...")
    # 1. Use PyWorld Harvest for robust F0 estimation (better than DIO for vocals)
    _f0, t = pw.harvest(y, sr)
    
    # 2. Extract Spectral Envelope and Aperiodicity (keeps the character/timbre of the voice)
    sp = pw.cheaptrick(y, _f0, t, sr)
    ap = pw.d4c(y, _f0, t, sr)
    
    print(f"Aligning pitch to scale: {scale}...")
    scale_degrees = get_scale_degrees(scale)
    
    # 3. Apply pitch correction to the F0 contour
    corrected_f0 = np.zeros_like(_f0)
    for i, pitch in enumerate(_f0):
        if pitch > 0:
            target_pitch = closest_pitch_from_scale(pitch, scale_degrees)
            # Interpolate between original pitch and target pitch based on correction_strength
            corrected_f0[i] = pitch + (target_pitch - pitch) * correction_strength
        else:
            corrected_f0[i] = 0.0 # Keep silent frames unvoiced
            
    # Apply a light median filter to smooth out drastic pitch jitters (prevents synthetic crackling)
    # Only smooth where voicing exists
    voiced_indices = corrected_f0 > 0
    if np.any(voiced_indices):
        smoothed_voiced = sig.medfilt(corrected_f0[voiced_indices], kernel_size=5)
        corrected_f0[voiced_indices] = smoothed_voiced

    print("Re-synthesizing audio...")
    # 4. Use PyWorld to synthesize the voice using corrected pitch + original timbre
    synthesized_audio = pw.synthesize(corrected_f0, sp, ap, sr)
    
    # Normalize to prevent clipping
    synthesized_audio = librosa.util.normalize(synthesized_audio)
    
    # Save the output file
    sf.write(output_path, synthesized_audio, sr)
    print(f"Saved autotuned audio to {output_path}!")

# --- Example Usage ---
if __name__ == "__main__":
    # Target Key examples: "C:maj", "A:min", "Eb:maj", "G#:min"
    input_vocals = "vocals.wav"  # Replace with your vocal WAV file path
    output_vocals = "vocals_autotuned.wav"
    
    # Set correction_strength to 1.0 for hard T-Pain snap, or 0.7 for natural pitch correction
    autotune_vocals(input_vocals, output_vocals, scale="C:maj", correction_strength=0.8)