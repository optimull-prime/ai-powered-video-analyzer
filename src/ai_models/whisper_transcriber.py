import logging
from pathlib import Path
import tempfile
import shutil
import os
import json

import whisper
import whisperx
import torch
from src.utils.device_utils import get_device_settings
from src.config import settings

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_path: str = "base"): # model_path can be size like "base", "small", etc.
        self.model_name = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{self.model_name}': {e}")
            # You might want to raise a custom exception or handle this gracefully
            raise

    def transcribe(self, audio_path: str, language: str = None) -> str:
        if not self.model:
            logger.error("Whisper model is not loaded. Cannot transcribe.")
            return "Error: Transcription model not loaded."

        temp_path = None
        try:
            # Convert to Path object and resolve it
            audio_file = Path(audio_path).resolve()
            
            # Add detailed 
            #  logging
            logger.debug("=== Whisper Audio File Debug Info ===")
            logger.debug(f"Input path: {audio_path}")
            logger.debug(f"Resolved path: {audio_file}")
            logger.debug(f"File exists: {audio_file.exists()}")
            
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            # Convert to string with forward slashes only when passing to whisper
            audio_path_str = str(audio_file).replace("\\", "/")
            logger.debug(f"Normalized path: {audio_path_str}")
            
            options = {}
            if language:
                options["language"] = language

            # Use the string version for whisper functions
            audio = whisper.audio.load_audio(audio_path_str)
            logger.debug(f"Audio loaded successfully, shape: {audio.shape}")


            # Use the string version for transcribe
            logger.debug("Starting transcription...")
            result = self.model.transcribe(audio_path_str, **options)
            if result is None:
                raise RuntimeError("Transcription failed - no result returned")
            logger.debug("Transcription completed successfully.")
            return result["text"]

        except Exception as e:
            logger.error(f"Error during transcription of {audio_path}: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            logger.error("Exception details:", exc_info=True)
            return f"Error during transcription: {e}"

class WhisperXTranscriber:
    def __init__(self, model_path: str = "base", device: str = None):
        """Initialize WhisperX transcriber.
        
        Args:
            model_path (str): Path to model or size name ("base", "small", etc.)
            device (str, optional): Override for device selection. 
        """
        self.model_name = model_path
        
        # Get optimal device settings if not overridden
        if device is None:
            self.device, self.compute_type, self.has_gpu = get_device_settings()
        else:
            self.device = device
            self.compute_type = "float32" if device == "cuda" else "int8"
            self.has_gpu = device == "cuda"
            
        self.model = None
        self.diarization_pipeline = None
        
        logger.info(f"WhisperX initializing with: device={self.device}, "
                   f"compute_type={self.compute_type}")
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading WhisperX model: {self.model_name} for device {self.device} and compute type {self.compute_type}")

            # Check hugging face token
            if not settings.HUGGING_FACE_TOKEN:
                logger.warning("No Hugging Face token found. Diarization may fail.  See README.")


            # Load model with appropriate settings
            model_kwargs = {
                "device": self.device,
                "compute_type": self.compute_type,
            }
            self.model = whisperx.load_model(self.model_name, **model_kwargs)
            
                
            self.diarization_pipeline = whisperx.DiarizationPipeline(
                use_auth_token=settings.HUGGING_FACE_TOKEN,
                device=self.device
            )
            
            logger.info(f"WhisperX model loaded successfully on {self.device} "
                       f"using {self.compute_type}")
        except Exception as e:
            logger.error(f"Failed to load WhisperX model '{self.model_name}': {e}")
            raise

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        if not self.model or not self.diarization_pipeline:
            logger.error("WhisperX model or diarization pipeline not loaded.")
            return {"error": "Transcription model not loaded."}

        try:
            audio_file = Path(audio_path).resolve()
            logger.debug(f"Processing audio file: {audio_file}")
            
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            # Initial transcription
            logger.debug("Starting WhisperX transcription...")
            audio = whisperx.load_audio(str(audio_file))
            result = self.model.transcribe(audio, language=language)

            # Align whisper output
            logger.debug("Aligning transcription...")
            result = whisperx.align(result["segments"], self.model, 
                                  audio, self.device, 
                                  return_char_alignments=False)

            # Diarize with speaker labels
            logger.debug("Performing diarization...")
            diarize_segments = self.diarization_pipeline(audio)
            
            # Assign speaker labels to segments
            logger.debug("Assigning speaker labels...")
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Format the result
            formatted_result = {
                "text": " ".join([segment["text"] for segment in result["segments"]]),
                "segments": result["segments"],
                "speaker_segments": [
                    {
                        "speaker": segment["speaker"],
                        "text": segment["text"],
                        "start": segment["start"],
                        "end": segment["end"]
                    }
                    for segment in result["segments"]
                ]
            }

            logger.info("WhisperX transcription completed successfully")
            return formatted_result

        except Exception as e:
            logger.error(f"Error during WhisperX transcription: {str(e)}")
            logger.error("Exception details:", exc_info=True)
            return {"error": f"Error during transcription: {e}"}

# Similar handler classes would be created for YOLO, BLIP, PANNs, and the LLM.
# Each would load its specific model and provide a clean interface.
# For example, llm_summarizer.py would interface with Ollama.