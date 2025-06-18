# AI-Powered Video Analyzer

A Python-based video analysis tool that uses multiple AI models to extract insights from video content, including transcription, object detection, and scene description.

## Overview

This project is based on the original work by [Arash Sajjadi](https://github.com/arashsajjadi/ai-powered-video-analyzer). It has been modified and enhanced to meet specific requirements while maintaining the core concept of AI-powered video analysis.

## Features

- Speech transcription using OpenAI's Whisper
- Object detection (planned)
- Scene description (planned)
- Audio event detection (planned)
- Content summarization (planned)

## Installation

1. Clone this repository:
2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. If you want to use speaker diarization (whisperx), make sure:
- CUDA is installed
- Torch is installed properly per the command generated on https://pytorch.org/
- You have an API key for Hugging Face
- Set up Hugging Face access:
   - Get your token from https://huggingface.co/settings/tokens
   - Create a `.env` file in the project root
   - Add your token: `HUGGING_FACE_TOKEN=your_token_here`
   - Ensure `.env` is in your `.gitignore`

4. Install in development mode:
```bash
pip install -e .
```

## Usage

Basic usage example:
```bash
python -m src.cli analyze video "path/to/video.mp4" --transcription-language en
```

## Project Structure

```
ai-powered-video-analyzer/
├── src/
│   ├── ai_models/       # AI model implementations
│   ├── core/           # Core processing logic
│   ├── cli/            # Command-line interface
│   └── config/         # Configuration management
├── tests/              # Test suite
└── setup.py           # Package configuration
```

## Configuration

Configuration is managed through environment variables and/or a settings file. See `src/config/settings.py` for available options.

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Credits

- Original concept and implementation by [Arash Sajjadi](https://github.com/arashsajjadi/ai-powered-video-analyzer)
- This project uses several open-source AI models:
  - [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition

## License

This project maintains the same license as the original work by Arash Sajjadi. Please refer to the original repository for license details.




