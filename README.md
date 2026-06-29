# Voice Assistant
Voice assistant was created to be a private, open source alternative to Google Home, Alexa, etc. As such it has no cloud/service provider dependencies beyond PyPi for library installs. It aims to be modular, so that each main component can run on its own server, or all parts can run on one machine. Components can be swapped out with custom code while leaving the rest of the system intact. New skills can be added by writing new python modules and registering them with the intent handler.

## Privacy through Local Processing
Voice Assistant uses OpenWakeWord for wake word detection, faster-whisper for Speech-to-Text (STT), a local LLM (running in Ollama) for intent extraction and Piper for Text-to-Speech (TTS). All these services can run on a single machine, or on dedicated hardware on your network. No external resources are required.

## Modularity
Each main component of the system passes data over a simple network interface to the next, orchestrated by a central python script that holds intent handlers and passes data to the appropriate next step. Each component can be swapped out separately for testing or to replace with a custom version. An example of this is the ollama-stub, which stands in for an LLM for testing in environments where access to a real LLM is impractical or expensive. The modular idea was initially forced by conflicting dependencies and grew into a useful way to develop a growing codebase in a more managable fashion.

## Skills
The skill system is not fully fledged, but the system has been designed to be easy to expand with additonal abilities. New skills provide skill code in a python script, then register an event handler in the orchestration script/LLM system prompt.

## A Star Trek feel
The default file names for models refer to Star Trek:TNG themed voices. The response text is intended to have the tone of the TNG Enterprise computer. However these are fully tweakable in the source code. Future versions will move many configuration options out into config files

# Install
Because of the modular nature and conflicting dependencies of some libraries on some platforms, there are four python environments to set up.

Clone the git repo:
`cd ~/projects   # or wherever suits you
`git clone https://github.com/reverse_atomic_roger/voice-assistant.git

Within each top-level directory, create a new python virtual environment, and install dependencies:
`python -m venv .venv
`.\.venv\Scripts\Activate ## on windows
`source .venv/bin/activate ##on *nix
`pip install -r requirements.txt

Create or acquire yourself some appropriate onnx voice models for speech recognition and generation. Create or acquire some wav files to act as acknowledgement and error sounds. Check any of the CONFIGURE constants at the top of each python file to point to the correct model files, and if necessary, change IP addresses for multi-device setups.

Each folder then has one python file that contains the main code and imports any other code, as required:
Wakeword -> satellite_main.py
Orchestration -> orchestration.py
STT -> stt_server.py
TTS -> tts_server.py