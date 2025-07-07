# BlindSpot VisionGuide System – Supporting Files

This folder contains the supporting materials for the manuscript:
"Empowering the Visually Impaired with AI: The BlindSpot VisionGuide System on Raspberry Pi"

## Folder Contents

1. **/model/**
   - Model files used for the image captioning module.
   - Includes `.safetensors` model files and a `.json` files.
   - https://drive.google.com/drive/folders/1ruwd5Alc_phzwP-Tu_ruFBW_RC5UtlUO?usp=drive_link 

2. **/code/**
   - Combined Python script: `Code.py`
     - This single file integrates all three modules:
       - Face Recognition (real-time detection using webcam)
       - Image Captioning
       - Online Newspaper Reading
     - Includes voice assistant integration to run modules via voice commands.

3. Online Newspaper Reading
	For this module in Code.py script fetches and filters the latest Indian news using the NewsAPI.
   > **Note:**  
   > - The news data is fetched dynamically from NewsAPI during runtime.  
   > - No static JSON or `.txt` file is stored for news.  
   > - Please supply your own API key by replacing `YOUR_API_KEY` in the script

4. **requirements.txt**
   - In the requirements.txt file, not all listed libraries are in use. I’ve tried and replaced some of them with alternatives. So, if you're planning to use the code, please keep that in mind.


## Face Recognition Note

The system uses real-time face recognition. Known faces are pre-encoded by capturing them through the Raspberry Pi camera. These encodings are stored internally and not in any CSV format.

## How to Run

1. Set up your Raspberry Pi with a camera, microphone, and speaker.
2. Install required packages using:
	pip install -r requirements.txt
3. Run the integrated voice-based system:
	code.py
4. Use voice commands such as:
- "run face module"
- "run caption module"
- "run news module"
- "exist from the code"




