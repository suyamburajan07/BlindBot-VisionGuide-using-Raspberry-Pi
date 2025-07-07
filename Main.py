import speech_recognition as sr
import pyttsx3
import face_recognition
import cv2
import numpy as np
import dlib
import sys, time
import requests
import json
import os
from datetime import datetime, timedelta
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt
import nltk


def perform_face_recognition():

  # Initialize text-to-speech engine
  engine = pyttsx3.init()
  voices = engine.getProperty('voices')
  engine.setProperty('voice', voices[1].id)  # Set default voice (optional)

  # Function to speak the recognized name
  def speak_name(name):
      engine.say(name)
      engine.runAndWait()

  # Initialize Dlib's face detector (HOG-based)
  detector = dlib.get_frontal_face_detector()

  # Simple Artificial Neural Network (ANN) for face recognition (example)
  class ANNClassifier:
      def __init__(self):
          self.embeddings_dict = {}
          self.threshold = 0.6  # Distance threshold for face recognition

      def load_embeddings(self, name, image_path):
          if not os.path.exists(image_path):
              raise FileNotFoundError(f"Image file '{image_path}' not found.")

          image = face_recognition.load_image_file(image_path)
          face_encodings = face_recognition.face_encodings(image)

          if len(face_encodings) == 0:
              raise ValueError(f"No face detected in image '{image_path}'.")

          embedding = face_encodings[0]
          self.embeddings_dict[name] = embedding

      def recognize_face(self, face_encoding):
          min_distance = float('inf')
          recognized_name = "Unknown"

          for name, known_embedding in self.embeddings_dict.items():
              distance = np.linalg.norm(face_encoding - known_embedding)
              if distance < min_distance:
                  min_distance = distance
                  recognized_name = name

          if min_distance > self.threshold:
              recognized_name = "Unknown"

          return recognized_name

  # Initialize the ANN classifier
  ann_classifier = ANNClassifier()

  # Load embeddings for known faces
  try:
      ann_classifier.load_embeddings("ganesh", "C:/Users/User/PycharmProjects/pythonProject1/venv/ganesh.jpeg")
      ann_classifier.load_embeddings("suyamburajan", "C:/Users/User/PycharmProjects/pythonProject1/venv/suyambu.jpg")
      ann_classifier.load_embeddings("Harry Potter",
                                     "C:/Users/User/PycharmProjects/pythonProject1/venv/harry_potter.jpeg")
  except Exception as e:
      print(f"Error loading embeddings: {e}")
      exit()

  # Get a reference to webcam #0
  print("[INFO] sampling frames from webcam...")
  engine.say("[INFO] sampling frames from webcam...")
  engine.runAndWait()
  video_capture = cv2.VideoCapture(0)

  # Define the duration for the timer in seconds (e.g., 10 minutes)
  duration = 1 * 10
  start_time = time.time()

  while True:
      # Check if the specified duration has passed
      elapsed_time = time.time() - start_time
      if elapsed_time > duration:
          print("Timer expired, exiting...")
          engine.say("Timer expired, exiting...")
          engine.runAndWait()
          break

      # Grab a single frame of video
      ret, frame = video_capture.read()

      if not ret:
          print("Failed to grab frame from webcam")
          break

      # Resize frame of video to 1/2 size for faster processing
      small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

      # Convert the image from BGR color to RGB color (face_recognition uses RGB)
      rgb_small_frame = small_frame[:, :, ::-1]

      # Detect faces using dlib's HOG detector
      faces = detector(rgb_small_frame, 1)  # Increasing the upsample value can help detect smaller faces

      # Loop through detected faces
      for face in faces:
          # Get the face coordinates
          (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

          # Extract face encoding using face_recognition library
          face_encodings = face_recognition.face_encodings(rgb_small_frame, [(y, x + w, y + h, x)])

          if len(face_encodings) > 0:
              face_encoding = face_encodings[0]
              # Recognize the face using ANN
              recognized_name = ann_classifier.recognize_face(face_encoding)

              # Speak the recognized name
              speak_name(recognized_name)

              # Draw a rectangle around the face
              cv2.rectangle(frame, (x * 2, y * 2), ((x + w) * 2, (y + h) * 2), (0, 255, 0), 2)

              # Draw a label with a name below the face
              cv2.rectangle(frame, (x * 2, (y + h + 15) * 2), ((x + w) * 2, (y + h) * 2), (0, 255, 0), cv2.FILLED)
              font = cv2.FONT_HERSHEY_DUPLEX
              cv2.putText(frame, recognized_name, (x * 2 + 6, (y + h) * 2 - 6), font, 1.0, (255, 255, 255), 1)

      # Display the resulting image
      cv2.imshow('Camera', frame)

      # Check if the 'q' key is pressed to quit
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  # Release handle to the webcam
  video_capture.release()
  cv2.destroyAllWindows()


def perform_image_captioning():

    def load_model(model_dir="./blip_model_large"):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            processor.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
        else:
            processor = BlipProcessor.from_pretrained(model_dir)
            model = BlipForConditionalGeneration.from_pretrained(model_dir)

        return processor, model

    def generate_caption(image_path, processor, model, device):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=60,
                repetition_penalty=2.5,
                length_penalty=1.0
            )

        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption, image

    def visualize_caption(image, caption):
        # Uncomment the following lines to enable visualization
        # plt.figure(figsize=(6, 6))
        # plt.imshow(image)
        # plt.axis("off")
        # plt.title(caption, fontsize=12, wrap=True)
        # plt.show()
        pass

    def capture_image():
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Could not access the camera.")
            return None

        print("Waiting for 2 seconds before capturing...")
        time.sleep(2)  # Wait for 2 seconds before capturing

        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture image.")
            return None

        image_path = "captured_image.jpg"
        cv2.imwrite(image_path, frame)
        cam.release()
        return image_path

    def speak_caption(caption):
        engine = pyttsx3.init()
        engine.setProperty('rate', 125)  # Reduce speed of speech
        engine.say(caption)
        engine.runAndWait()

    if __name__ == "__main__":
        nltk.download('punkt')
        processor, model = load_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        image_path = capture_image()
        if image_path:
            caption, image = generate_caption(image_path, processor, model, device)
            print(f"Generated Caption: {caption}")
            speak_caption(caption)

            # Uncomment the next line if visualization is needed
            # visualize_caption(image, caption)


def perform_newspaper_recognition():
    def speak_text(text):
        """Converts text to speech."""
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)  # Use female voice (optional)
        engine.setProperty('rate', 140)  # Adjust speech rate
        engine.say(text)
        engine.runAndWait()

    def get_latest_news(api_endpoint, api_key, country=None, sources=None, query=None, category=None, language="en"):
        """Fetches the latest news from a specified API with manual date filtering."""
        seven_days_ago = datetime.now() - timedelta(days=7)
        params = {
            "apiKey": api_key,
            "language": language,
            "sortBy": "publishedAt"  # Sort by newest first
        }
        if country:
            params["country"] = country
        if sources:
            params["sources"] = sources
        if query:
            params["q"] = query
        if category:
            params["category"] = category

        try:
            response = requests.get(api_endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            # Debugging: Print API response
            print("API Full Response:", json.dumps(data, indent=4))

            if "articles" in data and data["articles"]:
                # Exclude Google News (India) results
                filtered_articles = [
                    article for article in data["articles"]
                    if article["source"]["id"] != "google-news-in"
                ]

                # Manually filter articles by date (last 7 days)
                recent_articles = []
                for article in filtered_articles:
                    published_at = article.get("publishedAt", "")
                    if published_at:
                        article_date = datetime.strptime(published_at[:10], "%Y-%m-%d")
                        if article_date >= seven_days_ago:
                            recent_articles.append(article)

                if not recent_articles:
                    print("No relevant recent articles found.")
                return recent_articles
            else:
                print("No articles found for India.")
                return []

        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return []

    def get_indian_sources(api_key):
        """Fetches available Indian news sources."""
        API_SOURCES_ENDPOINT = "https://newsapi.org/v2/top-headlines/sources"
        params = {"apiKey": api_key, "country": "in"}
        try:
            response = requests.get(API_SOURCES_ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()
            sources = [source["id"] for source in data.get("sources", []) if source["id"] != "google-news-in"]
            return ",".join(sources) if sources else None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching sources: {e}")
            return None

    # API Configurations
    YOUR_API_KEY = "641439d0282f485ab91070f692d96eb7"  # Replace with your actual API key
    API_ENDPOINT = "https://newsapi.org/v2/top-headlines"

    print("Getting information from Indian newspapers...")
    speak_text("Getting information from Indian newspapers")

    # First, try to get Indian news using sources
    indian_sources = get_indian_sources(YOUR_API_KEY)
    if indian_sources:
        latest_news = get_latest_news(API_ENDPOINT, YOUR_API_KEY, sources=indian_sources)
    else:
        latest_news = get_latest_news(API_ENDPOINT, YOUR_API_KEY, country="in")

    # If no news is found, switch to 'everything' API with 'India' as the search term
    if not latest_news:
        print("Trying general search for 'India'...")
        API_ENDPOINT = "https://newsapi.org/v2/everything"
        latest_news = get_latest_news(API_ENDPOINT, YOUR_API_KEY, query="India")

    if latest_news:
        num_articles_to_read = 3  # Limit number of articles

        for i, article in enumerate(latest_news[:num_articles_to_read]):
            title = article.get('title', 'No title available')
            description = article.get('description', 'No description available')
            url = article.get('url', '')

            # Shorten description
            max_description_length = 100
            shortened_description = description[:max_description_length] + "..." if len(
                description) > max_description_length else description

            print(f"Title: {title}\nDescription: {shortened_description}\nURL: {url}\n{'-' * 50}")

            # Speak news
            speak_text(f"Title: {title}. Description: {shortened_description}")
    else:
        print("No recent news found for India. Please try again later.")
        speak_text("No recent news found for India. Please try again later.")


# Initialize recognizer and text-to-speech engine
r = sr.Recognizer()
engine = pyttsx3.init()

# Define a dictionary for custom response mapping
response_map = {
    "run the face module": "fine",  # Run face recognition after "welcome"
    "run the image captioning module": "fine",
    "run the newspaper module": "fine",
    "exit from the code": "ok",
}

# Flag to indicate if face recognition has already run
face_recognition_done = False

def SpeakText(text):
  """Speaks the given text using the text-to-speech engine."""
  engine.say(text)
  engine.runAndWait()

# Loop infinitely for user to speak
while True:

    # Exception handling for runtime errors
    try:
        # Use microphone as source
        with sr.Microphone() as source:

            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=0.2)

            # Listen for user input
            audio = r.listen(source)

            # Recognize audio using Google
            text = r.recognize_google(audio).lower()
            print("You said:", text)

            # Check for custom responses
            if text in response_map:
                # Only run face recognition if "welcome" is said and it hasn't been run before
                if text == "run the face module" and not face_recognition_done:
                    message = perform_face_recognition()
                    #SpeakText(message)
                    #print("Text output:", message)
                    face_recognition_done = True # Set flag to true after running face recognition

                if text == "run the image captioning module" and not face_recognition_done:
                    message = perform_image_captioning()
                    #SpeakText(message)
                    #print("Text output:", message)
                    face_recognition_done = True # Set flag to true after running face recognition
                    
                if text == "run the newspaper module" and not face_recognition_done:
                    message = perform_newspaper_recognition()
                    #SpeakText(text)
                    #print("Text output:", message)
                    face_recognition_done = True

                if text == "exit from the code" and not face_recognition_done:
                    print("Exiting program")
                    SpeakText("Exiting program")
                    break  # Exit the loop to terminate the program
                
                else:
                    # Speak the mapped response for other keywords or if welcome has already been said
                    response = response_map[text]
                    #SpeakText(response)
                    #print("Text output:", response)

            else:
                # No custom mapping, display original text
                print("------- ------- ------- -------")
                SpeakText("try again")

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("Could not understand audio")

    # Reset the face recognition flag at the end of each loop iteration
    face_recognition_done = False