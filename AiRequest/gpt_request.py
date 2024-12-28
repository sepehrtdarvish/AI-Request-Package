import base64
import PIL
from openai import OpenAI
from io import BytesIO
from abc import ABC, abstractmethod
import google.generativeai as genai

client = OpenAI(
  api_key='some key'
)


class AiRequest(ABC):
    def __init__(self, model):
        self.model=model

    @abstractmethod
    def send(self):
        pass


class GPTChatRequest(AiRequest):
    DEFAULT_MODEL = 'gpt-3.5-turbo'
    DEFAULT_TEMPERATURE = 1
    DEFAULT_TOP_P = 1
    DEFAULT_FREQUENCY_PENALTY = 0
    DEFAULT_PRESENCE_PENALTY = 0
    DEFAULT_STOP = None

    def __init__(self, model=None, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, stop=None):
        model = model or self.DEFAULT_MODEL
        temperature = temperature or self.DEFAULT_TEMPERATURE
        top_p = top_p or self.DEFAULT_TOP_P
        frequency_penalty = frequency_penalty or self.DEFAULT_FREQUENCY_PENALTY
        presence_penalty = presence_penalty or self.DEFAULT_PRESENCE_PENALTY
        stop = stop or self.DEFAULT_STOP
        super().__init__(model)
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def send(self, messages):
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop
        )
        return response


class TranscriptAudioRequest(AiRequest):
    DEFAULT_MODEL = "whisper-1"
    DEFAULT_LANGUAGE = "en"

    def __init__(self, model=None, language=None):
        model = model or self.DEFAULT_MODEL
        language = language or self.DEFAULT_LANGUAGE
        super().__init__(model)
        self.language = language

    def send(self, audio_file):
        transcript = client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
            response_format="text",
            language=self.language
        )
        return transcript


class SpeechAudioRequest(AiRequest):
    DEFAULT_MODEL = "tts-1"
    VOICES = [['alloy', 'echo', 'fable', 'onyx'], ['nova', 'shimmer']]
    DEFAULT_VOICE = VOICES[1][0]

    def __init__(self, voice=None, model=None):
        voice = voice or self.DEFAULT_VOICE
        model = model or self.DEFAULT_MODEL
        super().__init__(model)
        self.voice = voice

    def send(self, text):
        response = client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=text
        )
        speech_file = BytesIO(response.content)
        speech_file.seek(0)
        return speech_file


class ImageToTextRequest(AiRequest):
    DEFAULT_MODEL = "gemini-pro-vision"

    def __init__(self, model=None):
        model = model or self.DEFAULT_MODEL
        super().__init__(model)

    def send(self, prompt, image_data):
        img = BytesIO(base64.b64decode(image_data.split(',')[1]))
        img_ = PIL.Image.open(img)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content([prompt, img_])
        return response
