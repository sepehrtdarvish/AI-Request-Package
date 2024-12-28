from .gpt_request import GPTChatRequest, TranscriptAudioRequest, SpeechAudioRequest, ImageToTextRequest
from .exceptions import RequestTypeNotValid, AudioRequestTypeNotValid, AudioRequestTypeNotFound, RequestTypeNotFound

class AIRequestInterface():
    ALLOWED_REQUEST_TYPES = ['audio', 'image', 'chat'] 
    ALLOWED_AUDIO_TYPES = ['transcriptions', 'speech']

    def __init__(self, type, audio_request_type=None, **kwargs):
        self.request_type = self.__get_validated_request_type(type)
        self.request = self.__request_factory(type=type, audio_request_type=audio_request_type, **kwargs)


    def send(self,  **kwargs):
        return self.request.send(**kwargs)



    def __request_factory(self, type, audio_request_type, **kwargs):
        if type == 'audio':
            validated_audio_req_typ = self.__get_validated_audio_request_type(audio_request_type=audio_request_type)
            if validated_audio_req_typ == 'transcriptions':
                return TranscriptAudioRequest(**kwargs)
            if validated_audio_req_typ == 'speech':
                return SpeechAudioRequest(**kwargs)
        elif type == 'chat':
            return GPTChatRequest(**kwargs)
        elif type == 'image':
            return ImageToTextRequest(**kwargs)
            
            
    
    def __get_validated_request_type(self, request_type):
        if not request_type:
            raise RequestTypeNotValid
        elif request_type not in self.ALLOWED_REQUEST_TYPES:
            raise RequestTypeNotFound
        else:
            return request_type
    
    def __get_validated_audio_request_type(self, audio_request_type):
        if not audio_request_type:
            raise AudioRequestTypeNotFound
        elif audio_request_type not in self.ALLOWED_AUDIO_TYPES:
            raise AudioRequestTypeNotValid
        else:
            return audio_request_type
        