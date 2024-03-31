from .inference import InferenceEngine

class AbstractProcessor :
    pass

class Preprocessor(AbstractProcessor) :
    pass

class Postprocessor(AbstractProcessor) :
    pass

class Transformer :
    def __init__(self) :
        self.inferer = InferenceEngine()

    def preprocess() :
        pass

    def predict(self, prompt: str) :
        self.inferer.predict()

    def postprocess() :
        pass