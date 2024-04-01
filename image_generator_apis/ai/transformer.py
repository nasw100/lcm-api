import os

from abc import abstractmethod, ABCMeta
from pathlib import Path
from typing import Union

from .preprocess import ImagePreprocessor
from .inference import DiffusionEngine
from .postprocess import ImagePostprocessor

class Transformer(metaclass=ABCMeta) :
    @abstractmethod
    def preprocess(self) :
        pass
    
    @abstractmethod
    def predict(self) :
        pass

    @abstractmethod
    def postprocess(self) :
        pass

class ImageGeneratorTransformer(Transformer) :
    def __init__(self) :
        self.preprocessor = ImagePreprocessor()
        self.inferer = DiffusionEngine()
        self.postprocessor = ImagePostprocessor()

    def preprocess(self) :
        pass

    def predict(self, prompt: str) :
        self.predicted = self.inferer.predict(prompt)

    def postprocess(self, fp: Union[str, Path, os.PathLike]) :
        self.postprocessor(self.predicted).save(fp)