import os

from pathlib import Path
from PIL import Image
from typing import Union

from ..exceptions.ai import NoResultImage, NotSupportedMediaType


class Postprocessor() :
    def __init__(self) :
        self.result = None

    def __call__(self, predicted) -> "Postprocessor" :
        self.result = predicted
        return self

class ImagePostprocessor(Postprocessor) :
    def filter(self, value: int) :
        pass

    def save(self, fp: Union[str, Path, os.PathLike]) :
        if self.result :
            result_images = self.result.images
            if type(result_images[0]) == Image.Image :
                result_images[0].save(fp)

            else :
                raise NotSupportedMediaType(message="PIL Image is the only supported media type")
        
        else :
            raise NoResultImage(message="run predict and make an result before this function: ImagePostprocessor.save")

