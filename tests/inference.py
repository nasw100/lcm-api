import pytest
from PIL import Image
from pathlib import Path
from image_generator_apis.ai.inference import InferenceEngine

inferer = InferenceEngine()

def test_predict() :
    result = inferer.predict("A K-Pop band with beautiful flowers")
    result_image = result.images[0]
    assert type(result_image) == Image.Image

@pytest.mark.parametrize('fp', ['str-test-img.png', Path(__file__).parent / 'path-test-img.png'])
def test_save_result(fp) :
    assert inferer.save_result(fp) == None
