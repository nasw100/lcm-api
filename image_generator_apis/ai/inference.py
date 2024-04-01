from abc import abstractmethod, ABCMeta
from typing import Literal, Union, Any
from PIL import Image
from diffusers import DiffusionPipeline
from pathlib import Path

from ..exceptions.ai import ParameterOutOfRange, NoResultImage
from ..core.settings import get_envs

import os
import torch

settings = get_envs()


class DiffusionEngine :
    def __init__(self, 
                 model: str = "SimianLuo/LCM_Dreamshaper_v7", 
                 device: Literal["cuda", "mps", "cpu"] = "mps", 
                 torch_dtype: torch.dtype =torch.float32,
                 local: bool = True) :
        """Initializing Inference Engine using Diffusion Pipe

        Args:
            model (str, optional): model name. Defaults to "SimianLuo/LCM_Dreamshaper_v7".
            device (Literal[&quot;cuda&quot;, &quot;mps&quot;, &quot;cpu&quot;], optional): 
                cuda : nvidia 
                mps : mac apple silicon gpu
                cpu : cpu. 
                Defaults to "mps".
            torch_dtype (torch.dtype, optional): torch data type. Use float16 to save GPU Memory. Defaults to torch.float32.
            local (bool, optional): The PC whose memory is below 40GB is regarded as local. Defaults to True.
        """
        self.result = None
        self.pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(model)
        self.pipe.to(torch_device=device, torch_dtype=torch_dtype)

        if local :
            self.pipe.enable_attention_slicing()


    def predict(self, 
                prompt: str, 
                num_inference_steps: int = 4, 
                guidance_scale: float = 8.0, 
                lcm_origin_steps: int = 50, 
                output_type: Literal["pil"] = "pil") -> DiffusionPipeline :
        """Generate an image from a prompt

        Args:
            prompt (str): A text to generate an image.
            num_inference_steps (int, optional): Count of inference steps. Recommended 1 - 8, Allowed 1 - 50, Defaults to 4.
            guidance_scale (float, optional): . Defaults to 8.0.
            lcm_origin_steps (int, optional): . Defaults to 50.
            output_type (Literal[&quot;pil&quot;], optional): output type. Defaults to "pil".

        Returns:
            DiffusionPipeline: _description_
        """
        if num_inference_steps > 50 or num_inference_steps < 1 :
            raise ParameterOutOfRange(message="num inference steps can be set between 1 and 50")
        
        self.result =  self.pipe(prompt=prompt, 
                         num_inference_steps=num_inference_steps, 
                         guidance_scale=guidance_scale, 
                         lcm_origin_steps=lcm_origin_steps, 
                         output_type=output_type)
        
        return self.result
