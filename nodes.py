import io
import os
from .aihub_env import AIHUB_LORAS_DIR, AIHUB_LORAS_LOCALE_DIR, AIHUB_MODELS_DIR, AIHUB_MODELS_LOCALE_DIR, AIHUB_WORKFLOWS_DIR
from nodes import LoadImage, CheckpointLoaderSimple, LoraLoader, UNETLoader, LoraLoaderModelOnly, VAELoader, CLIPLoader, DualCLIPLoader
from folder_paths import get_filename_list
import json
import torch
import torchaudio
import random
import comfy.samplers
import safetensors.torch
from comfy.cli_args import args
from comfy_extras.nodes_audio import load as load_audio_file
from torch.nn.functional import interpolate
from comfy.utils import common_upscale

from PIL import Image

import numpy as np

#import sys

#custom_node_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "comfyui-videohelpersuite", "videohelpersuite", "load_images_nodes"))
#if custom_node_path not in sys.path:
#    sys.path.append(custom_node_path)

#import LoadVideoPath

#from server import PromptServer, BinaryEventTypes
from .server import AIHubServer

SERVER = AIHubServer()
SERVER.start_server()
print("AIHub Server Started")

LAST_MODEL_FILE = None
LAST_MODEL_FILE_IS_DIFFUSION_MODEL = False
LAST_MODEL = None
LAST_MODEL_CLIP = None
LAST_MODEL_VAE = None
LAST_MODEL_WEIGHT_DTYPE = None

LAST_VAE = None
LAST_VAE_FILE = None

LAST_CLIP_FILE = None
LAST_CLIP = None
LAST_CLIP_DUAL = False
LAST_CLIP_TYPE = None

CLIP_TYPES = [
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "mochi",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "sdxl",
    "flux",
    "hunyuan_video"
]

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]

class AIHubWorkflowController:
    """
    A necessary piece for a workflow to be able to be registered as an AIHub workflow you should
    have only one of this node in each workflow you make
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"tooltip": "The id for the workflow, must be unique among all the workflows", "default": "my_custom_workflow"}),
                "label": ("STRING", {"tooltip": "The label for the workflow to be used", "default": "my_custom_workflow"}),
                "description": ("STRING", {"tooltip": "The noise to use.", "multiline": True, "default": ""}),
                "category": ("STRING", {"tooltip": "A path to be used within the menu, separated by /", "default": "my_custom_workflow"}),
                "context": (["image", "video", "audio", "3d", "text"], {"tooltip": "The context of this workflow", "default": "image"}),
                "project_type": ("STRING", {"tooltip": "The project type is an arbitrary string that limits in which context the workflow can be used"}),
                "project_type_init": ("BOOLEAN", {"tooltip": "This workflow can initialize the given project type, provided an empty project folder"})
                # eg. project types I think about implementing, ltxv, sdxl-character
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "register"
    CATEGORY = "aihub/workflow"

    def register(self, id, label, description, category, context, project_type, project_type_init):
        return ()

### EXPOSES
class AIHubExposeInteger:
    """
    A utility node for exposing a configurable integer and its properties.
    """

    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_int"

    # The RETURN_TYPES defines the data types of the node's outputs.
    # It returns a single integer, which is the value of the "default" input.
    RETURN_TYPES = ("INT",)

    # The INPUT_TYPES method is where we define all the properties that the
    # user can edit. Each property from your data structure becomes a dictionary key.
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_integer", "tooltip": "A unique custom id for this workflow (it should be unique)"}),
                "label": ("STRING", {"default": "Exposed Integer", "tooltip": "This is the label that will appear in the field"}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "min": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647}),
                "max": ("INT", {"default": 100, "min": -2147483648, "max": 2147483647}),
                "step": ("INT", {"default": 1, "min": -2147483648, "max": 2147483647}),
                "value": ("INT", {"default": 10, "min": -2147483648, "max": 2147483647}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will make this option be hidden under advanced options for this workflow"}),
                "index": ("INT", {"default": 0, "tooltip": "this value is used for sorting the input fields when displaying, lower values will appear first"}),
            }
        }

    def get_exposed_int(self, label, tooltip, min, max, step, value, description, advanced, index):
        if (value < min):
            raise ValueError(f"Error: {id} should be greater or equal to {min}")
        if (value > max):
            raise ValueError(f"Error: {id} should be less or equal to {max}")
        return (value,)
    
class AIHubExposeSteps:
    """
    A utility node for exposing a configurable integer meant for using as steps, the difference
    between using a standard integer is that this one is able to default to whatever the model
    being used has as default steps

    Note that this defaulting is a client side feature, this node just exposes an integer
    """

    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_steps"

    # The RETURN_TYPES defines the data types of the node's outputs.
    # It returns a single integer, which is the value of the "default" input.
    RETURN_TYPES = ("INT",)

    # The INPUT_TYPES method is where we define all the properties that the
    # user can edit. Each property from your data structure becomes a dictionary key.
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "steps", "tooltip": "A unique custom id for this workflow (it should be unique)"}),
                "label": ("STRING", {"default": "Steps", "tooltip": "This is the label that will appear in the field"}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "value": ("INT", {"default": 10}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will make this option be hidden under advanced options for this workflow"}),
                "index": ("INT", {"default": 0, "tooltip": "this value is used for sorting the input fields when displaying, lower values will appear first"}),
                "unaffected_by_model_steps": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this steps value will not be affected by the model's default steps"}),
            }
        }

    def get_exposed_steps(self, id, label, tooltip, value, advanced, index, unaffected_by_model_steps):
        if (value < 0):
            raise ValueError(f"Error: {id} should be greater or equal to {0}")
        if (value > 150):
            raise ValueError(f"Error: {id} should be less or equal to {150}")
        return (value,)
    
class AIHubExposeProjectConfigInteger:
    """
    A utility node for exposing an integer from the project config.json
    """

    CATEGORY = "aihub/expose/config"
    FUNCTION = "get_exposed_int"

    RETURN_TYPES = ("INT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "boolean", "tooltip": "A unique custom id for this workflow."}),
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to expose, use dots for entering sublevels"}),
                "default": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "tooltip": "The default value of the field"}),
            },
            "hidden": {
                "value": ("INT", {"default": 0, "tooltip": "The value of the field"}),
            }
        }

    def get_exposed_int(self, id, field, default, value=None):
        return (value if value is not None else default,)
    
class AIHubExposeFloat:
    """
    A utility node for exposing a configurable float and its properties.
    """
    
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_float"
    RETURN_TYPES = ("FLOAT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "float", "tooltip": "A unique custom id for this workflow."}),
                "label": ("STRING", {"default": "Float", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip."}),
                "min": ("FLOAT", {"default": 0.0, "min": -1.0e+20, "max": 1.0e+20}),
                "max": ("FLOAT", {"default": 1.0, "min": -1.0e+20, "max": 1.0e+20}),
                "step": ("FLOAT", {"default": 0.05, "tooltip": "The step value for the float input.", "min": 0.001, "max": 1.0, "step": 0.001}),
                "value": ("FLOAT", {"default": 0.5, "min": -1.0e+20, "max": 1.0e+20}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will make this option be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
                "slider": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this float will be represented as a slider in the UI."}),
            }
        }

    def get_exposed_float(self, id, label, tooltip, min, max, step, value, advanced, index, slider):
        if (value < min):
            raise ValueError(f"Error: {id} should be greater or equal to {min}")
        if (value > max):
            raise ValueError(f"Error: {id} should be less or equal to {max}")
        return (value,)
    
class AIHubExposeCfg:
    """
    A utility node for exposing a configurable float for cfg
    the difference between using a standard float is that this one is able to default
    to whatever the model being used has as default cfg

    Note that this defaulting is a client side feature, this node just exposes a float
    """
    
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_cfg"
    RETURN_TYPES = ("FLOAT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "cfg", "tooltip": "A unique custom id for this workflow."}),
                "label": ("STRING", {"default": "Cfg", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip."}),
                "value": ("FLOAT", {"default": 3.0}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will make this option be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
                "unaffected_by_model_cfg": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this cfg value will not be affected by the model's default cfg"}),
            }
        }

    def get_exposed_cfg(self, id, label, tooltip, value, advanced, index, unaffected_by_model_cfg):
        if (value < 0.0):
            raise ValueError(f"Error: {id} should be greater or equal to {0.0}")
        if (value > 100.0):
            raise ValueError(f"Error: {id} should be less or equal to {100.0}")
        return (value,)
    
class AIHubExposeProjectConfigFloat:
    """
    A utility node for exposing a float from the project config.json
    """

    CATEGORY = "aihub/expose/config"
    FUNCTION = "get_exposed_float"

    RETURN_TYPES = ("FLOAT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "boolean", "tooltip": "A unique custom id for this workflow."}),
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to expose, use dots for entering sublevels"}),
                "default": ("FLOAT", {"default": 0.0, "min": -1.0e+20, "max": 1.0e+20, "tooltip": "The default value of the field"}),
            },
            "hidden": {
                "value": ("FLOAT", {"default": 0.0, "tooltip": "The value of the field"}),
            }
        }

    def get_exposed_float(self, id, field, default, value=None):
        return (value if value is not None else default,)
    
class AIHubExposeBoolean:
    """
    A utility node for exposing a configurable boolean and its properties.
    """
    
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_boolean"
    RETURN_TYPES = ("BOOLEAN",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "boolean", "tooltip": "A unique custom id for this workflow."}),
                "label": ("STRING", {"default": "Boolean", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip for this input."}),
                "value": ("BOOLEAN", {"default": False}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this option will be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            }
        }

    def get_exposed_boolean(self, id, label, tooltip, description, value, advanced, index):
        return (value,)
    
class AIHubExposeProjectConfigBoolean:
    """
    A utility node for exposing a boolean from the project config.json
    """

    CATEGORY = "aihub/expose/config"
    FUNCTION = "get_exposed_boolean"

    RETURN_TYPES = ("BOOLEAN",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "boolean", "tooltip": "A unique custom id for this workflow."}),
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to expose, use dots for entering sublevels"}),
                "default": ("BOOLEAN", {"default": True, "tooltip": "The default value of the field"}),
            },
            "hidden": {
                "value": ("BOOLEAN", {"default": True, "tooltip": "The value of the field"}),
            }
        }

    def get_exposed_boolean(self, id,field, default, value=None):
        return (value if value is not None else default,)
    
class AIHubExposeString:
    """
    A utility node for exposing a configurable string and its properties.
    """

    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_string"
    RETURN_TYPES = ("STRING",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "string", "tooltip": "A unique custom id for this workflow."}),
                "label": ("STRING", {"default": "String", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip."}),
                "minlen": ("INT", {"default": 0}),
                "maxlen": ("INT", {"default": 1000}),
                "value": ("STRING", {"default": ""}),
                "multiline": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this field will allow multiline input."}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this option will be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            }
        }

    def get_exposed_string(self, id, label, tooltip, minlen, maxlen, value, multiline, advanced, index):
        if (len(value) < minlen):
            raise ValueError(f"Error: {label} length should be greater or equal to {minlen}")
        if (len(value) > maxlen):
            raise ValueError(f"Error: {label} length should be less or equal to {maxlen}")
        return (value,)
    
class AIHubExposeProjectConfigString:
    """
    A utility node for exposing a string from the project config.json
    """

    CATEGORY = "aihub/expose/config"
    FUNCTION = "get_exposed_string"

    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "boolean", "tooltip": "A unique custom id for this workflow."}),
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to expose, use dots for entering sublevels"}),
                "default": ("STRING", {"default": "", "tooltip": "The default value of the field"}),
            },
            "hidden": {
                "value": ("STRING", {"default": "my-value", "tooltip": "The value of the field"}),
            }
        }

    def get_exposed_string(self, id, field, default, value=None):
        return (value if value is not None else default,)

class AIHubExposeStringSelection:
    """
    A utility node for exposing a string that can be selected from a list of strings of options.
    """

    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_selection"
    RETURN_TYPES = ("STRING",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "string_selection", "tooltip": "A unique custom id for this workflow."}),
                "label": ("STRING", {"default": "String Selection", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip."}),
                "options": ("STRING", {"multiline": True, "default": "option1\noption2\noption3", "tooltip": "A newline separated string array of values for the dropdown."}),
                "options_label": ("STRING", {"multiline": True, "default": "Option 1\nOption 2\nOption 3", "tooltip": "A newline separated string array of labels for the dropdown. Must match the number of options."}),
                "value": ("STRING", {"default": "option1"}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this option will be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            }
        }

    def get_exposed_selection(self, id, label, tooltip, options, options_label, value, advanced, index):
        optionsparsed = [o.strip() for o in options.split("\n") if o.strip()]
        if value not in optionsparsed:
            raise ValueError(f"Error: {label} The value given is not in the list of options")
        return (value,)
    
class AIHubExposeSeed:
    """
    A utility node for exposing a configurable seed to ensure that there are changes in the workflow
    that require a seed to be set
    """

    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_seed"
    RETURN_TYPES = ("INT",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "seed", "tooltip": "A unique custom id for this workflow."}),
                "label": ("STRING", {"default": "Seed", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this option will be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            }
        }

    def get_exposed_seed(self, id, label, tooltip, value, advanced, index):
        return (value, )
        
class AIHubExposeSampler:
    """
    An utility to expose the sampler to be selected
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_sampler"
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("SAMPLER",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "sampler", "tooltip": "A unique custom id for this workflow."}),
                "label": ("STRING", {"default": "Sampler", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "value": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Choose the sampler to use"}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this option will be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
                "unaffected_by_model_sampler": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this sampler value will not be affected by the model's default sampler"}),
            }
        }

    def get_exposed_sampler(self, id, label, tooltip, value, advanced, index, unaffected_by_model_sampler):
        return (value, )
    
class AIHubExposeScheduler:
    """
    An utility to expose the scheduler to be selected
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_scheduler"
    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("SCHEDULER",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "scheduler", "tooltip": "A unique custom id for this workflow."}),
                "label": ("STRING", {"default": "Scheduler", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "value": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The default value for the scheduler to use, can be any string"}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this option will be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
                "unaffected_by_model_scheduler": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this scheduler value will not be affected by the model's default scheduler"}),
            }
        }

    def get_exposed_scheduler(self, id, label, tooltip, value, advanced, index, unaffected_by_model_scheduler):
        return (value, )
    
class AIHubExposeExtendableScheduler:
    """
    An utility to expose the scheduler to be selected
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_scheduler"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("SCHEDULER",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "scheduler", "tooltip": "A unique custom id for this workflow."}),
                "label": ("STRING", {"default": "Scheduler", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "value": ("STRING", {"tooltip": "The default value for the scheduler, can be any string"}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this option will be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
                "unaffected_by_model_scheduler": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this scheduler value will not be affected by the model's default scheduler"}),
                "blacklist": ("STRING", {"default": "", "tooltip": "A newline separated list of schedulers that should not be selectable", "multiline": True}),
                "blacklist_all": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will blacklist all the schedulers"}),
                "extras": ("STRING", {"default": "", "tooltip": "A newline separated list of extra schedulers that will be added to what is left", "multiline": True}),
            }
        }

    def get_exposed_scheduler(self, id, label, tooltip, value, advanced, index, unaffected_by_model_scheduler, blacklist, blacklist_all, extras):
        return (value, )

class AIHubExposeImage:
    """
    An utility to expose an image to be used in the workflow
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_image"
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "POS_X", "POS_Y", "LAYER_ID", "WIDTH", "HEIGHT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_image", "tooltip": "A unique custom ID for this workflow."}),
                "label": ("STRING", {"default": "Image", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "type": ([
                    "current_layer",
                    "current_layer_at_image_intersection",
                    "merged_image",
                    "merged_image_without_current_layer",
                    "merged_image_current_layer_intersection",
                    "merged_image_current_layer_intersection_without_current_layer",
                    "upload",
                ], {"default": "upload", "tooltip": "The source of the image"}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            },
            "hidden": {
                "pos_x": ("INT", {"default": 0, "tooltip": "The X position of the layer in the canvas"}),
                "pos_y": ("INT", {"default": 0, "tooltip": "The Y position of the layer in the canvas"}),
                "layer_id": ("STRING", {"default": "", "tooltip": "The ID of the layer to use, only given if type is current_layer or previous_layer"}),
                "local_file": ("STRING",),
            },
        }

    def get_exposed_image(self, id, label, tooltip, type, index, pos_x=0, pos_y=0, layer_id="", local_file=None):
        image = None
        mask = None
        if local_file is not None:
            if (os.path.exists(local_file)):
                # Instantiate a LoadImage node and use its logic to load the file
                loader = LoadImage()
                # The load_image method returns a tuple, so we need to get the first element
                loaded_image_tuple = loader.load_image(local_file)
                image = loaded_image_tuple[0]
                mask = loaded_image_tuple[1]
            else:
                filenameOnly = os.path.basename(local_file)
                raise ValueError(f"Error: Image file not found: {filenameOnly}")
        else:
            raise ValueError("You must specify the local_file for this node to function")

        _, height, width, _ = image.shape
        
        return (image, mask, pos_x, pos_y, layer_id, width, height,)
    
class AIHubExposeFrame:
    """
    An utility to expose a video frame to be used in the workflow
    """

    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_frame"

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT", "FRAME_INDEX", "TOTAL_FRAMES",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_frame", "tooltip": "A unique custom ID for this workflow."}),
                "label": ("STRING", {"default": "Frame", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "frame_index_type": ([
                    "relative",
                    "absolute",
                ], {"default": "relative", "tooltip": "The index type of the frame"}),
                "frame_index": ("INT", {"default": 0, "tooltip": "The index of the frame to expose, it gets affected by relative or absolute type, note that if out of bounds it will clamp to the nearest valid frame"}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            },
            "hidden": {
                "frame": ("INT", {"default": 0, "tooltip": "The actual frame number in the video"}),
                "total_frames": ("INT", {"default": 1, "tooltip": "The total number of frames in the video"}),
                "local_file": ("STRING",),
            },
        }
    
    def get_exposed_frame(self, id, label, tooltip, frame_index_type, frame_index, index, frame=0, total_frames=1, local_file=None):
        image = None
        if local_file is not None:
            if (os.path.exists(local_file)):
                # Instantiate a LoadImage node and use its logic to load the file
                loader = LoadImage()
                # The load_image method returns a tuple, so we need to get the first element
                loaded_image_tuple = loader.load_image(local_file)
                image = loaded_image_tuple[0]
            else:
                filenameOnly = os.path.basename(local_file)
                raise ValueError(f"Error: Image file not found: {filenameOnly}")
        else:
            raise ValueError("You must specify the local_file for this node to function")

        _, height, width, _ = image.shape
        
        return (image, width, height, frame, total_frames,)
    
class AIHubExposeProjectImage:
    """
    An utility to expose an image from the project files
    """

    CATEGORY = "aihub/expose/config"
    FUNCTION = "get_exposed_image"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_image", "tooltip": "A unique custom ID for this workflow."}),
                "file_name": ("STRING", {"default": "image.png", "tooltip": "The name of the image as stored in the project files, including extension"}),
            },
            "hidden": {
                "local_file": ("STRING",),
            },
        }

    def get_exposed_image(self, id, file_name, local_file=None):
        image = None
        mask = None
        if local_file is not None:
            if (os.path.exists(local_file)):
                # Instantiate a LoadImage node and use its logic to load the file
                loader = LoadImage()
                # The load_image method returns a tuple, so we need to get the first element
                loaded_image_tuple = loader.load_image(local_file)
                image = loaded_image_tuple[0]
                mask = loaded_image_tuple[1]
            else:
                filenameOnly = os.path.basename(local_file)
                raise ValueError(f"Error: Image file not found: {filenameOnly}")
        else:
            raise ValueError("You must specify the local_file for this node to function")

        _, height, width, _ = image.shape
        
        return (image, mask, width, height,)
    
class AIHubExposeImageInfoOnly:
    """
    An utility to expose an image (info only) to be used in the workflow
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_image_info_only"
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("POS_X", "POS_Y", "LAYER_ID", "WIDTH", "HEIGHT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_image", "tooltip": "A unique custom ID for this workflow."}),
                "label": ("STRING", {"default": "Image", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "type": ([
                    "current_layer",
                    "current_layer_at_image_intersection",
                    "merged_image",
                    "merged_image_without_current_layer",
                    "merged_image_current_layer_intersection",
                    "merged_image_current_layer_intersection_without_current_layer",
                    "upload",
                ], {"default": "upload", "tooltip": "The source of the image"}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            },
            "hidden": {
                "pos_x": ("INT", {"default": 0, "tooltip": "The X position of the layer in the canvas"}),
                "pos_y": ("INT", {"default": 0, "tooltip": "The Y position of the layer in the canvas"}),
                "layer_id": ("STRING", {"default": "", "tooltip": "The layer id of the image."}),
                "value_width": ("INT", {"tooltip": "width of the image."}),
                "value_height": ("INT", {"tooltip": "height of the image."}),
            }
        }

    def get_exposed_image_info_only(self, id, label, tooltip, type, index, pos_x=0, pos_y=0, layer_id="", value_width=1024, value_height=1024):
        return (pos_x, pos_y, layer_id, value_width, value_height,)
    
class AIHubExposeImageBatch:
    """
    An utility to expose an image batch to be used in the workflow
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_image_batch"
    
    RETURN_TYPES = ("IMAGE", "MASK", "AIHUB_METADATA", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASKS", "METADATA", "WIDTH", "HEIGHT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_image_batch", "tooltip": "A unique custom ID for this workflow."}),
                "label": ("STRING", {"default": "Image Batch", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "type": (["all_frames", "all_layers_at_image_size", "upload"], {"default": "upload", "tooltip": "The source of the image batch"}),
                "minlen": ("INT", {"default": 0}),
                "maxlen": ("INT", {"default": 1000}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
                "metadata_fields": ("STRING", {"default": "", "multiline": True, "tooltip": "A newline separated list of metadata fields to include in the metadata JSON for each image in the batch," +
                                               " add a space with the type next to it, if not specified it will be considered integer, valid types are: INT, FLOAT, FLOAT, STRING and BOOLEAN." +
                                               " A second space and further allows for specifying modifiers, valid modifiers are SORTED, UNIQUE, NONZERO, NONEMPTY, MULTILINE. " +
                                               " for BOOLEAN it is also possible to use ONE_TRUE and ONE_FALSE as modifiers." +
                                               " It is also possible to add numeric validity modifiers with a colon, for example MAX:100, MAXLEN:100, MIN:0 or MINLEN:0 " +
                                               " But also a property name provided that property exist in the project and is an expose integer or expose project integer " +
                                               " For example: 'frame_number INT POSITIVE SORTED MAX:total_frames\nprompt_at_frame STRING NONEMPTY MULTILINE MAXLEN:100'"}),
                "metadata_fields_label": ("STRING", {"default": "", "multiline": True, "tooltip": "A newline separated list of labels for the metadata fields to include in the metadata JSON for each image in the batch. Must match the number of metadata fields."}),
            },
            "optional": {
                "normalizer": ("AIHUB_NORMALIZER", {"tooltip": "The method to use for normalizing the images in the batch, if not specified it will use the image with the most megapixels as the target size with a nearest-exact upscaler"}),
            },
            "hidden": {
                "local_files": ("STRING", {"default": "[]"}),
                "metadata": ("STRING", {"default": "[]", "tooltip": "The image batch metadata as a JSON string"}),
            }
        }

    def get_exposed_image_batch(self, id, label, tooltip, type, minlen, maxlen, index, metadata_fields, metadata_fields_label, normalizer=None, local_files=None, metadata="[]"):
        image_batch = None
        masks = None
        metadata = None
        width = 0
        height = 0

        normalizer = normalizer if normalizer is not None else Normalizer(0, 0, "nearest-exact")

        if local_files:
            # If a local_files string is provided, attempt to load the images.
            try:
                # Parse the JSON string into a Python list of filenames.
                filenames = json.loads(local_files)
                metadata = json.loads(metadata)
                if not isinstance(metadata, list):
                    raise ValueError("Error: data is not a valid JSON string encoding an array")
                if not isinstance(filenames, list):
                    raise ValueError("Error: local_files is not a valid JSON string encoding an array")
                if len(filenames) > maxlen:
                    raise ValueError(f"Error: {id} contains too many files")
                if len(metadata) != len(filenames):
                    raise ValueError(f"Error: {id} the number of files does not match the number of metadata entries")

                metadata = json.loads(metadata)
                if not isinstance(metadata, list):
                    raise ValueError("Error: metadata is not a valid JSON string encoding an array")

                metadata_fields_list = [line.strip() for line in metadata_fields.split("\n") if line.strip()]
                true_booleans = []
                false_booleans = []
                previous_value_of_sorted_field = {}
                for item in metadata:
                    if not isinstance(item, dict):
                        raise ValueError("Error: metadata items must be objects")
                    
                    for field_def in metadata_fields_list:
                        field_splitted = field_def.split(" ")
                        field_name = field_splitted[0]
                        field_type = field_splitted[1]
                        field_modifiers = field_splitted[2:] if len(field_splitted) > 2 else []

                        if field_type not in item:
                            raise ValueError(f"Error: metadata item is missing field '{field_def}'")
                        elif field_type == "INT" and not isinstance(item[field_name], int):
                            raise ValueError(f"Error: metadata field '{field_name}' is not of type INT")
                        elif field_type == "FLOAT" and not isinstance(item[field_name], float):
                            raise ValueError(f"Error: metadata field '{field_name}' is not of type FLOAT")
                        elif field_type == "STRING" and not isinstance(item[field_name], str):
                            raise ValueError(f"Error: metadata field '{field_name}' is not of type STRING")
                        elif field_type == "BOOLEAN" and not isinstance(item[field_name], bool):
                            raise ValueError(f"Error: metadata field '{field_name}' is not of type BOOLEAN")
                        
                        if "ONE_TRUE" in field_modifiers and field_type == "BOOLEAN" and item[field_name] is True and field_name in true_booleans:
                            raise ValueError(f"Error: metadata field '{field_name}' is marked as ONE_TRUE but multiple items have it set to true")
                        if "ONE_FALSE" in field_modifiers and field_type == "BOOLEAN" and item[field_name] is False and field_name in false_booleans:
                            raise ValueError(f"Error: metadata field '{field_name}' is marked as ONE_FALSE but multiple items have it set to false")
                        
                        if field_type == "BOOLEAN" and item[field_name] is True:
                            true_booleans.append(field_name)
                        elif field_type == "BOOLEAN" and item[field_name] is False:
                            false_booleans.append(field_name)

                        if "NONZERO" in field_modifiers and field_type in ["INT", "FLOAT"] and item[field_name] == 0:
                            raise ValueError(f"Error: metadata field '{field_name}' is marked as NONZERO but has a value of zero")
                        
                        if "NONEMPTY" in field_modifiers and field_type == "STRING" and item[field_name] == "":
                            raise ValueError(f"Error: metadata field '{field_name}' is marked as NONEMPTY but is empty")
                        
                        if "UNIQUE" in field_modifiers:
                            # check if the value is unique in the metadata list
                            values = [m[field_name] for m in metadata if field_name in m]
                            if values.count(item[field_name]) > 1:
                                raise ValueError(f"Error: metadata field '{field_name}' is marked as UNIQUE but has the duplicate value '{item[field_name]}'")

                        # find max and min modifiers
                        for modifier in field_modifiers:
                            if modifier.startswith("MAX:"):
                                max_value = modifier[4:]
                                # make sure max_value is a number
                                try:
                                    float(max_value)
                                except ValueError:
                                    continue

                                if field_type in ["INT", "FLOAT"]:
                                    if item[field_name] > float(max_value):
                                        raise ValueError(f"Error: metadata field '{field_name}' exceeds maximum value of {max_value}")
                            elif modifier.startswith("MIN:"):
                                min_value = modifier[4:]

                                # make sure min_value is a number
                                try:
                                    float(min_value)
                                except ValueError:
                                    continue

                                if field_type in ["INT", "FLOAT"]:
                                    if item[field_name] < float(min_value):
                                        raise ValueError(f"Error: metadata field '{field_name}' is below minimum value of {min_value}")
                            elif modifier.startswith("MINLEN:"):
                                min_value = modifier[6:]

                                # make sure min_value is a number
                                try:
                                    float(min_value)
                                except ValueError:
                                    continue

                                if field_type == "STRING":
                                    if len(item[field_name]) < int(min_value):
                                        raise ValueError(f"Error: metadata field '{field_name}' is below minimum length of {min_value}")
                            elif modifier.startswith("MAXLEN:"):
                                max_value = modifier[6:]

                                # make sure max_value is a number
                                try:
                                    float(max_value)
                                except ValueError:
                                    continue

                                if field_type == "STRING":
                                    if len(item[field_name]) > int(max_value):
                                        raise ValueError(f"Error: metadata field '{field_name}' exceeds maximum length of {max_value}")
                                    
                        if "SORTED" in field_modifiers:
                            if field_name in previous_value_of_sorted_field:
                                if item[field_name] < previous_value_of_sorted_field[field_name]:
                                    raise ValueError(f"Error: metadata field '{field_name}' is marked as SORTED but the items are not in sorted order")
                            previous_value_of_sorted_field[field_name] = item[field_name]

                loaded_images = []
                loaded_masks = []
                loader = LoadImage()
                
                for filename in filenames:
                    # Check if the file exists before trying to load it.
                    if os.path.exists(filename):
                        # Load each image and append it to the list.
                        loaded_img_tuple = loader.load_image(filename)
                        loaded_images.append(loaded_img_tuple[0])
                        loaded_masks.append(loaded_img_tuple[1])
                    else:
                        filenameOnly = os.path.basename(filename)
                        raise ValueError(f"Error: Image file not found: {filenameOnly}")
                    
                image_batch, masks, width, height = normalizer.normalize(loaded_images, loaded_masks)

                # Concatenate all the images into a single batch tensor.
                image_batch = torch.cat(loaded_images, dim=0)
                masks = torch.cat(loaded_masks, dim=0)
            except json.JSONDecodeError:
                # Handle invalid JSON input gracefully.
                raise ValueError("Error: local_files is not a valid JSON string encoding an array")
        else:
            # Return an empty placeholder if no input is provided.
            raise ValueError("You must specify local_files for this node to function")
        
        return (image_batch, masks, metadata, width, height,)
    
class AIHubExposeProjectImageBatch:
    """
    An utility to expose an image batch from the project files
    """

    CATEGORY = "aihub/expose/config"
    FUNCTION = "get_exposed_image_batch"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_image_batch", "tooltip": "A unique custom ID for this workflow."}),
                "file_name": ("STRING", {"default": "exposed_image_batch.png", "tooltip": "The filename of the image batch as stored in the project files, including extension, without the number counter"}),
                "indexes": ("STRING", {"default": "", "tooltip": "A comma separated list of indexes to load from the image batch" +
                                       " For example: '0,1,2' to load the first three images, or a range like '0-4' to load the first five images; negative indexes are supported as well"}),
            },
            "optional": {
                "normalizer": ("AIHUB_NORMALIZER", {"tooltip": "The method to use for normalizing the images in the batch, if not specified it will use the image with the most megapixels as the target size with a nearest-exact upscaler"}),
            },
            "hidden": {
                "local_files": ("STRING", {"default": "[]"}),
            }
        }

    def get_exposed_image_batch(self, id, file_name, indexes, normalizer=None, local_files=None):
        loaded_images = None
        loaded_masks = None
        if local_files:
            # If a local_files string is provided, attempt to load the images.
            try:
                # Parse the JSON string into a Python list of filenames.
                filenames = json.loads(local_files)
                if not isinstance(filenames, list):
                    raise ValueError("Error: local_files is not a valid JSON string encoding an array")
                
                loaded_images = []
                loaded_masks = []
                loader = LoadImage()
                
                for filename in filenames:
                    # Check if the file exists before trying to load it.
                    if os.path.exists(filename):
                        # Load each image and append it to the list.
                        loaded_img_tuple = loader.load_image(filename)
                        loaded_images.append(loaded_img_tuple[0])
                        loaded_masks.append(loaded_img_tuple[1])

            except json.JSONDecodeError:
                # Handle invalid JSON input gracefully.
                raise ValueError("Error: local_files is not a valid JSON string encoding an array")

        if not loaded_images:
            raise ValueError(f"Error: {id} no valid images found")

        normalizer = normalizer if normalizer is not None else Normalizer(0, 0, "nearest-exact")
        return normalizer.normalize(loaded_images, loaded_masks)

class AIHubExposeModel:
    """
    An utility to expose a model to be used in the workflow
    loras will be applied as well if they are defined
    as allowable by the model

    The model exposer is quite complicated as it combines a lot of loaders into a single compact package
    and it does so because it has to be able to handle a lot of different scenarios

    It doesn't have UI autocomplete in comfy itself for the models or loras, however you can leave the following fields blank as
    doing so will just make the client side UI pick the first one it gets as default

    model, loras, loras_strengths, loras_use_loader_model_only, is_diffusion_model, diffusion_model_weight_dtype, optional_vae, optional_clip, optional_clip_type
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_model"
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_model", "tooltip": "A unique custom id for this workflow (it should be unique)"}),
                "label": ("STRING", {"default": "Model", "tooltip": "This is the label that will appear in the field"}),
                "model": ("STRING", {"default": "", "tooltip": "The default model to use as base for this workflow"}),
                "loras": ("STRING", {"default": "", "tooltip": "The default, comma separated list of loras to apply to this model"}),
                "loras_strengths": ("STRING", {"default": "", "tooltip": "The default, comma separated list of lora strengths to apply to this model, it must match the number of loras given"}),
                "loras_use_loader_model_only": ("STRING", {"default": "", "tooltip": "The default, comma separated list of booleans for the loras, 't' or 'f' to apply only to the model and not the clip, it must match the number of loras given"}),
                "is_diffusion_model": ("BOOLEAN", {"default": True, "tooltip": "If set to true, it will load the model from the diffusion_models folder, if false it will load it from the checkpoints folder"}),
                "diffusion_model_weight_dtype": (WEIGHT_DTYPES, {"default": "default", "tooltip": "The weight dtype to use when loading the diffusion model, this is only used if is_diffusion_model is true"}),
                "limit_to_family": ("STRING", {"default": "", "tooltip": "The family of the model to be loaded is limited by this value"}),
                "limit_to_group": ("STRING", {"default": "", "tooltip": "The group of the model to be loaded is limited by this value"}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will make this option be hidden under advanced options for this workflow"}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
                "disable_loras_selection": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will disable the loras selection field in the UI"}),
                "disable_checkpoint_selection": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will disable the checkpoint selection field in the UI"}),
            },
            "optional": {
                "optional_vae": ("STRING", {"default": "", "tooltip": "The default for an optional VAE to load, if not given the VAE from the checkpoint will be used if available"}),
                "optional_clip": ("STRING", {"default": "", "tooltip": "The default for an optional CLIP to load, if not given the CLIP from the checkpoint will be used if available"}),
                "optional_clip_type": ("STRING", {"default": "", "tooltip": "The default for an optional CLIP to load, if not given the CLIP from the checkpoint will be used if available"}),
            }
        }

    def get_exposed_model(self, id, label, model, loras, loras_strengths, loras_use_loader_model_only, is_diffusion_model, diffusion_model_weight_dtype, limit_to_family, limit_to_group, tooltip, advanced, index,
                          disable_loras_selection, disable_checkpoint_selection, optional_vae="", optional_clip="", optional_clip_type=""):
        # first lets load the checkpoint
        (model_loaded, clip, vae) = AIHubUtilsLoadModel().load_model(
            model,
            is_diffusion_model,
            diffusion_model_weight_dtype,
        )

        if optional_vae is not None and optional_vae.strip() != "":
            (vae,) = AIHubUtilsLoadVAE().load_vae(optional_vae)

        if optional_clip is not None and optional_clip.strip() != "":
            if optional_clip_type is None or optional_clip_type.strip() == "":
                raise ValueError("Error: If optional_clip is given, optional_clip_type must be given as well")
            # now we need to comma separate to check if we have multiple clips to merge, we only take the first two in case
            # of multiple because that is the default to use DualClip
            clips = [c.strip() for c in optional_clip.split(",") if c.strip()]
            # ensure that only two clips are given exactly
            if len(clips) > 2:
                raise ValueError("Error: Only a max of two optional_clip entries are supported for DualClip")

            (clip,) = AIHubUtilsLoadCLIP().load_clip(clips[0], clips[1] if len(clips) > 1 else "", optional_clip_type)

        # now we have to apply the loras if given
        if loras and model_loaded is not None:
            lora_list = [l.strip() for l in loras.split(",") if l.strip()]
            strengths_list = [float(s.strip()) for s in loras_strengths.split(",") if s.strip()]
            use_loader_model_only_list = [s.strip() == "t" for s in loras_use_loader_model_only.split(",")]
            if len(strengths_list) != len(lora_list):
                raise ValueError("Error: The number of lora strengths must match the number of loras")
            if len(use_loader_model_only_list) != len(lora_list):
                raise ValueError("Error: The number of lora_use_loader_model_only values must match the number of loras")
            for lora, strength, use_loader_model_only in zip(lora_list, strengths_list, use_loader_model_only_list):
                model_loaded, clip = AIHubUtilsLoadLora().load_lora(model_loaded, clip, lora, use_loader_model_only, strength)

        return (model_loaded, clip, vae,)

class AIHubExposeModelSimple:
    """
    Exposes a model without any of the defaults or lora application logic
    it still exists under the hood but remains hidden
    """

    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_model"
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_model", "tooltip": "A unique custom id for this workflow (it should be unique)"}),
                "label": ("STRING", {"default": "Model", "tooltip": "This is the label that will appear in the field"}),
                "limit_to_family": ("STRING", {"default": "", "tooltip": "The family of the model to be loaded is limited by this value"}),
                "limit_to_group": ("STRING", {"default": "", "tooltip": "The group of the model to be loaded is limited by this value"}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will make this option be hidden under advanced options for this workflow"}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
                "disable_loras_selection": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will disable the loras selection field in the UI"}),
            },
            "hidden": {
                "model": ("STRING", {"default": "", "tooltip": "The default model to use as base for this workflow"}),
                "loras": ("STRING", {"default": "", "tooltip": "The default, comma separated list of loras to apply to this model"}),
                "loras_strengths": ("STRING", {"default": "", "tooltip": "The default, comma separated list of lora strengths to apply to this model, it must match the number of loras given"}),
                "loras_use_loader_model_only": ("STRING", {"default": "", "tooltip": "The default, comma separated list of booleans for the loras, 't' or 'f' to apply only to the model and not the clip, it must match the number of loras given"}),
                "is_diffusion_model": ("BOOLEAN", {"default": True, "tooltip": "If set to true, it will load the model from the diffusion_models folder, if false it will load it from the checkpoints folder"}),
                "diffusion_model_weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "tooltip": "The weight dtype to use when loading the diffusion model, this is only used if is_diffusion_model is true"}),
                "optional_vae": ("STRING", {"default": "", "tooltip": "The default for an optional VAE to load, if not given the VAE from the checkpoint will be used if available"}),
                "optional_clip": ("STRING", {"default": "", "tooltip": "The default for an optional CLIP to load, if not given the CLIP from the checkpoint will be used if available"}),
                "optional_clip_type": ("STRING", {"default": "", "tooltip": "The default for an optional CLIP to load, if not given the CLIP from the checkpoint will be used if available"}),
            },
        }

    def get_exposed_model(self, id, label, limit_to_family, limit_to_group, tooltip, advanced, index,
                          disable_loras_selection, model="", loras="", loras_strengths="", loras_use_loader_model_only="", is_diffusion_model=True, diffusion_model_weight_dtype="default",
                          optional_vae="", optional_clip="", optional_clip_type=""):
        
        return AIHubExposeModel().get_exposed_model(
            id,
            label,
            model,
            loras,
            loras_strengths,
            loras_use_loader_model_only,
            is_diffusion_model,
            diffusion_model_weight_dtype,
            limit_to_family,
            limit_to_group,
            tooltip,
            advanced,
            index,
            disable_loras_selection,
            False,
            optional_vae,
            optional_clip,
            optional_clip_type,
        )
    
class AIHubExposeProjectText:
    """
    An utility to expose text to be used in the workflow to expose
    text files stored in the project files
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_text"

    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_text", "tooltip": "A unique custom id for this workflow."}),
                "file_name": ("STRING", {"default": "text.txt", "tooltip": "The name of the text file as stored in the project files, including extension"}),
                "batch_index": ("STRING", {"default": "", "tooltip": "If the file belongs to a batch, the index of the latent to load, it must be a single integer, and it can be negative to count from the end"}),
            },
            "hidden": {
                "local_file": ("STRING", {"default": "", "tooltip": "A local file to load the text from, if given this will be used instead of loading from file"}),
            }
        }

    def get_exposed_text(self, id, file_name, batch_index, local_file=None):
        text = None
        if local_file is not None and local_file.strip() != "":
            if (os.path.exists(local_file)):
                with open(local_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                filenameOnly = os.path.basename(local_file)
                raise ValueError(f"Error: Text file not found: {filenameOnly}")
        else:
            raise ValueError("You must specify the local_file of the text for this node to function")

        return (text,)

class AIHubExposeProjectVideo:
    """
    An utility to expose video files to be used in the workflow
    Because this util does not have the capacity to process the video file itself
    into something usable, it merely exposes the file name to be used by other nodes
    that can handle video files
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_video"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("VIDEO_FILEPATH",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_video", "tooltip": "A unique custom id for this workflow."}),
                "file_name": ("STRING", {"default": "video.mp4", "tooltip": "The name of the video file as stored in the project files, including extension"}),
                "batch_index": ("STRING", {"default": "", "tooltip": "If the file belongs to a batch, the index of the latent to load, it must be a single integer, and it can be negative to count from the end"}),
            },
            "hidden": {
                "local_file": ("STRING", {"default": "", "tooltip": "A local file to load the video from, if given this will be used instead of loading from file"}),
            }
        }

    def get_exposed_video(self, id, file_name, batch_index, local_file=None):
        video_file = None
        if local_file is not None and local_file.strip() != "":
            if (os.path.exists(local_file)):
                video_file = local_file
            else:
                filenameOnly = os.path.basename(local_file)
                raise ValueError(f"Error: Video file not found: {filenameOnly}")
        else:
            raise ValueError("You must specify the local_file of the video for this node to function")

        return (video_file,)
    
class AIHubExposeVideo:
    """
    An utility to expose video files to be used in the workflow
    Because this util does not have the capacity to process the video file itself
    into something usable, it merely exposes the file name to be used by other nodes
    that can handle video files
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_video"

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("VIDEO_FILEPATH", "SEGMENT_ID")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_video", "tooltip": "A unique custom ID for this workflow."}),
                "label": ("STRING", {"default": "Video", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "type": (["current_segment", "merged_video", "upload"], {"default": "upload", "tooltip": "The source of the video"}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            },
            "hidden": {
                "segment_id": ("STRING", {"default": "", "tooltip": "The segment id of the video layer"}),
                "local_file": ("STRING", {"default": "", "tooltip": "A local file to load the video from, if given this will be used instead of loading from file"}),
            }
        }

    def get_exposed_video(self, id, label, tooltip, type, index, segment_id="", local_file=None):
        video_file = None
        if local_file is not None and local_file.strip() != "":
            if (os.path.exists(local_file)):
                video_file = local_file
            else:
                filenameOnly = os.path.basename(local_file)
                raise ValueError(f"Error: Video file not found: {filenameOnly}")
        else:
            raise ValueError("You must specify the local_file of the video for this node to function")

        return (video_file, segment_id)

class AIHubExposeProjectAudio:
    """
    An utility to expose audio files to be used in the workflow
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_audio"

    RETURN_TYPES = ("AUDIO",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_audio", "tooltip": "A unique custom id for this workflow."}),
                "file_name": ("STRING", {"default": "audio.mp3", "tooltip": "The name of the audio file as stored in the project files, including extension"}),
                "batch_index": ("STRING", {"default": "", "tooltip": "If the file belongs to a batch, the index of the latent to load, it must be a single integer, and it can be negative to count from the end"}),
            },
            "hidden": {
                "local_file": ("STRING", {"default": "", "tooltip": "A local file to load the audio from, if given this will be used instead of loading from file"}),
            }
        }

    def get_exposed_audio(self, id, file_name, batch_index, local_file=None):
        audio = None
        if local_file is not None and local_file.strip() != "":
            if (os.path.exists(local_file)):
                # The load_audio method returns a tuple, so we need to get the first element
                waveform, sample_rate = load_audio_file(local_file)
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            else:
                filenameOnly = os.path.basename(local_file)
                raise ValueError(f"Error: Audio file not found: {filenameOnly}")
        else:
            raise ValueError("You must specify the local_file of the audio for this node to function")
        return (audio,)
    
class AIHubExposeAudio:
    """
    An utility to expose audio files to be used in the workflow
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_audio"

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("AUDIO", "SEGMENT_ID")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_audio", "tooltip": "A unique custom ID for this workflow."}),
                "label": ("STRING", {"default": "Audio", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "type": (["current_segment", "merged_audio", "upload"], {"default": "upload", "tooltip": "The source of the audio"}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            },
            "hidden": {
                "segment_id": ("STRING", {"default": "", "tooltip": "The segment id of the audio layer"}),
                "local_file": ("STRING", {"default": "", "tooltip": "A local file to load the audio from, if given this will be used instead of loading from file"}),
            }
        }

    def get_exposed_audio(self, id, label, tooltip, type, index, segment_id="", local_file=None):
        audio = None
        if local_file is not None and local_file.strip() != "":
            if (os.path.exists(local_file)):
                # The load_audio method returns a tuple, so we need to get the first element
                waveform, sample_rate = load_audio_file(local_file)
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            else:
                filenameOnly = os.path.basename(local_file)
                raise ValueError(f"Error: Audio file not found: {filenameOnly}")
        else:
            raise ValueError("You must specify the local_file of the audio for this node to function")
        return (audio, segment_id)

class AIHubExposeProjectLatent:
    """
    An utility to expose latent files to be used in the workflow
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_latent"
    RETURN_TYPES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_latent", "tooltip": "A unique custom id for this workflow."}),
                "file_name": ("STRING", {"default": "latent.safetensors", "tooltip": "The name of the latent file as stored in the project files, including extension"}),
                "batch_index": ("STRING", {"default": "", "tooltip": "If the file belongs to a batch, the index of the latent to load, it must be a single integer, and it can be negative to count from the end"}),
            },
            "hidden": {
                "local_file": ("STRING", {"default": "", "tooltip": "A local file to load the latent from, if given this will be used instead of loading from file"}),
            }
        }
    
    def get_exposed_latent(self, id, file_name, batch_index, local_file=None):
        samples = None
        if local_file is not None and local_file.strip() != "":
            if (os.path.exists(local_file)):
                latent = safetensors.torch.load_file(local_file, device="cpu")
                multiplier = 1.0
                if "latent_format_version_0" not in latent:
                    multiplier = 1.0 / 0.18215
                samples = {"samples": latent["latent_tensor"].float() * multiplier}
            else:
                filenameOnly = os.path.basename(local_file)
                raise ValueError(f"Error: Latent file not found: {filenameOnly}")
        else:
            raise ValueError("You must specify the local_file of the latent for this node to function")
        return (samples,)

## Actions
class AIHubActionNewImage:
    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"
    
    RETURN_TYPES = ("IMAGE", "MASK",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "action": (["REPLACE", "APPEND"], {"default": "APPEND", "tooltip": "If append is selected, if the image exist it will be added a number to the name to make it unique, if replace is selected, the image will be replaced with the new one"}),
                "name": ("STRING", {"default": "new image", "tooltip": "The name of the image this is used internally"}),
            },
            "optional": {
                "mask": ("MASK",),
                "file_name": ("STRING", {"default": "", "tooltip": "The filename to use, if not given the name value will be used with a .png extension"}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    def run_action(self, image, action, name, mask=None, file_name=""):
        if not file_name:
            file_name = name
            if not file_name.lower().endswith(".png"):
                file_name += ".png"
            # replace spaces with underscores
            file_name = file_name.replace(" ", "_")

        c_image = image[0]
        # we have to convert this image to bytes and apply the mask if the mask is given
        if mask is not None:
            c_mask = mask[0]
            # convert mask to binary
            mask_np = c_mask.cpu().numpy()
            alpha_channel = Image.fromarray((mask_np * 255).clip(0, 255).astype(np.uint8), mode='L')
            i = 255. * c_image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.putalpha(alpha_channel)
        else:
            i = 255. * c_image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')

        SERVER.send_binary_data_to_current_client_sync(
            img_bytes.getvalue(), "image/png", {
                "action": "NEW_IMAGE",
                "width": c_image.shape[1],
                "height": c_image.shape[0],
                "type": "image/png",
                "file_action": action,
                "file_name": file_name,
                "name": name,
            },
        )

        return (image, mask)
    
class AIHubActionNewImageBatch:
    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "action": (["REPLACE", "APPEND"], {"default": "APPEND", "tooltip": "If append is selected, the image batch will extend the previous potentially existing images, if replace is selected, the image batch will replace all existing images"}),
                "name": ("STRING", {"default": "new image batch", "tooltip": "The name of the images to be used as filenames, a number will be added to each image to make it unique"}),
            },
            "optional": {
                "masks": ("MASK",),
                "file_name": ("STRING", {"default": "", "tooltip": "The filename to use as base extension included, if not given the name value will be used with a .png extension"}),
            }
        }

    def run_action(self, images, action, name, masks=None, file_name=""):
        if not file_name:
            file_name = name
            if not file_name.lower().endswith(".png"):
                file_name += ".png"
            # replace spaces with underscores
            file_name = file_name.replace(" ", "_")

        SERVER.send_json_to_current_client_sync(
            {
                "type": "PREPARE_BATCH",
                "file_name": file_name,
                "name": name,
                "file_action": action,
            }
        )

        c_images = images
        # we have to convert this image to bytes and apply the mask if the mask is given
        for j in range(c_images.shape[0]):
            # first we need to get the basename and the extension if any provided
            base_name, ext = os.path.splitext(name)
            # we don't use the extension, we always save as png
            file_name = base_name + "_" + str(j + 1) + ".png"
            # replace spaces with underscores
            file_name = file_name.replace(" ", "_")

            c_image = c_images[j]
            c_mask = masks[j] if masks is not None else None

            if c_mask is not None:
                # convert mask to binary
                mask_np = c_mask.cpu().numpy()
                alpha_channel = Image.fromarray((mask_np * 255).clip(0, 255).astype(np.uint8), mode='L')
                i = 255. * c_image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                img.putalpha(alpha_channel)
            else:
                i = 255. * c_image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')

            SERVER.send_binary_data_to_current_client_sync(
                img_bytes.getvalue(), "image/png", {
                    "action": "NEW_IMAGE",
                    "width": c_images.shape[2],
                    "height": c_images.shape[1],
                    "type": "image/png",
                    "file_action": "APPEND",
                    "file_name": file_name,
                    "name": name,
                    "count": c_images.shape[0],
                    "batch_index": j,
                    "batch_size": c_images.shape[0],
                },
        )

        return (images,)
    
class AIHubActionNewFrames:
    """
    Builds new frames for video from a batch of images
    effectively acts like as AIHubNewImageBatch but sends a special event
    telling it to use the images as frames for a video
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "action": (["REPLACE", "APPEND"], {"default": "REPLACE", "tooltip": "This is a file level action for file storage, if append is selected, the images will be added to existing ones, if replace is selected, existing images will be replaced"}),
                "name": ("STRING", {"default": "new image batch", "tooltip": "The name of the images to be used as filenames, a number will be added to each image to make it unique"}),
                "insert_index": ("INT", {"default": -1, "tooltip": "The starting index for the frames, where to place them, negative indexes are allowed to insert at the end","min": -2147483648, "max": 2147483647}),
                "insert_action": (["REPLACE", "APPEND", {"default": "REPLACE", "tooltip": "The action to execute on the video frames themselves that are being worked on"}])
            },
            "optional": {
                "file_name": ("STRING", {"default": "", "tooltip": "The filename to use as base extension included, if not given the name value will be used with a .png extension"}),
            }
        }

    def run_action(self, images, action, name, insert_index, insert_action, file_name=""):
        AIHubActionNewImageBatch().run_action(images, action, name, None, file_name)

        if not file_name:
            file_name = name
            if not file_name.lower().endswith(".png"):
                file_name += ".png"
            # replace spaces with underscores
            file_name = file_name.replace(" ", "_")

        SERVER.send_json_to_current_client_sync(
            {
                "type": "USE_AS_FRAMES",
                "file_name": file_name,
                "name": name,
                "file_action": action,
                "count": images.shape[0],
                "insert_index": insert_index,
                "insert_action": insert_action,
            }
        )
    
class AIHubActionNewLayer:
    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"
    
    RETURN_TYPES = ("IMAGE", "MASK",)
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pos_x": ("INT", {"default": 0, "tooltip": "The X position of the layer in the canvas"}),
                "pos_y": ("INT", {"default": 0, "tooltip": "The Y position of the layer in the canvas"}),
                "reference_layer_id": ("STRING", {"default": "", "tooltip": "The Layer ID regarding the position of the image, the meta-values __first__ and __last__ can be used to refer to the first and last layer respectively"}),
                "reference_layer_action": (["REPLACE", "NEW_BEFORE", "NEW_AFTER"], {"default": "NEW_AFTER", "tooltip": "Specify the action to execute at the given layer id"}),
                "name": ("STRING", {"default": "new layer", "tooltip": "The name of the layer this is used for the layer list in the UI"}),
                "action": (["APPEND", "REPLACE"], {"default": "REPLACE", "tooltip": "This refers to the file system level action, if append is selected, if the image exist it will be added a number" +
                                                   " to the name to make it unique, if replace is selected, the image will be replaced with the new one; since layers are supposed to be integrated in the project REPLACE is often best"}),
            },
            "optional": {
                "mask": ("MASK",),
                "file_name": ("STRING", {"default": "", "tooltip": "The filename to use, if not given the name value will be used with a .png extension"}),
            }
        }

    def run_action(self, image, pos_x, pos_y, reference_layer_id, reference_layer_action, name, action="REPLACE", mask=None, file_name=""):
        if not file_name:
            file_name = name
            if not file_name.lower().endswith(".png"):
                file_name += ".png"
            # replace spaces with underscores
            file_name = file_name.replace(" ", "_")

        c_image = image[0]
        # we have to convert this image to bytes and apply the mask if the mask is given
        if mask is not None:
            c_mask = mask[0]
            # convert mask to binary
            mask_np = c_mask.cpu().numpy()
            alpha_channel = Image.fromarray((mask_np * 255).clip(0, 255).astype(np.uint8), mode='L')
            i = 255. * c_image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.putalpha(alpha_channel)
        else:
            i = 255. * c_image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')

        SERVER.send_binary_data_to_current_client_sync(
            img_bytes.getvalue(), "image/png", {
                "action": "NEW_LAYER",
                "width": c_image.shape[1],
                "height": c_image.shape[0],
                "type": "image/png",
                "pos_x": pos_x,
                "pos_y": pos_y,
                "reference_layer_id": reference_layer_id,
                "reference_layer_action": reference_layer_action,
                "name": name,
                "file_name": file_name,
                "file_action": action,
            },
        )

        return (image, mask)
    
# Actions that are considered patches
# running these actions should create a patch to the project
class AIHubActionSetProjectConfigInteger:
    """
    A utility node for setting an integer from the project config.json
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"
    OUTPUT_NODE = True

    RETURN_TYPES = ()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to set, use dots for entering sublevels"}),
                "value": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "tooltip": "The value of the field to set"}),
            },
        }

    def run_action(self, field, value):
        SERVER.send_json_to_current_client_sync(
            {
                "type": "SET_CONFIG_VALUE",
                "field": field,
                "value": value,
            }
        )

        return ()
    
class AIHubActionSetProjectConfigFloat:
    """
    A utility node for setting an float from the project config.json
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"
    OUTPUT_NODE = True

    RETURN_TYPES = ()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to set, use dots for entering sublevels"}),
                "value": ("FLOAT", {"default": 0.0, "min": -1.0e+20, "max": 1.0e+20, "tooltip": "The value of the field to set"}),
            }
        }

    def run_action(self, field, value):
        SERVER.send_json_to_current_client_sync(
            {
                "type": "SET_CONFIG_VALUE",
                "field": field,
                "value": value,
            }
        )

        return ()
    
class AIHubActionSetProjectConfigBoolean:
    """
    A utility node for setting a boolean from the project config.json
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"

    RETURN_TYPES = ()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to set, use dots for entering sublevels"}),
                "value": ("BOOLEAN", {"default": True, "tooltip": "The value of the field to set"}),
            }
        }

    def run_action(self, field, value):
        SERVER.send_json_to_current_client_sync(
            {
                "type": "SET_CONFIG_VALUE",
                "field": field,
                "value": value,
            }
        )

        return ()
    
class AIHubActionSetProjectConfigString:
    """
    A utility node for setting a string from the project config.json
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"
    OUTPUT_NODE = True

    RETURN_TYPES = ()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to set, use dots for entering sublevels"}),
                "value": ("STRING", {"default": "", "tooltip": "The value of the field to set"}),
            }
        }

    def run_action(self, field, value):
        SERVER.send_json_to_current_client_sync(
            {
                "type": "SET_CONFIG_VALUE",
                "field": field,
                "value": value,
            }
        )

        return ()
    
class AIHubActionNewLatent:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "action": (["REPLACE", "APPEND"], {"default": "APPEND", "tooltip": "If append is selected, the latents will be added a number to the name to make it unique, if replace is selected, the latents will be replaced with the new one"}),
                "file_name": ("STRING", {"default": "new_latent.safetensors", "tooltip": "The name of the latents this is used for the filename value directly"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    CATEGORY = "aihub/actions"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "run_action"

    def run_action(self, samples, action, file_name, prompt=None, extra_pnginfo=None):
        # support save metadata for latent sharing
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = None
        if not args.disable_metadata:
            metadata = {"prompt": prompt_info}
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        output = {}
        output["latent_tensor"] = samples["samples"].contiguous()
        output["latent_format_version_0"] = torch.tensor([])

        bytes = None
        if metadata is not None:
            bytes = safetensors.torch.save(output, metadata=metadata)
        else:
            bytes = safetensors.torch.save(output)
        
        SERVER.send_binary_data_to_current_client_sync(
            bytes, "application/octet-stream", {
                "action": "NEW_LATENT",
                "file_name": file_name,
                "file_action": action,
                "type": "application/octet-stream",
            }
        )
        return ()

class AIHubActionNewAudio:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "action": (["REPLACE", "APPEND"], {"default": "APPEND", "tooltip": "If append is selected, the audio will be added a number to the name to make it unique, if replace is selected, the audio will be replaced with the new one"}),
                "name": ("STRING", {"default": "new audio", "tooltip": "The name of the audio to be used"}),
            },
            "optional": {
                "format": (["wav", "ogg", "flac"], {"default": "wav", "tooltip": "The format to save the audio in"}),
                "file_name": ("STRING", {"default": "", "tooltip": "The filename to use, with the extension, if not given the name value will be used with the given format extension"}),
            }
        }
    
    CATEGORY = "aihub/actions"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "run_action"

    def run_action(self, audio, action, name, format="wav", file_name=""):
        if not file_name:
            file_name = name
            if not file_name.lower().endswith(f".{format}"):
                file_name += f".{format}"
            # replace spaces with underscores
            file_name = file_name.replace(" ", "_")

        waveform = audio["waveform"][0]
        sample_rate = audio["sample_rate"]
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, sample_rate, format=format.upper())
        audio_bytes = buffer.getvalue()

        SERVER.send_binary_data_to_current_client_sync(
            audio_bytes, f"audio/{format}", {
                "action": "NEW_AUDIO",
                "file_name": file_name,
                "file_action": action,
                "name": name,
                "type": f"audio/{format}",
            }
        )
        return ()
    
class AIHubActionNewAudioSegment:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "action": (["REPLACE", "APPEND"], {"default": "APPEND", "tooltip": "If append is selected, the audio will be added a number to the name to make it unique, if replace is selected, the audio will be replaced with the new one"}),
                "name": ("STRING", {"default": "new audio", "tooltip": "The name of the audio to be used"}),
                "reference_segment_id": ("STRING", {"default": "", "tooltip": "The reference segment id of the audio layer"}),
                "reference_segment_action": (["REPLACE", "NEW_BEFORE", "NEW_AFTER", "MERGE"], {"default": "NEW_AFTER", "tooltip": "Specify the action to execute at the given segment id"}),
            },
            "optional": {
                "format": (["wav", "ogg", "flac"], {"default": "wav", "tooltip": "The format to save the audio in"}),
                "file_name": ("STRING", {"default": "", "tooltip": "The filename to use, with the extension, if not given the name value will be used with the given format extension"}),
            }
        }
    
    CATEGORY = "aihub/actions"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "run_action"

    def run_action(self, audio, action, name, reference_segment_id, reference_segment_action, format="wav", file_name=""):
        if not file_name:
            file_name = name
            if not file_name.lower().endswith(f".{format}"):
                file_name += f".{format}"
            # replace spaces with underscores
            file_name = file_name.replace(" ", "_")

        waveform = audio["waveform"][0]
        sample_rate = audio["sample_rate"]
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, sample_rate, format=format.upper())
        audio_bytes = buffer.getvalue()

        SERVER.send_binary_data_to_current_client_sync(
            audio_bytes, f"audio/{format}", {
                "action": "NEW_AUDIO_SEGMENT",
                "file_name": file_name,
                "file_action": action,
                "name": name,
                "type": f"audio/{format}",
                "reference_segment_id": reference_segment_id,
                "reference_segment_action": reference_segment_action,
            }
        )
        return ()

class AIHubActionNewVideo:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING", {"default": "new_video.mp4", "tooltip": "The path of the video file to be loaded"}),
                "mime_type": ("STRING", {"default": "video/mp4", "tooltip": "The mime type of the video"}),
                "action": (["REPLACE", "APPEND"], {"default": "APPEND", "tooltip": "If append is selected, the video will be added a number to the name to make it unique, if replace is selected, the video will be replaced with the new one"}),
                "name": ("STRING", {"default": "new video", "tooltip": "The name of the video to be used"}),
            },
            "optional": {
                "file_name": ("STRING", {"default": "", "tooltip": "The filename to use, with the extension, if not given the name value will be used with the given format extension based on the mime type"}),
            }
        }
    
    CATEGORY = "aihub/actions"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "run_action"

    def run_action(self, input_file, mime_type, action, name, file_name=""):
        if not os.path.exists(input_file):
            raise ValueError(f"Error: Video file not found: {input_file}")
        
        if not file_name:
            file_name = name
            mime_type_splitted = mime_type.split("/")
            extension = mime_type_splitted[1] if len(mime_type_splitted) > 1 else "mp4"
            if not file_name.lower().endswith(f".{extension}"):
                file_name += f".{extension}"
            # replace spaces with underscores
            file_name = file_name.replace(" ", "_")

        video_bytes = None
        with open(input_file, "rb") as f:
            video_bytes = f.read()

        SERVER.send_binary_data_to_current_client_sync(
            video_bytes, mime_type, {
                "action": "NEW_VIDEO",
                "file_name": file_name,
                "file_action": action,
                "name": name,
                "type": mime_type,
            }
        )
        return ()
    
class AIHubActionNewVideoSegment:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_file": ("STRING", {"default": "new_video.mp4", "tooltip": "The path of the video file to be loaded"}),
                "mime_type": ("STRING", {"default": "video/mp4", "tooltip": "The mime type of the video"}),
                "name": ("STRING", {"default": "new video", "tooltip": "The name of the video to be used"}),
                "reference_segment_id": ("STRING", {"default": "", "tooltip": "The reference segment id of the video layer"}),
                "reference_segment_action": (["REPLACE", "NEW_BEFORE", "NEW_AFTER"], {"default": "NEW_AFTER", "tooltip": "Specify the action to execute at the given segment id"}),
            },
            "optional": {
                "file_name": ("STRING", {"default": "", "tooltip": "The filename to use, with the extension, if not given the name value will be used with the given format extension based on the mime type"}),
            }
        }
    
    CATEGORY = "aihub/actions"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "run_action"

    def run_action(self, input_file, mime_type, name, reference_segment_id, reference_segment_action, file_name=""):
        if not os.path.exists(input_file):
            raise ValueError(f"Error: Video file not found: {input_file}")

        if not file_name:
            mime_type_splitted = mime_type.split("/")
            extension = mime_type_splitted[1] if len(mime_type_splitted) > 1 else "mp4"
            file_name = name
            if not file_name.lower().endswith(f".{extension}"):
                file_name += f".{extension}"
            # replace spaces with underscores
            file_name = file_name.replace(" ", "_")

        video_bytes = None
        with open(input_file, "rb") as f:
            video_bytes = f.read()

        SERVER.send_binary_data_to_current_client_sync(
            video_bytes, mime_type, {
                "action": "NEW_VIDEO_SEGMENT",
                "file_name": file_name,
                "name": name,
                "type": mime_type,
                "reference_segment_id": reference_segment_id,
                "reference_segment_action": reference_segment_action,
            }
        )
        return ()

class AIHubActionNewText:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "New Text", "tooltip": "The text to be added"}),
                "action": (["REPLACE", "APPEND"], {"default": "APPEND", "tooltip": "If append is selected, the text will be added a number to the name to make it unique, if replace is selected, the text will be replaced with the new one"}),
                "name": ("STRING", {"default": "new text", "tooltip": "The name of the text to be used"}),
                "mime_type": ("STRING", {"default": "text/plain", "tooltip": "The mime type of the text"}),
            },
            "optional": {
                "file_name": ("STRING", {"default": "", "tooltip": "The filename to use, with the extension, if not given the name value will be used with the given format extension based on the mime type"}),
            }
        }
    
    CATEGORY = "aihub/actions"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "run_action"

    def run_action(self, text, action, name, mime_type="text/plain", file_name=""):
        if not file_name:
            file_name = name
            mime_type_splitted = mime_type.split("/")
            extension = mime_type_splitted[1] if len(mime_type_splitted) > 1 else "txt"

            # special cases for known mime types where the second part is not the extension
            if mime_type == "text/markdown":
                extension = "md"
            elif mime_type == "text/plain":
                extension = "txt"

            if not file_name.lower().endswith(f".{extension}"):
                file_name += f".{extension}"
            # replace spaces with underscores
            file_name = file_name.replace(" ", "_")

        text_bytes = text.encode('utf-8')

        SERVER.send_binary_data_to_current_client_sync(
            text_bytes, mime_type, {
                "action": "NEW_TEXT",
                "file_name": file_name,
                "file_action": action,
                "name": name,
                "type": mime_type,
            }
        )
        return ()

## UTILS

class AIHubUtilsCropMergedImageToLayerSize:
    """
    A utility node for cropping a merged image to the size of a given layer
    """

    CATEGORY = "aihub/utils"
    FUNCTION = "crop_merged_image_to_layer_size"

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("IMAGE", "MASK",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "merged_image": ("IMAGE",),
                "layer_pos_x": ("INT", {"default": 0, "tooltip": "The X position of the layer in the canvas", "min": -10000, "max": 10000}),
                "layer_pos_y": ("INT", {"default": 0, "tooltip": "The Y position of the layer in the canvas", "min": -10000, "max": 10000}),
                "layer_width": ("INT", {"default": 512, "tooltip": "The width of the layer", "min": 1, "max": 10000}),
                "layer_height": ("INT", {"default": 512, "tooltip": "The height of the layer", "min": 1, "max": 10000}),
            },
            "optional": {
                "merged_mask": ("MASK", {"optional": True, "tooltip": "The mask of the merged image"}),
            }
        }

    def crop_merged_image_to_layer_size(self, merged_image, layer_pos_x, layer_pos_y, layer_width, layer_height, merged_mask=None):
        _, merged_height, merged_width, _ = merged_image.shape

        # Calculate cropping coordinates
        x1 = max(0, layer_pos_x)
        y1 = max(0, layer_pos_y)
        x2 = min(merged_width, layer_pos_x + layer_width)
        y2 = min(merged_height, layer_pos_y + layer_height)

        # Crop the image
        cropped_image = merged_image[:, y1:y2, x1:x2, :]

        cropped_mask = None
        if merged_mask is not None:
            cropped_mask = merged_mask[:, y1:y2, x1:x2]

        return (cropped_image, cropped_mask)
    
class AIHubUtilsFitLayerToMergedImage:
    """
    A utility node for fitting a layer inside the bounds of a given merged image
    cropping position and size if necessary
    """

    CATEGORY = "aihub/utils"
    FUNCTION = "fit_layer_to_merged_image"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "POS_X", "POS_Y", "WIDTH", "HEIGHT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "merged_image_width": ("INT", {"default": 1024, "tooltip": "The width of the merged image"}),
                "merged_image_height": ("INT", {"default": 1024, "tooltip": "The height of the merged image"}),
                "layer_pos_x": ("INT", {"default": 0, "tooltip": "The X position of the layer in the canvas", "min": -10000, "max": 10000}),
                "layer_pos_y": ("INT", {"default": 0, "tooltip": "The Y position of the layer in the canvas", "min": -10000, "max": 10000}),
                "layer_image": ("IMAGE", {"tooltip": "The image of the layer"}),
            },
            "optional": {
                "layer_mask": ("MASK", {"tooltip": "The mask of the layer"}),
            }
        }

    def fit_layer_to_merged_image(self, merged_image_width, merged_image_height, layer_pos_x, layer_pos_y, layer_image, layer_mask=None):
        _, layer_height, layer_width, _ = layer_image.shape

        # layer bbox
        layer_x1 = layer_pos_x
        layer_y1 = layer_pos_y
        layer_x2 = layer_pos_x + layer_width
        layer_y2 = layer_pos_y + layer_height
        # merged image bbox
        merged_x1 = 0
        merged_y1 = 0
        merged_x2 = merged_image_width
        merged_y2 = merged_image_height

        # make an intersection of the two bboxes to get the new bbox of the layer
        new_x1 = max(layer_x1, merged_x1)
        new_y1 = max(layer_y1, merged_y1)
        new_x2 = min(layer_x2, merged_x2)
        new_y2 = min(layer_y2, merged_y2)
        new_width = new_x2 - new_x1
        new_height = new_y2 - new_y1

        # apply the new bbox to the layer image and mask
        if new_width <= 0 or new_height <= 0:
            raise ValueError("Error: The layer is completely outside the bounds of the merged image")
        
        # calculate the crop coordinates relative to the layer image
        crop_x1 = new_x1 - layer_x1
        crop_y1 = new_y1 - layer_y1
        crop_x2 = crop_x1 + new_width
        crop_y2 = crop_y1 + new_height
        cropped_image = layer_image[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
        cropped_mask = None
        if layer_mask is not None:
            cropped_mask = layer_mask[:, crop_y1:crop_y2, crop_x1:crop_x2]

        return (cropped_image, cropped_mask, new_x1, new_y1, new_width, new_height)
    
class AIHubUtilsFloatToInt:
    """
    A utility node for converting a float to an integer
    """

    CATEGORY = "aihub/utils"
    FUNCTION = "float_to_int"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "tooltip": "The float to convert to an integer"}),
            }
        }

    def float_to_int(self, value):
        try:
            int_value = int(value)
        except ValueError:
            raise ValueError(f"Error: Could not convert float to integer: {value}")
        return (int_value,)
    
class AIHubUtilsStrToFloat:
    """
    A utility node for converting a string to a float
    """

    CATEGORY = "aihub/utils"
    FUNCTION = "str_to_float"

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"default": "0.0", "tooltip": "The string to convert to a float"}),
            }
        }

    def str_to_float(self, value):
        try:
            float_value = float(value)
        except ValueError:
            raise ValueError(f"Error: Could not convert string to float: {value}")
        return (float_value,)
    
class AIHubUtilsStrToVector:
    """
    A utility node for converting a comma separated string to a vector of floats
    supports up to 3 dimensions
    """

    CATEGORY = "aihub/utils"
    FUNCTION = "str_to_vector"

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("FLOAT_X", "FLOAT_Y", "FLOAT_Z",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"default": "0.0,0.0,0.0", "tooltip": "The comma separated string to convert to a vector of floats"}),
            }
        }

    def str_to_vector(self, value):
        try:
            float_list = [float(v.strip()) for v in value.split(",") if v.strip()]
        except ValueError:
            raise ValueError(f"Error: Could not convert string to vector of floats: {value}")
        if len(float_list) >= 3:
            return (float_list[0], float_list[1], float_list[2],)
        elif len(float_list) == 2:
            return (float_list[0], float_list[1], 0.0,)
        elif len(float_list) == 1:
            return (float_list[0], 0.0, 0.0,)
    
class AIHubUtilsLoadModel:
    """
    Utility to load a model from the given filename, uses the same logic as expose checkpoint
    but allows to select if is_diffusion_model and the weight dtype
    """
    CATEGORY = "aihub/utils"
    FUNCTION = "load_model"

    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "", "tooltip": "The model to load"}),
                "is_diffusion_model": ("BOOLEAN", {"default": True, "tooltip": "If set to true, it will load the model from the diffusion_models folder, if false it will load it from the checkpoints folder"}),
                "diffusion_model_weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "tooltip": "The weight dtype to use when loading the diffusion model, this is only used if is_diffusion_model is true"}),
            },
        }

    def load_model(self, model, is_diffusion_model, diffusion_model_weight_dtype):
        model_loaded = None
        clip = None
        vae = None

        global LAST_MODEL
        global LAST_MODEL_CLIP
        global LAST_MODEL_VAE
        global LAST_MODEL_FILE_IS_DIFFUSION_MODEL
        global LAST_MODEL_FILE
        global LAST_MODEL_WEIGHT_DTYPE
        if model:
            if is_diffusion_model:
                if LAST_MODEL_FILE == model and LAST_MODEL_FILE_IS_DIFFUSION_MODEL == is_diffusion_model and LAST_MODEL_WEIGHT_DTYPE == diffusion_model_weight_dtype:
                    print(f"Reusing already loaded diffusion model {model} with weight dtype {diffusion_model_weight_dtype}")
                    model_loaded = LAST_MODEL
                else:
                    print("Using UNETLoader to load the diffusion model " + str(model) + " with weight dtype " + str(diffusion_model_weight_dtype))
                    loader = UNETLoader()
                    model_loaded, = loader.load_unet(model, diffusion_model_weight_dtype)
                    LAST_MODEL_WEIGHT_DTYPE = diffusion_model_weight_dtype
            else:
                if LAST_MODEL_FILE == model and LAST_MODEL_FILE_IS_DIFFUSION_MODEL == is_diffusion_model:
                    print(f"Reusing already loaded checkpoint model {model}")
                    model_loaded = LAST_MODEL
                    clip = LAST_MODEL_CLIP
                    vae = LAST_MODEL_VAE
                else:
                    print("Using CheckpointLoaderSimple to load the model " + str(model))
                    loader = CheckpointLoaderSimple()
                    model_loaded, clip, vae = loader.load_checkpoint(model)

            LAST_MODEL = model_loaded
            LAST_MODEL_CLIP = clip
            LAST_MODEL_VAE = vae
            LAST_MODEL_FILE = model
            LAST_MODEL_FILE_IS_DIFFUSION_MODEL = is_diffusion_model

            if model_loaded is None:
                raise ValueError(f"Error: Could not load the model checkpoint: {model}")
        else:
            print("No model specified so it cannot be loaded")
            
        return (model_loaded, clip, vae,)
    
class AIHubUtilsLoadVAE:
    """
    Utility to load a VAE from the given filename, uses the same logic as expose VAE
    """
    CATEGORY = "aihub/utils"
    FUNCTION = "load_vae"

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("STRING", {"default": "", "tooltip": "The VAE to load"}),
            },
        }

    def load_vae(self, vae):
        vae_loaded = None

        global LAST_VAE
        global LAST_VAE_FILE
        if vae:
            if LAST_VAE_FILE == vae:
                print(f"Reusing already loaded VAE {vae}")
                vae_loaded = LAST_VAE
            else:
                print("Using VAELoader to load the VAE " + str(vae))
                loader = VAELoader()
                vae_loaded, = loader.load_vae(vae)

            LAST_VAE = vae_loaded
            LAST_VAE_FILE = vae

            if vae_loaded is None:
                raise ValueError(f"Error: Could not load the VAE: {vae}")
        else:
            print("No VAE specified so it cannot be loaded")
            
        return (vae_loaded,)
    
class AIHubUtilsLoadCLIP:
    """
    Utility to load a CLIP from the given filename, uses the same logic as expose CLIP
    """
    CATEGORY = "aihub/utils"
    FUNCTION = "load_clip"

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("CLIP",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_1": ("STRING", {"default": "", "tooltip": "The CLIP to load"}),
                "clip_2": ("STRING", {"default": "", "tooltip": "The Second CLIP to load, if given, a Dual CLIP will be created"}),
                "type": (CLIP_TYPES, {"default": "stable_diffusion", "tooltip": "The type of the CLIP to load"}),
                "device": (["default", "cpu"], {"default": "default", "tooltip": "The device to load the CLIP on"}),
            },
        }

    def load_clip(self, clip_1, clip_2, type, device):
        clip_loaded = None

        global LAST_CLIP_FILE
        global LAST_CLIP
        global LAST_CLIP_DUAL
        global LAST_CLIP_TYPE
        clip_1_stripped = clip_1.strip() if clip_1 is not None else ""
        clip_2_stripped = clip_2.strip() if clip_2 is not None else ""

        if clip_1_stripped:
            if clip_2_stripped:
                # now we need to comma separate to check if we have multiple clips to merge, we only take the first two in case
                # of multiple because that is the default to use DualClip
                if LAST_CLIP_FILE == f"{clip_1_stripped},{clip_2_stripped}" and LAST_CLIP_TYPE == type and LAST_CLIP_DUAL == True:
                    print(f"Reusing already loaded Dual CLIP {clip_1_stripped},{clip_2_stripped} with type {type}")
                    clip_loaded = LAST_CLIP
                else:
                    clip_loader = DualCLIPLoader()
                    clip_loaded, = clip_loader.load_clip(clip_1_stripped, clip_2_stripped, type, device=device)
                LAST_CLIP_DUAL = True
            else:
                if LAST_CLIP_FILE == clip_1_stripped and LAST_CLIP_TYPE == type and LAST_CLIP_DUAL == False:
                    print(f"Reusing already loaded CLIP {clip_1_stripped} with type {type}")
                    clip_loaded = LAST_CLIP
                else:
                    print(type(clip_1))
                    print(f"Using CLIPLoader to load the CLIP {clip_1_stripped} with type {type}")
                    clip_loader = CLIPLoader()
                    clip_loaded, = clip_loader.load_clip(clip_1_stripped, type, device=device)
                LAST_CLIP_DUAL = False

            LAST_CLIP = clip_loaded
            LAST_CLIP_FILE = f"{clip_1_stripped},{clip_2_stripped}" if clip_2 and clip_2.strip() != "" else clip_1_stripped
            LAST_CLIP_TYPE = type

            if clip_loaded is None:
                raise ValueError(f"Error: Could not load the CLIP: {clip_1_stripped}")
        else:
            print("No CLIP specified so it cannot be loaded")
            
        return (clip_loaded,)
    
class AIHubUtilsLoadLora:
    """
    Utility to load a Lora from the given filename, uses the same logic as expose Lora
    """
    CATEGORY = "aihub/utils"
    FUNCTION = "load_lora"

    RETURN_TYPES = ("MODEL", "CLIP",)
    RETURN_NAMES = ("MODEL", "CLIP",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora": ("STRING", {"default": "", "tooltip": "The Lora to load"}),
                "use_loader_model_only": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will only apply the lora to the model and not to the clip"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "tooltip": "The strength of the lora to apply"})
            },
        }
    
    def load_lora(self, model, clip, lora, use_loader_model_only, strength):
        if lora and lora.strip() != "":
            if use_loader_model_only:
                print(f"Applying lora {lora} with strength {strength} to model only")
                lora_loader = LoraLoaderModelOnly()
                model, = lora_loader.load_lora_model_only(model, lora, strength)
                # clip remains unchanged
            else:
                print(f"Applying lora {lora} with strength {strength} to model and clip")
                lora_loader = LoraLoader()
                model, clip = lora_loader.load_lora(model, clip, lora, strength, strength)

        return (model, clip,)
    
class AIHubUtilsMetadataMap:
    """
    Takes AIHub metadata from the image batch expose and maps it to a string output
    """
    CATEGORY = "aihub/utils"
    FUNCTION = "metadata_map"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "metadata": ("AIHUB_METADATA", {"tooltip": "The metadata string from the image batch expose node"}),
                "field": ("STRING", {"default": "my_field", "tooltip": "The field to concatenate from the metadata"}),
                "separator": ("STRING", {"default": ",", "tooltip": "The separator to use between multiple values"}),
                "true_value": ("STRING", {"default": "t", "tooltip": "The value to use for true boolean values"}),
                "false_value": ("STRING", {"default": "f", "tooltip": "The value to use for false boolean values"}),
            }
        }

    def metadata_map(self, metadata, field, separator, true_value, false_value):
        if not metadata or metadata.strip() == "":
            return ("",)
        
        try:
            metadata_json = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error: Could not parse metadata JSON: {e}")

        if not isinstance(metadata_json, list):
            raise ValueError("Error: Metadata JSON is not a list")

        values = []
        for item in metadata_json:
            if field in item:
                value = item[field]
                if isinstance(value, bool):
                    value = true_value if value else false_value
                elif isinstance(value, (int, float)):
                    value = str(value)
                elif isinstance(value, str):
                    pass  # already a string
                else:
                    value = str(value)  # convert other types to string
                values.append(value)

        result = separator.join(values)
        return (result,)

class Normalizer:
    def __init__(self, normalize_at_width=0, normalize_at_height=0, normalize_upscale_method="nearest-exact"):
        self.normalize_at_width = normalize_at_width
        self.normalize_at_height = normalize_at_height
        self.normalize_upscale_method = normalize_upscale_method

    def normalize(self, images, masks, is_tensor=False):
        normalize_width = self.normalize_at_width
        normalize_height = self.normalize_at_height
        normalize_megapixels = (normalize_width * normalize_height) / 1_000_000

        if is_tensor:
            # Convert images and masks to list of tensors
            images = [img for img in images]
            if masks is not None:
                masks = [mask for mask in masks]

            if normalize_width == 0 or normalize_height == 0:
                raise ValueError("Error: When using tensor images, both normalize_at_width and normalize_at_height must be greater than 0")
        elif normalize_width == 0 and normalize_height == 0:
            for i in range(len(images)):
                img = images[i]
                _, h, w, _ = img.shape
                image_megapixels = (w * h) / 1_000_000
                if image_megapixels > normalize_megapixels:
                    normalize_width = w
                    normalize_height = h
                    break

        width = normalize_width
        height = normalize_height

        for i in range(len(images)):
            img = images[i]
            mask = None if masks is None else masks[i]
            _, h, w, _ = img.shape
                    
            if w != normalize_width or h != normalize_height:
                samples = img.movedim(-1, 1)  # NHWC -> NCHW
                samples = common_upscale(
                    samples, normalize_width, normalize_height, self.normalize_upscale_method, crop="center"
                )
                images[i] = samples.movedim(1, -1)  # NCHW -> NHWC
                if mask is not None:
                    mask = interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(height, width), mode='bilinear', align_corners=True)
                    crop_x = (normalize_width - w) // 2
                    crop_y = (normalize_height - h) // 2
                    mask = mask[:, :, crop_y:crop_y+normalize_height, crop_x:crop_x+normalize_width]
                    masks[i] = mask

        # Concatenate all the images into a single batch tensor.
        image_batch = torch.cat(images, dim=0)
        masks_batch = None if masks is None else torch.cat(masks, dim=0)

        return (image_batch, masks_batch, width, height)

class AIHubUtilsNewNormalizer:
    """
    A utility node for creating a normalizer to normalize images in expose nodes
    """
    CATEGORY = "aihub/utils"
    FUNCTION = "new_normalizer"
    RETURN_TYPES = ("AIHUB_NORMALIZER",)
    RETURN_NAMES = ("NORMALIZER",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "normalize_at_width": ("INT", {"default": 0, "tooltip": "If greater than 0, it will resize all images to this width, requiring normalize_at_height to be set, by default normalization is otherwise done at the image with most megapixels"}),
                "normalize_at_height": ("INT", {"default": 0, "tooltip": "If greater than 0, it will resize all images to this height, requiring normalize_at_width to be set, by default normalization is otherwise done at the image with most megapixels"}),
                "normalize_upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "nearest-exact", "tooltip": "The method to use when upscaling images"}),
            }
        }
    
    def new_normalizer(self, normalize_at_width, normalize_at_height, normalize_upscale_method):
        if ((normalize_at_width == 0) or (normalize_at_height == 0)) and ((normalize_at_width != 0) or (normalize_at_height != 0)):
            raise ValueError("Error: Both normalize_at_width and normalize_at_height must be set to values greater than 0, or both must be set to 0 to disable fixed size normalization")
        
        normalizer = Normalizer(normalize_at_width, normalize_at_height, normalize_upscale_method)
        return (normalizer,)
    
class AIHubUtilsScaleImageAndMasks:
    """
    A utility node for running a normalizer to scale images and masks
    """
    CATEGORY = "aihub/utils"
    FUNCTION = "run_normalizer"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The list of images to normalize"}),
                "normalize_at_width": ("INT", {"default": 0, "tooltip": "Must be greater than 0"}),
                "normalize_at_height": ("INT", {"default": 0, "tooltip": "Must be greater than 0"}),
                "normalize_upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "nearest-exact", "tooltip": "The method to use when upscaling images"}),
            },
            "optional": {
                "masks": ("MASK", {"optional": True, "tooltip": "The list of masks to normalize, if given must match the number of images"}),
            }
        }
    
    def run_normalizer(self, images, normalize_at_width, normalize_at_height, normalize_upscale_method, masks=None):
        if masks is None:
            pass
        elif len(masks) != len(images):
            raise ValueError("Error: The number of masks must match the number of images")
        
        normalizer = Normalizer(normalize_at_width, normalize_at_height, normalize_upscale_method)
        return normalizer.normalize(images, masks, is_tensor=True)
    
# === META NODES ===

def get_filename_list_for_aihub_folder(folder):
    result = []
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            result.append(os.path.splitext(file)[0])
    return result

class AIHubMetaSetExportedModelImage:
    """
    Meta node utility to set the image of an exported model
    """
    CATEGORY = "aihub/meta"
    FUNCTION = "set_exported_model_image"

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (get_filename_list_for_aihub_folder(AIHUB_MODELS_DIR), {"default": "", "tooltip": "The id of the model to set the image for"}),
                "image": ("IMAGE", {"tooltip": "The image to set for the model"}),
            }
        }
    
    def set_exported_model_image(self, model, image):
        if image is None or len(image) == 0:
            raise ValueError("Error: No image specified")

        image_filename = model + ".png"

        if image is not None:
            # save the image to the models folder with the name of the model id
            c_image = image[0]
            i = 255. * c_image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(os.path.join(AIHUB_MODELS_DIR, image_filename), format='PNG')

        return ()
    
class AIHubMetaExportModel:
    """
    Meta node utility to export a model to its json file so it can be used by
    the client side UI
    """
    CATEGORY = "aihub/meta"
    FUNCTION = "export_model"

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (get_filename_list("checkpoints") + get_filename_list("diffusion_models"), {"tooltip": "The model to export"}),
                "weight_dtype": (WEIGHT_DTYPES, {"default": "default", "tooltip": "The weight dtype to use when loading the diffusion model, this is only useful if the model is a diffusion model"}),
                "context": (["image", "video", "audio", "3d", "text"], {"default": "image", "tooltip": "The context to use for the model", "multiline": True}),
                "name": ("STRING", {"default": "", "tooltip": "The name to use for the model in a human readable format, if not given the model filename will be used"}),
                "description": ("STRING", {"default": "", "tooltip": "The description to use for the model, this is shown in the UI", "multiline": True}),
                "family": ("STRING", {"default": "sdxl", "tooltip": "The family to use for the model, this is used for groupping models in the UI"}),
                "group": ("STRING", {"default": "my_checkpoint_name", "tooltip": "The group to use for the model, this is used for groupping models in the UI"}),
                "vae": ([""] + get_filename_list("vae"), {"default": "", "tooltip": "The VAE to use for the model, if not given the VAE from the checkpoint will be used"}),
                "clip_1": ([""] + get_filename_list("text_encoders"), {"default": "", "tooltip": "The CLIP to use for the model, if not given the CLIP from the checkpoint will be used"}),
                "clip_2": ([""] + get_filename_list("text_encoders"), {"default": "", "tooltip": "The second CLIP to use for the model, if not given a single CLIP will be used"}),
                "clip_type": (CLIP_TYPES, {"default": "stable_diffusion", "tooltip": "The type of the CLIP to use"}),
                "default_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "tooltip": "The default CFG scale to use for the model"}),
                "default_steps": ("INT", {"default": 20, "min": 1, "max": 100, "tooltip": "The default number of steps to use for the model"}),
                "default_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", "tooltip": "The default scheduler to use for the model"}),
                "default_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_sde", "tooltip": "The default sampler to use for the model"})
            },
            "optional": {
                "optional_image": ("IMAGE", {"default": "", "tooltip": "An optional image to use for the model"})
            }
        }
    
    def export_model(self, model, weight_dtype, context, name, description, family, group, vae, clip_1, clip_2, clip_type, default_cfg, default_steps, default_scheduler, default_sampler, optional_image=None):
        if not model or model.strip() == "":
            raise ValueError("Error: No model specified")
        if not family or family.strip() == "":
            raise ValueError("Error: No family specified")
        
        id = os.path.splitext(os.path.basename(model))[0]
        if not name or name.strip() == "":
            name = id

        is_diffusion_model = model in get_filename_list("diffusion_models")
        
        model_JSON = {
            "id": id,
            "name": name,
            "file": model,
            "family": family,
            "description": description,
            "context": context,
            "is_diffusion_model": is_diffusion_model,
            "default_cfg": default_cfg,
            "default_steps": default_steps,
            "default_scheduler": default_scheduler,
            "default_sampler": default_sampler
        }

        if is_diffusion_model:
            model_JSON["diffusion_model_weight_dtype"] = weight_dtype

        if group and group.strip() != "":
            model_JSON["group"] = group

        if vae and vae.strip() != "":
            model_JSON["vae_file"] = vae

        if clip_1 and clip_1.strip() != "" and (not clip_2 or clip_2.strip() == ""):
            model_JSON["clip_file"] = clip_1
            model_JSON["clip_type"] = clip_type
        elif clip_2 and clip_2.strip() != "" and (not clip_1 or clip_1.strip() == ""):
            model_JSON["clip_file"] = clip_2
            model_JSON["clip_type"] = clip_type
        elif clip_2 and clip_2.strip() != "" and clip_1 and clip_1.strip() != "":
            model_JSON["clip_file"] = clip_1 + "," + clip_2
            model_JSON["clip_type"] = clip_type

        image_filename = id + ".png"

        if optional_image is not None:
            # save the image to the models folder with the name of the model id
            c_image = optional_image[0]
            i = 255. * c_image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(os.path.join(AIHUB_MODELS_DIR, image_filename), format='PNG')

        json_filename = id + ".json"
        with open(os.path.join(AIHUB_MODELS_DIR, json_filename), "w", encoding="utf-8") as f:
            json.dump(model_JSON, f, indent=4)

        with open(os.path.join(AIHUB_MODELS_LOCALE_DIR, "default", json_filename), "w", encoding="utf-8") as f:
            json.dump({"name": name, "description": description}, f, indent=4)

        return ()
    
class AIHubMetaExportLora:
    """
    Meta node utility to export a lora to its json file so it can be used by
    the client side UI
    """
    CATEGORY = "aihub/meta"
    FUNCTION = "export_lora"

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": (get_filename_list("loras"), {"default": "", "tooltip": "The lora to export"}),
                "name": ("STRING", {"default": "", "tooltip": "The name to use for the lora in a human readable format, if not given the lora filename will be used"}),
                "description": ("STRING", {"default": "", "tooltip": "The description to use for the lora, this is shown in the UI", "multiline": True}),
                "context": (["image", "video", "audio", "3d", "text"], {"default": "image", "tooltip": "The context to use for the lora"}),
                "limit_to_family": ("STRING", {"default": "sdxl", "tooltip": "The lora can only apply to models tagged with this family"}),
                "limit_to_group": ("STRING", {"default": "", "tooltip": "The lora can only apply to models tagged with this group"}),
                "limit_to_model": ([""] + get_filename_list("checkpoints") + get_filename_list("diffusion_models"), {"default": "", "tooltip": "The lora can only apply to this specific model"}),
                "default_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "tooltip": "The default strength to use for the lora"}),
                "use_loader_model_only": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will only apply the lora to the model and not to the clip"}),
            },
            "optional": {
                "optional_image": ("IMAGE", {"default": "", "tooltip": "An optional image to use for the lora"})
            }
        }
    
    def export_lora(self, lora, name, description, context, limit_to_family, limit_to_group, limit_to_model, default_strength, use_loader_model_only, optional_image=None):
        if not lora or lora.strip() == "":
            raise ValueError("Error: No lora specified")
        
        id = os.path.splitext(os.path.basename(lora))[0]
        if not name or name.strip() == "":
            name = id

        lora_JSON = {
            "id": id,
            "file": lora,
            "context": context,
            "description": description,
            "default_strength": default_strength,
            "use_loader_model_only": use_loader_model_only
        }

        if limit_to_family and limit_to_family.strip() != "":
            lora_JSON["limit_to_family"] = limit_to_family
        
        if limit_to_group and limit_to_group.strip() != "":
            if not limit_to_family or limit_to_family.strip() == "":
                raise ValueError("Error: limit_to_family must be set if limit_to_group is set")
            lora_JSON["limit_to_group"] = limit_to_group

        if limit_to_model and limit_to_model.strip() != "":
            model_id = os.path.splitext(os.path.basename(limit_to_model))[0]
            lora_JSON["limit_to_model"] = model_id

        image_filename = id + ".png"
        if optional_image is not None:
            # save the image to the models folder with the name of the model id
            c_image = optional_image[0]
            i = 255. * c_image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(os.path.join(AIHUB_LORAS_DIR, image_filename), format='PNG')

        json_filename = id + ".json"
        with open(os.path.join(AIHUB_LORAS_DIR, json_filename), "w", encoding="utf-8") as f:
            json.dump(lora_JSON, f, indent=4)

        with open(os.path.join(AIHUB_LORAS_LOCALE_DIR, "default", json_filename), "w", encoding="utf-8") as f:
            json.dump({"name": name, "description": description}, f, indent=4)

        return ()
    
class AIHubMetaSetExportedLoraImage:
    """
    Meta node utility to set the image of an exported lora
    """
    CATEGORY = "aihub/meta"
    FUNCTION = "set_exported_lora_image"

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": (get_filename_list_for_aihub_folder(AIHUB_LORAS_DIR), {"default": "", "tooltip": "The id of the lora to set the image for"}),
                "image": ("IMAGE", {"tooltip": "The image to set for the lora"}),
            }
        }

    def set_exported_lora_image(self, lora, image):
        if image is None or len(image) == 0:
            raise ValueError("Error: No image specified")

        image_filename = lora + ".png"

        if image is not None:
            # save the image to the lora folder with the name of the lora id
            c_image = image[0]
            i = 255. * c_image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(os.path.join(AIHUB_LORAS_DIR, image_filename), format='PNG')

        return ()
    
class AIHubMetaSetExportedWorkflowImage:
    """
    Meta node utility to set the image of an exported workflow
    """
    CATEGORY = "aihub/meta"
    FUNCTION = "set_exported_workflow_image"

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "workflow": (get_filename_list_for_aihub_folder(AIHUB_WORKFLOWS_DIR), {"default": "", "tooltip": "The id of the workflow to set the image for"}),
                "image": ("IMAGE", {"tooltip": "The image to set for the workflow"}),
            }
        }

    def set_exported_workflow_image(self, workflow, image):
        if image is None or len(image) == 0:
            raise ValueError("Error: No image specified")

        image_filename = workflow + ".png"

        if image is not None:
            # save the image to the workflows folder with the name of the workflow id
            c_image = image[0]
            i = 255. * c_image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(os.path.join(AIHUB_WORKFLOWS_DIR, image_filename), format='PNG')

        return ()