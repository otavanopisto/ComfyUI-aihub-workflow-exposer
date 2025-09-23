import io
import os
from nodes import LoadImage, CheckpointLoaderSimple, LoraLoader, UNETLoader, LoraLoaderModelOnly, VAELoader, CLIPLoader, DualCLIPLoader
import json
import torch
import random
import comfy.samplers

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
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to expose, use dots for entering sublevels"}),
                "value": ("INT", {"default": 0, "tooltip": "The value of the field"}),
                "default": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "tooltip": "The default value of the field"}),
            }
        }

    def get_exposed_int(self, field, value, default):
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
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to expose, use dots for entering sublevels"}),
                "value": ("FLOAT", {"default": 0.0, "tooltip": "The value of the field"}),
                "default": ("FLOAT", {"default": 0.0, "min": -1.0e+20, "max": 1.0e+20, "tooltip": "The default value of the field"}),
            }
        }

    def get_exposed_float(self, field, value, default):
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
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to expose, use dots for entering sublevels"}),
                "value": ("BOOLEAN", {"default": True, "tooltip": "The value of the field"}),
                "default": ("BOOLEAN", {"default": True, "tooltip": "The default value of the field"}),
            }
        }

    def get_exposed_boolean(self, field, value, default):
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
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to expose, use dots for entering sublevels"}),
                "value": ("STRING", {"default": "my-value", "tooltip": "The value of the field"}),
                "default": ("STRING", {"default": "", "tooltip": "The default value of the field"}),
            }
        }

    def get_exposed_string(self, field, value, default):
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
                    "merged_image",
                    "merged_image_without_current_layer",
                    "upload",
                ], {"default": "upload", "tooltip": "The source of the image"}),
                "pos_x": ("INT", {"default": 0, "tooltip": "The X position of the layer in the canvas"}),
                "pos_y": ("INT", {"default": 0, "tooltip": "The Y position of the layer in the canvas"}),
                "layer_id": ("STRING", {"default": "", "tooltip": "The ID of the layer to use, only given if type is current_layer or previous_layer"}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            },
            "optional": {
                "value": ("IMAGE",),
                "value_mask": ("MASK",),
                "local_file": ("STRING",),
            },
        }

    def get_exposed_image(self, id, label, tooltip, type, pos_x, pos_y, layer_id, index, value=None, value_mask=None, local_file=None):
        image = None
        mask = None
        if value is not None:
            image = value
            mask = value_mask
        elif local_file is not None:
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
            raise ValueError("You must specify either local_file or the value of the image for this node to function")

        _, height, width, _ = image.shape
        
        return (image, mask, pos_x, pos_y, layer_id, width, height,)
    
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
                "type": (["current_layer", "merged_image", "merged_image_without_current_layer", "upload"], {"default": "upload", "tooltip": "The source of the image"}),
                "pos_x": ("INT", {"default": 0, "tooltip": "The X position of the layer in the canvas"}),
                "pos_y": ("INT", {"default": 0, "tooltip": "The Y position of the layer in the canvas"}),
                "layer_id": ("INT", {"default": 0, "tooltip": "The Z (Layer) position of the image."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            },
            "optional": {
                "value": ("IMAGE",),
                "value_width": ("INT", {"tooltip": "width of the image."}),
                "value_height": ("INT", {"tooltip": "height of the image."}),
            }
        }

    def get_exposed_image_info_only(self, id, label, tooltip, type, pos_x, pos_y, layer_id, index, value=None, value_width=1024, value_height=1024):
        image = None
        height = value_height
        width = value_width
        if value is not None:
            image = value
            _, height, width, _ = image.shape
        
        return (pos_x, pos_y, layer_id, width, height,)
    
class AIHubExposeImageBatch:
    """
    An utility to expose an image batch to be used in the workflow
    """
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_image_batch"
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_image_batch", "tooltip": "A unique custom ID for this workflow."}),
                "label": ("STRING", {"default": "Image Batch", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "type": (["new_images", "all_reference_frames", "new_reference_frames"]),
                "minlen": ("INT", {"default": 0}),
                "maxlen": ("INT", {"default": 1000}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            },
            "optional": {
                "values": ("IMAGE",),
                "masks": ("MASK",),
                "local_files": ("STRING", {"default": "[]"}),
            }
        }

    def get_exposed_image_batch(self, id, label, tooltip, type, maxlen, index, values=None, values_masks=None, local_files=None):
        image_batch = None
        masks = None
        if values is not None:
            # If an image batch is passed from a connected node, use it directly.
            image_batch = values
            masks = values_masks
            if image_batch.shape[0] > maxlen:
                raise ValueError(f"Error: {id} contains too many files")
            if masks is not None and masks.shape[0] != image_batch.shape[0]:
                raise ValueError(f"Error: {id} the number of masks does not match the number of images")
            
        elif local_files:
            # If a local_files string is provided, attempt to load the images.
            try:
                # Parse the JSON string into a Python list of filenames.
                filenames = json.loads(local_files)
                if not isinstance(filenames, list):
                    raise ValueError("Error: local_files is not a valid JSON string encoding an array")
                if len(filenames) > maxlen:
                    raise ValueError(f"Error: {id} contains too many files")
                
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
                
                # Concatenate all the images into a single batch tensor.
                image_batch = torch.cat(loaded_images, dim=0)
                masks = torch.cat(loaded_masks, dim=0)
            except json.JSONDecodeError:
                # Handle invalid JSON input gracefully.
                raise ValueError("Error: local_files is not a valid JSON string encoding an array")
        else:
            # Return an empty placeholder if no input is provided.
            raise ValueError("You must specify either local_files or the values of the images for this node to function")
        
        return (image_batch, masks,)
    
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
                "diffusion_model_weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "tooltip": "The weight dtype to use when loading the diffusion model, this is only used if is_diffusion_model is true"}),
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
                          disable_loras_selection, disable_checkpoint_selection, optional_vae=None, optional_clip=None, optional_clip_type=None):
        # first lets load the checkpoint using the comfy CheckpointLoaderSimple
        model_loaded = None
        clip = None
        vae = None
        if model:
            if is_diffusion_model:
                print("Using UNETLoader to load the diffusion model " + str(model) + " with weight dtype " + str(diffusion_model_weight_dtype))
                loader = UNETLoader()
                model_loaded, = loader.load_unet(model, diffusion_model_weight_dtype)
            else:
                print("Using CheckpointLoaderSimple to load the model " + str(model))
                loader = CheckpointLoaderSimple()
                model_loaded, clip, vae = loader.load_checkpoint(model)
            
            if model_loaded is None:
                raise ValueError(f"Error: Could not load the model checkpoint: {model}")
            
            # now we have to apply the loras if given
            if loras:
                lora_list = [l.strip() for l in loras.split(",") if l.strip()]
                strengths_list = [float(s.strip()) for s in loras_strengths.split(",") if s.strip()]
                use_loader_model_only_list = [s.strip() == "t" for s in loras_use_loader_model_only.split(",")]
                if len(strengths_list) != len(lora_list):
                    raise ValueError("Error: The number of lora strengths must match the number of loras")
                if len(use_loader_model_only_list) != len(lora_list):
                    raise ValueError("Error: The number of lora_use_loader_model_only values must match the number of loras")
                for lora, strength, use_loader_model_only in zip(lora_list, strengths_list, use_loader_model_only_list):
                    if not use_loader_model_only:
                        print(f"Applying lora {lora} with strength {strength} to model and clip")
                        lora_loader = LoraLoader()
                        model_loaded, clip = lora_loader.load_lora(model_loaded, clip, lora, strength, strength)
                    else:
                        print(f"Applying lora {lora} with strength {strength} to model only")
                        lora_loader = LoraLoaderModelOnly()
                        model_loaded, = lora_loader.load_lora_model_only(model_loaded, lora, strength)
        else:
            print("No model specified so it cannot be loaded")

        if optional_vae is not None and optional_vae.strip() != "":
            print("Using VAELoader to load the VAE " + str(optional_vae))
            vae_loader = VAELoader()
            (vae,) = vae_loader.load_vae(optional_vae)

        if optional_clip is not None and optional_clip.strip() != "":
            if optional_clip_type is None or optional_clip_type.strip() == "":
                raise ValueError("Error: If optional_clip is given, optional_clip_type must be given as well")
            # now we need to comma separate to check if we have multiple clips to merge, we only take the first two in case
            # of multiple because that is the default to use DualClip
            if "," in optional_clip:
                clips = [c.strip() for c in optional_clip.split(",") if c.strip()]
                clips = clips[:2]
                print(f"Using DualCLIPLoader to load the CLIPs {clips[0]} and {clips[1]} with type {optional_clip_type}")
                clip_loader = DualCLIPLoader()
                (clip,) = clip_loader.load_clip(clips[0], clips[1], optional_clip_type)
            else:
                print(type(optional_clip))
                print(f"Using CLIPLoader to load the CLIP {optional_clip} with type {optional_clip_type}")
                clip_loader = CLIPLoader()
                (clip,) = clip_loader.load_clip(optional_clip, optional_clip_type)

        return (model_loaded, clip, vae,)

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
                "name": ("STRING", {"default": "new image", "tooltip": "The name of the image this is used for the filename value"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    def run_action(self, image, action, name, mask=None):
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
                "name": name,
            },
        )

        return (image, mask)
    
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
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    def run_action(self, image, pos_x, pos_y, reference_layer_id, reference_layer_action, name, mask=None):
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
            },
        )

        return (image, mask)
    
# Actions that are considered patches
# running these actions should create a patch to the project
class AIHubPatchActionSetProjectConfigInteger:
    """
    A utility node for setting an integer from the project config.json
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
                "value": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "tooltip": "The value of the field to set"}),
            },
        }

    def run_action(self, field, value):
        #server = PromptServer.instance
        #server.send_sync(
        #    "AIHUB_PATCH_ACTION_SET_CONFIG_INTEGER",
        #    {
        #        "field": field,
        #        "value": value,
        #    },
        #    server.client_id,
        #)

        return ()
    
class AIHubPatchActionSetProjectConfigFloat:
    """
    A utility node for setting an float from the project config.json
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
                "value": ("FLOAT", {"default": 0.0, "min": -1.0e+20, "max": 1.0e+20, "tooltip": "The value of the field to set"}),
            }
        }

    def run_action(self, field, value):
        #server = PromptServer.instance
        #server.send_sync(
        #    "AIHUB_PATCH_ACTION_SET_CONFIG_FLOAT",
        #    {
        #        "field": field,
        #        "value": value,
        #    },
        #    server.client_id,
        #)

        return ()
    
class AIHubPatchActionSetProjectConfigBoolean:
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
        #server = PromptServer.instance
        #server.send_sync(
        #    "AIHUB_PATCH_ACTION_SET_CONFIG_BOOLEAN",
        #    {
        #        "field": field,
        #        "value": value,
        #    },
        #    server.client_id,
        #)

        return ()
    
class AIHubPatchActionSetProjectConfigString:
    """
    A utility node for setting a string from the project config.json
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
                "value": ("STRING", {"default": "", "tooltip": "The value of the field to set"}),
            }
        }

    def run_action(self, field, value):
        #server = PromptServer.instance
        #server.send_sync(
        #    "AIHUB_PATCH_ACTION_SET_CONFIG_STRING",
        #    {
        #        "field": field,
        #        "value": value,
        #    },
        #    server.client_id,
        #)

        return ()
    
class AIHubPatchActionSetFile:
    pass

class AIHubPatchActionSetImage:
    pass

class AIHubPatchActionSetVideo:
    pass

class AIHubPatchActionSetAudio:
    pass

class AIHubPatchActionSetPreviewImage:
    pass

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
                "layer_image": ("IMAGE", {"optional": True, "tooltip": "The image of the layer"}),
            },
            "optional": {
                "layer_mask": ("MASK", {"optional": True, "tooltip": "The mask of the layer"}),
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
    
class AIHubUtilsStrToInt:
    """
    A utility node for converting a string to an integer
    """

    CATEGORY = "aihub/utils"
    FUNCTION = "str_to_int"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"default": "0", "tooltip": "The string to convert to an integer"}),
            }
        }

    def str_to_int(self, value):
        try:
            int_value = int(value)
        except ValueError:
            raise ValueError(f"Error: Could not convert string to integer: {value}")
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