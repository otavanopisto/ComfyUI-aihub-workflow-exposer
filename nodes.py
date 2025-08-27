import os
from nodes import LoadImage
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
print("AIHub Server Started ===========================================================================================================================================================================")

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
                "context": (["image", "video", "audio"], {"tooltip": "The context of this workflow", "default": "image"}),
                "project_type": ("STRING", {"tooltip": "The project type is an arbitrary string that limits in which context the workflow can be used"}),
                "project_type_init": ("BOOLEAN", {"tooltip": "This workflow can initialize the given project type, provided an empty project folder"})
                # eg. project types I think about implementing, ltxv, sdxl-character
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "register"
    CATEGORY = "aihub/workflow"

    def register():
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
                "min": ("INT", {"default": 0}),
                "max": ("INT", {"default": 100}),
                "step": ("INT", {"default": 1}),
                "start": ("INT", {"default": 0}),
                "value": ("INT", {"default": 10}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will make this option be hidden under advanced options for this workflow"}),
                "index": ("INT", {"default": 0, "tooltip": "this value is used for sorting the input fields when displaying, lower values will appear first"}),
            }
        }

    def get_exposed_int(self, label, tooltip, min, max, step, start, value, description, advanced, index):
        if (value < min):
            raise ValueError(f"Error: {id} should be greater or equal to {min}")
        if (value > max):
            raise ValueError(f"Error: {id} should be less or equal to {max}")
        return (value,)
    
class AIHubExposeConfigInteger:
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
                "default": ("INT", {"default": 0, "tooltip": "The default value of the field"}),
            }
        }

    def get_exposed_int(self, field, value):
        return (value,)
    
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
                "min": ("FLOAT", {"default": 0.0}),
                "max": ("FLOAT", {"default": 1.0}),
                "step": ("FLOAT", {"default": 0.01}),
                "start": ("FLOAT", {"default": 0.0}),
                "value": ("FLOAT", {"default": 0.5}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, it will make this option be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            }
        }

    def get_exposed_float(self, id, label, tooltip, min, max, step, start, value, advanced, index):
        if (value < min):
            raise ValueError(f"Error: {id} should be greater or equal to {min}")
        if (value > max):
            raise ValueError(f"Error: {id} should be less or equal to {max}")
        return (value,)
    
class AIHubExposeConfigFloat:
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
                "default": ("FLOAT", {"default": 0.0, "tooltip": "The default value of the field"}),
            }
        }

    def get_exposed_float(self, field, value):
        return (value,)
    
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
    
class AIHubExposeConfigBoolean:
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

    def get_exposed_boolean(self, field, value):
        return (value,)
    
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
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this option will be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            }
        }

    def get_exposed_string(self, id, label, tooltip, minlen, maxlen, value, advanced, index):
        if (len(value) < minlen):
            raise ValueError(f"Error: {label} length should be greater or equal to {minlen}")
        if (len(value) > maxlen):
            raise ValueError(f"Error: {label} length should be less or equal to {maxlen}")
        return (value,)
    
class AIHubExposeConfigString:
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

    def get_exposed_string(self, field, value):
        return (value,)

class AIHubExposeStringSelection:
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
                "value": (["random", "fixed"], {"default": "random", "tooltip": "Choose whether to use a random or a fixed seed."}),
                "value_fixed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this option will be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            }
        }

    def get_exposed_seed(self, id, label, tooltip, value, value_fixed, advanced, index):
        if value == "random":
            return (random.randint(0, 0xffffffffffffffff),)
        else:
            return (value_fixed,)
        
class AIHubExposeSampler:
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
            }
        }

    def get_exposed_sampler(self, id, label, tooltip, value, value_fixed, advanced, index):
        return value
    
class AIHubExposeScheduler:
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_sampler"
    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("SCHEDULER",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "scheduler", "tooltip": "A unique custom id for this workflow."}),
                "label": ("STRING", {"default": "Scheduler", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "value": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Choose the scheduler to use"}),
                "advanced": ("BOOLEAN", {"default": False, "tooltip": "If set to true, this option will be hidden under advanced options for this workflow."}),
                "index": ("INT", {"default": 0, "tooltip": "This value is used for sorting the input fields when displaying; lower values will appear first."}),
            }
        }

    def get_exposed_sampler(self, id, label, tooltip, value, value_fixed, advanced, index):
        return value

class AIHubExposeImage:
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_image"
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "POS_X", "POS_Y", "POS_Z", "WIDTH", "HEIGHT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_image", "tooltip": "A unique custom ID for this workflow."}),
                "label": ("STRING", {"default": "Image", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "type": (["current_layer", "merged_image_below_current_layer", "merged_image_until_current_layer", "merged_image", "previous_layer", "upload"], {"default": "upload", "tooltip": "The source of the image"}),
                "value_pos_x": ("INT", {"default": 0, "tooltip": "The X position of the layer in the canvas"}),
                "value_pos_y": ("INT", {"default": 0, "tooltip": "The Y position of the layer in the canvas"}),
                "value_pos_z": ("INT", {"default": 0, "tooltip": "The Z (Layer) position of the image."}),
            },
            "optional": {
                "value": ("IMAGE",),
                "local_file": ("STRING",),
            },
        }

    def get_exposed_image(self, id, label, tooltip, type, value_pos_x, value_pos_y, value_pos_z, value=None, local_filename=None):
        image = None
        if value is not None:
            image = value
        elif local_filename is not None and os.path.exists(local_filename):
            # Instantiate a LoadImage node and use its logic to load the file
            loader = LoadImage()
            # The load_image method returns a tuple, so we need to get the first element
            loaded_image_tuple = loader.load_image(local_filename)
            image = loaded_image_tuple[0]
        else:
            raise ValueError("You must specify either local_filename (hidden) or the value of the image for this node to function")

        _, height, width, _ = image.shape
        
        return (image, value_pos_x, value_pos_y, value_pos_z, width, height,)
    
class AIHubExposeImageInfoOnly:
    CATEGORY = "aihub/expose"
    FUNCTION = "get_exposed_image_info_only"
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("POS_X", "POS_Y", "POS_Z", "WIDTH", "HEIGHT",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("STRING", {"default": "exposed_image", "tooltip": "A unique custom ID for this workflow."}),
                "label": ("STRING", {"default": "Image", "tooltip": "This is the label that will appear in the field."}),
                "tooltip": ("STRING", {"default": "", "tooltip": "An optional tooltip"}),
                "type": (["current_layer", "merged_image_below_current_layer", "merged_image_until_current_layer", "merged_image", "previous_layer", "upload"], {"default": "upload", "tooltip": "The source of the image"}),
                "value_pos_x": ("INT", {"default": 0, "tooltip": "The X position of the layer in the canvas"}),
                "value_pos_y": ("INT", {"default": 0, "tooltip": "The Y position of the layer in the canvas"}),
                "value_pos_z": ("INT", {"default": 0, "tooltip": "The Z (Layer) position of the image."}),
            },
            "optional": {
                "value": ("IMAGE",),
                "value_width": ("INT", {"tooltip": "width of the image."}),
                "value_height": ("INT", {"tooltip": "height of the image."}),
            }
        }

    def get_exposed_image_info_only(self, id, label, tooltip, type, value_pos_x, value_pos_y, value_pos_z, value=None, value_width=1024, value_height=1024):
        image = None
        height = value_height
        width = value_width
        if value is not None:
            image = value
            _, height, width, _ = image.shape
        
        return (value_pos_x, value_pos_y, value_pos_z, width, height,)
    
class AIHubExposeImageBatch:
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
                "maxlen": ("INT", {"default": 1000}),
            },
            "optional": {
                "values": ("IMAGE",),
                "local_filenames": ("STRING", {"default": "[]"}),
            }
        }

    def get_exposed_image_batch(self, id, label, tooltip, type, maxlen, values=None, local_filenames=None):
        image_batch = None
        if values is not None:
            # If an image batch is passed from a connected node, use it directly.
            image_batch = values
            if image_batch.shape[0] > maxlen:
                raise ValueError(f"Error: {id} contains too many files")
            
        elif local_filenames:
            # If a local_filenames string is provided, attempt to load the images.
            try:
                # Parse the JSON string into a Python list of filenames.
                filenames = json.loads(local_filenames)
                if not isinstance(filenames, list):
                    raise ValueError("Error: local_filenames is not a valid JSON string encoding an array")
                if len(filenames) > maxlen:
                    raise ValueError(f"Error: {id} contains too many files")
                
                loaded_images = []
                loader = LoadImage()
                
                for filename in filenames:
                    # Check if the file exists before trying to load it.
                    if os.path.exists(filename):
                        # Load each image and append it to the list.
                        loaded_images.append(loader.load_image(filename)[0])
                    else:
                        raise ValueError(f"Error: Image file not found: {filename}")
                
                # Concatenate all the images into a single batch tensor.
                image_batch = torch.cat(loaded_images, dim=0)
            except json.JSONDecodeError:
                # Handle invalid JSON input gracefully.
                raise ValueError("Error: local_filenames is not a valid JSON string encoding an array")
        else:
            # Return an empty placeholder if no input is provided.
            raise ValueError("You must specify either local_filenames (hidden) or the values of the images for this node to function")
        
        return (image_batch,)

## Actions
class AIHubActionNewImage:
    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"
    
    RETURN_TYPES = ("IMAGE", )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    def run_action(image):
        image = image[0]
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        SERVER.send_binary_data_to_current_client(
            img, "image/png", {
                "action": "NEW_IMAGE",
                "width": image.shape[3],
                "height": image.shape[2],
                "type": "image/png",
            },
        )

        return (image,)
    
class AIHubActionNewLayer:
    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"
    
    RETURN_TYPES = ("IMAGE", )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pos_x": ("INT", {"default": 0, "tooltip": "The X position of the layer in the canvas"}),
                "pos_y": ("INT", {"default": 0, "tooltip": "The Y position of the layer in the canvas"}),
                "pos_z": ("INT", {"default": 0, "tooltip": "The Z (Layer) position of the image."}),
                "replace": ("BOOLEAN", {"default": False, "tooltip": "If true, it will replace the layer at the given Z position, otherwise it will insert a new layer at that position"}),
            }
        }
    
    def run_action(image, pos_x, pos_y, pos_z, replace):
        image = image[0]
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        SERVER.send_binary_data_to_current_client(
            img, "image/png",{
                "action": "NEW_LAYER",
                "width": image.shape[3],
                "height": image.shape[2],
                "type": "image/png",
                "pos_x": pos_x,
                "pos_y": pos_y,
                "pos_z": pos_z,
                "replace": replace,
            },
        )

        return (image,)
    
class AIHubActionSetProgressStatus:
    """
    A utility node for setting the progress status for the progressbar for the client
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*", {"forceInput": True}),
                "status": ("STRING", {"default": "", "tooltip": "The status to specify"}),
            }
        }

    def run_action(self, value):
        #server = PromptServer.instance
        #server.send_sync(
        #    "AIHUB_ACTION_SET_PROGRESS_STATUS",
        #    {
        #        "status": value,
        #    },
        #    server.client_id,
        #)

        return (value)
    
# Actions that are considered patches
# running these actions should create a patch to the project
class AIHubPatchActionSetConfigInteger:
    """
    A utility node for setting an integer from the project config.json
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"

    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to set, use dots for entering sublevels"}),
                "value": ("INT", {"default": 0, "tooltip": "The value of the field to set"}),
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
    
class AIHubPatchActionSetConfigFloat:
    """
    A utility node for setting an float from the project config.json
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"

    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "field": ("STRING", {"default": "my-field", "tooltip": "The field to set, use dots for entering sublevels"}),
                "value": ("FLOAT", {"default": 0.0, "tooltip": "The value of the field to set"}),
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
    
class AIHubPatchActionSetConfigBoolean:
    """
    A utility node for setting a boolean from the project config.json
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"

    RETURN_TYPES = ()

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
    
class AIHubPatchActionSetConfigString:
    """
    A utility node for setting a string from the project config.json
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"

    RETURN_TYPES = ()

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

class AIHubDevWebsocketDebug:
    """
    A developer tool in order to test websocket information from within comfyui, you should change the seed in order
    to retrieve new information
    """

    CATEGORY = "aihub/actions"
    FUNCTION = "run_action"

    RETURN_TYPES = ("*", )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "retrieve": (["LAST_PING_QUEUE_VALUE"], {"tooltip": "The element to retrieve"}),
                "seed": ("STRING", {"default": "42", "tooltip": "A random seed in order to trigger "}),
            }
        }

    def run_action(self, seed):
        return (SERVER.LAST_PING_QUEUE_VALUE, )