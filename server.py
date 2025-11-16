import asyncio
import threading
import json
import uuid
from os import path, listdir, environ, makedirs
from copy import deepcopy
import tempfile
from time import sleep
import re
from functools import partial
from nodes import interrupt_processing
import comfy.samplers
from shutil import rmtree
import threading
from folder_paths import get_filename_list

from aiohttp import web
from server import PromptServer
from execution import validate_prompt, SENSITIVE_EXTRA_DATA_KEYS

from .aihub_env import AIHUB_DIR, AIHUB_LORAS_LOCALE_DIR, AIHUB_MODELS_DIR, AIHUB_LORAS_DIR, AIHUB_MODELS_LOCALE_DIR, AIHUB_WORKFLOWS_DIR, AIHUB_WORKFLOWS_LOCALE_DIR

WEB_SOCKET_SERVER_PORT = 8000
SERVER_THREAD = None
SERVER_RUNNING_FLAG = threading.Event()

AIHUB_COLD = environ.get("AIHUB_COLD", "0") == "1"
AIHUB_PERSIST_TEMPFILES = environ.get("AIHUB_PERSIST_TEMPFILES", "0") == "1"

WORKFLOWS_CACHE_RAW = None
WORKFLOWS_CACHE_VALID = None
WORKFLOWS_AIHUB_SUMMARY = {}

LORAS_CACHE_RAW = None
LORAS_CACHE_CLEANED = {}

MODELS_CACHE_RAW = None
MODELS_CACHE_CLEANED = {}

AIHUB_TEMP_DIRECTORY_ENV = environ.get("AIHUB_TEMP_DIR", None)
AIHUB_TEMP_DIRECTORY = AIHUB_TEMP_DIRECTORY_ENV if AIHUB_TEMP_DIRECTORY_ENV is not None else tempfile.gettempdir()

AIHUB_MAX_MESSAGE_SIZE = int(environ.get("AIHUB_MAX_MESSAGE_SIZE", 50 * 1024 * 1024))  # 50 MB

@PromptServer.instance.routes.post("/aihub_workflows")
async def handle_workflow_add(request):
    # get the json data from the request
    try:
        data = await request.json()
    except Exception as e:
        return web.json_response({"error": "Invalid JSON data"}, status=400)
    
    workflow_id = None

    try:
        for node in data.values():
            if node.get("class_type", None) == "AIHubWorkflowController":
                workflow_id = node.get("inputs", {}).get("id", None)
                break
    except Exception as e:
        return web.json_response({"error": "Invalid workflow data"}, status=400)
    
    # save the json data to the aihub directory
    with open(path.join(AIHUB_WORKFLOWS_DIR, f"{workflow_id}.json"), "w") as f:
        json.dump(data, f)

    return web.json_response({"status": "ok"})

@PromptServer.instance.routes.post("/aihub_workflows/{workflow_id}/locale/{locale}")
async def handle_workflow_locale_add(request):
    # get the json data from the request
    try:
        data = await request.json()
    except Exception as e:
        return web.json_response({"error": "Invalid JSON data"}, status=400)
    
    workflow_id = request.match_info.get("workflow_id", None)
    locale = request.match_info.get("locale", None)
    
    # save the json data to the aihub directory
    locale_folder = path.join(AIHUB_WORKFLOWS_LOCALE_DIR, locale)
    if not path.exists(locale_folder):
        makedirs(locale_folder)
    with open(path.join(locale_folder, f"{workflow_id}.json"), "w") as f:
        json.dump(data, f)

    return web.json_response({"status": "ok"})

@PromptServer.instance.routes.post("/aihub_workflows/{workflow_id}/image")
async def handle_workflow_image_add(request):
    # for this we just need to get the binary data from the request
    # get the json data from the request
    try:
        data = await request.read()
    except Exception as e:
        return web.json_response({"error": "Invalid image data"}, status=400)
    
    workflow_id = request.match_info.get("workflow_id", None)

    if workflow_id is None or workflow_id.strip() == "" or ".." in workflow_id or "/" in workflow_id or "\\" in workflow_id:
        return web.json_response({"error": "Invalid workflow id"}, status=400)

    # save the image data to the aihub directory
    with open(path.join(AIHUB_WORKFLOWS_DIR, f"{workflow_id}.png"), "wb") as f:
        f.write(data)

    return web.json_response({"status": "ok"})

@PromptServer.instance.routes.get("/aihub_list_models_and_loras")
async def handle_list_models(request):
    return web.json_response({"checkpoints": get_filename_list("checkpoints"), "diffusion_models": get_filename_list("diffusion_models"),
                               "loras": get_filename_list("loras")})

class AIHubServer:
    """
    the server class handles the websocket server and the workflow queue
    file uploads and downloads, as well as the workflow validation and processing
    """
    CURRENTLY_RUNNING = None
    WORKFLOW_REQUEST_QUEUE = []
    QUEUE_LOCK = threading.Lock()

    LAST_PING_QUEUE_VALUE = None

    awaiting_tasks_amount = 0
    awaiting_tasks_done_flag = None
    awaiting_tasks_lock = threading.Lock()

    loop = None

    def __init__(self):
        # create folder ../../aihub if it doesnt't exist
        if not path.exists(AIHUB_DIR):
            makedirs(AIHUB_DIR)

        # create folder ../../aihub/workflows if it doesn't exist
        if not path.exists(AIHUB_WORKFLOWS_DIR):
            makedirs(AIHUB_WORKFLOWS_DIR)

        # create folder ../../aihub/loras if it doesn't exist
        if not path.exists(AIHUB_LORAS_DIR):
            makedirs(AIHUB_LORAS_DIR)

        # create folder ../../aihub/models if it doesn't exist
        if not path.exists(AIHUB_MODELS_DIR):
            makedirs(AIHUB_MODELS_DIR)

        # create folder ../../aihub/workflows if it doesn't exist
        if not path.exists(AIHUB_WORKFLOWS_LOCALE_DIR):
            makedirs(AIHUB_WORKFLOWS_LOCALE_DIR)

        # create folder ../../aihub/loras if it doesn't exist
        if not path.exists(AIHUB_LORAS_LOCALE_DIR):
            makedirs(AIHUB_LORAS_LOCALE_DIR)

        # create folder ../../aihub/models if it doesn't exist
        if not path.exists(AIHUB_MODELS_LOCALE_DIR):
            makedirs(AIHUB_MODELS_LOCALE_DIR)

        originalQueueUpdatedFn = PromptServer.instance.queue_updated
        originalSendSyncFn = PromptServer.instance.send_sync
        PromptServer.instance.queue_updated = partial(self.queue_updated_override, originalQueueUpdatedFn)
        PromptServer.instance.send_sync = partial(self.send_sync_override, originalSendSyncFn)
        
        return
    
    def send_sync_override(self, originalFn, event, data, sid=None):
        originalFn(event, data, sid)

        if (event == "progress_state" and self.CURRENTLY_RUNNING is not None):
            prompt_id_to_check = self.CURRENTLY_RUNNING["id"]
            if data is not None and data.get("prompt_id", None) == prompt_id_to_check:
                # this is a dictionary where the key is the node id and the value is the node data
                # we need to get the data of a node that has a property named state that equals to "running"
                all_nodes = data.get("nodes", {})
                for node_id, node_data in all_nodes.items():
                    if node_data.get("state", "") == "running":
                        workflow = self.CURRENTLY_RUNNING['workflow']
                        node_info = workflow.get(node_id, {})
                        node_name = node_info.get("_meta", {}).get("title", node_info.get("class_type", "Unknown"))
                        future = asyncio.run_coroutine_threadsafe(
                                self.CURRENTLY_RUNNING["ws"].send_json({
                                    'type': 'WORKFLOW_STATUS',
                                    'id': prompt_id_to_check,
                                    'workflow_id': self.CURRENTLY_RUNNING['workflow_id'],
                                    'node_id': node_id,
                                    'node_name': node_name,
                                    'progress': node_data.get("value", 0),
                                    'total': node_data.get("max", 1),
                                }), self.loop
                            )
                        break
    
    def queue_updated_override(self, originalFn):
        # this should still have the self of the PromptServer instance because it is a bound method
        originalFn()

        queue_info = PromptServer.instance.get_queue_info()
        task_completed = False
        if (queue_info is not None and "exec_info" in queue_info and "queue_remaining" in queue_info["exec_info"]):
            queue_remaining = queue_info["exec_info"]["queue_remaining"]
            task_completed = queue_remaining == 0

        # {'exec_info': {'queue_remaining': 0}}
        prompt_id_to_check = self.CURRENTLY_RUNNING["id"] if self.CURRENTLY_RUNNING is not None else None
        if (prompt_id_to_check is not None and task_completed):
            prompt_info = PromptServer.instance.prompt_queue.get_history(prompt_id=prompt_id_to_check)

            if prompt_info and prompt_id_to_check in prompt_info:
                prompt_specific_data = prompt_info[prompt_id_to_check]
                status = prompt_specific_data["status"]

                if self.awaiting_tasks_amount > 0:
                    self.awaiting_tasks_done_flag.wait(timeout=30)

                if ("status_str" in status and status["status_str"] == "error"):
                    error_message = "Unknown error"
                    if "messages" in status:
                        error_message = ", ".join([data.get('exception_message', "").replace("\n", "") 
                            for v, data in status['messages'] 
                            if v == 'execution_error'])
                    try:
                        asyncio.run(self.CURRENTLY_RUNNING["ws"].send_json({
                            'type': 'WORKFLOW_FINISHED',
                            'id': prompt_id_to_check,
                            'workflow_id': self.CURRENTLY_RUNNING['workflow_id'],
                            'error': True,
                            'error_message': error_message,
                        }))
                    except Exception as e:
                        print(f"Error sending workflow finished message, maybe user closed connection? {e}")
                    self.CURRENTLY_RUNNING = None
                    
                else:
                    try:
                        asyncio.run(self.CURRENTLY_RUNNING["ws"].send_json({
                            'type': 'WORKFLOW_FINISHED',
                            'id': prompt_id_to_check,
                            'workflow_id': self.CURRENTLY_RUNNING['workflow_id'],
                            'error': False,
                        }))
                    except Exception as e:
                        print(f"Error sending workflow finished message, maybe user closed connection? {e}")
                    self.CURRENTLY_RUNNING = None
        
        if (task_completed):
            if (self.CURRENTLY_RUNNING is not None):
                print("Warning: queue remaining is 0 but currently running is not None")
                return
                
            self.CURRENTLY_RUNNING = None
            
            asyncio.run(self.process_next_workflow_in_queue())
        else:
            print("Queue not empty or no current workflow running, not processing next workflow")
    
    def retrieve_checkpoints_raw(self):
        """
        Retrieves all checkpoints json information from the aihub/models directory.
        checkpoint json files are expected to have the following attributes:
        """

        if (AIHUB_COLD and MODELS_CACHE_RAW is not None):
            return MODELS_CACHE_RAW
        
        # first let's read the files in the ComfyUI models directory
        print(f"Loading models from directory: {AIHUB_MODELS_DIR}")

        # if the directory does not exist, return empty list
        if path.exists(AIHUB_MODELS_DIR):
            # otherwise, read the files
            files = listdir(AIHUB_MODELS_DIR)
            models = []
            for file in files:
                if file.endswith(".json"):
                    with open(path.join(AIHUB_MODELS_DIR, file), "r", encoding="utf-8") as f:
                        try:
                            model_data = json.load(f)
                            models.append(model_data)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON from model file: {file}")

            if AIHUB_COLD:
                MODELS_CACHE_RAW = models

            return models
        return []
    
    def retrieve_checkpoints_cleaned(self, locale=None):
        """
        Retrieves all checkpoints json information but only what is important for the client.
        """

        locale = locale if locale is not None else "default"
        if (AIHUB_COLD and locale in MODELS_CACHE_CLEANED):
            return MODELS_CACHE_CLEANED[locale]

        raw_models = self.retrieve_checkpoints_raw()
        cleaned_models = []

        for model in raw_models:
            id = model.get("id", "None")
            locale = locale.lower().replace("-", "_")
            potential_locale_file = path.join(AIHUB_MODELS_LOCALE_DIR, locale, id + ".json")
            if "_" in locale and not path.exists(potential_locale_file):
                potential_locale_file = path.join(AIHUB_MODELS_LOCALE_DIR, locale.split("_")[0], id + ".json")
            locale_data = model
            if path.exists(potential_locale_file):
                with open(potential_locale_file, "r", encoding="utf-8") as f:
                    try:
                        locale_data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from model locale file: {potential_locale_file}")

            cleaned_model = {
                "id": model.get("id", None), #required
                "file": model.get("file", None), # required the checkpoint file or diffusion model file
                "vae_file": model.get("vae_file", None), #optional, the vae file to use with this model
                "clip_file": model.get("clip_file", None), #optional, the clip file to use with this model, can be a comma separated list of two clips for using a dual clip
                "clip_type": model.get("clip_type", None), #optional, the clip type to use with this model
                "name": locale_data.get("name", model.get("name", None)), # required the name of the model as will be seen by the user
                "group": model.get("group", None), # optional, a group name to group models together, can also be used to limit loras
                "family": model.get("family", None), # required the family of the model, e.g. "stable-diffusion", "sdxl", "other" these are used to limit loras
                "context": model.get("context", None), # required the context of the model, e.g. "image", "video", "3d", "text", "audio" these are used to limit models depending on the context
                "is_diffusion_model": model.get("is_diffusion_model", None), # required, boolean, if true it will be loaded as a diffusion model instead of a checkpoint
                "diffusion_model_weight_dtype": model.get("diffusion_model_weight_dtype", "default"), # optional, if is_diffusion_model is true, this can be used to specify the weight dtype to use when loading the diffusion model
                "description": locale_data.get("description", model.get("description", "")), # required, a description of the model
                "default_cfg": model.get("default_cfg", None), # optional, the default cfg value to use with this model
                "default_steps": model.get("default_steps", None), # optional, the default steps value to use with this model
                "default_sampler": model.get("default_sampler", None), # optional, the default sampler to use with this model
                "default_scheduler": model.get("default_scheduler", None), # optional, the default scheduler to use with this model
            }
            cleaned_models.append(cleaned_model)

        if AIHUB_COLD:
            MODELS_CACHE_CLEANED[locale] = cleaned_models

        return cleaned_models
    
    def retrieve_loras_raw(self):
        """
        Retrieves all loras information from the aihub/loras directory.
        """

        if (AIHUB_COLD and LORAS_CACHE_RAW is not None):
            return LORAS_CACHE_RAW
        
        # first let's read the files in the ComfyUI loras directory
        print(f"Loading loras from directory: {AIHUB_LORAS_DIR}")

        # if the directory does not exist, return empty list
        if path.exists(AIHUB_LORAS_DIR):
            # otherwise, read the files
            files = listdir(AIHUB_LORAS_DIR)
            loras = []
            for file in files:
                if file.endswith(".json"):
                    with open(path.join(AIHUB_LORAS_DIR, file), "r", encoding="utf-8") as f:
                        try:
                            lora_data = json.load(f)
                            loras.append(lora_data)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON from lora file: {file}")

            if AIHUB_COLD:
                LORAS_CACHE_RAW = loras

            return loras
        return []

    def retrieve_loras_cleaned(self, locale=None):
        """
        Retrieves all loras information but only what is important for the client.
        Basically all but the path that specifies where the lora is in the server
        """

        locale = locale if locale is not None else "default"
        if (AIHUB_COLD and locale in LORAS_CACHE_CLEANED):
            return LORAS_CACHE_CLEANED[locale]
        
        raw_loras = self.retrieve_loras_raw()
        cleaned_loras = []

        for lora in raw_loras:
            id = lora.get("id", "None")
            locale = locale.lower().replace("-", "_")
            potential_locale_file = path.join(AIHUB_LORAS_LOCALE_DIR, locale, id + ".json")
            if "_" in locale and not path.exists(potential_locale_file):
                potential_locale_file = path.join(AIHUB_LORAS_LOCALE_DIR, locale.split("_")[0], id + ".json")
            locale_data = lora
            if path.exists(potential_locale_file):
                with open(potential_locale_file, "r", encoding="utf-8") as f:
                    try:
                        locale_data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from model locale file: {potential_locale_file}")
            cleaned_lora = {
                "id": lora.get("id", None), # required, the unique identifier for the lora
                "file": lora.get("file", None), # required, the lora filename in the loras directory
                "name": locale_data.get("name", lora.get("name", None)), # required, the name of the lora as will be seen by the user
                "context": lora.get("context", None), #required, the context in which this lora can be used, e.g. "image", "video", "3d", "text", "audio"
                "description": locale_data.get("description", lora.get("description", "")), #required a description of the lora
                "default_strength": lora.get("default_strength", None), #optional, the default strength the lora will have when applied
                "limit_to_model": lora.get("limit_to_model", None), #optional, limits to a specific model id
                "limit_to_family": lora.get("limit_to_family", None), #required, limits to which family this lora can be applied
                "limit_to_group": lora.get("limit_to_group", None), #optional, limits to which model group this lora can be applied
                "use_loader_model_only": lora.get("use_loader_model_only", False), # if true, the lora will use the LoaderModelOnly node to load the lora and apply it to the model, so it will not affect the clip embeddings
            }
            cleaned_loras.append(cleaned_lora)

        if AIHUB_COLD:
            LORAS_CACHE_CLEANED[locale] = cleaned_loras

        return cleaned_loras
    
    def retrieve_workflows_raw(self):
        """
        Retrieves all workflow JSON files from the specified directory.
        1. Reads all JSON files in the ComfyUI workflows directory.
        2. Parses them into Python dictionaries.
        3. Caches the results if AIHUB_COLD is enabled.
        4. Returns a list of workflow dictionaries.
        """
        if (AIHUB_COLD and WORKFLOWS_CACHE_RAW is not None):
            return WORKFLOWS_CACHE_RAW
        
        # first let's read the files in the ComfyUI workflows directory
        print(f"Loading workflows from directory: {AIHUB_WORKFLOWS_DIR}")

        # if the directory does not exist, return empty list
        if path.exists(AIHUB_WORKFLOWS_DIR):
            # otherwise, read the files
            files = listdir(AIHUB_WORKFLOWS_DIR)
            workflows = []
            for file in files:
                if file.endswith(".json"):
                    with open(path.join(AIHUB_WORKFLOWS_DIR, file), "r", encoding="utf-8") as f:
                        try:
                            workflow_data = json.load(f)
                            workflows.append(workflow_data)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON from workflow file: {file}")

            if AIHUB_COLD:
                WORKFLOWS_CACHE_RAW = workflows

            return workflows
        return []
    
    def retrieve_valid_workflows(self):
        """
        Provides the valid workflows that can be used with AIHub.
        For that they need to have a AIHubWorkflowController node.
        Those workflows are filtered from the raw workflows.
        However they may not all be usable as they may be corrupted or missing
        required nodes.
        """

        # in order to validate the workflows, we need to check if they contain a specific node
        # that establishes them as valid for AIHub processing.
        # the node type we are looking for is "AIHubWorkflowController"

        if (AIHUB_COLD and WORKFLOWS_CACHE_VALID is not None):
            return WORKFLOWS_CACHE_VALID
        
        raw_workflows = self.retrieve_workflows_raw()
        valid_workflows = []

        for workflow in raw_workflows:
            if workflow is None or not isinstance(workflow, dict):
                continue
            for key, node in workflow.items():
                if node.get("class_type", None) == "AIHubWorkflowController":
                    valid_workflows.append(workflow)
                    break

        if AIHUB_COLD:
            WORKFLOWS_CACHE_VALID = valid_workflows

        return valid_workflows
    
    def retrieve_valid_workflow_aihub_summary_from(self, workflow, locale=None):
        """
        Retrieves the aihub summary for a specific workflow.
        This is used to provide the client with the basic information
        about the workflow without sending the whole workflow data.
        """
        # we build the basic data structure for the workflow summary
        workflow_summary = {"expose":{}, "conditions": []}

        workflow_locale_patch = None
        if locale is not None and locale != "default":
            id = None
            for key, node in workflow.items():
                # check if it is a AIHubWorkflowController or AIHubExpose node
                # those are the only nodes we care about for the summary
                if node.get("class_type") == "AIHubWorkflowController":
                    id = node.get("inputs", {}).get("id", None)
                    break

            locale = locale.lower().replace("-", "_")
            potential_locale_file = path.join(AIHUB_WORKFLOWS_LOCALE_DIR, locale, id + ".json")
            if "_" in locale and not path.exists(potential_locale_file):
                potential_locale_file = path.join(AIHUB_WORKFLOWS_LOCALE_DIR, locale.split("_")[0], id + ".json")
            if path.exists(potential_locale_file):
                with open(potential_locale_file, "r", encoding="utf-8") as f:
                    try:
                        workflow_locale_patch = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from workflow locale file: {potential_locale_file}")

        # we will iterate through the nodes to find the AIHubWorkflowController node
        # and the AIHubExpose nodes, and extract their parameters
        for nodeId, node in workflow.items():

            # check if it is a AIHubWorkflowController or AIHubExpose node
            # those are the only nodes we care about for the summary
            if node.get("class_type") == "AIHubWorkflowController" or node.get("class_type", "").startswith("AIHubExpose") or node.get("class_type") == "AIHubAddRunCondition":
                # the basic data structure for the node summary
                data = node.get("inputs", {})
                data_patch = {}
                if workflow_locale_patch is not None:
                    # make a shallow copy of data which is a dictionary
                    data = data.copy()
                    # apply the locale patch to the data
                    data_patch = workflow_locale_patch.get(nodeId, {})
                    for key, value in data_patch.items():
                        if key in ["description", "name", "tooltip", "label", "options_label", "category", "metadata_fields_label", "error"]:
                            data[key] = value

                # if it is a controller node, we copy all the data to the workflow summary
                if node.get("class_type") == "AIHubWorkflowController":
                    # put every property of data into workflow_summary
                    for key in data:
                        workflow_summary[key] = data[key]

                elif node.get("class_type") == "AIHubAddRunCondition":
                    # add a new condition in the conditions list
                    workflow_summary["conditions"].append(data)

                # if it is an AIHubExpose node, we add it to the expose list
                else:
                    # add a new expose in the expose list
                    id = data.get("id", None)

                    # store the expose data
                    workflow_summary["expose"][id] = {
                        "type": node.get("class_type"),
                        "data": data,
                    }

        return workflow_summary
    
    def retrieve_valid_workflows_aihub_summary(self, locale=None):
        """
        The aihub summary is what is sent to the client when they request the workflows list.
        It contains the basic information for the client to use the workflow without sending
        the whole workflow data.
        """

        # the aihub summary is a simplified version of the valid workflows
        # it contains the basic information for the client to use the workflow
        # without sending the entire workflow data, basically the workflow summary will
        # contain only the nodes that start with AIHubExpose and the value of the parameters
        # as key value pairs

        original_locale = locale
        locale = locale if locale is not None else "default"
        if (AIHUB_COLD and locale in WORKFLOWS_AIHUB_SUMMARY):
            return WORKFLOWS_AIHUB_SUMMARY[locale]
        
        valid_workflows = self.retrieve_valid_workflows()
        aihub_summaries = {}

        for workflow in valid_workflows:
            # we build the basic data structure for the workflow summary
            workflow_summary = self.retrieve_valid_workflow_aihub_summary_from(workflow, locale=original_locale)

            # get the workflow id
            workflow_id = workflow_summary.get("id")

            # check for duplicate workflow ids
            if aihub_summaries.get(workflow_id) is not None:
                print("Warning: duplicate workflow id found, overwriting previous workflow with id " + workflow_id)

            # store the workflow summary
            aihub_summaries[workflow_id] = workflow_summary

        if AIHUB_COLD:
            WORKFLOWS_AIHUB_SUMMARY[locale] = aihub_summaries
                    
        return aihub_summaries
    
    def retrieve_workflow_by_id(self, workflow_id):
        """
        For a given workflow id, retrieves the full workflow data.
        This is used when a client requests to run a specific workflow
        """

        valid_workflows = None
        if AIHUB_COLD and WORKFLOWS_CACHE_VALID is not None:
            valid_workflows = WORKFLOWS_CACHE_VALID
        else:
            valid_workflows = self.retrieve_valid_workflows()

        #retrieve a specific workflow by the controller id
        for workflow in valid_workflows:
             for key, node in workflow.items():
                if node.get("class_type", None) == "AIHubWorkflowController":
                    if node.get("inputs", {}).get("id", None) == workflow_id:
                        return workflow

        return None
    
    def validate_and_process_workflow_request(self, socket_file_dir, request):
        """
        Validates and processes a workflow request from a client.
        This requires the socket_file_dir to handle file uploads.
        The request must contain a valid workflow_id and expose parameters.
        The function returns a tuple of (processed_workflow, is_valid, message).
        """

        # a valid request must contain a workflow_id that matches a valid workflow
        if "workflow_id" not in request:
            return None, False, "Missing workflow_id"
        
        # it must also contain expose parameters
        if "expose" not in request:
            return None, False, "Missing expose parameters"
        
        # get the workflow by id
        workflow = self.retrieve_workflow_by_id(request["workflow_id"])
        if workflow is None:
            return None, False, "Invalid workflow_id"
        
        # make a deep copy of the workflow to modify
        workflow_copy = deepcopy(workflow)
        
        # get the workflow summary to validate the expose parameters
        workflow_summary = self.retrieve_valid_workflow_aihub_summary_from(workflow)
        if workflow_summary is None:
            return None, False, "Workflow summary not found, corrupted workflow?"
        
        # validate that all expose parameters are present
        for expose in workflow_summary["expose"].values():
            # get the id of the expose
            expose_id = expose["data"].get("id", None)

            # id is required
            if expose_id is None:
                return None, False, "Corrupted workflow, expose node missing id"
            
            # check that the expose id is present in the request
            if expose_id not in request["expose"]:
                return None, False, f"Missing parameter for expose id {expose_id}"
            
            for key, node in workflow_copy.items():
                class_type = node.get("class_type", "")
                if class_type.startswith("AIHubExpose") and node.get("inputs", {}).get("id", None) == expose_id:
                    if isinstance(request["expose"][expose_id], dict) and "local_file" in request["expose"][expose_id]:
                        # local_file is a special case, it must represent a local file path within a subdirectory that is
                        # specific for the given websocket session, so we must check that it is alphanumeric and does not contain any path traversal characters
                        local_file_path = request["expose"][expose_id]["local_file"]

                        if local_file_path is None:
                            # allow it
                            workflow_copy[key]["inputs"]["local_file"] = None
                            continue

                        local_file_path_unmodified = local_file_path
                        #check that it is a string and that fits the 0-9 A-Z a-z _-.
                        if not isinstance(local_file_path, str) or not re.match(r'^[0-9A-Za-z_\-\.]+$', local_file_path):
                            return None, False, f"Invalid local_file path for expose id {expose_id}, must be alphanumeric dots and dashes only not {local_file_path_unmodified}"
                    
                        # if all checks pass, we need to convert the local_file_path to a full path
                        value_to_set = path.join(socket_file_dir, local_file_path)

                        # check that the file exists
                        if not path.exists(value_to_set) or not path.isfile(value_to_set):
                            return None, False, f"File not found for expose id {expose_id} at file {local_file_path_unmodified}"

                        workflow_copy[key]["inputs"]["local_file"] = value_to_set

                        for prop_key, prop_value in request["expose"][expose_id].items():
                            if prop_key != "local_file":
                                workflow_copy[key]["inputs"][prop_key] = prop_value
                    elif isinstance(request["expose"][expose_id], dict) and "local_files" in request["expose"][expose_id]:
                        if not isinstance(request["expose"][expose_id], dict) or "local_files" not in request["expose"][expose_id]:
                            return None, False, f"Invalid parameter for expose id {expose_id}, must be an object with a local_files property"
                        
                        # local_file is a special case, it must represent a local file path within a subdirectory that is
                        # specific for the given websocket session, so we must check that it is alphanumeric and does not contain any path traversal characters
                        local_file_paths = request["expose"][expose_id]["local_files"]

                        if not isinstance(local_file_paths, list):
                            return None, False, f"Invalid local_files value for expose id {expose_id}, must be a list of file names"
                        
                        validated_file_paths = []
                        for local_file_path in local_file_paths:
                            local_file_path_unmodified = local_file_path
                            if not isinstance(local_file_path, str) or not re.match(r'^[0-9A-Za-z_\-\.]+$', local_file_path):
                                return None, False, f"Invalid local_file path for expose id {expose_id}, must be alphanumeric dots and dashes only not {local_file_path_unmodified}"
                            # if all checks pass, we need to convert the local_file_path to a full path
                            full_path = path.join(socket_file_dir, local_file_path)

                            # check that the file exists
                            if not path.exists(full_path) or not path.isfile(full_path):
                                return None, False, f"File not found for expose id {expose_id} at file {local_file_path_unmodified}"
                            
                            validated_file_paths.append(full_path)

                        value_to_set = json.dumps(validated_file_paths)

                        workflow_copy[key]["inputs"]["local_files"] = value_to_set

                        for prop_key, prop_value in request["expose"][expose_id].items():
                            if prop_key != "local_files":
                                workflow_copy[key]["inputs"][prop_key] = prop_value
                    elif isinstance(request["expose"][expose_id], dict):
                        # for other types of expose nodes, we just copy every property that is exposed
                        # into the inputs of the node
                        for prop_key, prop_value in request["expose"][expose_id].items():
                            workflow_copy[key]["inputs"][prop_key] = prop_value

                        # security is not necessarily a concern here because the class will
                        # only use the properties it expects
                    elif "value" in workflow_copy[key]["inputs"]:
                        workflow_copy[key]["inputs"]["value"] = request["expose"][expose_id]
                    else:
                        # set the value property by default even if it does not exist
                        # may be hidden field
                        workflow_copy[key]["inputs"]["value"] = request["expose"][expose_id]

        return workflow_copy, True, "Workflow validated and prepared"
    
    async def on_websocket_connect(self, request):
        """
        Handles a new WebSocket connection and messages.
        """
        global AIHUB_MAX_MESSAGE_SIZE
        ws = web.WebSocketResponse(max_msg_size=AIHUB_MAX_MESSAGE_SIZE)
        await ws.prepare(request)

        # make a directory for this websocket connection to store files, use the Temp directory for the operating system
        socket_file_dir = path.join(AIHUB_TEMP_DIRECTORY, "comfyui_socket_aihub_files", str(uuid.uuid4()))
        if not path.exists(socket_file_dir):
            makedirs(socket_file_dir)

        print(f"New WebSocket connection")
        PREVIOUS_UPLOAD_HEADER = None

        locale = request.headers.get("locale", None)

        try:

            # TODO validation
            # validate_websocket_user()

            await ws.send_json({
                'type': 'INFO_LIST',
                'workflows': self.retrieve_valid_workflows_aihub_summary(locale=locale),
                'models': self.retrieve_checkpoints_cleaned(locale=locale),
                'loras': self.retrieve_loras_cleaned(locale=locale),
                'samplers': comfy.samplers.KSampler.SAMPLERS,
                'schedulers': comfy.samplers.KSampler.SCHEDULERS
            })

            # Asynchronously wait for and process messages from the client
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    print(f"Received message: {msg.data}")
                    try:
                        data = json.loads(msg.data)

                        if not isinstance(data, dict):
                            await ws.send_json({'type': 'ERROR', 'message': 'Invalid JSON format, must be an object'})
                            continue
                        elif 'type' not in data:
                            await ws.send_json({'type': 'ERROR', 'message': 'Missing request type in JSON'})
                            continue

                        elif 'ping' in data and data["type"] == "PING":
                            await ws.send_json({'type': 'PONG', 'value': data["ping"]})
                            self.LAST_PING_QUEUE_VALUE = str(data["ping"])

                        elif data["type"] == "FILE_CHECK_EXISTS":
                            file_to_check = data["filename"] if "filename" in data else None
                            if file_to_check is None:
                                await ws.send_json({'type': 'ERROR', 'message': 'Missing file name to check'})
                                continue
                            if not isinstance(file_to_check, str) or not re.match(r'^[0-9A-Za-z_\-\.]+$', file_to_check):
                                await ws.send_json({'type': 'ERROR', 'message': 'Invalid file name to check, must be alphanumeric dots and dashes only'})
                                continue
                            full_path = path.join(socket_file_dir, file_to_check)
                            exists = path.exists(full_path) and path.isfile(full_path)
                            await ws.send_json({'type': 'CHECK_EXISTS_STATUS', 'file': file_to_check, 'exists': exists})
                            continue

                        elif data["type"] == "FILE_UPLOAD":
                            if "filename" not in data or not data["filename"].isalnum():
                                await ws.send_json({'type': 'ERROR', 'message': 'Invalid binary header'})
                                continue
                            if not re.match(r'^[0-9A-Za-z_\-\.]+$', data["filename"]):
                                await ws.send_json({'type': 'ERROR', 'message': 'Invalid filename in binary header, must be alphanumeric dots and dashes only'})
                                continue
                            # now we got to check a if-not-exist flag
                            if "if_not_exists" in data and data["if_not_exists"] == True:
                                full_path = path.join(socket_file_dir, data["filename"])
                                if path.exists(full_path) and path.isfile(full_path):
                                    await ws.send_json({'type': 'FILE_UPLOAD_SKIP', 'file': data["filename"]})
                                    PREVIOUS_UPLOAD_HEADER = None
                                    continue
                            PREVIOUS_UPLOAD_HEADER = data
                            await ws.send_json({'type': 'UPLOAD_ACK', 'file': data["filename"]})
                            continue

                        elif 'cancel' in data and data["type"] == "WORKFLOW_OPERATION":
                            cancelled_current = False
                            what_to_cancel = data['cancel']
                            if type(what_to_cancel) is str:
                                if (self.CURRENTLY_RUNNING is not None and self.CURRENTLY_RUNNING["ws"] == ws and self.CURRENTLY_RUNNING["id"] == what_to_cancel):
                                    await self.CURRENTLY_RUNNING["ws"].send_json({
                                        'type': 'WORKFLOW_FINISHED',
                                        'id': self.CURRENTLY_RUNNING["id"],
                                        'workflow_id': self.CURRENTLY_RUNNING["workflow_id"],
                                        'error': True,
                                        "error_message": "Workflow run cancelled by user",
                                        "cancelled": True
                                    })
                                    self.cancel_current_run()
                                    cancelled_current = True
                                else:
                                    with self.QUEUE_LOCK:
                                        for i in range(0, len(self.WORKFLOW_REQUEST_QUEUE)):
                                            if self.WORKFLOW_REQUEST_QUEUE[i]["id"] == what_to_cancel and self.WORKFLOW_REQUEST_QUEUE[i]["ws"] == ws:
                                                await self.WORKFLOW_REQUEST_QUEUE[i]["ws"].send_json({
                                                    'type': 'WORKFLOW_FINISHED',
                                                    'id': self.WORKFLOW_REQUEST_QUEUE[i]["id"],
                                                    'workflow_id': self.WORKFLOW_REQUEST_QUEUE[i]["workflow_id"],
                                                    'error': True,
                                                    "error_message": "Workflow run cancelled by user",
                                                    "cancelled": True
                                                })
                                                del self.WORKFLOW_REQUEST_QUEUE[i]
                                                break
                                
                                        for i in range(0, len(self.WORKFLOW_REQUEST_QUEUE)):
                                            await self.WORKFLOW_REQUEST_QUEUE[i]["ws"].send_json({
                                                'type': 'WORKFLOW_AWAIT',
                                                'id': self.WORKFLOW_REQUEST_QUEUE[i]["id"],
                                                'workflow_id': self.WORKFLOW_REQUEST_QUEUE[i]["workflow_id"],
                                                'before_this': i
                                            })

                            if cancelled_current:
                                asyncio.create_task(self.process_next_workflow_in_queue())

                        elif 'workflow_id' not in data and data["type"] == "WORKFLOW_OPERATION":
                            await ws.send_json({'type': 'ERROR', 'message': 'Missing workflow_id to execute'})

                        elif 'workflow_id' in data and data["type"] == "WORKFLOW_OPERATION":
                            # validate before queueing
                            validated_workflow, valid, message = self.validate_and_process_workflow_request(socket_file_dir, data)
                            if not valid:
                                await ws.send_json({'type': 'ERROR', 'message': message, 'workflow_id': data.get('workflow_id', None)})
                                continue
                            # Place the prompt in the queue for the ComfyUI node to pick up.
                            # We use a lock to ensure this is a thread-safe operation.
                            run_id = str(uuid.uuid4())
                            with self.QUEUE_LOCK:
                                self.WORKFLOW_REQUEST_QUEUE.append({
                                    'id': run_id,
                                    'request': data,
                                    'ws': ws,
                                    'workflow': validated_workflow,
                                    'workflow_id': data['workflow_id'],
                                    'file_dir': socket_file_dir,
                                })

                            # Send a confirmation back to the external client
                            await ws.send_json({
                                'type': 'WORKFLOW_AWAIT',
                                'id': run_id,
                                'workflow_id': data["workflow_id"],
                                'before_this': len(self.WORKFLOW_REQUEST_QUEUE) - 1
                            })

                            if self.CURRENTLY_RUNNING is None:
                                asyncio.create_task(self.process_next_workflow_in_queue())

                        else:
                            await ws.send_json({'type': 'ERROR', 'message': 'Unknown request type'})

                    except json.JSONDecodeError:
                        await ws.send_json({'type': 'ERROR', 'message': 'Invalid JSON'})
                
                elif msg.type == web.WSMsgType.ERROR:
                    print(f'WebSocket connection closed with exception {ws.exception()}')

                elif msg.type == web.WSMsgType.BINARY:
                    print('WebSocket recieved a file')
                    # we now save the file to the socket_file_dir with the name specified in the previous header
                    if PREVIOUS_UPLOAD_HEADER is None:
                        await ws.send_json({'type': 'ERROR', 'message': 'Missing upload header before file upload'})
                        continue

                    file_name = PREVIOUS_UPLOAD_HEADER.get("filename", None)
                    full_path = path.join(socket_file_dir, file_name)

                    print('File to be saved at ' + full_path)

                    try:
                        with open(full_path, "wb") as f:
                            f.write(msg.data)
                        await ws.send_json({'type': 'FILE_UPLOAD_SUCCESS', 'file': file_name})
                    except Exception as e:
                        await ws.send_json({'type': 'ERROR', 'message': 'Error saving file'})
                        print(f"Error saving file {file_name}: {e}")

                    PREVIOUS_UPLOAD_HEADER = None
                else:
                    print('WebSocket received unknown message type')
                    await ws.send_json({'type': 'ERROR', 'message': 'Unknown message type'})
                    

        finally:
            print(f"WebSocket connection closed by client")

            with self.QUEUE_LOCK:
                # We first find the keys of the prompts to remove.
                for i in range(0, len(self.WORKFLOW_REQUEST_QUEUE)):
                    if self.WORKFLOW_REQUEST_QUEUE[i]["ws"] == ws:
                        del self.WORKFLOW_REQUEST_QUEUE[i]
                
                for i in range(0, len(self.WORKFLOW_REQUEST_QUEUE)):
                    self.WORKFLOW_REQUEST_QUEUE[i]["ws"].send_json({
                        'type': 'WORKFLOW_AWAIT',
                        'id': self.WORKFLOW_REQUEST_QUEUE[i]["id"],
                        'workflow_id': self.WORKFLOW_REQUEST_QUEUE[i]["workflow_id"],
                        'before_this': i
                    })

                if self.CURRENTLY_RUNNING and self.CURRENTLY_RUNNING["ws"] == ws:
                    self.cancel_current_run()
                    asyncio.create_task(self.process_next_workflow_in_queue())

            if not AIHUB_PERSIST_TEMPFILES:
                # clean up the temporary directory for this websocket connection
                try:
                    if path.exists(socket_file_dir):
                        rmtree(socket_file_dir)
                except Exception as e:
                    print(f"Error cleaning up socket file directory {socket_file_dir}: {e}")

        return ws

    async def run_websocket_server(self, app, host, port):
        """
        Starts the aiohttp web server.
        """
        print(f"Attempting to start WebSocket server on port {port}...")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)

        await site.start()
        SERVER_RUNNING_FLAG.set()
        print("WebSocket server is running and ready for connections.")
        # This will block until the runner is stopped
        #await runner.cleanup()
        while True:
            await asyncio.sleep(3600)
    
    def start_server_in_thread(self):
        """
        A function to start the aiohttp event loop and server in a separate thread.
        This prevents it from blocking the main ComfyUI process.
        """
        # Create the aiohttp application
        app = web.Application()
        app.router.add_get('/ws', self.on_websocket_connect)

        # add an endpoint to get raw files, this is also useful for retrieving the
        # image files and it is mainly what it is used for
        app.router.add_static('/workflows/', path=AIHUB_WORKFLOWS_DIR, show_index=True)
        app.router.add_static('/models/', path=AIHUB_MODELS_DIR, show_index=True)
        app.router.add_static('/loras/', path=AIHUB_LORAS_DIR, show_index=True)

        # Start the server
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.run_websocket_server(app, "0.0.0.0", WEB_SOCKET_SERVER_PORT))
        self.loop.close()
    
    def start_server(self):
        """
        Checks if the server is running and starts it in a new thread if it's not.
        This is called when the custom node is first imported.
        """
        global SERVER_THREAD
        if SERVER_THREAD is None or not SERVER_THREAD.is_alive():

            # Start the server in a separate daemon thread
            SERVER_THREAD = threading.Thread(target=self.start_server_in_thread, daemon=True)
            SERVER_THREAD.start()
            print("Spawned new WebSocket server thread.")
            # Wait for the server to confirm it has started
            SERVER_RUNNING_FLAG.wait(timeout=10)

            #threading.Thread(target=self.process_messages, daemon=True).start()
            #print("Process of messages thread started.")

    async def process_next_workflow_in_queue(self):
        with self.QUEUE_LOCK:
            if len(self.WORKFLOW_REQUEST_QUEUE) > 0 and self.CURRENTLY_RUNNING is None:
                self.CURRENTLY_RUNNING = self.WORKFLOW_REQUEST_QUEUE.pop()

                for i in range(0, len(self.WORKFLOW_REQUEST_QUEUE)):
                    await self.WORKFLOW_REQUEST_QUEUE[i]["ws"].send_json({
                        'type': 'WORKFLOW_AWAIT',
                        'id': self.WORKFLOW_REQUEST_QUEUE[i]["id"],
                        'workflow_id': self.WORKFLOW_REQUEST_QUEUE[i].request["workflow_id"],
                        'before_this': i
                    })
            
                await self.process_current()

    def cancel_current_run(self):
        interrupt_processing()
        self.CURRENTLY_RUNNING = None

    async def process_current(self):
        self.awaiting_tasks_lock.acquire()
        self.awaiting_tasks_amount = 0
        self.awaiting_tasks_lock.release()
        self.awaiting_tasks_done_flag = threading.Event()

        if not self.CURRENTLY_RUNNING:
            return
        
        print("Processing current workflow...")
        # we will disable validation because it is not really useful (let it crash if the workflow is invalid, these should be curated workflows already)
        # and the original validate_prompt function is not good at validating because it makes errors where
        # there shouldn't be any so we use a custom function to fix this behaviour

        #valid = await validate_prompt(self.CURRENTLY_RUNNING["id"], self.CURRENTLY_RUNNING["workflow"], None)
        #print(valid)
        
        #if valid[0]:
        await self.CURRENTLY_RUNNING["ws"].send_json({
            'type': 'WORKFLOW_START',
            'workflow_id': self.CURRENTLY_RUNNING["request"]["workflow_id"],
            'id': self.CURRENTLY_RUNNING["id"],
        })
        #outputs_to_execute = valid[2]
        # we will however need to find all the output nodes to know which ones to execute
        # for that we will have to find for all actionable nodes (nodes that start with AIHubAction)
        # and get their ids, then we will pass those ids to the prompt queue to execute

        outputs_to_execute = []
        for key, node in self.CURRENTLY_RUNNING["workflow"].items():
            if node.get("class_type", "").startswith("AIHubAction"):
                outputs_to_execute.append(key)

        print(f"Executing workflow expecting {len(outputs_to_execute)} output nodes.", outputs_to_execute)

        number = PromptServer.instance.number
        PromptServer.instance.number += 1
        sensitive = {}
        extra_data = self.CURRENTLY_RUNNING["request"].get("extra_data", {})
        for sensitive_val in SENSITIVE_EXTRA_DATA_KEYS:
            if sensitive_val in extra_data:
                sensitive[sensitive_val] = extra_data.pop(sensitive_val)
        PromptServer.instance.prompt_queue.put((number, self.CURRENTLY_RUNNING["id"], self.CURRENTLY_RUNNING["workflow"], extra_data, outputs_to_execute, sensitive))
        #else:
        #    await self.CURRENTLY_RUNNING["ws"].send_json({
        #        'type': 'ERROR',
        #        'workflow_id': self.CURRENTLY_RUNNING["request"]["workflow_id"],
        #        'id': self.CURRENTLY_RUNNING["id"],
        #        'node_errors': valid[3],
        #        'message': f'Error validating workflow: {json.dumps(valid[1])}',
        #    })
        #    self.CURRENTLY_RUNNING = None

    async def send_binary_data(self, workflow_id, id, ws, binary_data, data_type, action_data):
        """
        Sends binary data (e.g., images, files) to the client via websocket.
        """
        # Send a header message to indicate incoming binary data
        await ws.send_json({
            "type": "FILE",
            'workflow_id': workflow_id,
            'id': id,
            "data_type": data_type,
            "action": action_data,
        })
        # Send the binary data itself
        await ws.send_bytes(binary_data)

    async def send_binary_data_to_current_client(self, binary_data, data_type, action_data):
        return await self.send_binary_data(self.CURRENTLY_RUNNING["workflow_id"], self.CURRENTLY_RUNNING["id"], self.CURRENTLY_RUNNING["ws"], binary_data, data_type, action_data)
    
    async def send_json(self, workflow_id, id, ws, data):
        data["workflow_id"] = workflow_id
        data["id"] = id
        """
        Sends JSON data to the client via websocket.
        """
        await ws.send_json(data)

    async def send_json_to_current_client(self, data):
        return await self.send_json(self.CURRENTLY_RUNNING["workflow_id"], self.CURRENTLY_RUNNING["id"], self.CURRENTLY_RUNNING["ws"], data)
    
    async def send_status_message_to_current_client(self, status_message, extra_data=None):
        data = {
            "type": "STATUS",
            "message": status_message
        }
        if extra_data:
            data.update(extra_data)
        return await self.send_json(self.CURRENTLY_RUNNING["workflow_id"], self.CURRENTLY_RUNNING["id"], self.CURRENTLY_RUNNING["ws"], data)
    
    def send_binary_data_to_current_client_sync(self, binary_data, data_type, action):
        future = asyncio.run_coroutine_threadsafe(
            self.send_binary_data_to_current_client(binary_data, data_type, action),
            self.loop
        )
        self.awaiting_tasks_lock.acquire()
        self.awaiting_tasks_amount += 1
        self.awaiting_tasks_lock.release()

        def on_done(t):
            self.awaiting_tasks_lock.acquire()
            self.awaiting_tasks_amount -= 1
            self.awaiting_tasks_lock.release()

            if self.awaiting_tasks_amount == 0:
                self.awaiting_tasks_done_flag.set()

        future.add_done_callback(on_done)

    def send_json_to_current_client_sync(self, data):
        future = asyncio.run_coroutine_threadsafe(
            self.send_json_to_current_client(data),
            self.loop
        )
        self.awaiting_tasks_lock.acquire()
        self.awaiting_tasks_amount += 1
        self.awaiting_tasks_lock.release()

        def on_done(t):
            self.awaiting_tasks_lock.acquire()
            self.awaiting_tasks_amount -= 1
            self.awaiting_tasks_lock.release()

            if self.awaiting_tasks_amount == 0:
                self.awaiting_tasks_done_flag.set()

        future.add_done_callback(on_done)