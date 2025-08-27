import asyncio
import threading
import json
import uuid
from os import path, listdir, environ, makedirs, remove, rmdir
from copy import deepcopy
import tempfile
from time import sleep

from aiohttp import web

WEB_SOCKET_SERVER_PORT = 8000
SERVER_THREAD = None
SERVER_RUNNING_FLAG = threading.Event()

AIHUB_COLD_WORKFLOWS = environ.get("AIHUB_COLD_WORKFLOWS", "0") == "1"
AIHUB_WORKFLOWS_DIR = environ.get("AIHUB_WORKFLOWS_DIR", None)
AIHUB_PERSIST_TEMPFILES = environ.get("AIHUB_PERSIST_TEMPFILES", False)

WORKFLOWS_CACHE_RAW = None
WORKFLOWS_CACHE_VALID = None
WORKFLOWS_AIHUB_SUMMARY = None

AIHUB_TEMP_DIRECTORY_ENV = environ.get("AIHUB_TEMP_DIR", None)
AIHUB_TEMP_DIRECTORY = AIHUB_TEMP_DIRECTORY_ENV if AIHUB_TEMP_DIRECTORY_ENV is not None else tempfile.gettempdir()

class AIHubServer:
    """
    the server class handles the websocket server and the workflow queue
    file uploads and downloads, as well as the workflow validation and processing
    """
    CURRENTLY_RUNNING = None
    WORKFLOW_REQUEST_QUEUE = []
    QUEUE_LOCK = threading.Lock()

    LAST_PING_QUEUE_VALUE = None

    def __init__(self):
        return
    
    def retrieve_workflows_raw(self):
        """
        Retrieves all workflow JSON files from the specified directory.
        1. Reads all JSON files in the ComfyUI workflows directory.
        2. Parses them into Python dictionaries.
        3. Caches the results if AIHUB_COLD_WORKFLOWS is enabled.
        4. Returns a list of workflow dictionaries.
        """
        if (AIHUB_COLD_WORKFLOWS and WORKFLOWS_CACHE_RAW is not None):
            return WORKFLOWS_CACHE_RAW
        
        # first let's read the files in the ComfyUI workflows directory
        workflows_dir = path.join(AIHUB_WORKFLOWS_DIR if AIHUB_WORKFLOWS_DIR is not None else path.dirname(path.abspath(__file__)), "..", "..", "user", "default", "workflows")
        print(f"Loading workflows from directory: {workflows_dir}")

        # if the directory does not exist, return empty list
        if path.exists(workflows_dir):
            # otherwise, read the files
            files = listdir(workflows_dir)
            workflows = []
            for file in files:
                if file.endswith(".json"):
                    with open(path.join(workflows_dir, file), "r", encoding="utf-8") as f:
                        try:
                            workflow_data = json.load(f)
                            # we will add this file name to the workflow data for reference
                            # for errors and debugging
                            workflow_data["__source_file"] = file
                            workflows.append(workflow_data)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON from workflow file: {file}")

            if AIHUB_COLD_WORKFLOWS:
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

        if (AIHUB_COLD_WORKFLOWS and WORKFLOWS_CACHE_VALID is not None):
            return WORKFLOWS_CACHE_VALID
        
        raw_workflows = self.retrieve_workflows_raw()
        valid_workflows = []

        for workflow in raw_workflows:
            if "nodes" in workflow:
                for node in workflow["nodes"]:
                    if node.get("type") == "AIHubWorkflowController":
                        valid_workflows.append(workflow)
                        break

        if AIHUB_COLD_WORKFLOWS:
            WORKFLOWS_CACHE_VALID = valid_workflows

        return valid_workflows
    
    def retrieve_valid_workflows_aihub_summary_cleaned(self):
        """
        Retrieves all aihub summaries but removes any properties that start with __
        within the expose data.
        """

        summaries = self.retrieve_valid_workflows_aihub_summary()
        cleaned_summaries = deepcopy(summaries)

        for workflow_id in cleaned_summaries:
            expose = cleaned_summaries[workflow_id].get("expose", {})
            for expose_id in expose:
                expose_info = expose[expose_id]
                keys_to_remove = [key for key in expose_info if key.startswith("__")]
                for key in keys_to_remove:
                    del expose_info[key]

                expose_data = expose_info.get("data", {})
                keys_to_remove = [key for key in expose_data if key.startswith("__")]
                for key in keys_to_remove:
                    del expose_data[key]

        return cleaned_summaries
    
    def retrieve_valid_workflows_aihub_summary(self):
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

        if (AIHUB_COLD_WORKFLOWS and WORKFLOWS_AIHUB_SUMMARY is not None):
            return WORKFLOWS_AIHUB_SUMMARY
        
        valid_workflows = self.retrieve_valid_workflows()
        aihub_summaries = {}

        for workflow in valid_workflows:
            # we build the basic data structure for the workflow summary
            workflow_summary = {"expose":{}}

            # keep this as a flag to skip corrupted workflows
            corrupted = False

            # we will iterate through the nodes to find the AIHubWorkflowController node
            # and the AIHubExpose nodes, and extract their parameters
            if "nodes" in workflow:

                # iterate through nodes with index to be able to reference them later
                for i in range(0, len(workflow["nodes"])):

                    # no need to continue if already corrupted
                    if corrupted:
                        break

                    # get the node
                    node = workflow["nodes"][i]

                    # check if it is a AIHubWorkflowController or AIHubExpose node
                    # those are the only nodes we care about for the summary
                    if node.get("type") == "AIHubWorkflowController" or node.get("type").startswith("AIHubExpose"):
                        # the basic data structure for the node summary
                        data = {}

                        # we will iterate through the inputs to find the ones with widgets
                        # since those specify the data structure for the expose, such as data type, name, etc...
                        widgetindex = -1
                        # this is where comfy stores the values for the widgets
                        widgetvalues = node.get("widgets_values", [])

                        # now we loop through the inputs
                        for input in node.get("inputs", []):
                            # if it has a widget, we care about it
                            if "widget" in input:
                                # increment the widget index
                                widgetindex += 1

                                # if the widget index is out of range, the workflow is corrupted
                                if (widgetindex >= len(widgetvalues)):
                                    print("Warning: widget index out of range for workflow input, corrupted workflow on comfyui id " +
                                          workflow.get("__source_file", "unknown") + " node id " + str(node.get("id", "unknown")))
                                    corrupted = True
                                    break

                                # get the widget value
                                widgetvalue = widgetvalues[widgetindex]

                                # store the value in the data structure with the input name as key
                                input_name = input["name"]
                                data[input_name] = widgetvalue

                                # special handling for value and local_file inputs
                                # we need to store the widget index for those inputs
                                # local_file takes priority over value if both are present
                                if input_name == "value" or input_name == "local_file" or input_name == "values" or input_name == "local_files":
                                    if ("__widget_is_file" in data and data["__widget_is_file"] is True):
                                        # already found a local_file input, skip value input
                                        continue

                                    # otherwise store the widget index and if it is a file input
                                    # this will ensure that local_file takes priority over value
                                    data["__widget_index"] = widgetindex
                                    data["__widget_is_file"] = input_name == "local_file" or input_name == "local_files"
                                    data["__widget_is_multiple_files"] = input_name == "local_files"

                                # special handling for value_fixed input
                                # this is used for the ExposeSeed node to indicate the fixed value of the seed
                                if input_name == "value_fixed":
                                    data["__widget_value_fixed_index"] = widgetvalue
                                    data["__widget_value_fixed"] = True

                        # if the workflow is already corrupted, skip further processing
                        if corrupted:
                            break

                        # if it is a controller node, we copy all the data to the workflow summary
                        if node.get("type") == "AIHubWorkflowController":
                            # put every property of data into workflow_summary
                            for key in data:
                                workflow_summary[key] = data[key]

                        # if it is an AIHubExpose node, we add it to the expose list
                        else:
                            # add a new expose in the expose list
                            id = data.get("id", None)

                            # if the widget index is -1, it means we did not find a value or local_file input
                            # which means the workflow is corrupted
                            if data.get("__widget_index", -1) == -1:
                                print("Warning: missing value or local_file input for expose node, corrupted workflow on comfyui id " +
                                    workflow.get("__source_file", "unknown") + " node id " + str(node.get("id", "unknown")))
                                corrupted = True
                                break

                            # id is required and must be unique
                            if id is None:
                                print("Warning: missing id for expose node, corrupted workflow on comfyui id " +
                                      workflow.get("__source_file", "unknown") + " node id " + str(node.get("id", "unknown")))
                                corrupted = True
                                break
                            if id in workflow_summary["expose"]:
                                print("Warning: duplicate expose id found, corrupted workflow on comfyui id " +
                                      workflow.get("__source_file", "unknown") + " node id " + str(node.get("id", "unknown")))
                                corrupted = True
                                break

                            # store the expose data
                            workflow_summary["expose"][id] = {
                                "type": node.get("type"),
                                "data": data,
                                # special property to reference the node index in the workflow
                                "__node_index": i,
                            }

            # if the workflow is not corrupted, we add it to the summaries list
            if not corrupted:
                # workflow id is required
                if "id" not in workflow_summary:
                    print("Warning: missing id for workflow, corrupted workflow on comfyui id " +
                          workflow.get("__source_file", "unknown"))
                    continue

                # get the workflow id
                workflow_id = workflow_summary.get("id")

                # check for duplicate workflow ids
                if aihub_summaries.get(workflow_id) is not None:
                    print("Warning: duplicate workflow id found, overwriting previous workflow with id " + workflow_id)

                # store the workflow summary
                aihub_summaries[workflow_id] = workflow_summary

        if AIHUB_COLD_WORKFLOWS:
            WORKFLOWS_AIHUB_SUMMARY = aihub_summaries
                    
        return aihub_summaries
    
    def retrieve_workflow_by_id(self, workflow_id):
        """
        For a given workflow id, retrieves the full workflow data.
        This is used when a client requests to run a specific workflow
        """

        valid_workflows = None
        if AIHUB_COLD_WORKFLOWS and WORKFLOWS_CACHE_VALID is not None:
            valid_workflows = WORKFLOWS_CACHE_VALID
        else:
            valid_workflows = self.retrieve_valid_workflows()

        #retrieve a specific workflow by the controller id
        for workflow in valid_workflows:
            if "nodes" in workflow:
                for node in workflow["nodes"]:

                    # it must have a AIHubWorkflowController node
                    if node.get("type") == "AIHubWorkflowController":
                        widgetindex = -1
                        widgetvalues = node.get("widgets_values", [])
                        for input in node.get("inputs", []):
                            if "widget" in input:
                                widgetindex += 1

                                if (widgetindex >= len(widgetvalues)):
                                    break

                                if input["name"] != "id":
                                    continue

                                widgetvalue = widgetvalues[widgetindex]

                                if widgetvalue == workflow_id:
                                    return workflow
        return None
    
    def retrieve_workflow_summary_by_id(self, workflow_id):
        """
        For a given workflow id, retrieves the workflow summary data.
        This is used when a client requests to run a specific workflow
        """
        aihub_summaries = None
        if AIHUB_COLD_WORKFLOWS and WORKFLOWS_AIHUB_SUMMARY is not None:
            aihub_summaries = WORKFLOWS_AIHUB_SUMMARY
        else:
            aihub_summaries = self.retrieve_valid_workflows_aihub_summary()
        
        return aihub_summaries.get(workflow_id, None)
    
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
        workflow_summary = self.retrieve_workflow_summary_by_id(request["workflow_id"])
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
            
            # get these metadata for the expose
            # the index is used to set the value in the correct widget
            # and whether it is a file input
            # this matters for security reasons
            widget_index = expose["data"].get("__widget_index", -1)
            widget_is_file = expose["data"].get("__widget_is_file", False)
            widget_is_multiple_files = expose["data"].get("__widget_is_file", False)
            widget_is_seed = expose["data"].get("__widget_value_fixed", False)
            widget_is_seed_value_fixed_index = expose["data"].get("__widget_value_fixed_index", -1)
            
            # so now we can validate the parameter value
            value_to_set = request["expose"][expose_id]
            value_to_set_fixed = None
            # if it is a file input, we need to validate that the value is a valid file path
            # within the socket_file_dir and prevent path traversal attacks
            if widget_is_file and not widget_is_multiple_files:
                # local_file is a special case, it must represent a local file path within a subdirectory that is
                # specific for the given websocket session, so we must check that it is alphanumeric and does not contain any path traversal characters
                local_file_path = request["expose"][expose_id]
                local_file_path_unmodified = local_file_path
                if not isinstance(local_file_path, str) or not local_file_path.isalnum():
                    return None, False, f"Invalid local_file path for expose id {expose_id}, must be alphanumeric only"
            
                # if all checks pass, we need to convert the local_file_path to a full path
                value_to_set = path.join(socket_file_dir, local_file_path)

                # check that the file exists
                if not path.exists(value_to_set) or not path.isfile(value_to_set):
                    return None, False, f"File not found for expose id {expose_id} at file {local_file_path_unmodified}"
            elif widget_is_file and widget_is_multiple_files:
                # local_file is a special case, it must represent a local file path within a subdirectory that is
                # specific for the given websocket session, so we must check that it is alphanumeric and does not contain any path traversal characters
                local_file_paths = request["expose"][expose_id]

                if not isinstance(local_file_paths, list):
                    return None, False, f"Invalid local_files value for expose id {expose_id}, must be a list of alphanumeric file names"
                
                validated_file_paths = []
                for local_file_path in local_file_paths:
                    local_file_path_unmodified = local_file_path
                    if not isinstance(local_file_path, str) or not local_file_path.isalnum():
                        return None, False, f"Invalid local_file path for expose id {expose_id}, must be alphanumeric only"
                    # if all checks pass, we need to convert the local_file_path to a full path
                    full_path = path.join(socket_file_dir, local_file_path)

                    # check that the file exists
                    if not path.exists(full_path) or not path.isfile(full_path):
                        return None, False, f"File not found for expose id {expose_id} at file {local_file_path_unmodified}"
                    
                    validated_file_paths.append(full_path)

                value_to_set = json.dumps(validated_file_paths)
            if widget_is_seed:
                value_to_set = value_to_set.value
                value_to_set_fixed = value_to_set.value_fixed

            # set the value in the workflow copy
            # these values are just set as they come
            # get the node index to modify
            node_index = expose.get("__node_index", -1)
            if node_index == -1 or node_index >= len(workflow_copy["nodes"]):
                return None, False, "Corrupted workflow, expose node index out of range"
                
            node = workflow_copy["nodes"][node_index]
            widgetvalues = node.get("widgets_values", [])
            if widget_index == -1 or widget_index >= len(widgetvalues):
                return None, False, "Corrupted workflow, expose widget index out of range"
                
            widgetvalues[widget_index] = value_to_set

            if widget_is_seed:
                if widget_is_seed_value_fixed_index == -1 or widget_is_seed_value_fixed_index >= len(widgetvalues):
                    return None, False, "Corrupted workflow, expose seed fixed value index out of range"
                widgetvalues[widget_is_seed_value_fixed_index] = value_to_set_fixed

            node["widgets_values"] = widgetvalues
            workflow_copy["nodes"][node_index] = node
        
        return workflow_copy, True, "Workflow validated and prepared"
    
    async def on_websocket_connect(self, request):
        """
        Handles a new WebSocket connection and messages.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # make a directory for this websocket connection to store files, use the Temp directory for the operating system
        socket_file_dir = path.join(AIHUB_TEMP_DIRECTORY, "comfyui_socket_aihub_files", str(uuid.uuid4()))
        if not path.exists(socket_file_dir):
            makedirs(socket_file_dir)

        print(f"New WebSocket connection")
        PREVIOUS_BINARY_HEADER = None

        try:

            # TODO validation
            # validate_websocket_user()

            await ws.send_json({
                'type': 'INFO_LIST',
                'workflows': self.retrieve_valid_workflows_aihub_summary_cleaned(),
            })

            # Asynchronously wait for and process messages from the client
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    print(f"Received message: {msg.data}")
                    try:
                        data = json.loads(msg.data)

                        if 'ping' in data:
                            await ws.send_json({'type': 'PONG', 'value': data["ping"]})
                            self.LAST_PING_QUEUE_VALUE = str(data["ping"])

                        elif "check_exists" in data:
                            file_to_check = data["check_exists"]
                            if not isinstance(file_to_check, str) or not file_to_check.isalnum():
                                await ws.send_json({'type': 'ERROR', 'message': 'Invalid file name to check, must be alphanumeric only'})
                                continue
                            full_path = path.join(socket_file_dir, file_to_check)
                            exists = path.exists(full_path) and path.isfile(full_path)
                            await ws.send_json({'type': 'CHECK_EXISTS_STATUS', 'file': file_to_check, 'exists': exists})
                            continue

                        elif "binary_header" in data:
                            new_header = data["binary_header"]
                            if not isinstance(new_header, dict) or "type" not in new_header or "filename" not in new_header or not new_header["filename"].isalnum():
                                await ws.send_json({'type': 'ERROR', 'message': 'Invalid binary header'})
                                continue
                            PREVIOUS_BINARY_HEADER = data["binary_header"]
                            continue

                        elif 'workflow_id' not in data:
                            await ws.send_json({'type': 'ERROR', 'message': 'Missing workflow_id to execute'})

                        else:
                            # validate before queueing
                            validated_workflow, valid, message = self.validate_and_process_workflow_request(socket_file_dir, data)
                            if not valid:
                                await ws.send_json({'type': 'ERROR', 'message': message})
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
                                    'file_dir': socket_file_dir,
                                })

                            # Send a confirmation back to the external client
                            await ws.send_json({
                                'type': 'WORKFLOW_AWAIT',
                                'id': run_id,
                                'workflow_id': data.workflow_id,
                                'before_this': len(self.WORKFLOW_REQUEST_QUEUE) - 1
                            })

                    except json.JSONDecodeError:
                        await ws.send_json({'type': 'ERROR', 'message': 'Invalid JSON'})
                
                elif msg.type == web.WSMsgType.ERROR:
                    print('WebSocket connection closed with exception')

                elif msg.type == web.WSMsgType.BINARY:
                    print('WebSocket recieved a file')
                    # we now save the file to the socket_file_dir with the name specified in the previous header
                    if PREVIOUS_BINARY_HEADER is None:
                        await ws.send_json({'type': 'ERROR', 'message': 'Missing binary header before file upload'})
                        continue

                    file_name = PREVIOUS_BINARY_HEADER.get("filename", None)
                    full_path = path.join(socket_file_dir, file_name)

                    try:
                        with open(full_path, "wb") as f:
                            f.write(msg.data)
                        await ws.send_json({'type': 'FILE_UPLOAD_SUCCESS', 'file': file_name})
                    except Exception as e:
                        await ws.send_json({'type': 'ERROR', 'message': 'Error saving file'})
                        print(f"Error saving file {file_name}: {e}")

                    PREVIOUS_BINARY_HEADER = None
                else:
                    print('WebSocket received unknown message type')
                    await ws.send_json({'type': 'ERROR', 'message': 'Unknown message type'})
                    

        finally:
            print(f"WebSocket connection closed by client")

            with self.QUEUE_LOCK:
                # We first find the keys of the prompts to remove.
                for i in range(0, len(self.WORKFLOW_REQUEST_QUEUE)):
                    if self.WORKFLOW_REQUEST_QUEUE[i].ws == ws:
                        del self.WORKFLOW_REQUEST_QUEUE[i]
                
                for i in range(0, len(self.WORKFLOW_REQUEST_QUEUE)):
                    self.WORKFLOW_REQUEST_QUEUE[i].ws.send_json({
                        'status': 'WORKFLOW_AWAIT',
                        'before_this': i
                    })

                if self.CURRENTLY_RUNNING and self.CURRENTLY_RUNNING.ws == ws:
                    self.cancel_current_run()

            if not AIHUB_PERSIST_TEMPFILES:
                # clean up the temporary directory for this websocket connection
                try:
                    if path.exists(socket_file_dir):
                        for f in listdir(socket_file_dir):
                            file_path = path.join(socket_file_dir, f)
                            if path.isfile(file_path):
                                remove(file_path)
                        rmdir(socket_file_dir)
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

        # Start the server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_websocket_server(app, "0.0.0.0", WEB_SOCKET_SERVER_PORT))
        loop.close()
    
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

            threading.Thread(target=self.process_messages, daemon=True).start()
            print("Process of messages thread started.")

    def process_messages(self):
        while True:
            with self.QUEUE_LOCK:
                if len(self.WORKFLOW_REQUEST_QUEUE) > 0 and self.CURRENTLY_RUNNING is None:
                    self.CURRENTLY_RUNNING = self.WORKFLOW_REQUEST_QUEUE.pop()

                    for i in range(0, len(self.WORKFLOW_REQUEST_QUEUE)):
                        self.WORKFLOW_REQUEST_QUEUE[i].ws.send_json({
                            'status': 'WORKFLOW_AWAIT',
                            'id': self.WORKFLOW_REQUEST_QUEUE[i].id,
                            'workflow_id': self.WORKFLOW_REQUEST_QUEUE[i].request.workflow_id,
                            'before_this': i
                        })
            
                    self.process_current()

            # recheck every 100ms
            sleep(0.1)

    def cancel_current_run(self):
        # TODO cancelling the current running workflow operation
        pass

    def process_current(self):
        #TODO do processing of the current queued workflow operation
        pass

    async def send_binary_data(self, ws, binary_data, data_type, action_data):
        """
        Sends binary data (e.g., images, files) to the client via websocket.
        """
        # Send a header message to indicate incoming binary data
        await ws.send_json({
            "type": "FILE",
            "type": data_type,
            "action": action_data,
        })
        # Send the binary data itself
        await ws.send_bytes(binary_data)

    async def send_binary_data_to_current_client(self, binary_data, data_type, action_data):
        return await self.send_binary_data(self.CURRENTLY_RUNNING["ws"], binary_data, data_type, action_data)
    
    async def send_json(self, ws, data):
        """
        Sends JSON data to the client via websocket.
        """
        await ws.send_json(data)

    async def send_json_to_current_client(self, data):
        return await self.send_json(self.CURRENTLY_RUNNING["ws"], data)
    
    async def send_status_message_to_current_client(self, status_message, extra_data=None):
        data = {
            "type": "STATUS",
            "message": status_message
        }
        if extra_data:
            data.update(extra_data)
        return await self.send_json(self.CURRENTLY_RUNNING["ws"], data)