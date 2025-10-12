import { app } from "../../scripts/app.js";
import { ComfyDialog } from "../../scripts/ui.js";
import { api } from "../../scripts/api.js";

function uploadFileFixed(mimeType) {
    return new Promise((resolve, reject) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = mimeType;
        input.onchange = () => {
            const file = input.files[0];
            if (file) {
                resolve(file);
            } else {
                resolve(null);
            }
        };
        input.addEventListener('error', (e) => { reject(e); });
        input.addEventListener('abort', (e) => { resolve(null); });
        input.addEventListener('cancel', (e) => { resolve(null); });
        input.click();
    });
}

const postExported = (json) => fetch("/aihub_workflows", { method: "POST", body: JSON.stringify(json) });

function idValidator(id) {
    // workflow and expose ids must be alphanumeric or underscores, and between 3 and 50 characters
    const regex = /^[a-zA-Z0-9_]{3,50}$/;
    return regex.test(id);
}

function getWorkflowId(exported) {
    // Find the workflow controller node and return its workflow_id input value
    for (const nodeId in exported) {
        const node = exported[nodeId];
        if (node.class_type === "AIHubWorkflowController") {
            return [node.inputs.id, node.inputs.project_type, node.inputs.project_init];
        }
    }
    return [null, null, null];
}

function validateWorkflow(exported) {
    const [workflowId, projectType, projectInit] = getWorkflowId(exported);
    if (!workflowId || workflowId.trim() === "") {
        const dialog = new ComfyDialog()
        dialog.show("Validation Error: The workflow must have a valid workflow id set in the AIHubWorkflowController node");
        return false;
    }
    
    if (projectType && projectType.trim() !== "" && !idValidator(projectType)) {
        const dialog = new ComfyDialog()
        dialog.show("Validation Error: The project type is invalid, it must be alphanumeric or underscores, and between 3 and 50 characters");
        return false;
    }

    if (projectInit && !projectType) {
        const dialog = new ComfyDialog()
        dialog.show("Validation Error: The project init is set but the project type is not, you must set a project type if you set a project init");
        return false;
    }

    // Perform validation on the exported workflow
    // ensure that the exported workflow has at least one output action node that starts with AIHubAction
    const hasOutputAction = Object.keys(exported).some(nodeId => {
        const node = exported[nodeId];
        return node.class_type.startsWith("AIHubAction");
    });

    if (!hasOutputAction) {
        const dialog = new ComfyDialog()
        dialog.show("Validation Error: The workflow must contain at least one output action node");
        return false;
    }

    // ensure that each AIHubExpose that has an id set has a unique id
    const ids = new Set();
    const repeating_ids = new Set();
    Object.keys(exported).forEach(nodeId => {
        const node = exported[nodeId];
        if (node.class_type.startsWith("AIHubExpose") && node.inputs.id && node.inputs.id.trim() !== "") {
            if (ids.has(node.inputs.id.trim())) {
                repeating_ids.add(node.inputs.id.trim());
            }
            ids.add(node.inputs.id.trim());
        }
    });

    const invalidIds = Array.from(ids).filter(id => {
        return !idValidator(id);
    });

    if (invalidIds.length > 0) {
        const dialog = new ComfyDialog()
        dialog.show("Validation Error: The following expose ids are invalid: " + invalidIds.join(", "));
        return false;
    }

    if (repeating_ids.size > 0) {
        const dialog = new ComfyDialog()
        dialog.show("Validation Error: The workflow contains repeating expose ids: " + Array.from(repeating_ids).join(", "));
        return false;
    }

    const shouldNotHaveExposeProjectTypes = projectInit || !projectType;
    const exposeProjectTypes = Object.keys(exported).filter(nodeId => {
        const node = exported[nodeId];
        return node.class_type.startsWith("AIHubExposeProject");
    });

    if (shouldNotHaveExposeProjectTypes && exposeProjectTypes.length > 0) {
        const dialog = new ComfyDialog()
        dialog.show("Validation Error: The workflow has a project init or no project type, it should not contain any AIHubExposeProject type nodes");
        return false;
    }

    // TODO validate AIHubExposeModel and AIHubExposeModelSimple nodes to see if the values are valid models and loras
    // TODO validate AIHubExposeImageBatch node to see if metadata fields are valid
    // TODO validate that no AIHubExpose nodes have connected values to their inputs since they must be static
    // TODO these validations should allow for exceptions as only id and file_name are required to be static, other fields can be connected

    return workflowId;
}

app.registerExtension({
    name: "comfyui-aihub-workflow-exposer",
    nodeCreated(node) {
        if (node.comfyClass === "AIHubWorkflowController") {
            node.addWidget("button", "Validate Workflow", "EXPORT", async () => {
                // call comfyui native export API function
                const exported = await app.graphToPrompt();
                const workflowId = validateWorkflow(exported.output)
                if (workflowId) {
                    const dialog = new ComfyDialog()
                    dialog.show("Validation Successful: The workflow is valid and ready for export.");
                    dialog.textElement.style.color = "green";
                }
            });
            node.addWidget("button", "Export to AIHub Workflows", "EXPORT", async () => {
                // call comfyui native export API function
                const exported = await app.graphToPrompt();
                const workflowId = validateWorkflow(exported.output);
                if (workflowId) {
                    await postExported(exported.output);
                    // this is a file object
                    try {
                        const potentialImage = await uploadFileFixed("image/png");
                        if (potentialImage) {
                            await fetch("/aihub_workflows/" + workflowId + "/image", { method: "POST", body: potentialImage });
                            const dialog = new ComfyDialog()
                            dialog.show("Export Successful: The workflow was exported with an image.");
                            dialog.textElement.style.color = "green";
                        } else {
                            const dialog = new ComfyDialog()
                            dialog.show("Export Successful: The workflow was exported without an image, if an image already exists it was not changed.");
                            dialog.textElement.style.color = "green";
                        }
                    } catch (e) {
                        const dialog = new ComfyDialog()
                        dialog.show("Export Successful: The workflow was exported without an image due to an error: " + e.message);
                        dialog.textElement.style.color = "orange";
                    }
                }
            });
        }
    }
});