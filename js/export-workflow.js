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
const postExportedForLocale = (json, workflow_id, locale) => fetch(`/aihub_workflows/${workflow_id}/locale/${locale}`, { method: "POST", body: JSON.stringify(json) });

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

const validTypes = ["INT", "FLOAT", "BOOLEAN", "STRING"];
const validModifiersForType = {
    "INT": ["SORTED", "UNIQUE"],
    "FLOAT": ["SORTED", "UNIQUE"],
    "BOOLEAN": ["ONE_TRUE", "ONE_FALSE"],
    "STRING": ["UNIQUE", "MULTILINE"]
}

const validSpecialModifiersForType = {
    "INT": ["MAX", "MIN", "DEFAULT", "MAXOFFSET", "MINOFFSET"],
    "FLOAT": ["MAX", "MIN", "DEFAULT", "MAXOFFSET", "MINOFFSET"],
    "BOOLEAN": ["DEFAULT"],
    "STRING": ["MAXLEN", "MINLEN", "MAXLENOFFSET", "MINLENOFFSET"]
}

const validSpecialModifiersForTypeSupportExposeAsValue = {
    "INT": ["MAX", "MIN"],
    "FLOAT": ["MAX", "MIN"],
    "BOOLEAN": [],
    "STRING": ["MAXLEN", "MINLEN"]
}

const validSpecialModifiersValueTypeValidatorForType = {
    "INT": parseInt,
    "FLOAT": parseFloat,
    "BOOLEAN": null,
    "STRING": parseInt
}

const validSpecialModifiersValueTypeNodeClassType = {
    "INT": ["AIHubExposeInteger", "AIHubExposeProjectInteger"],
    "FLOAT": ["AIHubExposeInteger", "AIHubExposeFloat", "AIHubExposeProjectFloat", "AIHubExposeProjectInteger"],
    "BOOLEAN": null,
    "STRING": ["AIHubExposeInteger", "AIHubExposeProjectInteger"]
}

function validateMetadataFieldLine(nodeInputId, line, lineNumber, exported) {
    if (!line || line.trim() === "") {
        const dialog = new ComfyDialog()
        dialog.show("Validation Error: The metadata field line in node " + nodeInputId + " is empty at line " + (lineNumber + 1));
        return false;
    }
    const parts = line.split(" ").map(s => s.trim()).filter(s => s !== "");
    const fieldId = parts[0];
    const type = parts[1];
    const modifiers = parts.slice(2);
    if (!idValidator(fieldId)) {
        const dialog = new ComfyDialog()
        dialog.show("Validation Error: The metadata field id '" + fieldId + "' in node " + nodeInputId + " is invalid, it must be alphanumeric or underscores, and between 3 and 50 characters at line " + (lineNumber + 1));
        return false;
    }
    
    if (!validTypes.includes(type)) {
        const dialog = new ComfyDialog()
        dialog.show("Validation Error: The metadata field type '" + type + "' in node " + nodeInputId + " is invalid, it must be one of: " + validTypes.join(", ") + " at line " + (lineNumber + 1));
        return false;
    }

    let max = null
    let min = null
    let maxlen = null
    let minlen = null
    let one_true = false
    let one_false = false
    for (const modifier of modifiers) {
        const specialModifier = modifier.split(":")[0];
        if (!validModifiersForType[type].includes(modifier) && !validSpecialModifiersForType[type].includes(specialModifier)) {
            const dialog = new ComfyDialog()
            dialog.show("Validation Error: The metadata field modifier '" + modifier + "' in node " + nodeInputId + " is invalid for type " + type + ", valid modifiers are: " + validModifiersForType[type].join(", ") +
                (validSpecialModifiersForType[type].length > 0 ? " and special modifiers: " + validSpecialModifiersForType[type].join(", ") : "") + " at line " + (lineNumber + 1));
            return false;
        }
        const isSpecialModifier = validSpecialModifiersForType[type].includes(specialModifier);
        if (isSpecialModifier) {
            const valuePart = modifier.split(":")[1];
            if (!valuePart) {
                const dialog = new ComfyDialog()
                dialog.show("Validation Error: The metadata field special modifier '" + modifier + "' in node " + nodeInputId + " is missing a value after the colon, or missing the colon entirely");
                return false;
            }
            const valueValidator = validSpecialModifiersValueTypeValidatorForType[type];
            if (valueValidator) {
                const value = valueValidator(valuePart);
                if (isNaN(value)) {
                    if (!validSpecialModifiersForTypeSupportExposeAsValue[type].includes(specialModifier)) {
                        const dialog = new ComfyDialog()
                        dialog.show("Validation Error: The metadata field special modifier '" + modifier + "' in node " + nodeInputId + " is not supported for expose as value");
                        return false;
                    }
                    // check if there is a node with that id and of the correct class type
                    const nodeClassTypes = validSpecialModifiersValueTypeNodeClassType[type];
                    let found = false;
                    for (const nodeId in exported) {
                        const node = exported[nodeId];
                        if (nodeClassTypes.includes(node.class_type) && node.inputs.id === valuePart) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        const dialog = new ComfyDialog()
                        dialog.show("Validation Error: The metadata field special modifier '" + modifier + "' in node " + nodeInputId + " has an invalid value '" + valuePart + "', it must be a valid node id or a valid number for the type");
                        return false;
                    }
                } else {
                    // store the value for further validation if needed
                    if (specialModifier === "MAX") {
                        max = value;
                    } else if (specialModifier === "MIN") {
                        min = value;
                    } else if (specialModifier === "MAXLEN") {
                        maxlen = value;
                    } else if (specialModifier === "MINLEN") {
                        minlen = value;
                    }
                }
            }
        } else {
            if (type === "BOOLEAN") {
                if (modifier === "ONE_TRUE") {
                    one_true = true;
                } else if (modifier === "ONE_FALSE") {
                    one_false = true;
                }
            }
        }

        // check if no conflicting modifiers are set
        if (type === "BOOLEAN" && one_true && one_false) {
            const dialog = new ComfyDialog()
            dialog.show("Validation Error: The metadata field in node " + nodeInputId + " has conflicting modifiers ONE_TRUE and ONE_FALSE set at line " + (lineNumber + 1));
            return false;
        } else if ((type === "INT" || type === "FLOAT") && max !== null && min !== null && max < min) {
            const dialog = new ComfyDialog()
            dialog.show("Validation Error: The metadata field in node " + nodeInputId + " has conflicting modifiers MAX and MIN set, MAX is less than MIN at line " + (lineNumber + 1));
            return false;
        } else if (type === "STRING" && maxlen !== null && minlen !== null && maxlen < minlen) {
            const dialog = new ComfyDialog()
            dialog.show("Validation Error: The metadata field in node " + nodeInputId + " has conflicting modifiers MAXLEN and MINLEN set, MAXLEN is less than MINLEN at line " + (lineNumber + 1));
            return false;
        }
    }

    return true;
}

function validateWorkflow(exported, modelsAndLoras) {
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
        if (node.class_type.startsWith("AIHubExpose") && node.inputs.id && typeof node.inputs.id === "string" && node.inputs.id.trim() !== "") {
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


    const allGood = Object.keys(exported).every(nodeId => {
        const node = exported[nodeId];
        if (!node.class_type.startsWith("AIHub")) {
            return true; // skip validation for non-AIHub nodes
        }
        const nodeIdValue = node.inputs.id;
        if ("batch_index" in node.inputs) {
            // batch_index is a string, check if that string value is a valid integer
            const batchIndex = node.inputs.batch_index;
            if (batchIndex && isNaN(parseInt(batchIndex))) {
                const dialog = new ComfyDialog()
                dialog.show("Validation Error: The batch_index value in node " + nodeIdValue + " is not a valid integer");
                return false;
            }
        }

        if ("indexes" in node.inputs) {
            // indexes is a string, check if that string value is a valid comma separated list of integers or ranges
            const indexes = node.inputs.indexes;
            if (indexes) {
                const parts = indexes.split(",");
                for (const part of parts) {
                    const rangeParts = part.split(":");
                    if (rangeParts.length > 2) {
                        const dialog = new ComfyDialog()
                        dialog.show("Validation Error: The indexes value in node " + nodeIdValue + " is not a valid range");
                        return false;
                    }
                    for (const rangePart of rangeParts) {
                        if (isNaN(parseInt(rangePart))) {
                            const dialog = new ComfyDialog()
                            dialog.show("Validation Error: The indexes value in node " + nodeIdValue + " is not a valid integer");
                            return false;
                        }
                    }
                }
            }
        }

        if ("file_name" in node.inputs) {
            // file_name must not contain special characters
            const fileName = node.inputs.file_name;
            if (fileName) {
                const regex = /^[a-zA-Z_\-\.]+$/;
                if (!regex.test(fileName)) {
                    const dialog = new ComfyDialog()
                    // may not have nodeIdValue set because Action nodes do not require an id
                    if (typeof nodeIdValue === "undefined" || nodeIdValue === null || nodeIdValue === "") {
                        dialog.show("Validation Error: The file_name value in node class " + node.class_type + " contains invalid characters, only letters, underescores and dashes are allowed: " + fileName);
                    } else {
                        dialog.show("Validation Error: The file_name value in node " + nodeIdValue + " contains invalid characters, only letters, underescores and dashes are allowed: " + fileName);
                    }
                    return false;
                }
            }
        }

        if ("options_label" in node.inputs) {
            // check that options_label is a newline separated list of labels matching the number of options
            const optionsLabel = node.inputs.options_label;
            const options = node.inputs.options;
            if (optionsLabel) {
                if (optionsLabel.split("\n").length !== options.split("\n").length) {
                    const dialog = new ComfyDialog()
                    dialog.show("Validation Error: The options_label in node " + nodeIdValue + " does not match the number of options, empty lines count as labels and options");
                    return false;
                }
            }
        }

        const allPassed = ["max_expose_id", "min_expose_id", "maxlen_expose_id", "minlen_expose_id"].every(key => {
            if (key in node.inputs) {
                const exposeId = node.inputs[key];
                if (exposeId && typeof exposeId !== "string") {
                    const dialog = new ComfyDialog()
                    dialog.show("Validation Error: The " + key + " value in node " + nodeIdValue + " must be a string representing the expose id");
                    return false;
                }
                if (!exposeId) {
                    return true; // empty expose id is allowed
                }
                // find the expose to see if it exists and is of the correct type
                let found = false;
                for (const otherNodeId in exported) {
                    const otherNode = exported[otherNodeId];
                    if (otherNode.class_type.startsWith("AIHubExpose") && otherNode.inputs.id === exposeId) {
                        found = true;
                        // Check if the exposed node is of the correct type
                        if (otherNode.class_type !== "AIHubExposeInteger" && otherNode.class_type !== "AIHubExposeProjectConfigInteger" && otherNode.class_type !== "AIHubExposeFloat" && otherNode.class_type !== "AIHubExposeProjectConfigFloat") {
                            const dialog = new ComfyDialog()
                            dialog.show("Validation Error: The " + key + " value in node " + nodeIdValue + " is not of the correct type, it must be an exposed integer or float");
                            return false;
                        }
                    }
                }

                if (!found) {
                    const dialog = new ComfyDialog()
                    dialog.show("Validation Error: The " + key + " value in node " + nodeIdValue + " does not correspond to any exposed integer or float");
                    return false;
                }
            }

            return true
        });

        if (!allPassed) {
            return false;
        }

        if (node.class_type === "AIHubExposeModel" || node.class_type === "AIHubExposeModelSimple") {
            const potentialModel = node.inputs.model;
            if (potentialModel && !modelsAndLoras.checkpoints.includes(potentialModel) && !modelsAndLoras.diffusion_models.includes(potentialModel)) {
                const dialog = new ComfyDialog()
                dialog.show("Validation Error: The model '" + potentialModel + "' in node " + nodeIdValue + " is not available on the server");
                return false;
            } else if (potentialModel) {
                // check if the model is a diffusion model
                const isDiffusionModel = modelsAndLoras.diffusion_models.includes(potentialModel);
                if (isDiffusionModel && node.class_type === "AIHubExposeModel" && !node.inputs.is_diffusion_model) {
                    const dialog = new ComfyDialog()
                    dialog.show("Validation Error: The model '" + potentialModel + "' in node " + nodeIdValue + " is a diffusion model, but the is_diffusion_model input is not set to true");
                    return false;
                }
            }

            const potentialLoras = node.inputs.loras || "";
            const potentialLorasStrengths = node.inputs.loras_strengths || "";
            const potentialLorasUseLoaderModelOnly = node.inputs.loras_use_loader_model_only || "";
            if (potentialLoras || potentialLorasStrengths || potentialLorasUseLoaderModelOnly) {
                const loraList = potentialLoras.split(",").map(s => s.trim()).filter(s => s !== "");
                const loraStrengthList = potentialLorasStrengths.split(",").map(s => s.trim()).filter(s => s !== "");
                const loraUseLoaderModelOnlyList = potentialLorasUseLoaderModelOnly.split(",").map(s => s.trim()).filter(s => s !== "");

                if (loraStrengthList.length > 0 && loraStrengthList.length !== loraList.length) {
                    const dialog = new ComfyDialog()
                    dialog.show("Validation Error: The number of lora strengths does not match the number of loras in node " + nodeIdValue);
                    return false;
                }

                if (loraUseLoaderModelOnlyList.length > 0 && loraUseLoaderModelOnlyList.length !== loraList.length) {
                    const dialog = new ComfyDialog()
                    dialog.show("Validation Error: The number of lora use_loader_model_only flags does not match the number of loras in node " + nodeIdValue);
                    return false;
                }

                // ensure all the loras strengths are valid floats between 0 and 1
                for (const strength of loraStrengthList) {
                    const strengthValue = parseFloat(strength);
                    if (isNaN(strengthValue) || strengthValue < 0 || strengthValue > 1) {
                        const dialog = new ComfyDialog()
                        dialog.show("Validation Error: The lora strength '" + strength + "' in node " + nodeIdValue + " is not a valid float between 0 and 1");
                        return false;
                    }
                }

                // ensure all the loras use_loader_model_only are either t or f
                for (const useLoaderModelOnly of loraUseLoaderModelOnlyList) {
                    if (useLoaderModelOnly !== "t" && useLoaderModelOnly !== "f") {
                        const dialog = new ComfyDialog()
                        dialog.show("Validation Error: The lora use_loader_model_only flag '" + useLoaderModelOnly + "' in node " + nodeIdValue + " is not valid, must be 't' or 'f'");
                        return false;
                    }
                }

                for (const lora of loraList) {
                    if (!modelsAndLoras.loras.includes(lora)) {
                        const dialog = new ComfyDialog()
                        dialog.show("Validation Error: The lora '" + lora + "' in node " + nodeIdValue + " is not available on the server");
                        return false;
                    }
                }
            }
        }

        if ("metadata_fields_label" in node.inputs) {
            // check that metadata_fields_label is a newline separated list of labels matching the number of metadata fields
            const metadataFieldsLabel = node.inputs.metadata_fields_label;
            const metadataFields = node.inputs.metadata_fields;
            if (metadataFieldsLabel || metadataFields) {
                const allValidHere = metadataFieldsLabel.split("\n").every((v, index) => {
                    if (!v.trim()) {
                        const dialog = new ComfyDialog()
                        dialog.show("Validation Error: The metadata_fields_label in node " + nodeIdValue + " has an empty metadata field label at line " + index);
                        return false;
                    }
                    return true;
                })
                if (!allValidHere) {
                    return false;
                }
                if (metadataFieldsLabel.split("\n").length !== metadataFields.split("\n").length) {
                    const dialog = new ComfyDialog()
                    dialog.show("Validation Error: The metadata_fields_label in node " + nodeIdValue + " does not match the number of metadata fields, empty lines count as labels and fields");
                    return false;
                }
            }
        }

        if ("metadata_fields" in node.inputs) {
            // validate if metadata fields are valid
            const metadataFields = node.inputs.metadata_fields;
            const lines = metadataFields.split("\n");
            if (metadataFields.length !== 0) {
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    const valid = validateMetadataFieldLine(nodeIdValue, line, i, exported);
                    if (!valid) {
                        return false;
                    }
                }
            }
        }

        if (node.class_type.startsWith("AIHubExpose")) {
            // validate that no AIHubExpose nodes have connected values to their inputs since they must be static
            // connected values are basically set up as arrays in the inputs object
            const exceptions = ["normalizer"]
            for (const inputKey in node.inputs) {
                const inputValue = node.inputs[inputKey];
                if (Array.isArray(inputValue) && !exceptions.includes(inputKey)) {
                    const dialog = new ComfyDialog()
                    dialog.show("Validation Error: The input '" + inputKey + "' in node " + nodeIdValue + " is connected, but expose nodes must have static values for all basic inputs");
                    return false;
                }
            }
        }

        return true;
    });
    if (!allGood) {
        return false;
    }

    return workflowId;
}

function cleanWorkflowForLocaleData(exported) {
    // remove all nodes that are not AIHubExpose* or AIHubWorkflowController
    const cleaned = {};
    Object.keys(exported).forEach(nodeId => {
        const node = exported[nodeId];
        if (node.class_type.startsWith("AIHubExpose") || node.class_type === "AIHubWorkflowController" || node.class_type === "AIHubAddRunCondition") {
            cleaned[nodeId] = {}
            Object.keys(node.inputs).forEach(key => {
                if (["description", "name", "tooltip", "label", "options_label", "category", "metadata_fields_label", "error"].includes(key)) {
                    cleaned[nodeId][key] = node.inputs[key];
                }
            });
        }
    });
    return cleaned;
}

app.registerExtension({
    name: "comfyui-aihub-workflow-exposer",
    nodeCreated(node) {
        if (node.comfyClass === "AIHubWorkflowController") {
            node.addWidget("button", "Validate Workflow", "EXPORT", async () => {
                // call comfyui native export API function
                const exported = await app.graphToPrompt();
                const modelsAndLoras = await fetch("/aihub_list_models_and_loras").then(res => res.json());
                const workflowId = validateWorkflow(exported.output, modelsAndLoras)
                if (workflowId) {
                    const dialog = new ComfyDialog()
                    dialog.show("Validation Successful: The workflow is valid and ready for export.");
                    dialog.textElement.style.color = "green";
                }
            });
            node.addWidget("button", "Export to AIHub Workflows", "EXPORT", async () => {
                // call comfyui native export API function
                const exported = await app.graphToPrompt();
                const modelsAndLoras = await fetch("/aihub_list_models_and_loras").then(res => res.json());
                const workflowId = validateWorkflow(exported.output, modelsAndLoras);
                if (workflowId) {
                    await postExported(exported.output);
                    await postExportedForLocale(cleanWorkflowForLocaleData(exported.output), workflowId, "default");
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