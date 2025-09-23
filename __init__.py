from .nodes import *

NODE_CLASS_MAPPINGS = {
    "AIHubWorkflowController": AIHubWorkflowController,

    "AIHubExposeInteger": AIHubExposeInteger,
    "AIHubExposeFloat": AIHubExposeFloat,
    "AIHubExposeBoolean": AIHubExposeBoolean,
    "AIHubExposeString": AIHubExposeString,
    "AIHubExposeStringSelection": AIHubExposeStringSelection,
    "AIHubExposeSeed": AIHubExposeSeed,
    "AIHubExposeImage": AIHubExposeImage,
    "AIHubExposeImageInfoOnly": AIHubExposeImageInfoOnly,
    "AIHubExposeImageBatch": AIHubExposeImageBatch,
    "AIHubExposeScheduler": AIHubExposeScheduler,
    "AIHubExposeExtendableScheduler": AIHubExposeExtendableScheduler,
    "AIHubExposeSampler": AIHubExposeSampler,
    "AIHubExposeCfg": AIHubExposeCfg,
    "AIHubExposeSteps": AIHubExposeSteps,
    "AIHubExposeModel": AIHubExposeModel,

    "AIHubExposeProjectConfigInteger": AIHubExposeProjectConfigInteger,
    "AIHubExposeProjectConfigFloat": AIHubExposeProjectConfigFloat,
    "AIHubExposeProjectConfigBoolean": AIHubExposeProjectConfigBoolean,
    "AIHubExposeProjectConfigString": AIHubExposeProjectConfigString,

    "AIHubPatchActionSetProjectConfigInteger": AIHubPatchActionSetProjectConfigInteger,
    "AIHubPatchActionSetProjectConfigFloat": AIHubPatchActionSetProjectConfigFloat,
    "AIHubPatchActionSetProjectConfigBoolean": AIHubPatchActionSetProjectConfigBoolean,
    "AIHubPatchActionSetProjectConfigString": AIHubPatchActionSetProjectConfigString,

    "AIHubActionNewImage": AIHubActionNewImage,
    "AIHubActionNewLayer": AIHubActionNewLayer,

    "AIHubUtilsCropMergedImageToLayerSize": AIHubUtilsCropMergedImageToLayerSize,
    "AIHubUtilsFitLayerToMergedImage": AIHubUtilsFitLayerToMergedImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIHubWorkflowController": "AIHub Workflow Controller",

    "AIHubExposeInteger": "AIHub Expose Integer",
    "AIHubExposeFloat": "AIHub Expose Float",
    "AIHubExposeBoolean": "AIHub Expose Boolean",
    "AIHubExposeString": "AIHub Expose String",
    "AIHubExposeStringSelection": "AIHub Expose String Selection",
    "AIHubExposeSeed": "AIHub Expose Seed",
    "AIHubExposeImage": "AIHub Expose Image",
    "AIHubExposeImageInfoOnly": "AIHub Expose Image (Info Only)",
    "AIHubExposeImageBatch": "AIHub Expose Image Batch",
    "AIHubExposeScheduler": "AIHub Expose Scheduler",
    "AIHubExposeExtendableScheduler": "AIHub Expose Extendable Scheduler",
    "AIHubExposeSampler": "AIHub Expose Sampler",
    "AIHubExposeCfg": "AIHub Expose CFG",
    "AIHubExposeSteps": "AIHub Expose Steps",
    "AIHubExposeModel": "AIHub Expose Model",

    "AIHubExposeProjectConfigInteger": "AIHub Expose Project Config Integer",
    "AIHubExposeProjectConfigFloat": "AIHub Expose Project Config Float",
    "AIHubExposeProjectConfigBoolean": "AIHub Expose Project Config Boolean",
    "AIHubExposeProjectConfigString": "AIHub Expose Project Config String",

    "AIHubActionNewImage": "AIHub Action New Image",
    "AIHubActionNewLayer": "AIHub Action New Layer",

    "AIHubPatchActionSetProjectConfigInteger": "AIHub Patch Action Set Project Config Integer",
    "AIHubPatchActionSetProjectConfigFloat": "AIHub Patch Action Set Project Config Float",
    "AIHubPatchActionSetProjectConfigBoolean": "AIHub Patch Action Set Project Config Boolean",
    "AIHubPatchActionSetProjectConfigString": "AIHub Patch Action Set Project Config String",

    "AIHubUtilsCropMergedImageToLayerSize": "AIHub Utils Crop Merged Image To Layer Size",
    "AIHubUtilsFitLayerToMergedImage": "AIHub Utils Fit Layer To Merged Image",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]