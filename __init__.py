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
    "AIHubExposeSampler": AIHubExposeSampler,

    "AIHubExposeConfigInteger": AIHubExposeConfigInteger,
    "AIHubExposeConfigFloat": AIHubExposeConfigFloat,
    "AIHubExposeConfigBoolean": AIHubExposeConfigBoolean,
    "AIHubExposeConfigString": AIHubExposeConfigString,

    "AIHubPatchActionSetConfigInteger": AIHubPatchActionSetConfigInteger,
    "AIHubPatchActionSetConfigFloat": AIHubPatchActionSetConfigFloat,
    "AIHubPatchActionSetConfigBoolean": AIHubPatchActionSetConfigBoolean,
    "AIHubPatchActionSetConfigString": AIHubPatchActionSetConfigString,
    
    "AIHubActionNewImage": AIHubActionNewImage,
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
    "AIHubExposeSampler": "AIHub Expose Sampler",

    "AIHubExposeConfigInteger": "AIHub Expose Config Integer",
    "AIHubExposeConfigFloat": "AIHub Expose Config Float",
    "AIHubExposeConfigBoolean": "AIHub Expose Config Boolean",
    "AIHubExposeConfigString": "AIHub Expose Config String",

    "AIHubActionNewImage": "AIHub Action New Image",

    "AIHubPatchActionSetConfigInteger": "AIHub Patch Action Set Config Integer",
    "AIHubPatchActionSetConfigFloat": "AIHub Patch Action Set Config Float",
    "AIHubPatchActionSetConfigBoolean": "AIHub Patch Action Set Config Boolean",
    "AIHubPatchActionSetConfigString": "AIHub Patch Action Set Config String",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]