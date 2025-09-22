from .vantage_project import VantageProject
from .vantage_server import preload  
from .single_looper import VantageSingleLooperI2V
from .dual_looper import VantageDualLooperI2V

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "VantageProject": VantageProject,
    "VantageI2VSingleLooper": VantageSingleLooperI2V,
    "VantageI2VDualLooper": VantageDualLooperI2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VantageProject": "Vantage Project Loader",
    "VantageI2VSingleLooper": "Vantage I2V Single Model Looper",
    "VantageI2VDualLooper": "Vantage I2V Dual Model Looper",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "preload"]

