# bgc_models/__init__.py

# 1. Import all your models here
from .npzd_sasai import Model_NPZD
from .nemuro import Model_NEMURO
from .no_sms import Model_NO_SMS
from .fennel06 import Model_Fennel06
from .flexpft import Model_FlexPFT
# from .enemuro_yoshie import Model_eNEMURO       # Future models...

# 2. Create a Registry (Dictionary mapping string names to Classes)
MODEL_REGISTRY = {
    "NPZD": Model_NPZD,
    "NEMURO": Model_NEMURO,
    "NO_SMS": Model_NO_SMS,
    "Fennel06": Model_Fennel06,
    "FlexPFT": Model_FlexPFT,
    # "eNEMURO": Model_eNEMURO
}

# 3. Create a helper function (The "Factory")
def get_model(model_name, nz, ny, nx, water_mask, params=None):
    """
    Fetches and initializes the requested BGC model.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found! Available models: {available}")
    
    # Grab the class from the dictionary and instantiate it
    ModelClass = MODEL_REGISTRY[model_name]
    return ModelClass(nz, ny, nx, water_mask, params)
