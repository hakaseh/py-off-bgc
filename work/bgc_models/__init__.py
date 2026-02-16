# bgc_models/__init__.py

# 1. Import all your models here
from .npzd_sasai import Model_NPZD
# from .nemuro_kishi import Model_NEMURO   # Uncomment when ready
# from .enemuro_yoshie import Model_eNEMURO       # Future models...

# 2. Create a Registry (Dictionary mapping string names to Classes)
MODEL_REGISTRY = {
    "NPZD": Model_NPZD,
    # "NEMURO": Model_NEMURO,
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