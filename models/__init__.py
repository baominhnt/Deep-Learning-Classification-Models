from .densenet import DenseNet121
from .mobilenet import MobileNetV2
from .efficientnet import EfficientNetB0

def build_model(model_config):
    """Factory builder that instantiates custom neural network architectures 
    dynamically based on configuration definitions."""
    model_name = model_config['name'].lower()
    num_classes = model_config['num_classes']
    
    if model_name == "densenet":
        return DenseNet121(num_classes=num_classes, drop_rate=model_config.get('drop_rate', 0.2))
    elif model_name == "mobilenet":
        return MobileNetV2(num_classes=num_classes, width_mult=model_config.get('width_mult', 1.0))
    elif model_name == "efficientnet":
        return EfficientNetB0(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model architecture request: '{model_name}'")