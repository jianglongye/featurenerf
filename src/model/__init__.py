from .models import PixelNeRFNet
from .models_embed import PixelNeRFEmbedNet
from .models_semantic import PixelNeRFSemanticNet


def make_model(conf, *args, **kwargs):
    """Placeholder to allow more model types"""
    model_type = conf.get_string("type", "pixelnerf")  # single
    if model_type == "pixelnerf":
        net = PixelNeRFNet(conf, *args, **kwargs)
    elif model_type == "pixelnerfembed":
        net = PixelNeRFEmbedNet(conf, *args, **kwargs)
    elif model_type == "pixelnerfsemantic":
        net = PixelNeRFSemanticNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
