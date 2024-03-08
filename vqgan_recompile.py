import torch
import importlib
from omegaconf import OmegaConf
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    if sd is not None:
        model.load_state_dict(sd)
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}

def load_model(config, ckpt, gpu, eval_mode):
    # load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
        print(f"loaded model from global step {global_step}.")
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"], gpu=gpu, eval_mode=eval_mode)["model"]
    return model, global_step

if __name__ == "__main__":
    ckpt = "vqgan.ckpt"
    base_configs = "vqgan_config.yaml"

    configs = OmegaConf.load(base_configs)
    model, global_step = load_model(configs, ckpt, gpu=False, eval_mode=True)

    encoder = torch.nn.Sequential(model.encoder, model.quant_conv)
    decoder = torch.nn.Sequential(model.post_quant_conv, model.decoder)
    quantizer = model.quantize
    torch.save((encoder, decoder, quantizer), "vqvae_models.pth")
