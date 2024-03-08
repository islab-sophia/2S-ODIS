import numpy as np
import torch
def to_code(x,encoder,quantizer,device):
    if x.device != device:
        x = x.to(device)
    if len(x.shape)==3:
        x = x.unsqueeze(0)
    out = encoder(x)
    _, _, (_, _, code) = quantizer(out)
    return code.cpu().detach().numpy()

def code_to_img(z_q,quantizer,decoder,device,shape=(16,32)):
    if type(z_q) == np.ndarray:
        z_q = torch.from_numpy(z_q)
    z_q = z_q.reshape(-1,*shape).to(device)
    x = quantizer.get_codebook_entry(z_q, (-1, *shape, 256))
    recon = decoder(x)
    return torch.clip(recon.cpu().detach()/2+0.5,0,1)