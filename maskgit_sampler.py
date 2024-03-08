import math
import numpy as np
import torch
import torch.nn.functional as F

class MaskGitSampler:
    def __init__(self,codebook_num,sampling_iter=16,is_second_stage=False):
        self.codebook_num = codebook_num
        self.gamma = lambda r:np.cos(r * np.pi / 2)
        self.CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf])
        self.sampling_iter = sampling_iter
        self.is_second_stage = is_second_stage

    def to(self,device):
        self.CONFIDENCE_OF_KNOWN_TOKENS = self.CONFIDENCE_OF_KNOWN_TOKENS.to(device)
        return self

    def noise(self,token):
        shape = token.shape
        token = token.clone()
        if self.is_second_stage:
            token= token.reshape(shape[0]//26,-1)
        else:
            token= token.reshape(shape[0],-1)
        r = math.floor(self.gamma(np.random.uniform()) * token.shape[1])
        sample = torch.rand(token.shape, device=token.device).topk(r, dim=1).indices
        mask = torch.zeros(token.shape, dtype=torch.bool, device=token.device)
        mask.scatter_(dim=1, index=sample, value=True)
        token[mask] = self.codebook_num
        return token.reshape(*shape)

    def mask_by_random_topk(self,mask_len, probs, temperature=0.1):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(probs.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        masking = (confidence < cut_off)
        return masking

    def sampling(self,model,batch_size,device,cond):
        if self.is_second_stage:
            token = self.codebook_num*torch.ones((batch_size*26,16,16),device=device,dtype=torch.int)
            flat_shape = (token.shape[0]//26,-1)
        else:
            token = self.codebook_num*torch.ones((batch_size,16,32),device=device,dtype=torch.int)
            flat_shape = (token.shape[0],-1)
        shape = token.shape
        unknown_number_in_the_beginning = torch.sum(token.reshape(*flat_shape) == self.codebook_num, dim=-1)
        for i in range(self.sampling_iter):
            token_reshaped = token.reshape(*flat_shape)
            logits = model(token_reshaped,*cond)
            if self.is_second_stage:
                logits = logits.permute(0,2,3,1)
            logits = logits.reshape(*flat_shape,self.codebook_num)
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()
            unknown_map = (token_reshaped == self.codebook_num)
            sampled_ids = torch.where(unknown_map, sampled_ids,token_reshaped)
            ratio = 1. * (i + 1) / self.sampling_iter
            mask_ratio = self.gamma(ratio)
            probs = F.softmax(logits, dim=-1)
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1),-1)
            selected_probs = torch.where(unknown_map, selected_probs,self.CONFIDENCE_OF_KNOWN_TOKENS)
            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio),1)
            mask_len = torch.maximum(torch.zeros_like(mask_len),
                                     torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True) - 1,mask_len))
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=4.5 * (1. - ratio))
            token = torch.where(masking, self.codebook_num, sampled_ids)
            token = token.reshape(shape)
        return token
