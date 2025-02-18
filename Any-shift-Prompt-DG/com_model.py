import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import yaml
import pdb
# from dassl.engine import TRAINER_REGISTRY, TrainerX
# from dassl.metrics import compute_accuracy
# from dassl.utils import load_pretrained_weights, load_checkpoint
# from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg['MODEL']['BACKBONE']['NAME']
    # backbone_name = "RN50"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # pdb.set_trace()
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # ??

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, doms_pro=False, target_dom='sketch', learnctx=True):
        super().__init__()
        self.doms_pro = doms_pro
        self.learnctx = learnctx
        self.domains = ['art', 'photo', 'cartoon', 'sketch']
        # pdb.set_trace()
        if target_dom == 'art_painting':
            target_dom = 'art'
        self.domains.remove(target_dom)
        self.domains.append(target_dom)
        # pdb.set_trace()
        n_cls = len(classnames)
        n_ctx = cfg['TRAINER']['COCOOP']['N_CTX']
        ctx_init = cfg['TRAINER']['COCOOP']['CTX_INIT']
        # pdb.set_trace()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if ctx_init:
        #     # use given words to initialize context vectors
        #     ctx_init = ctx_init.replace("_", " ")
        #     n_ctx = len(ctx_init.split(" "))
        #     prompt = clip.tokenize(ctx_init)
        #     with torch.no_grad():
        #         embedding = clip_model.token_embedding(prompt).type(dtype)
        #     ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefix = ctx_init
        # else:
        #     # random initialization
        #     ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        #     nn.init.normal_(ctx_vectors, std=0.02)
        #     prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        if cfg['TRAINER']['COCOOP']['PREC'] == "fp16":
            # pdb.set_trace()
            self.meta_net.half()

        # pdb.set_trace()
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        ##############################
        # if self.doms_pro:
        #     self.prefixs = []
        #     self.suffixs = []
        #     self.ctxs = []
        #     tokenized_prompts = []
        #     for i in range(len(self.domains)):
        #         # pdb.set_trace()
        #         prompt_prefix = 'a ' + self.domains[i] + ' of a'

        #         prompts = [prompt_prefix + " " + name + "." for name in classnames] # start - a - photo - of - a - dog - . - end
        #                                                                     # 0       1     2      3   4    5    6    7
        #         dom_tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        #         # pdb.set_trace()
        #         # dom_tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        #         with torch.no_grad():
        #             embedding = clip_model.token_embedding(dom_tokenized_prompts).type(dtype)

        #         # These token vectors will be saved when in save_model(),
        #         # but they should be ignored in load_model() as we want to use
        #         # those computed using the current class names
        #         self.prefixs.append(embedding[:, :1, :])  # SOS
        #         self.ctxs.append(embedding[:, 1:1 + n_ctx, :])
        #         self.suffixs.append(embedding[:, 1 + n_ctx :, :])  # CLS, EOS     remove 1,2,3,4, e.g., a photo of a ?
        #         tokenized_prompts.append(dom_tokenized_prompts)

        ###############################
        # else:
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # start - a - photo - of - a - dog - . - end
                                                                            # 0       1     2      3   4    5    6    7
        # pdb.set_trace()
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token", embedding)  # SOS
        # self.register_buffer("token_suffix", embedding[:, 1:, :])  # CLS, EOS     remove 1,2,3,4, e.g., a photo of a ?
        # pdb.set_trace()

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # pdb.set_trace()
    
    # def construct_prompts(self, ctx, prefix, suffix, label=None):
    #     # dim0 is either batch_size (during training) or n_cls (during testing)
    #     # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
    #     # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
    #     # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

    #     if label is not None:
    #         prefix = prefix[label]
    #         suffix = suffix[label]

    #     prompts = torch.cat(
    #         [
    #             prefix,  # (dim0, 1, dim)
    #             ctx,     # (dim0, n_ctx, dim)
    #             suffix,  # (dim0, *, dim)
    #         ],
    #         dim=1,
    #     )

    #     return prompts

    def forward(self, im_features, domid=None):
        # if not self.doms_pro:
        # prefix = self.token_prefix
        # # pdb.set_trace()
        # suffix = self.token_suffix
        # if not self.learnctx:
        #     ctx = torch.zeros(self.ctx.size()).cuda()
        # else:
        #     ctx = self.ctx                     # (n_ctx, ctx_dim)  learnable
        prompts = self.token              # 7 * 77 * 512
        # pdb.set_trace()
        # if not self.learnctx:
        #     ctx = self.ctxs[domid][0].cuda()
        # else:
        #     ctx = self.ctx + self.ctxs[domid][0].cuda()                    # (n_ctx, ctx_dim)  learnable
        # pdb.set_trace()
        # bias = self.meta_net(im_features)  # (batch, ctx_dim)
        # bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        # ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        # ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)  (128, 4, 512)
        # ctx_shifted = bias.repeat(1, ctx.size()[0], 1)
        
        # Use instance-conditioned context tokens for all classes
        # prompts = []
        # for ctx_shifted_i in ctx_shifted:  # for each sample in one batch
        #     ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)   # 7 * 4 * 512
        #     pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)  7 * 77 * 512
        #     prompts.append(pts_i)
        # prompts = torch.stack(prompts) # 128 * 7 * 77 * 512
        # # pdb.set_trace()
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, doms_pro=False, target_dom='sketch', learnctx=True):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, doms_pro, target_dom, learnctx)
        self.doms_pro = doms_pro
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None, dom=None):
        if not self.doms_pro:
            tokenized_prompts = self.tokenized_prompts
        else:
            tokenized_prompts = self.tokenized_prompts[dom]
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # pdb.set_trace()

        prompts = self.prompt_learner(image_features, dom)
        # pdb.set_trace()
        
        logits = []
        for imf_i in image_features:
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # pdb.set_trace()
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        
        # if self.prompt_learner.training:
        #     return F.cross_entropy(logits, label)
        
        return logits