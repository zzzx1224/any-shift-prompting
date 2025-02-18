from distutils.errors import DistutilsGetoptError
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
    backbone_name = cfg["MODEL"]["BACKBONE"]["NAME"]
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


class vi_prompt(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 16, output_dim),
            # nn.ReLU(inplace=True),
        )
        # self.mulayer = nn.Linear(output_dim, output_dim)
        # self.sigmalayer = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # pdb.set_trace()
        xall = self.layers(x)

        # x_mean = x.mean(0, keepdim=True)
        # xall = torch.cat([x_mean, x], 0)
        # xall = self.layers(xall)
        #
        # x_mu = self.mulayer(xall)
        # x_sig = F.softplus(self.sigmalayer(xall))
        #
        # d_mu = x_mu[0]
        # d_sig = x_sig[0]
        # x_mu = x_mu[1:]
        # x_sig = x_sig[1:]
        #
        # # pdb.set_trace()
        # self.kld = self.kl_divergence(
        #     x_mu,
        #     x_sig,
        #     d_mu.unsqueeze(0).repeat(x.size()[0], 1),
        #     d_sig.unsqueeze(0).repeat(x.size()[0], 1),
        # )
        #
        # return x_mu, x_sig, d_mu, d_sig, self.kld
        return xall

    # KL divergence
    def kl_divergence(self, mu_q, sigma_q, mu_p, sigma_p):

        var_q = sigma_q**2 + 1e-6
        var_p = sigma_p**2 + 1e-6

        component1 = torch.log(var_p) - torch.log(var_q)
        component2 = var_q / var_p
        component3 = (mu_p - mu_q).pow(2) / var_p

        KLD = 0.5 * torch.sum((component1 - 1 + component2 + component3), 1)
        return KLD



class visualpromptlearner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
            # nn.ReLU(inplace=True),
        )
        # self.mulayer = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # pdb.set_trace()
        # x_mean = x.mean(0, keepdim=True)
        # xall = torch.cat([x_mean, x], 0)
        x = self.layers(x)
        # x = self.mulayer(x)

        return x


class visualpromptlearners(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
        )
        self.mulayer = nn.Linear(output_dim, output_dim)
        self.siglayer = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # pdb.set_trace()
        # x_mean = x.mean(0, keepdim=True)
        # xall = torch.cat([x_mean, x], 0)
        x = self.layers(x)
        xmu = self.mulayer(x)
        xsig = F.softplus(self.siglayer(x))

        if self.training:
            eps = xmu.new(xmu.size()).normal_()
            x = xmu + xsig * eps
        else:
            x = xmu

        return x



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class TaskTokenTransformer(nn.Module):
    def __init__(self, input_dim: int, layers: int, heads: int, output_dim: int, taskpres=True):  # 512, 2, 8, 512
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = taskpres

        self.class_embedding = torch.zeros(1, output_dim).cuda()
        self.class_embedding_mu = nn.Parameter(torch.empty((1, output_dim), dtype=torch.float32).normal_(0., 0.1), requires_grad=True)
        self.class_embedding_sig = nn.Parameter(torch.empty((1, output_dim), dtype=torch.float32).normal_(0., 0.1), requires_grad=True)
        self.ln_pre = LayerNorm(output_dim)
        self.ln_post = LayerNorm(output_dim)
        self.positional_embedding = nn.Parameter(torch.empty((3, output_dim), dtype=torch.float32).normal_(0., 0.1),
                                            requires_grad=True)

        self.transformer = Transformer(output_dim, layers, heads)

        self.mu_layer = nn.Linear(output_dim, output_dim)
        self.sigma_layer = nn.Linear(output_dim, output_dim)


    def forward(self, fx: torch.Tensor, claprompts: torch.Tensor, targets: torch.Tensor):
        # pdb.set_trace()
        self.class_embedding = self.class_embedding_mu + F.softplus(self.class_embedding_sig) * torch.randn_like(self.class_embedding_mu)
        x = torch.cat([self.class_embedding.to(fx.dtype) + torch.zeros(fx.shape[0], 1, fx.shape[-1], dtype=fx.dtype, device=fx.device), fx.unsqueeze(1)], dim=1)
        x = x + self.positional_embedding[:1]

        xall = torch.cat([self.class_embedding.to(fx.dtype) + self.positional_embedding[0].unsqueeze(0), fx + self.positional_embedding[1]], dim=0)
        # pdb.set_trace()

        x = torch.cat([x, claprompts.unsqueeze(0).repeat(x.size()[0], 1, 1) + self.positional_embedding[-1]], dim=1)

        # xall = torch.cat([xall, claprompts + self.positional_embedding[-1]], dim=0)
        if self.training:
            xall = torch.cat([xall, claprompts[targets.unique()] + self.positional_embedding[-1]], dim=0)
        else:
            xall = torch.cat([xall, claprompts + self.positional_embedding[-1]], dim=0)

        # x = torch.cat([xall.unsqueeze(0), x], dim=0)
        # pdb.set_trace()

        x = self.ln_pre(x)

        x = self.transformer(x)

        x = self.ln_post(x[:, 0, :])

        xall = self.ln_post(xall.unsqueeze(0))

        xall = self.transformer(xall)

        xall = self.ln_post(xall[:, 0, :])

        x = torch.cat([xall, x], dim=0)

        x_mu = self.mu_layer(x)
        x_sig = F.softplus(self.sigma_layer(x))

        p_mu = x_mu[0]
        p_sig = x_sig[0]
        q_mu = x_mu[1:]
        q_sig = x_sig[1:]

        self.kld = self.kl_divergence(
            q_mu,
            q_sig,
            p_mu.unsqueeze(0).repeat(fx.size()[0], 1),
            p_sig.unsqueeze(0).repeat(fx.size()[0], 1),
        )

        if self.training:
            eps = x_mu.new(x_mu.size()).normal_()
            pall = x_mu + x_sig * eps
        else:
            pall = x_mu
        # pall = x_mu + x_sig * torch.randn_like(x_mu)
        ppp = pall[0]
        qp = pall[1:]

        # if self.proj is not None:
        #     x = x @ self.proj
        # pdb.set_trace()
        if self.residual:
            qpi = qp + fx
            qpt = qp.unsqueeze(1) + claprompts.unsqueeze(0)
            ppi = ppp.unsqueeze(0).repeat(qp.size()[0], 1) + fx
            ppt = (ppp.unsqueeze(0) + claprompts).unsqueeze(0).repeat(qp.size()[0], 1, 1)
        else:
            qpi = qp
            qpt = qp.unsqueeze(1)
            ppi = ppp.unsqueeze(0).repeat(qp.size()[0], 1)
            ppt = (ppp).unsqueeze(0).repeat(qp.size()[0], 1).unsqueeze(1)

        return qpi, qpt, ppi, ppt, self.kld.sum()

        # KL divergence

    def kl_divergence(self, mu_q, sigma_q, mu_p, sigma_p):

        var_q = sigma_q ** 2 + 1e-6
        var_p = sigma_p ** 2 + 1e-6

        component1 = torch.log(var_p) - torch.log(var_q)
        component2 = var_q / var_p
        component3 = (mu_p - mu_q).pow(2) / var_p

        KLD = 0.5 * torch.sum((component1 - 1 + component2 + component3), 1)
        return KLD


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )  # ??

        return x


class PromptLearner(nn.Module):
    def __init__(
        self,
        cfg,
        classnames,
        clip_model,
        doms_pro=False,
        target_dom="sketch",
        learnctx=True,
        vpro=False,
        dataset="PACS",
    ):
        super().__init__()
        self.doms_pro = doms_pro
        self.learnctx = learnctx
        self.vpro = vpro
        # self.domains = ['art', 'photo', 'cartoon', 'sketch']
        # if dataset == "PACS":
        #     self.domains = ["artpainting", "photo", "cartoon", "quickdraw"]
        # elif dataset == "office":
        #     self.domains = ["art", "clipart", "product", "real"]
        # elif dataset == "vlcs":
        #     self.domains = ["CALTECH", "LABELME", "PASCAL", "SUN"]
        # elif dataset == "domainnet":
        #     self.domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        # else:
        #     self.domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        # # pdb.set_trace()
        # if target_dom == "art_painting":
        #     target_dom = "artpainting"
        # if target_dom == "sketch":
        #     target_dom = "quickdraw"
        # if target_dom == "real_World":
        #     target_dom = "real"
        # self.domains.remove(target_dom)
        # self.domains.append(target_dom)
        # pdb.set_trace()
        n_cls = len(classnames)
        n_ctx = cfg["TRAINER"]["COCOOP"]["N_CTX"]
        ctx_init = cfg["TRAINER"]["COCOOP"]["CTX_INIT"]
        # pdb.set_trace()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        # pdb.set_trace()

        if vpro:
            self.meta_net = vi_prompt(vis_dim, ctx_dim)
        else:
            self.meta_net = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                        ("relu", nn.ReLU(inplace=True)),
                        ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                    ]
                )
            )

        if cfg["TRAINER"]["COCOOP"]["PREC"] == "fp16":
            # pdb.set_trace()
            self.meta_net.half()

        # pdb.set_trace()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        ##############################
        if self.doms_pro:
            self.prefixs = []
            self.suffixs = []
            self.ctxs = []
            tokenized_prompts = []
            for i in range(len(self.domains)):
                # pdb.set_trace()
                prompt_prefix = "a " + self.domains[i] + " of a"

                prompts = [
                    prompt_prefix + " " + name + "." for name in classnames
                ]  # start - a - photo - of - a - dog - . - end
                # 0       1     2      3   4    5    6    7
                dom_tokenized_prompts = torch.cat(
                    [clip.tokenize(p) for p in prompts]
                )  # (n_cls, n_tkn)
                # pdb.set_trace()
                # dom_tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(dom_tokenized_prompts).type(
                        dtype
                    )

                # These token vectors will be saved when in save_model(),
                # but they should be ignored in load_model() as we want to use
                # those computed using the current class names
                self.prefixs.append(embedding[:, :1, :])  # SOS
                self.ctxs.append(embedding[:, 1 : 1 + n_ctx, :])
                self.suffixs.append(
                    embedding[:, 1 + n_ctx :, :]
                )  # CLS, EOS     remove 1,2,3,4, e.g., a photo of a ?
                tokenized_prompts.append(dom_tokenized_prompts)

        ###############################
        else:
            prompts = [
                prompt_prefix + " " + name + "." for name in classnames
            ]  # start - a - photo - of - a - class - . - end
            # 0       1     2      3   4    5    6    7
            # pdb.set_trace()
            tokenized_prompts = torch.cat(
                [clip.tokenize(p) for p in prompts]
            )  # (n_cls, n_tkn)
            # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS  'start'
            self.register_buffer(
                "token_suffix", embedding[:, 1 + n_ctx :, :]
            )  # CLS, EOS     remove 1,2,3,4, e.g., "a photo of a"
            # pdb.set_trace()

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # pdb.set_trace()

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(
        self, im_features, domid=None, feat_pro=True, training=True, mc_times=5
    ):
        if not self.doms_pro:
            prefix = self.token_prefix
            # pdb.set_trace()
            suffix = self.token_suffix
            if not self.learnctx:
                ctx = torch.zeros(self.ctx.size()).cuda()
            else:
                ctx = self.ctx  # (n_ctx, ctx_dim)  learnable
        else:
            prefix = self.prefixs[domid].cuda()
            # pdb.set_trace()
            suffix = self.suffixs[domid].cuda()
            # pdb.set_trace()
            if not self.learnctx:
                ctx = self.ctxs[domid][0].cuda()
            else:
                ctx = (
                    self.ctx + self.ctxs[domid][0].cuda()
                )  # (n_ctx, ctx_dim)  learnable
        # pdb.set_trace()
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        if not feat_pro:
            bias = torch.zeros(bias.size()).cuda()
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)  (128, 4, 512)
        # ctx_shifted = bias.repeat(1, ctx.size()[0], 1)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:  # for each sample in one batch
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(
                self.n_cls, -1, -1
            )  # 7 * 4 * 512
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)  7 * 77 * 512
            prompts.append(pts_i)
        prompts = torch.stack(prompts)  # 128 * 7 * 4 * 512

        return prompts
            # pdb.set_trace()


class CustomCLIP(nn.Module):
    def __init__(
        self,
        cfg,
        classnames,
        clip_model,
        doms_pro=False,
        target_dom="sketch",
        learnctx=True,
        vpro=False,
        opendg=False,
        dataset="PACS",
        imgencoder=None,
        imgp = False,
        taskp = False,
        taskpres= True,
        vptmethod = 'mean',
        transl = 2
    ):
        super().__init__()
        self.prompt_learner = PromptLearner(
            cfg, classnames, clip_model, doms_pro, target_dom, learnctx, vpro, dataset
        )
        self.doms_pro = doms_pro
        self.vpro = vpro
        self.taskp = taskp
        self.vptmethod = vptmethod
        self.imgp = imgp
        if opendg:
            if dataset == "PACS":
                self.tra_cls_num = 6
            elif dataset == "office":
                self.tra_cls_num = 54
            elif dataset == 'domainnet':
                self.tra_cls_num = 300
        else:
            if dataset == "PACS":
                self.tra_cls_num = 7
            elif dataset == "office":
                self.tra_cls_num = 65
            elif dataset == "vlcs":
                self.tra_cls_num = 5
            elif dataset == "domainnet":
                self.tra_cls_num = 345
            elif dataset == "living17":
                self.tra_cls_num = 17
            elif dataset == "entity30":
                self.tra_cls_num = 30
        # self.dataset = dataset
        # pdb.set_trace()
        self.opendg = opendg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        if imgencoder:
            self.image_encoder = imgencoder
        else:
            self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        print('visual class-based prompt generation method is ' + vptmethod + '\n')

        if vptmethod == 'no' or vptmethod == 'cat' or vptmethod == 'macat' or vptmethod == 'mcat':
            self.visualpromptlearner = visualpromptlearner(512, 768)
            self.claposembedding = nn.Parameter(torch.empty((1, 768), dtype=torch.float32).normal_(0., 0.1),
                                                requires_grad=True)

        elif vptmethod == 'nos' or vptmethod == 'cats':
            self.visualpromptlearner = visualpromptlearners(512, 768)

        else:
            self.visualpromptlearner = visualprompttranser(512, 768, method=vptmethod)

        self.class_prompts = torch.cat([clip.tokenize('a photo of a ' + name) for name in classnames])
        # self.class_prompts = torch.cat([clip.tokenize('an image of ' + name + ' a') for name in classnames])
        self.testclasstokens = clip_model.token_embedding(self.class_prompts).type(self.dtype).cuda().detach()   # 7 * 77 * 512

        if self.taskp:
            self.tasktokenlearner = TaskTokenTransformer(512, transl, 8, 512, taskpres)

        # pdb.set_trace()

    def forward(self, image, label, dom=None, featpro=True, training=True, mc_times=5, claforimg=None, opentra='all'):
        if not self.doms_pro:
            tokenized_prompts = self.tokenized_prompts
        else:
            tokenized_prompts = self.tokenized_prompts[dom]
        logit_scale = self.logit_scale.exp()

        # pdb.set_trace()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).detach()
        # pdb.set_trace()

        if opentra == 'all' and self.training:
            claprompts = self.text_encoder(self.testclasstokens[: self.tra_cls_num],
                                           self.class_prompts[: self.tra_cls_num])
        else:
            claprompts = self.text_encoder(self.testclasstokens, self.class_prompts)

        claprompts = claprompts / claprompts.norm(dim=-1, keepdim=True)

        # if self.taskp:
        image_featuresq, clapromptsq, image_featuresp, clapromptsp, kld = self.tasktokenlearner(image_features, claprompts, label)
        # pdb.set_trace()
        clapromptsq = clapromptsq / clapromptsq.norm(dim=-1, keepdim=True)
        image_featuresq = image_featuresq / image_featuresq.norm(dim=-1, keepdim=True)
        clapromptsp = clapromptsp / clapromptsp.norm(dim=-1, keepdim=True)
        image_featuresp = image_featuresp / image_featuresp.norm(dim=-1, keepdim=True)

        clapromptsq = clapromptsq.view(-1, 512)
        clapromptsq = self.visualpromptlearner(clapromptsq)
        clapromptsq = clapromptsq.view(image_features.size()[0], -1, 768)

        clapromptsp = clapromptsp.view(-1, 512)
        clapromptsp = self.visualpromptlearner(clapromptsp)
        clapromptsp = clapromptsp.view(image_features.size()[0], -1, 768)

        clapromptsq = clapromptsq + self.claposembedding
        clapromptsp = clapromptsp + self.claposembedding

        image_features1q = self.image_encoder(image.type(self.dtype), clapromptsq, combine='cat')
        image_features1q = image_features1q / image_features1q.norm(dim=-1, keepdim=True)
        image_features1p = self.image_encoder(image.type(self.dtype), clapromptsp, combine='cat')
        image_features1p = image_features1p / image_features1p.norm(dim=-1, keepdim=True)


        if not self.vpro:
            prompts = self.prompt_learner(image_features, dom, featpro)
            logits = []
            for pts_i, imf_i in zip(prompts, image_features1):
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # pdb.set_trace()
                if self.opendg and training:
                    text_features = text_features[: self.w]
                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)
            logits = torch.stack(logits)
            dlogits = 0
            kld = torch.zeros(1).cuda()
        else:
            prompts = self.prompt_learner(
                image_featuresq, dom, featpro, training, mc_times
            )
            dprompts = self.prompt_learner(
                image_featuresp, dom, featpro, training, mc_times
            )
            # pdb.set_trace()
            logits = []
            dlogits = []

            if training:
                prompts = prompts.view(
                    image_features.size()[0],
                    mc_times,
                    prompts.size()[1],
                    prompts.size()[2],
                    prompts.size()[3],
                )
                dprompts = dprompts.view(
                    image_features.size()[0],
                    mc_times,
                    dprompts.size()[1],
                    dprompts.size()[2],
                    dprompts.size()[3],
                )
                for pts_i, ppts_i, imf_i, imf_p in zip(prompts, dprompts, image_features1q, image_features1p):
                    for pts_ij in pts_i:
                        text_features = self.text_encoder(pts_ij, tokenized_prompts)
                        text_features = text_features / text_features.norm(
                            dim=-1, keepdim=True
                        )
                        if self.opendg:
                            text_features = text_features[: self.tra_cls_num]
                        # pdb.set_trace()
                        l_i = logit_scale * imf_i @ text_features.t()
                        logits.append(l_i)
                    for dpts_j in ppts_i:  # 10
                        text_features = self.text_encoder(dpts_j, tokenized_prompts)
                        text_features = text_features / text_features.norm(
                            dim=-1, keepdim=True
                        )
                        if self.opendg:
                            text_features = text_features[: self.tra_cls_num]
                        # pdb.set_trace()
                        dl_i = logit_scale * imf_p @ text_features.t()
                        # dl_i = logit_scale * imf_i @ text_features.t()
                        dlogits.append(dl_i)
                logits = torch.stack(logits)
                dlogits = torch.stack(dlogits)
                # pdb.set_trace()
            else:
                for pts_i, ppts_i, imf_i in zip(prompts, dprompts, image_features1q):
                    text_features = self.text_encoder(pts_i, tokenized_prompts)
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )
                    # pdb.set_trace()
                    l_i = logit_scale * imf_i @ text_features.t()
                    logits.append(l_i)

                    dtext_features = self.text_encoder(ppts_i, tokenized_prompts)
                    dtext_features = dtext_features / dtext_features.norm(
                        dim=-1, keepdim=True
                    )
                    # pdb.set_trace()
                    dl_i = logit_scale * imf_i @ dtext_features.t()
                    dlogits.append(dl_i)

                logits = torch.stack(logits)
                dlogits = torch.stack(dlogits)
                kld = 0
        # pdb.set_trace()

        # if self.prompt_learner.training:
        #     return F.cross_entropy(logits, label)

        return logits, dlogits, kld
