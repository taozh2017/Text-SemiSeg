import torch
import torch.nn as nn
import numpy as np
from model.CLIP.clip import tokenize
from model.CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):

        super().__init__()

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection


    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        # x,_,_ = self.transformer(x.half())
        x = self.transformer(x.half())
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self,
                 args,
                 classnames,
                 n_ctx, # prompt max len
                 CSC, # True or False multi prompt
                 class_token_position, # cls position
                 clip_model):

        super().__init__()
        n_cls = len(classnames)
        ctx_dim = clip_model.ln_final.weight.shape[0] #

        self.ctx={}

        # for cls in prompts:
        #     for position in class_token_position:
        #         if CSC:
        #             ctx_vectors = torch.empty(len(prompts[cls]), n_ctx, ctx_dim).to(clip_model.device)
        #         else:
        #             ctx_vectors = torch.empty(n_ctx, ctx_dim).to(clip_model.device)
        #         nn.init.normal_(ctx_vectors, std=0.02)
        #         self.ctx['{}_{}'.format(cls,position)]=nn.Parameter(ctx_vectors,requires_grad=True)

        # for position in class_token_position:
        if CSC:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim).to(args.device)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim).to(args.device)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        # self.ctx['{}_{}'.format(cls, position)] = nn.Parameter(ctx_vectors,requires_grad=True)
        
        self.ctx = nn.Parameter(ctx_vectors, requires_grad=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # self.ctx = nn.ParameterDict(self.ctx)  # to be optimized
        
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(args.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        # _tokenizer = SimpleTokenizer()
        # prompts_split={cls: [prompt.replace("_", " ")  for prompt in prompts[cls]] for cls in prompts}
        # prompts_lens= {cls: [ len(_tokenizer.encode(prompt)) for prompt in prompts_split[cls]] for cls in prompts_split}
        # prompts_learnable_tokens = {cls:[prompt_prefix + " " + prompt + "." for prompt in prompts_split[cls]] for cls in prompts_split}
        # tokenized_prompts = {cls:torch.cat([tokenize(prompt) for prompt in prompts_learnable_tokens[cls]]).to(clip_model.device) for cls in prompts_learnable_tokens}
        # with torch.no_grad():
        #     embeddings = {cls:clip_model.token_embedding(tokenized_prompts[cls])  for cls in tokenized_prompts}
        # self.register_embeddings={}
        # for cls in embeddings:
        #     self.register_embeddings['{}_token_prefix'.format(cls)]=embeddings[cls][:, :1, :]
        #     self.register_embeddings['{}_token_suffix'.format(cls)]=embeddings[cls][:, 1 + n_ctx :, :]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position


    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    

class PromptMaker(nn.Module):

    def __init__(self,
                 args,
                 prompts,
                 clip_model,
                 n_ctx: int=8,  # prompt max len
                 CSC: bool= True,  # True or False multi prompt
                 class_token_position: list="end",  # cls position
                 ):

        super().__init__()
        # assert 'normal' in prompts and 'abnormal' in prompts
        # for position in class_token_position:
        # assert class_token_position in ['end','middle','front']

        self.prompt_learner = PromptLearner(args, prompts, n_ctx, CSC, class_token_position, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.class_token_position = class_token_position
        self.text_encoder = TextEncoder(clip_model)

    def forward(self,):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        # text_features=[]
        # for cls in prompts:
        #     class_embedding = self.text_encoder(prompts[cls], tokenized_prompts[cls].repeat(len(self.class_token_position), 1))
        #     class_embedding = class_embedding.mean(dim=0)
        #     class_embedding = class_embedding / class_embedding.norm()
        #     text_features.append(class_embedding)
        # text_features = torch.stack(text_features, dim=1)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features