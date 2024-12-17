import torch
import torch.nn.functional as F
import itertools
import copy
import abc
import tqdm

class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def between_steps(self):
        return

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0:
            h = attn.shape[0]
            self.forward(attn[h // 2:], is_cross, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn


class AttentionStore(AttentionControl):
    def __init__(self, size=16):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.size=size

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1]<= self.size*self.size:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.step_store[key][i] + \
                        self.attention_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):

        def forward(x, encoder_hidden_states=None,attention_mask=None):
            q = self.to_q(x)
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            out = self.to_out[0](out)
            out = self.to_out[1](out)
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count
    # print("registered {} layers!!!!".format(cross_att_count))


def reset_attention_control(model):
    def ca_forward(self):
        def forward(x, encoder_hidden_states=None,attention_mask=None):
            q = self.to_q(x)
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            out = self.to_out[0](out)
            out = self.to_out[1](out)
            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_)
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])

def aggregate_attention(attention_store, from_where, is_cross,bsz,attention_size=16):
    out = []
    attention_maps = attention_store.get_average_attention()
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == attention_size*attention_size:
                maps = item.reshape(bsz,-1,attention_size*attention_size, item.shape[-1])
                out.append(maps)
    out = torch.cat(out, dim=1)
    out = out.sum(1) / out.shape[1]
    return out



def get_latents(model, images):
    with torch.no_grad():
        latents = model.vae.encode(images).latent_dist.sample()
        latents = latents * model.vae.config.scaling_factor
    return latents


def tunning_prompts(model,images,encoder_hidden_states=None):
    bsz = images.shape[0]
    loss_itr = {
        "loss_prompt_tunning": []
    }
    if encoder_hidden_states is None:
        input_ids = model.tokenizer(
            "",
            truncation=True,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(1, 1)
        encoder_hidden_states = model.text_encoder(input_ids.to(model.device))[0].detach()
    with torch.no_grad():
        latents = model.vae.encode(images).latent_dist.sample()
        latents = latents * model.vae.config.scaling_factor
        latents = latents.detach()
    encoder_hidden_states.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [encoder_hidden_states],
        lr=1e-3
    )
    
    for i in range(30):
        optimizer.zero_grad()
        for b in range(bsz):
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (latents.shape[0],), device=model.device)
            
            noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)
            model_pred = model.unet(noisy_latents, timesteps, torch.cat([encoder_hidden_states]*bsz,dim=0)).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")/bsz
            loss.backward()
        optimizer.step()
    encoder_hidden_states.requires_grad_(False)
    return encoder_hidden_states,loss_itr
    
    

def protect_epoch(encoder_hidden_states, images,model,args):
    bsz = images.shape[0]
    encoder_hidden_states= encoder_hidden_states.detach()
    ori_images = images.clone()
    
    loss_itr = {
        "loss_pgd": []
    }
    
    controller = AttentionStore(args.attention_size)
    register_attention_control(model.unet, controller)

    target_cross = None
    
    for itr in range(args.pgd_itrs):
        images.requires_grad_(True)
        model.vae.zero_grad()
        model.unet.zero_grad()
        for b in range(bsz):
            images_single = images[b].unsqueeze(dim=0)
            controller.reset()
            
            latents = model.vae.encode(images_single).latent_dist.sample()
            latents = latents * model.vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (latents.shape[0],), device=model.device)
            
            noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)
            model_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            
            cross_attention_maps = aggregate_attention(controller,from_where=["down", "up"],is_cross=True,bsz=1)*1e+2
            cross_attention_maps = torch.clamp(cross_attention_maps,min=0,max=1)
    

            
            if target_cross is not None:
                pass
            else:
                target_cross = torch.zeros_like(cross_attention_maps).detach()
            
            if model.scheduler.config.prediction_type == "epsilon":
                noise = noise
            elif model.scheduler.config.prediction_type == "v_prediction":
                noise = model.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {model.scheduler.config.prediction_type}")    
            cross_attention_loss = -F.mse_loss(cross_attention_maps.float(), target_cross.float(), reduction="mean")
            adv_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss =  cross_attention_loss + adv_loss
            loss = loss/bsz
            loss.backward()
        alpha = args.pgd_alpha
        eps = args.pgd_eps

        images = images + alpha * images.grad.sign()
        eta = torch.clamp(images - ori_images, min=-eps, max=+eps)
        images = torch.clamp(ori_images + eta, min=-1, max=+1).detach_()
        loss_itr["loss_pgd"].append(float(loss.detach().item()))
    loss_itr["loss_pgd"] = loss_itr["loss_pgd"]
    reset_attention_control(model.unet)
    return images, loss_itr



def tunning_epoch(encoder_hidden_states, images, model, args):
    bsz = images.shape[0]
    params_to_optimize = itertools.chain(model.unet.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    encoder_hidden_states = encoder_hidden_states.detach()
    images = images.detach()

    loss_itr = {
        "loss_tunning": []
    }
    for step in range(args.tunning_steps):
        model.unet.train()
        for b in range(bsz): 
            images_single = images[b].unsqueeze(dim=0)

            latents = model.vae.encode(images_single).latent_dist.sample()
            latents = latents * model.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, model.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)

            # Predict the noise residual
            model_pred = model.unet(noisy_latents, timesteps,
                                    encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if model.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif model.scheduler.config.prediction_type == "v_prediction":
                target = model.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {model.scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")/bsz

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params_to_optimize, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad()
    return loss_itr


def defense(images, model, args):
    model_copy = copy.deepcopy(model)
    loss_all = []
    original_images = images.clone()
    encoder_hidden_states,loss_itr = tunning_prompts(model_copy,images)
    loss_all.append({
            "epoch": 0,
            "loss_itr": loss_itr
        })
    images, loss_itr = protect_epoch(encoder_hidden_states, original_images, model_copy, args)
    loss_all.append({
        "epoch": 0,
        "loss_itr": loss_itr
    })
    for epoch in tqdm.tqdm(range(args.epoches)):
        loss_itr = tunning_epoch(encoder_hidden_states, images, model_copy, args)
        loss_all.append({
            "epoch": epoch,
            "loss_itr": loss_itr
        })
        images, loss_itr = protect_epoch(encoder_hidden_states, images, model_copy, args)
        loss_all.append({
            "epoch": epoch+1,
            "loss_itr": loss_itr
        })
        encoder_hidden_states,loss_itr = tunning_prompts(model_copy,images,encoder_hidden_states)
        loss_all.append({
            "epoch": epoch+1,
            "loss_itr": loss_itr
        })

    return images, loss_all