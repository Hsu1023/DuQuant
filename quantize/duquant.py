import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_mistral_layer import QuantMistralDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import *
from quantize.const import CLIPMIN
import pdb



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     


def duquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower() or "vicuna" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
            "down_proj":"down",
        }
        layer_name_prefix = "model.layers"
    elif "mistral" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantMistralDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
            "down_proj":"down",
        }
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for llama/Llama-2/Llama-3/Vicuna/Mistral now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    input_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                input_ids.append(batch[0])
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "vicuna" in args.net.lower() or "mistral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise ValueError("Only support for llama/Llama-2/Llama-3/Vicuna/Mistral now")
    torch.cuda.empty_cache()
    
    quant_inps = inps
    rotate_inps = copy.copy(inps).mean(dim=0)

    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None


    if args.resume:
        duquant_parameters = torch.load(os.path.join(args.resume, f"duquant_parameters.pth"))
    else:
        duquant_parameters = {}

    for i in range(len(layers)):
        for name in ['q', 'k', 'v', 'gate', 'up', 'down', 'o']:
            exec(f"args.{name}_weight_quant_params = copy.copy(args.weight_quant_params)")
            exec(f"args.{name}_act_quant_params = copy.copy(args.act_quant_params)")
        args.q_quant_params = copy.copy(args.act_quant_params)
        args.k_quant_params = copy.copy(args.act_quant_params)

        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i]
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)        
        if torch.cuda.device_count() > 1:
            qlayer.mlp.to("cuda:1")

        if args.quant_method == 'duquant':
            set_init_duquant_params_state(qlayer, True)

        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0 :
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        
        if is_llama or args.abits == 16:
            use_shift = False  # deactivate channel-wise shifting for llama model and weight-only quantization
        
        if args.resume:
            # raise NotImplementedError
            qlayer.load_state_dict(duquant_parameters[i], strict=False)
            print(duquant_parameters[i].keys())

        if args.smooth:
            if duquant_parameters.get(i):
                qlayer.load_smooth_params(duquant_parameters[i], dev)
            else:
                qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype), requires_grad=False))
                for name,module in qlayer.named_modules():
                    if isinstance(module, QuantLinear):
                        for key in pairs.keys():
                            if key in name:
                                act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=CLIPMIN)
                                weight = module.weight.abs().max(dim=0)[0].clamp(min=CLIPMIN)
                                scale = (act.pow(args.alpha)/weight.to(act.device).pow(1-args.alpha)).clamp(min=CLIPMIN)
                                if use_shift and not is_llama:
                                    shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                                else:
                                    shift = torch.zeros_like(scale)
                                if key not in ['down_proj'] and args.smooth_epochs > 0:
                                    qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                                    qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                else:
                                    qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift, requires_grad=False))
                                    qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale, requires_grad=False))
        
        if args.smooth_epochs > 0:
            assert args.smooth
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":smooth_parameters(qlayer, use_shift),"lr":args.let_lr},],weight_decay=args.wd)
            set_requires_grad(get_post_parameters(qlayer), False)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.smooth_epochs):
                loss_list = []
                norm_list = []
                
                print(qlayer.qkt_smooth_scale)
                for j in range(args.nsamples//args.batch_size):  
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=smooth_parameters(qlayer)).cpu()
                    norm_list.append(norm.data)

            clear_temp_variable(qlayer)
        
            set_requires_grad(get_post_parameters(qlayer), True)
            del optimizer
        qlayer.half()
        try:
            with torch.no_grad():
                qlayer.qkt_smooth_scale.clamp_(min=0.5)
        except:
            pass
        smooth_and_let_inplace(qlayer, args)

        # real smooth and quantization      
        if args.quant_method == 'duquant':
            set_init_duquant_params_state(qlayer, False)
            set_quant_state(qlayer, weight_quant=True, act_quant=True)
            if duquant_parameters.get(i):
                qlayer.load_duquant_params(duquant_parameters[i], dev)
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        set_registered_x_none(qlayer)
                        rotate_inps = qlayer(rotate_inps.unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0][0]
            qlayer.register_duquant_params()
            set_init_duquant_params_state(qlayer, True)
        
        if args.let:
            set_quant_state(qlayer, weight_quant=True, act_quant=True)
            if duquant_parameters.get(i):
                qlayer.load_post_parameter(duquant_parameters[i], dev)
            else:
                qlayer.register_parameter("qkt_post_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
                for name,module in qlayer.named_modules():
                    if isinstance(module, QuantLinear):
                        for key in pairs.keys():
                            if key in name:
                                act = module.act_quantizer.recorded_x_max.clamp(min=CLIPMIN)
                                weight = module.weight_quantizer.recorded_x_max.clamp(min=CLIPMIN)
                                scale = (act.pow(args.let_alpha)/weight.to(act.device).pow(1-args.let_alpha)).clamp(min=0.8)
                                if key not in ['down_proj']:
                                    qlayer.register_parameter(f"{pairs[key]}_post_scale",torch.nn.Parameter(scale, requires_grad=False))
                                else:
                                    qlayer.register_parameter(f"{pairs[key]}_post_scale",torch.nn.Parameter(scale))
        
        # training
        if duquant_parameters.get(i):
            if args.lwc:
                qlayer.load_lwc_params(duquant_parameters[i], dev)
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr},],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):

                def check_nan_parameters(model_):
                        for param in model_.parameters():
                            if torch.isnan(param).any():
                                return True
                        return False
                original_parameters = [param.clone() for param in get_post_parameters(qlayer)]
                loss_list = []
                norm_list = []
                
                for j in range(args.nsamples//args.batch_size):  
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        post_rotate_quant_temporary(qlayer, args)
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        break
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters= get_post_parameters(qlayer)).cpu()

                    norm_list.append(norm.data)
                
                if check_nan_parameters(qlayer):
                    print('detect NaN', epochs)
                    loss.backward()
                    optimizer.zero_grad()
                    with torch.no_grad():
                        for param, original_param in zip(get_post_parameters(qlayer), original_parameters):
                            param.copy_(original_param)
                torch.cuda.empty_cache()
            clear_temp_variable(qlayer)
            del optimizer
        
        post_quant_inplace(qlayer, args)
        # obtain output of full-precision model

        qlayer.half()
        quant_inplace(qlayer)
        set_quant_state(qlayer, weight_quant=False, act_quant=True)

        if args.epochs>0 :
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            duquant_parameters[i] = duquant_state_dict(qlayer)
            if args.save_dir:
                torch.save(duquant_parameters, os.path.join(args.save_dir, f"duquant_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            duquant_parameters[i] = duquant_state_dict(qlayer)
            if args.save_dir:
                torch.save(duquant_parameters, os.path.join(args.save_dir, f"duquant_parameters.pth"))

        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    
    return model

