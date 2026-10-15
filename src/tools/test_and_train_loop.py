import torch
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import v2
from torch.distributions import Uniform
from torch.cuda.amp import GradScaler, autocast
from src.tools.transforms import *
from src.tools.losses import OverlapLoss, DetSNRLoss, CorrelationLoss, GradientDifferenceLoss, CharbonnierLoss, LogMagnitudeL1Loss, OptSNRLoss, CrossWaveletPowerLoss
from src.tools.metrics import *

def train_loop(dataloader, 
               model, 
               loss_fn, 
               optimizer, 
               batch_size, 
               train_history, 
               scaler, 
               device, 
               use_amp=True, 
               max_abs_tensor=None, 
               ema=None,
               pred_overlap_history=[],
               residual_overlap_history=[],
               clipper = None):
    #Initialise losses
    train_loss=0.#Sum of losses across all batches
    no_batches = len(dataloader)
    size = len(dataloader.dataset)
    # Set model to training mode to ensure things like BN and dropout etc. are on
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #Convert all data to channels last format
        X = X
        y = y
        #Set optimiser zero grad before any backward passes
        optimizer.zero_grad(set_to_none=True)
        #Forward/backward pass on mask
        '''ONLY model predictions in low precision'''
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            #Forward pass to get predicted masks
            pred_denoised = model(X)
        
        '''Loss and metrics computations in higher precision'''
        with torch.autocast(device_type=device, enabled=False):
            #Compute criterion
            loss = loss_fn(pred_denoised.float(), y)
            #Compute metrics
            overlap = overlap_AET_batchwise(pred_denoised.float(), y, dataloader.dataset.dt, None)
        #Scale mask loss, and do backwards pass
        scaler.scale(loss).backward()#retain_graph=True
        scaler.unscale_(optimizer)
        if clipper is not None:
            clipper.step()        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Adjust parameters
        scaler.step(optimizer)#optimizer.step()
        scaler.update()
        if ema is not None: ema.update()
        # '''Use switch EMA variant'''
        # model.update_model_with_ema()
        #Record all losses
        train_loss += loss.item()
        #Record all metrics
        train_overlap += overlap.item()
        # Report the current loss of a given batch if desired
        if batch % 10 == 0:#10
            loss, current = loss, batch * batch_size + len(X)#.item()
            print(f"loss: {loss:.6E}  [{current}/{size}]")

    #Report average loss by dividing by no. of batches
    final_loss= train_loss/no_batches
    print(f"Avg. total training loss: {final_loss:.6E}\n")

    #Report average metrics by dividing by no. of batches
    final_overlap= train_overlap/no_batches
    print(f"Avg. overlap: {final_overlap:.6E}\n")

    #Append loss to history
    train_history.append(final_loss)

    #Append metrics to history
    train_overlap.append(final_overlap)

    #print allocated memory
    print(f"Allocated memory (not including spikes): {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9} GB")

def val_loop(dataloader, 
             model, 
             loss_fn, 
             val_history, 
             scaler, 
             device, 
             use_amp=True, 
             max_abs_tensor=None, 
             pred_overlap_history=[], 
             input_overlap_history=[],
             ema=None,
             val_history_EMA=[],
             pred_overlap_history_EMA=[],
             residual_overlap_history=[],
             residual_overlap_history_EMA=[],
             ):
    #EMA will just be the regular model if we don't want it
    if ema is None: ema=model
    # Set model to evaluation mode to disable things like BN, dropouts etc.
    model.eval()
    no_batches = len(dataloader)
    #Initialise model metrics
    val_loss = 0.
    #Initialise EMA model metrics
    val_loss_EMA = 0.
    val_mask_loss_EMA=0.
    # Disable gradient computations for the validation loop as we don't need it
    with torch.inference_mode():#torch.no_grad()
        for X, y in dataloader:
            #Convert all data to channels last format
            X = X
            y = y
            #Forward/backward pass on mask
            '''ONLY predict signal in low precision'''
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                #Forward pass to get predicted masks
                pred_denoised = model(X)#pred_denoised
                #Forward pass EMA model to get predicted masks
                pred_denoised_EMA = ema.ema_model(X)
            
            '''Loss computations in higher precision'''
            with torch.autocast(device_type=device, enabled=False):
                #Compute criterion
                loss = loss_fn(pred_denoised, y)
                loss_EMA = loss_fn(pred_denoised_EMA, y)
                #Compute metrics
                overlap = overlap_AET_batchwise(pred_denoised.float(), y, dataloader.dataset.dt, None)
                overlap_EMA = overlap_AET_batchwise(pred_denoised_EMA.float(), y, dataloader.dataset.dt, None)
            #No gradient scaling since no backwards pass
            #Record all model losses
            val_loss += loss.item()
            val_loss_EMA += loss_EMA.item()
            #Record all metrics
            val_overlap += overlap.item()
            val_overlap_EMA += overlap_EMA.item()

    #Report average model per batch losses
    final_loss= val_loss/no_batches
    final_loss_EMA= val_loss_EMA/no_batches
    print(f"Validation loss: {final_loss:.6E}")
    print(f"Validation EMA loss: {final_loss_EMA:.6E}")

    #Report average metrics by dividing by no. of batches
    final_overlap= val_overlap/no_batches
    final_overlap_EMA= val_overlap_EMA/no_batches
    print(f"Avg. overlap: {final_overlap:.6E}\n")
    print(f"Avg. overlap: {final_overlap_EMA:.6E}\n")

    #Append model val. history
    val_history.append(final_loss)
    val_history_EMA.append(final_loss_EMA)

    #Append metrics to history
    pred_overlap_history.append(final_overlap)
    pred_overlap_history_EMA.append(final_overlap_EMA)
