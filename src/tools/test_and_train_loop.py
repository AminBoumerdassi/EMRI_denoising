import torch
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import v2
from torch.distributions import Uniform
from torch.cuda.amp import GradScaler, autocast
from src.tools.transforms import *
from src.tools.losses import OverlapLoss, DetSNRLoss, CorrelationLoss, GradientDifferenceLoss, CharbonnierLoss, LogMagnitudeL1Loss, OptSNRLoss, CrossWaveletPowerLoss

from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt

#Initialise some extra loss functions
'''Let's also try swapping out the mismatch loss for the det SNR loss. Maybe it works better?'''
'''Then, we'll also try SSIM loss on the masked data. No SNR or overlap etc.'''
overlap_loss_fn = OverlapLoss(eps=1e-21)#eps=1e-4
gradient_loss_fn = GradientDifferenceLoss(eps=1e-13)
SNR_loss_fn = DetSNRLoss()
log_mag_loss = LogMagnitudeL1Loss(epsilon=1e-13)
opt_SNR_loss = OptSNRLoss(eps=1e-8)
XWT_power_loss = CrossWaveletPowerLoss(scale_factor=1.0, eps=1e-13)#eps=1e-8

def si_sdr_loss_time_freq(preds, target, eps=1e-12):
    """
    Computes the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    Args:
        preds: Predicted signal (Batch, Time)
        target: Ground truth signal (Batch, Time)
    Returns:
        The average negative SI-SDR across the batch.
    """
    # 1. Zero-mean the signals
    target = target - torch.mean(target, dim=(-2,-1), keepdim=True)
    preds = preds - torch.mean(preds, dim=(-2,-1), keepdim=True)

    # 2. Project target onto preds to find the scale (alpha)
    # alpha = <preds, target> / ||target||^2
    alpha = (preds * target).sum((-2,-1)) / (((target ** 2).sum((-2,-1))) + eps)
    target = target * alpha.unsqueeze(-1).unsqueeze(-1)
    res = preds - target
    losses = 10 * torch.log10(
        (target ** 2).sum((-2,-1)) / ((res ** 2).sum((-2,-1)) + eps) + eps
    )

    # Return negative for minimization
    return -torch.mean(losses)

def Overlap_tf_Metric(input, target, eps=1e-4):
    '''
    Return the average overlap in a given batch of data. 
    This is ONLY valid when the input and target are already whitened.
    '''
    #Cast objects to float64 precision
    input_64 = input.double()
    target_64 = target.double()
    input_dot_target = torch.nansum(input_64*target_64, dim=(-2,-1))
    target_dot_target = torch.nansum(target_64*target_64, dim=(-2,-1))
    input_dot_input = torch.nansum(input_64*input_64, dim=(-2,-1))
    #Calculate denominator terms individually
    input_norm  = torch.sqrt(input_dot_input + eps)#torch.sqrt(torch.clamp(input_dot_input, min=self.eps))
    target_norm = torch.sqrt(target_dot_target + eps)#torch.sqrt(torch.clamp(target_dot_target, min=self.eps))
    #Calculate the match
    match = input_dot_target / (input_norm * target_norm)#input_dot_target * torch.rsqrt(target_dot_target * input_dot_input + self.eps)
    match = torch.mean(match)
    return match.float()


#mismatch_loss_fn = DetSNRLoss()#MismatchLoss()
# SNR_loss_fn = DetSNRLoss()

#Define how often to change loss weights
epochs_to_make_updates = 3#1

# Initialise softadapt object for loss reweighting
'''accuracy order is no_batches*epochs_to_make_updates - 2.
Cannot be odd if >5'''
softadapt_object = LossWeightedSoftAdapt(beta=0.1, accuracy_order=(32*epochs_to_make_updates-2))

# Initialize lists to keep track of loss values over the epochs we defined above
values_of_component_1 = []
values_of_component_2 = []
values_of_component_3 = []

# Initializing adaptive weights to all ones.
'''SNR loss greatly dominates by around 1e-5 relative to mask loss.
So try weighting the SNR loss by 1e-5'''
adapt_weights = torch.tensor([1.0, 1.0, 1.0])#1e-1

#Initialise masking objects
time_masking = TimeMasking(int(0.05*512), iid_masks=True, p=0.3)#.to(device)
freq_masking = FrequencyMasking(int(0.05*512), iid_masks=True)#.to(device)

#Initialise random uniform sampler for Gaussian noise level
rand_noise_std = Uniform(0.01, 0.05)

#Initialise random shifts in time axis
rand_time_shift = v2.RandomAffine(0,translate=(0.05,0))

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
    train_mask_loss=0.
    train_refinement_loss=0.
    train_SNR_loss=0.
    train_prediction_overlap= 0.0
    train_residual_overlap= 0.0
    no_batches = len(dataloader)
    size = len(dataloader.dataset)
    # Set model to training mode to ensure things like BN and dropout etc. are on
    model.train()
    # # Change 4: Make sure `epochs_to_make_change` have passed before calling SoftAdapt.
    global values_of_component_1
    global values_of_component_2
    global values_of_component_3
    global adapt_weights
    # if len(train_history) % epochs_to_make_updates == 0 and len(train_history) != 0:# 
    #     adapt_weights = softadapt_object.get_component_weights(torch.tensor(values_of_component_1), 
    #                                                             torch.tensor(values_of_component_2), 
    #                                                             verbose=False,
    #                                                             )
    #     # Resetting the lists to start fresh (this part is optional)
    #     values_of_component_1 = []
    #     values_of_component_2 = []

    for batch, (X, y, residual, mask_X, mask_residual) in enumerate(dataloader):
        #Convert all data to channels last format
        X = X.to(memory_format=torch.channels_last)
        y = y.to(memory_format=torch.channels_last)
        mask_X = mask_X.to(memory_format=torch.channels_last)
        mask_residual = mask_residual.to(memory_format=torch.channels_last)
        #Set optimiser zero grad before any backward passes
        optimizer.zero_grad(set_to_none=True)
        #Forward/backward pass on mask
        '''ONLY mask estimation in low precision. Not applying the mask, or calculating losses!'''
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            '''NOTE: we are masking the data in time and frequency! Turn off later!'''
            # #Generate light Gaussian noise at randomised noise levels
            # gaussian_noise = rand_noise_std.sample((X.shape[0],))[:,None,None,None].to(device) * torch.randn_like(X, device=device)
            # #Randomly set some samples to not have Gaussian noise
            # idxs = torch.argwhere(torch.randint(0, 2, (X.shape[0],))==1).flatten()
            # #Add light Gaussian noise
            # X[idxs,...] = X[idxs,...] + gaussian_noise[idxs,...]
            # #Randomly set some samples to have light dropout
            # idxs = torch.argwhere(torch.randint(0, 2, (X.shape[0],))==1).flatten()
            # #Add light dropout
            # X[idxs,...] = torch.nn.functional.dropout(X[idxs,...], 0.05)
            # #Randomly set some samples to have light time masking
            # idxs = torch.argwhere(torch.randint(0, 2, (X.shape[0],))==1).flatten()
            # #Add light time masking
            # X[idxs,...] = time_masking(X[idxs,...])
            # #Randomly set some samples to have light freq. masking
            # idxs = torch.argwhere(torch.randint(0, 2, (X.shape[0],))==1).flatten()
            # #Add light freq. masking
            # X[idxs,...] = freq_masking(X[idxs,...])
            # #Randomly set some samples to have a small time shift
            # idxs = torch.argwhere(torch.randint(0, 2, (X.shape[0],))==1).flatten()
            # #Add a shift in the time axis
            # X[idxs,...] = rand_time_shift(X[idxs,...])
            '''Dataloader is already preprocessing the data, and calculating masks'''
            #Forward pass to get predicted masks
            pred_masks, _ = model(X)#pred_denoised
        
        # '''Preprocessing to be done in float64 precision'''
        # #Define parameters for applying mask, and doing backtransforms                    
        # max_abs_tensor= 53.0#106.0#
        # inverse=True
        # asinh_scale= 1e23#1e46#
        # max_abs= 80.0#45.0
        # scaling_factor= 1e28#1e14
        # #Backtransform input, target, and residual
        # X_bt = preprocess_signal(X.double(), 
        #                     asinh_scale=asinh_scale, 
        #                     # asinh_shift=asinh_shift, 
        #                     max_absolute=max_abs_tensor,
        #                     inverse = inverse
        #                     )#.float()
        # y_bt = preprocess_signal(y.double(), 
        #                     asinh_scale=asinh_scale, 
        #                     # asinh_shift=asinh_shift, 
        #                     max_absolute=max_abs_tensor,
        #                     inverse = inverse
        #                     )#.float()
        # residual_bt = preprocess_signal(residual.double(), 
        #                     asinh_scale=asinh_scale, 
        #                     # asinh_shift=asinh_shift, 
        #                     max_absolute=max_abs_tensor,
        #                     inverse = inverse
        #                     )#.float()
        # #Apply the denoising mask to the backtransformed data
        # pred_denoised_bt = X_bt * preprocess_mask(pred_masks.double(), 
        #                             max_abs=max_abs,
        #                             scaling_factor=scaling_factor,
        #                     #  power_factor=power_factor, 
        #                         inverse=inverse
        #                         )#.float()
        # #Calculate predicted backtransformed residual
        # pred_residual_bt = X_bt - pred_denoised_bt
        # #Calculate the predictions and residuals in the transformed domain
        # pred_denoised = preprocess_signal(pred_denoised_bt, 
        #                     asinh_scale=asinh_scale, 
        #                     # asinh_shift=asinh_shift, 
        #                     max_absolute=max_abs_tensor,
        #                     inverse = False
        #                     ).float()
        # pred_residual = preprocess_signal(pred_residual_bt, 
        #                     asinh_scale=asinh_scale, 
        #                     # asinh_shift=asinh_shift, 
        #                     max_absolute=max_abs_tensor,
        #                     inverse = False
        #                     ).float()
        # # max_abs_tensor= torch.tensor([30.990543, 30.842495], device=device)[:,None,None]#torch.as_tensor([37.57473783, 36.71401857], device=device)[None,:,None,None]
        with torch.autocast(device_type=device, enabled=False):
            '''Loss computations and preprocessing in higher precision'''
            #Define parameters for applying mask, and doing backtransforms                    
            max_abs_tensor= 53.0#106.0#
            inverse=True
            asinh_scale= 1e23#1e46#
            max_abs= 80.0#45.0
            scaling_factor= 1e28#1e14
            #Backtransform input, target, and residual
            X_bt = preprocess_signal(X.double(), 
                                asinh_scale=asinh_scale, 
                                # asinh_shift=asinh_shift, 
                                max_absolute=max_abs_tensor,
                                inverse = inverse
                                )#.float()
            y_bt = preprocess_signal(y.double(), 
                                asinh_scale=asinh_scale, 
                                # asinh_shift=asinh_shift, 
                                max_absolute=max_abs_tensor,
                                inverse = inverse
                                )#.float()
            residual_bt = preprocess_signal(residual, 
                                asinh_scale=asinh_scale, 
                                # asinh_shift=asinh_shift, 
                                max_absolute=max_abs_tensor,
                                inverse = inverse
                                ).float()
            #Apply the denoising mask to the backtransformed data
            pred_denoised_bt = X_bt * preprocess_mask(pred_masks.double(), 
                                        max_abs=max_abs,
                                        scaling_factor=scaling_factor,
                                #  power_factor=power_factor, 
                                    inverse=inverse
                                    )#.float()
            #Calculate predicted backtransformed residual
            pred_residual_bt = X_bt - pred_denoised_bt
            #Calculate the predictions and residuals in the transformed domain
            pred_denoised = preprocess_signal(pred_denoised_bt, 
                                asinh_scale=asinh_scale, 
                                # asinh_shift=asinh_shift, 
                                max_absolute=max_abs_tensor,
                                inverse = False
                                ).float()
            pred_residual = preprocess_signal(pred_residual_bt, 
                                asinh_scale=asinh_scale, 
                                # asinh_shift=asinh_shift, 
                                max_absolute=max_abs_tensor,
                                inverse = False
                                ).float()
            # max_abs_tensor= torch.tensor([30.990543, 30.842495], device=device)[:,None,None]#torch.as_tensor([37.57473783, 36.71401857], device=device)[None,:,None,None]
            #Compute mask loss
            '''An idea for noise-aware loss weighting: 
            something simple - log(|h+g|) / log(|h|) + eps
            or even simpler: 5* weighting wherever g>h'''
            region_weighting = None#torch.abs(mask_X)#None#torch.abs(X/(y+1e-8))#1 + torch.log(1 + torch.abs(residual_bt)/(torch.abs(y_bt)+1e-8))#torch.log(torch.abs(X_bt)) / torch.log((torch.abs(y_bt)+1e-8))#None#torch.abs(mask_X)#/torch.amax(torch.abs(mask_X), dim=(-1,-2), keepdim=True)#torch.abs(y)/torch.abs(X)#torch.abs(mask_X)
            #Enforce the pixel-wise accuracy of each prediction
            mask_loss = loss_fn(pred_masks, mask_X, region_weighting = region_weighting, normalise=False)#pred_EMRI, y,#pred_EMRI_bt, y_bt,
            #Penalise the model for moving in a way that does not maximise the EMRI overlap
            mask_loss = mask_loss + 0.001 * overlap_loss_fn(pred_denoised_bt, y_bt)#0.1
            mask_loss = mask_loss + 0.001 * overlap_loss_fn(pred_residual_bt, residual_bt)#0.1
            #Penalise the model for blurry denoising
            #mask_loss = mask_loss + 0.01 * gradient_loss_fn(pred_denoised_bt, y_bt)
            #mask_loss = mask_loss + 0.01 * gradient_loss_fn(pred_residual_bt, residual_bt)
            #Penalise the model for incorrect XWT power
            '''mask_loss = mask_loss + 0.01 * XWT_power_loss(pred_denoised_bt, y_bt)#0.1
            mask_loss = mask_loss + 0.01 * XWT_power_loss(pred_residual_bt, residual_bt)#0.1'''
            # #Penalise the model for incorrect STFT magnitudes
            '''Added region weighting for the log mag on denoised'''
            mask_loss = mask_loss + 0.001 * log_mag_loss(pred_denoised_bt, y_bt, region_weighting = region_weighting)#0.1
            '''Added region weighting on log mag loss on residual'''
            mask_loss = mask_loss + 0.001 * log_mag_loss(pred_residual_bt, residual_bt)#, region_weighting = torch.abs(mask_residual), 0.1
            #Calculate the average overlap for batch inputs and predictions
            prediction_overlap = Overlap_tf_Metric(pred_denoised_bt, y_bt, eps=1e-21)
            residual_overlap = Overlap_tf_Metric(pred_residual_bt, residual_bt, eps=1e-21)
            #input_overlap = Overlap_tf_Metric(X_bt, y_bt)
        #Scale mask loss, and do backwards pass
        scaler.scale(mask_loss).backward()#retain_graph=True
        #Detach mask for stage two
        #pred_mask_detached = pred_masks.detach()
        #Free un-detached prediction from memory
        #del pred_masks, pred_EMRI_mask, pred_residual_mask
        #Forward/backward pass on refined signal
        '''Temporarily disabling this'''
        refinement_loss = torch.tensor([0.0])
        #pred_signal = 0.0
        # with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
        #     #Forward pass to get predicted signal
        #     pred_signal = model.predict_refined_signal(X, pred_mask_detached)#(X)
        #     #Compute signal loss
        #     region_weighting = None#torch.abs(mask_X)#/torch.amax(torch.abs(mask_X), dim=(-1,-2), keepdim=True)#torch.abs(y)/torch.abs(X)#torch.abs(mask_X)
        #     '''Temporarily disabling this!'''
        #     refinement_loss = overlap_loss_fn(pred_signal, y, region_weighting = region_weighting)
        #     refinement_loss = refinement_loss# + 0.01 * gradient_loss_fn(pred_signal, y)
        # #Scale signal loss and do backwards pass
        # scaler.scale(refinement_loss).backward(retain_graph=False)
        #Unscale gradient before gradient clipping
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
        loss = mask_loss.item()# + refinement_loss.item()
        train_mask_loss += mask_loss.item()#weighted_mask_loss
        train_refinement_loss += refinement_loss.item()#weighted_overlap_loss
        train_loss += loss#.item()
        train_prediction_overlap += prediction_overlap.item()
        train_residual_overlap += residual_overlap.item()
        # Report the current loss of a given batch if desired
        if batch % 10 == 0:#10
            loss, current = loss, batch * batch_size + len(X)#.item()
            print(f"loss: {loss:.6E}  [{current}/{size}]")

    #Report average loss by dividing by no. of batches
    final_loss= train_loss/no_batches
    final_mask_loss= train_mask_loss/no_batches
    final_refinement_loss= train_refinement_loss/no_batches
    final_prediction_overlap = train_prediction_overlap/no_batches
    final_residual_overlap = train_residual_overlap/no_batches

    print(f"Avg. total training loss: {final_loss:.6E}\n Component losses: {final_mask_loss:.6E} , {final_refinement_loss:.6E}")
    print(f"Training  overlap: {final_prediction_overlap:.3E}")
    print(f"Residual  overlap: {final_residual_overlap:.3E}")


    #Append loss to history
    train_history.append(final_loss)
    pred_overlap_history.append(final_prediction_overlap)
    residual_overlap_history.append(final_residual_overlap)

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
    val_mask_loss=0.
    val_refinement_loss=0.
    val_SNR_loss=0.
    val_prediction_overlap= 0.0
    val_residual_overlap= 0.0
    val_input_overlap = 0.0
    #Initialise EMA model metrics
    val_loss_EMA = 0.
    val_mask_loss_EMA=0.
    val_refinement_loss_EMA=0.
    val_SNR_loss_EMA=0.
    val_prediction_overlap_EMA= 0.0
    val_input_overlap_EMA = 0.0
    val_residual_overlap_EMA = 0.0
    global adapt_weights
    # Disable gradient computations for the validation loop as we don't need it
    with torch.inference_mode():#torch.no_grad()
        for X, y, residual, mask_X, mask_residual in dataloader:
            #Convert all data to channels last format
            X = X.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.channels_last)
            mask_X = mask_X.to(memory_format=torch.channels_last)
            mask_residual = mask_residual.to(memory_format=torch.channels_last)
            #Forward/backward pass on mask
            '''ONLY mask estimation in low precision. Not applying the mask, or calculating losses!'''
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                '''Dataloader is already preprocessing the data, and calculating masks'''
                #Forward pass to get predicted masks
                pred_masks, _ = model(X)#pred_denoised
                #Forward pass EMA model to get predicted masks
                pred_masks_EMA, _ = ema.ema_model(X)#pred_denoised_EMA
            
            # '''Loss computations and preprocessing in higher precision'''
            # #Define parameters for applying mask, and doing backtransforms                    
            # max_abs_tensor= 53.0#106.0#
            # inverse=True
            # asinh_scale= 1e23#1e46#
            # max_abs= 80.0#45.0
            # scaling_factor= 1e28#1e14
            # #Backtransform input, target, and residual
            # X_bt = preprocess_signal(X.double(), 
            #                     asinh_scale=asinh_scale, 
            #                     # asinh_shift=asinh_shift, 
            #                     max_absolute=max_abs_tensor,
            #                     inverse = inverse
            #                     )#.float()
            # y_bt = preprocess_signal(y.double(), 
            #                     asinh_scale=asinh_scale, 
            #                     # asinh_shift=asinh_shift, 
            #                     max_absolute=max_abs_tensor,
            #                     inverse = inverse
            #                     )#.float()
            # residual_bt = preprocess_signal(residual.double(), 
            #                     asinh_scale=asinh_scale, 
            #                     # asinh_shift=asinh_shift, 
            #                     max_absolute=max_abs_tensor,
            #                     inverse = inverse
            #                     )#.float()
            # #Apply the denoising mask
            # pred_denoised_bt = X_bt * preprocess_mask(pred_masks.double(), 
            #                             max_abs=max_abs,
            #                             scaling_factor=scaling_factor,
            #                     #  power_factor=power_factor, 
            #                     inverse=inverse
            #                     )#.float()
            # pred_denoised_EMA_bt = X_bt * preprocess_mask(pred_masks_EMA.double(), 
            #                             max_abs=max_abs,
            #                             scaling_factor=scaling_factor,
            #                     #  power_factor=power_factor, 
            #                     inverse=inverse
            #                     )#.float()
            # #Calculate predicted backtransformed residual
            # pred_residual_bt = X_bt - pred_denoised_bt
            # pred_residual_EMA_bt = X_bt - pred_denoised_EMA_bt
            # #Calculate the predictions and residuals in the transformed domain
            # pred_denoised = preprocess_signal(pred_denoised_bt, 
            #                     asinh_scale=asinh_scale, 
            #                     # asinh_shift=asinh_shift, 
            #                     max_absolute=max_abs_tensor,
            #                     inverse = False
            #                     ).float()
            # pred_denoised_EMA = preprocess_signal(pred_denoised_EMA_bt, 
            #                     asinh_scale=asinh_scale, 
            #                     # asinh_shift=asinh_shift, 
            #                     max_absolute=max_abs_tensor,
            #                     inverse = False
            #                     ).float()
            # pred_residual = preprocess_signal(pred_residual_bt, 
            #                     asinh_scale=asinh_scale, 
            #                     # asinh_shift=asinh_shift, 
            #                     max_absolute=max_abs_tensor,
            #                     inverse = False
            #                     ).float()
            # pred_residual_EMA = preprocess_signal(pred_residual_EMA_bt, 
            #                     asinh_scale=asinh_scale, 
            #                     # asinh_shift=asinh_shift, 
            #                     max_absolute=max_abs_tensor,
            #                     inverse = False
            #                     ).float()
            with torch.autocast(device_type=device, enabled=False):
                '''Loss computations and preprocessing in higher precision'''
                #Define parameters for applying mask, and doing backtransforms                    
                max_abs_tensor= 53.0#106.0#
                inverse=True
                asinh_scale= 1e23#1e46#
                max_abs= 80.0#45.0
                scaling_factor= 1e28#1e14
                #Backtransform input, target, and residual
                X_bt = preprocess_signal(X.double(), 
                                    asinh_scale=asinh_scale, 
                                    # asinh_shift=asinh_shift, 
                                    max_absolute=max_abs_tensor,
                                    inverse = inverse
                                    )#.float()
                y_bt = preprocess_signal(y.double(), 
                                    asinh_scale=asinh_scale, 
                                    # asinh_shift=asinh_shift, 
                                    max_absolute=max_abs_tensor,
                                    inverse = inverse
                                    )#.float()
                residual_bt = preprocess_signal(residual, 
                                    asinh_scale=asinh_scale, 
                                    # asinh_shift=asinh_shift, 
                                    max_absolute=max_abs_tensor,
                                    inverse = inverse
                                    ).float()
                #Apply the denoising mask
                pred_denoised_bt = X_bt * preprocess_mask(pred_masks.double(), 
                                            max_abs=max_abs,
                                            scaling_factor=scaling_factor,
                                    #  power_factor=power_factor, 
                                    inverse=inverse
                                    )#.float()
                pred_denoised_EMA_bt = X_bt * preprocess_mask(pred_masks_EMA.double(), 
                                            max_abs=max_abs,
                                            scaling_factor=scaling_factor,
                                    #  power_factor=power_factor, 
                                    inverse=inverse
                                    )#.float()
                #Calculate predicted backtransformed residual
                pred_residual_bt = X_bt - pred_denoised_bt
                pred_residual_EMA_bt = X_bt - pred_denoised_EMA_bt
                #Calculate the predictions and residuals in the transformed domain
                pred_denoised = preprocess_signal(pred_denoised_bt, 
                                    asinh_scale=asinh_scale, 
                                    # asinh_shift=asinh_shift, 
                                    max_absolute=max_abs_tensor,
                                    inverse = False
                                    ).float()
                pred_denoised_EMA = preprocess_signal(pred_denoised_EMA_bt, 
                                    asinh_scale=asinh_scale, 
                                    # asinh_shift=asinh_shift, 
                                    max_absolute=max_abs_tensor,
                                    inverse = False
                                    ).float()
                pred_residual = preprocess_signal(pred_residual_bt, 
                                    asinh_scale=asinh_scale, 
                                    # asinh_shift=asinh_shift, 
                                    max_absolute=max_abs_tensor,
                                    inverse = False
                                    ).float()
                pred_residual_EMA = preprocess_signal(pred_residual_EMA_bt, 
                                    asinh_scale=asinh_scale, 
                                    # asinh_shift=asinh_shift, 
                                    max_absolute=max_abs_tensor,
                                    inverse = False
                                    ).float()
                #Compute mask loss
                '''Maybe switch to 1 + torch.abs(mask_X) otherwise noisy regions get lost.'''
                region_weighting = None#torch.abs(mask_X)#torch.abs(X/(y+1e-8))#1 + torch.log(1 + torch.abs(residual_bt)/(torch.abs(y_bt)+1e-8))#torch.log(torch.abs(X_bt)) / torch.log(torch.abs(y_bt)+1e-8)#None#torch.abs(mask_X)#/torch.amax(torch.abs(mask_X), dim=(-1,-2), keepdim=True)#torch.abs(y)/torch.abs(X)#torch.abs(mask_X)
                '''Fit for the EMRI and glitch mask simultaneously'''
                #Enforce the pixel-wise accuracy of each prediction
                mask_loss = loss_fn(pred_masks, mask_X, region_weighting = region_weighting, normalise=False)#pred_EMRI, y,#pred_EMRI_bt, y_bt,
                mask_loss_EMA = loss_fn(pred_masks_EMA, mask_X, region_weighting = region_weighting, normalise=False)#pred_EMRI, y,#pred_EMRI_bt, y_bt,                
                #Penalise the model for moving in a way that does not maximise the EMRI overlap
                mask_loss = mask_loss + 0.001 * overlap_loss_fn(pred_denoised_bt, y_bt)#0.1
                mask_loss = mask_loss + 0.001 * overlap_loss_fn(pred_residual_bt, residual_bt)#0.1
                #mask_loss_EMA = mask_loss_EMA + 0.1 * overlap_loss_fn(pred_denoised_EMA_bt, y_bt)#0.5
                #mask_loss_EMA = mask_loss_EMA + 0.1 * overlap_loss_fn(pred_residual_EMA_bt, residual_bt)#0.5
                #Penalise the model for blurry denoising
                #mask_loss = mask_loss + 0.01 * gradient_loss_fn(pred_denoised_bt, y_bt)
                #mask_loss_EMA = mask_loss_EMA + 0.01 * gradient_loss_fn(pred_denoised_EMA_bt, y_bt)
                #mask_loss = mask_loss + 0.01 * gradient_loss_fn(pred_residual_bt, residual_bt)
                #mask_loss_EMA = mask_loss_EMA + 0.01 * gradient_loss_fn(pred_residual_EMA_bt, residual_bt)
                # #Penalise the model for incorrect STFT magnitudes
                '''Added region weighting for the log mass on denoised'''
                mask_loss = mask_loss + 0.001 * log_mag_loss(pred_denoised_bt, y_bt, region_weighting=region_weighting)
                mask_loss_EMA = mask_loss_EMA + 0.001 * log_mag_loss(pred_denoised_EMA_bt, y_bt, region_weighting=region_weighting)
                '''Added region weighting on log mag loss on residual'''
                mask_loss = mask_loss + 0.001 * log_mag_loss(pred_residual_bt, residual_bt)#, region_weighting = torch.abs(mask_residual)
                mask_loss_EMA = mask_loss_EMA + 0.001 * log_mag_loss(pred_residual_EMA_bt, residual_bt)#, region_weighting = torch.abs(mask_residual)
                #Penalise the model for incorrect XWT powers
                '''mask_loss = mask_loss + 0.01 * XWT_power_loss(pred_denoised_bt, y_bt)#0.1
                mask_loss = mask_loss + 0.01 * XWT_power_loss(pred_residual_bt, residual_bt)#0.1
                mask_loss_EMA = mask_loss_EMA + 0.01 * XWT_power_loss(pred_denoised_EMA_bt, y_bt)#0.1
                mask_loss_EMA = mask_loss_EMA + 0.01 * XWT_power_loss(pred_residual_EMA_bt, residual_bt)#0.1'''
            #Calculate the average overlap for batch inputs and predictions
            prediction_overlap = Overlap_tf_Metric(pred_denoised_bt, y_bt, eps=1e-21)
            prediction_overlap_EMA = Overlap_tf_Metric(pred_denoised_EMA_bt, y_bt, eps=1e-21)
            residual_overlap = Overlap_tf_Metric(pred_residual_bt, residual_bt, eps=1e-21)
            residual_overlap_EMA = Overlap_tf_Metric(pred_residual_EMA_bt, residual_bt, eps=1e-21)
            input_overlap = Overlap_tf_Metric(X_bt, y_bt, eps=1e-21)
            #No gradient scaling since no backwards pass
            #No detaching of mask
            #Forward/backward pass on refined signal
            '''Temporarily disabling this part'''
            # refinement_loss = torch.tensor([0.0])
            #Record all model losses and metrics
            loss = mask_loss.item()
            val_mask_loss += mask_loss.item()
            # val_refinement_loss += refinement_loss.item()
            val_loss += loss
            val_prediction_overlap += prediction_overlap.item()
            val_input_overlap += input_overlap.item()
            val_residual_overlap += residual_overlap.item()
            #Record all EMA model losses and metrics
            loss_EMA = mask_loss_EMA.item()
            val_mask_loss_EMA += mask_loss_EMA.item()
            # val_refinement_loss += refinement_loss_.item()
            val_loss_EMA += loss_EMA
            val_prediction_overlap_EMA += prediction_overlap_EMA.item()
            val_residual_overlap_EMA += residual_overlap_EMA.item()

    #Report average model per batch losses and metrics
    final_loss= val_loss/no_batches
    final_mask_loss= val_mask_loss/no_batches
    final_refinement_loss= val_refinement_loss/no_batches
    final_prediction_overlap = val_prediction_overlap/no_batches
    final_input_overlap = val_input_overlap/no_batches
    final_residual_overlap = val_residual_overlap/no_batches
    print(f"Validation loss: {final_loss:.6E}\n Component losses: {final_mask_loss:.6E} , {final_refinement_loss:.6E}")
    print(f"Validation overlap: {final_prediction_overlap:.3E}")
    print(f"Residual overlap: {final_residual_overlap:.3E}")

    #Report average EMA model per batch losses and metrics
    final_loss_EMA= val_loss_EMA/no_batches
    final_mask_loss_EMA= val_mask_loss_EMA/no_batches
    final_refinement_loss_EMA= val_refinement_loss_EMA/no_batches
    final_prediction_overlap_EMA = val_prediction_overlap_EMA/no_batches
    final_residual_overlap_EMA = val_residual_overlap_EMA/no_batches
    print(f"Validation EMA loss: {final_loss_EMA:.6E}\n Component losses: {final_mask_loss_EMA:.6E} , {final_refinement_loss_EMA:.6E}")
    print(f"Validation EMA overlap: {final_prediction_overlap_EMA:.3E}")
    print(f"Residual EMA overlap: {final_residual_overlap_EMA:.3E}")

    #Append model val. history
    val_history.append(final_loss)
    pred_overlap_history.append(final_prediction_overlap)
    residual_overlap_history.append(final_residual_overlap)
    input_overlap_history.append(final_input_overlap)
    #Append EMA model val. history
    val_history_EMA.append(final_loss_EMA)
    pred_overlap_history_EMA.append(final_prediction_overlap_EMA)
    residual_overlap_history_EMA.append(final_residual_overlap_EMA)