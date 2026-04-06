import torch
from torch.cuda.amp import GradScaler, autocast
from src.tools.transforms import *
from src.tools.losses import OverlapLoss, DetSNRLoss, CorrelationLoss, GradientDifferenceLoss, CharbonnierLoss

from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt

#Initialise some extra loss functions
'''Let's also try swapping out the mismatch loss for the det SNR loss. Maybe it works better?'''
'''Then, we'll also try SSIM loss on the masked data. No SNR or overlap etc.'''
overlap_loss_fn = CharbonnierLoss()#GradientDifferenceLoss()#OverlapLoss()
gradient_loss_fn = GradientDifferenceLoss()
# SNR_loss_fn = DetSNRLoss()

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

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, train_history, scaler, device, use_amp=True, max_abs_tensor=None):
    train_loss=0.#Sum of losses across all batches
    train_mask_loss=0.
    train_refinement_loss=0.
    train_SNR_loss=0.
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

    for batch, (X, y, mask_X) in enumerate(dataloader):
        #Convert all data to channels last format
        X = X.to(memory_format=torch.channels_last)
        y = y.to(memory_format=torch.channels_last)
        mask_X = mask_X.to(memory_format=torch.channels_last)
        #Set optimiser zero grad before any backward passes
        optimizer.zero_grad(set_to_none=True)
        #Forward/backward pass on mask
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            '''Dataloader is already preprocessing the data, and calculating masks'''
            #Forward pass to get predicted mask
            pred_mask = model.predict_coarse_mask(X)#model(X)
            #Compute mask loss
            region_weighting = None#torch.abs(mask_X)#/torch.amax(torch.abs(mask_X), dim=(-1,-2), keepdim=True)#torch.abs(y)/torch.abs(X)#torch.abs(mask_X)
            '''Fit directly for the EMRI rather than the mask!!!'''
            mask_loss = loss_fn(pred_mask, y, region_weighting = region_weighting)#mask_X
        #Scale mask loss, and do backwards pass
        scaler.scale(mask_loss).backward()#retain_graph=True
        #Detach mask for stage two
        pred_mask_detached = pred_mask.detach()
        #Free un-detached prediction from memory
        del pred_mask
        #Forward/backward pass on refined signal
        '''Temporarily disabling this'''
        refinement_loss = torch.tensor([0.0])
        pred_signal = 0.0
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Adjust parameters
        scaler.step(optimizer)#optimizer.step()
        scaler.update()
        #Record all losses
        loss = mask_loss.item()# + refinement_loss.item()
        train_mask_loss += mask_loss.item()#weighted_mask_loss
        train_refinement_loss += refinement_loss.item()#weighted_overlap_loss
        train_loss += loss#.item()
        # pred_mask, pred_signal = model(X)#X_transformed
        #Calculate region weighting
        #Calculate component unweighted loss
        # overlap_loss = overlap_loss_fn(pred_signal, y, region_weighting = region_weighting)###pred_mask, mask_X
        # snr_loss = SNR_loss_fn(pred_signal, y)
        # Keeping track of each unweighted loss component
        # values_of_component_1.append(mask_loss.item())
        # values_of_component_2.append(overlap_loss.item())
        # values_of_component_3.append(snr_loss.item())
        #Calculate weighted component losses
        # weighted_mask_loss = adapt_weights[0] * mask_loss
        # weighted_overlap_loss = adapt_weights[1] * overlap_loss
        # weighted_snr_loss = adapt_weights[2] * snr_loss
        # Backtransform the predicted signal
        # pred_signal_backtransformed = normalise(pred_signal, scale_factor=max_abs_tensor, forward=False)
        # pred_signal_backtransformed = asinh_transform(pred_signal_backtransformed, a=1 / 1e-13, c=0.0, forward=False)
        #Calculate network matched filter SNR of the predicted signal
        # pred_channelwise_SNR = torch.nansum((pred_signal_backtransformed * y), dim=(-2,-1)) / target_channelwise_SNR#(pred_signal * y_transformed)
        # pred_SNR = torch.linalg.vector_norm((torch.nansum((pred_signal * X_transformed), dim=(-2,-1)) / target_channelwise_SNR), dim=-1)#torch.sqrt(torch.sum((torch.nansum((pred_signal * X_transformed), dim=(1,2)) / target_channelwise_SNR)**2))
        #Compute the weighted loss
        '''
        Need to be a bit smart here.
        Pixel-wise accuracy of the predicted mask was useful.
        Probably even more useful would be structural similarity of the mask.
        Leave out the predicted waveform for now. It doesn't seem to optimise well.
        Let's consider: Charbonnier loss of mask + SSIM of mask.
        Det. SNR is not optimising, maybe because we're using the signal.
        Leave it out for now.'''
        # loss = mask_loss + overlap_loss#weighted_mask_loss + weighted_overlap_loss# + weighted_snr_loss# + SNR_loss_fn(pred_signal, y_transformed)
        # Accumulate component losses over the epoch
        # Report the current loss of a given batch if desired
        if batch % 10 == 0:#10
            loss, current = loss, batch * batch_size + len(X)#.item()
            print(f"loss: {loss:.6E}  [{current}/{size}]")
        #Free all un-needed variables
        del pred_signal, mask_loss, refinement_loss, pred_mask_detached

    #Report average loss by dividing by no. of batches
    final_loss= train_loss/no_batches
    final_mask_loss= train_mask_loss/no_batches
    final_refinement_loss= train_refinement_loss/no_batches

    print(f"Avg. total training loss: {final_loss:.6E}\n Component losses: {final_mask_loss:.6E} , {final_refinement_loss:.6E}")

    #Append loss to history
    train_history.append(final_loss)

    #print allocated memory
    print(f"Allocated memory (not including spikes): {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9} GB")

def val_loop(dataloader, model, loss_fn, val_history, scaler, device, use_amp=True, max_abs_tensor=None):
    # Set model to evaluation mode to disable things like BN, dropouts etc.
    model.eval()
    no_batches = len(dataloader)
    val_loss = 0.
    val_mask_loss=0.
    val_refinement_loss=0.
    val_SNR_loss=0.
    global adapt_weights
    # Disable gradient computations for the validation loop as we don't need it
    with torch.inference_mode():#torch.no_grad()
        for X, y, mask_X in dataloader:#batch, enumerate()
            #Convert all data to channels last format
            X = X.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.channels_last)
            mask_X = mask_X.to(memory_format=torch.channels_last)
            #Forward/backward pass on mask
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                '''Dataloader is already preprocessing the data, and calculating masks'''
                #Forward pass to get predicted mask
                pred_mask = model.predict_coarse_mask(X)#model(X)
                #Compute mask loss
                region_weighting = None#torch.abs(mask_X)#/torch.amax(torch.abs(mask_X), dim=(-1,-2), keepdim=True)#torch.abs(y)/torch.abs(X)#torch.abs(mask_X)
                '''Directly fitting the EMRI'''
                mask_loss = loss_fn(pred_mask, y, region_weighting = region_weighting)#mask_X
            #No gradient scaling since no backwards pass
            #No detaching of mask
            #Forward/backward pass on refined signal
            '''Temporarily disabling this part'''
            refinement_loss = torch.tensor([0.0])
            pred_signal = 0.0
            # with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            #     #Forward pass to get predicted signal
            #     pred_signal = 0.0#model.predict_refined_signal(X, pred_mask)#(X)
            #     #Compute signal loss
            #     region_weighting = None#torch.abs(mask_X)#/torch.amax(torch.abs(mask_X), dim=(-1,-2), keepdim=True)#torch.abs(y)/torch.abs(X)#torch.abs(mask_X)
            #     refinement_loss = 0.0#overlap_loss_fn(pred_signal, y, region_weighting = region_weighting)
            #     refinement_loss = refinement_loss# + 0.01 * gradient_loss_fn(pred_signal, y)

            #No gradient scaling since no backward pass
            #Record all losses
            loss = mask_loss.item()# + refinement_loss.item()
            val_mask_loss += mask_loss.item()#weighted_mask_loss
            val_refinement_loss += refinement_loss.item()#weighted_overlap_loss
            val_loss += loss#.item()
            del pred_signal, mask_loss, refinement_loss, pred_mask
            #Transform and normalise input and target
            # X_transformed = asinh_transform(X, a=1 / 1e-13, c=0.0, forward=True)
            # X_transformed= normalise(X_transformed, scale_factor=max_abs_tensor, forward=True)
            # y_transformed = asinh_transform(y, a=1 / 1e-13, c=0.0, forward=True)
            # y_transformed= normalise(y_transformed, scale_factor=max_abs_tensor, forward=True)
            # #Also transform just the noise for later
            # corruption = X - y
            # corruption_transformed = asinh_transform(corruption, a=1 / 1e-13, c=0.0, forward=True)
            # corruption_transformed = normalise(corruption_transformed, scale_factor=max_abs_tensor, forward=True)
            # #Calculate target mask
            # target_mask = y_transformed/X_transformed
            #Calculate target network SNR i.e. the optimal SNR
            # target_channelwise_SNR = torch.sqrt(torch.nansum((y * y), dim=(-2,-1)))#y_transformed * y_transformed
            # target_SNR = torch.linalg.vector_norm(target_channelwise_SNR, dim=-1)#torch.sqrt(torch.sum(target_channelwise_SNR**2))
            #Free all un-needed variables

    #Report average per batch loss
    final_loss= val_loss/no_batches
    final_mask_loss= val_mask_loss/no_batches
    final_refinement_loss= val_refinement_loss/no_batches
    print(f"Validation loss: {final_loss:.6E}\n Component losses: {final_mask_loss:.6E} , {final_refinement_loss:.6E}")

    #Append val. history
    val_history.append(final_loss)