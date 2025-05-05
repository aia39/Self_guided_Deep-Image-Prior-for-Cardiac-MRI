"""
Created on Fri May  2 15:17:25 2025

@author: maistiak
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.fft as fft
from tqdm.notebook import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as ssim
import os
from torch.autograd import Variable
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py
import argparse

from pytorch3dunet.unet3d.model import get_model, UNet3D, ResidualUNet3D, ResidualUNetSE3D


'''
#########################
### Helper functions ####
#########################
'''

#### From h5py format to torch format #####
def h5py2torch(data):
    r = data['real']
    im = data['imag']
    compl = r + 1j*im
    compl = torch.from_numpy(compl).type(dtype=torch.complex64)
    return compl


def fft_with_shifts(img):
    return fft.fftshift(fft.fft2(fft.ifftshift(img)))

def ifft_with_shifts(ksp):
    return fft.fftshift(fft.ifft2(fft.ifftshift(ksp)))

def ksp_and_mps_to_gt(ksp, mps):
    gt = mps.conj() * ifft_with_shifts(ksp)
    gt = torch.sum(gt, axis=1)
    return gt

def mps_and_gt_to_ksp(mps, gt):
    ksp = fft_with_shifts(mps * gt)
    return ksp

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   ##Loading device



def get_mask_path(main_path, mask_suffix='_mask_ktGaussian24.mat'):
    '''
    Finding masks directory name from the kspace directory
    
    Parameters
    ----------
    -- main_path : string (path for the fully sampled kspace)
    -- mask_suffix : string (last suffix for specific type of mask)
    
    Returns:
    --------
    mask_path : string (mask path from the kspace path)
    '''
    mask_path = main_path.replace('FullSample', 'Mask_TaskAll')

    if mask_path.endswith('.mat'):
        mask_path = mask_path[:-4] + mask_suffix
    else:
        raise ValueError("Expected .mat file at the end of the path.")

    return mask_path



#######################
### Data Loading ######
#######################

def data_loading(ksp_direc, mask_type, acc_factor, slice_num, st_frame, end_frame, show= True):
    '''
    Accessing the fully sampled kspace and masks
    
    Parameters
    ----------
    -- ksp_direc : string  (path for fully sampled kspace)
    -- mask_type : character
    'r' = kt-radial
    'g' = kt-gaussian cartesian 
    'u' = kt-uniform cartesian
    -- acc_factor : int  (8|16|24)
    -- slice_num : int (#slice)
    -- st_frame : int (starting frame to clip)
    -- end_frame : int (ending frame to clip)
    -- show : True|False  (showing undersampled kspace)
    
    Returns
    -------
    -- gt_full : torch  , shape:(#frames, #coils, xdim, ydim); containing fully sampled kspace for all frames
    -- mask : torch   , shape::(#frames, xdim, ydim); containing undersampling masks for all frames 

    '''
    
    ###### Fully sampled K-space #####
    direc = ksp_direc
    
    if mask_type == 'r':
        direc_m = get_mask_path(direc, f'_mask_ktRadial{acc_factor}.mat')
    elif mask_type == 'g':
        direc_m = get_mask_path(direc, f'_mask_ktGaussian{acc_factor}.mat')
    else:
        #direc_m = f'/v/raid1b/backup/maistiak/cmrxrecon25/ChallengeData/MultiCoil/Perfusion/TrainingSet/Mask_TaskAll/Center00{center_nmbr}/{scanner_nmbr}/{patient_number}/perfusion_mask_Uniform{acc_factor}.mat'
        direc_m = get_mask_path(direc, f'_mask_Uniform{acc_factor}.mat')
    
    with h5py.File(direc, "r") as f:      
        cell_array_name = "kspace"
        
        cell_array = f[cell_array_name]
        gt_full = h5py2torch(cell_array[()])
        print(f'Full kspace shape is {gt_full.shape}')
    
    ######################
    #### Loading mask ####
    ######################
    
    with h5py.File(direc_m, 'r') as f:
        m = f['mask']
        mask = m[()]
        mask = torch.from_numpy(mask)
        print(f'Mask shape is {mask.shape}')
    
    ### indexing and clipping ###
    mask = mask[st_frame:end_frame,:,:]
    gt_full = gt_full[st_frame:end_frame, slice_num,:,:]
        
    if show:
        eps = 1e-10
        under = (gt_full * torch.unsqueeze(mask, axis = 1))
        full = gt_full.to(device)   
        fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
        for j in range(full.shape[1]):
            ax1.imshow(np.log(np.abs(under[0,j,:,:].cpu())+eps), alpha=0.5, cmap='gray')
            ax1.imshow(np.log(mask[j,:,:]+eps), alpha=0.5, cmap='gray')
            
            ax2.imshow(np.log(np.abs(full[0,j,:,:].cpu())+eps), alpha=0.5, cmap='gray')
            ax2.imshow(np.log(mask[j,:,:]+eps), alpha=0.5, cmap='gray')
            plt.show()
            
    return gt_full, mask


def calc_sensmap(ksp_full):
    '''
    Calculating sensitivity maps based on fully sampled kspace
    
    Parameters
    ----------
    ksp_full : torch
        FUll kspace, shape (#frames, #coils, xdim, ydim)

    Returns
    -------
    sens_maps
        shape:(#frames, #coils, xdim, ydim); containing sensitivity maps for all frames

    '''
    def estimate_sens_maps_torch(full_ksp, show_sens = True, show_img = True):
        """
        Estimate coil sensitivity maps from centered k-space using PyTorch.
    
        Parameters:
        -----------
            kspace: torch.complex64 ,shape (#frames, #coils, xdim, ydim)
    
        Returns:
        --------
            sens_maps: torch.complex64 ,shape (#frames, #coils, xdim, ydim)
        """
        #kspace = kspace[35,:,:,:]
        #kspace = kspace.mean(dim=0)
        coil_imgs = ifft_with_shifts(full_ksp)  # shape: (frames, coils, x, y)
        
        rss = torch.sqrt(torch.sum(torch.abs(coil_imgs) ** 2, dim=1, keepdim=True))  # shape: (frames, 1, x, y)
        sens_maps = coil_imgs / (rss + 1e-8)  # avoid divide by zero
        
        if show_sens:
            for it in range(sens_maps.shape[1]):
                plt.imshow((torch.abs(sens_maps[0,it,:,:])), cmap='gray')
                plt.title('Sensitivity maps')
                plt.colorbar()
                plt.show()
        
        if show_img:
            frame = 16
            ksp_iff = ksp_and_mps_to_gt(full_ksp, sens_maps)
            plt.imshow((torch.abs(ksp_iff[frame,:,:])), cmap='gray')
            plt.title('Undersampled/Masked noisy kspace')
            plt.colorbar()
            plt.show()
                
        return sens_maps
    
    coil_maps = estimate_sens_maps_torch(ksp_full)
    print(f'Coil sensitivity shape is {coil_maps.shape}')
    
    return coil_maps
    

def show_results(epoch, view_frame, out, img_under, gt1, out_avg, ref, pred_ksp, avg_psnr, psnr, ssim_score, l1_norm, rmse):
    '''
    Showing predicted outputs, inputs, comparison and other metrics
    
    Parameters
    ----------
    -- epoch : int
    -- view_frame: int  (#frame which we want to show in every epoch)
    -- out: torch , shape (#frame, xdim, ydim)   (predicted image from UNet)
    -- img_under: torch, shape (#frame, xdim, ydim)    (ifft2 image after multiplying full kspace with mask)
    -- gt1: torch , shape (#frame, xdim, ydim)   (ifft2 image from fully sampled kspace)
    -- out_avg: torch , shape (#frame, xdim, ydim)   (avg weighted predicted img)
    -- ref: torch , shape (1, 2, #frame, xdim, ydim)   (input image to UNet, here 2 channel is for real and imaginary values)
    -- pred_ksp: torch , shape (#frame, #coil, xdim, ydim)    (fft2 of predicted image from unet)
    -- avg_psnr: float  
    -- psnr: float
    -- ssim_score: float (SSIM score for designated epoch)
    -- l1_norm: float
    -- rmse: float (Normalized RMSE score for designated epoch)
    
    Returns
    -------
    None

    '''
    print('Epoch: ', epoch)
    
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    
    # First subplot with title 'DIP recon'
    plt.subplot(231)
    img1 = axs[0,0].imshow(out[view_frame,:,:].cpu().numpy(), cmap='gray', vmin=0, vmax=out[view_frame,:,:].cpu().numpy().max() * 0.5) 
    divider1 = make_axes_locatable(axs[0,0])   #to match height of colorbar with the subfigures
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img1, cax=cax1)
    axs[0,0].set_title(f'Recon_frame')
    axs[0,0].set_xticks([])  # Remove x-axis ticks
    axs[0,0].set_yticks([])  # Remove y-axis ticks

    plt.subplot(232)
    acquired_img = torch.abs(img_under)/torch.max(torch.abs(img_under))  #acquired_img_ifft
    # acquired_img = torch.abs(acquired_img)

    img2 = axs[0,1].imshow(acquired_img[view_frame,:,:].cpu().numpy(), cmap='gray', vmin=0, vmax=acquired_img[view_frame,:,:].cpu().numpy().max() * 0.5)
    divider2 = make_axes_locatable(axs[0,1])   #to match height of colorbar with the subfigures
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img2, cax=cax2)
    axs[0,1].set_title('Noisy/Undersampled/input')
    axs[0,1].set_xticks([])  # Remove x-axis ticks
    axs[0,1].set_yticks([])  # Remove y-axis ticks
    
    # Second subplot with title 'Ground Truth'
    plt.subplot(233)
    img3 = axs[0,2].imshow(gt1[view_frame,:,:].numpy(), cmap='gray', vmin=0, vmax=gt1[view_frame,:,:].numpy().max() * 0.5)
    divider3 = make_axes_locatable(axs[0,2])   #to match height of colorbar with the subfigures
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img3, cax=cax3)
    axs[0,2].set_title('Groundtruth')
    axs[0,2].set_xticks([])  # Remove x-axis ticks
    axs[0,2].set_yticks([])  # Remove y-axis ticks

    
    plt.subplot(234)
    img4 = axs[1,0].imshow(np.abs(out_avg[view_frame,:,:].cpu().numpy()), cmap='gray', vmin=0, vmax=np.abs(out_avg[view_frame,:,:].cpu().numpy()).max() * 0.5)
    divider4 = make_axes_locatable(axs[1,0])   #to match height of colorbar with the subfigures
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img4, cax=cax4)
    axs[1,0].set_title('Avg recon')
    #axs[1,0].set_title('GT Kspace')
    
    axs[1,0].set_xticks([])  # Remove x-axis ticks
    axs[1,0].set_yticks([])  # Remove y-axis ticks

    plt.subplot(235)
    img5 = axs[1,1].imshow(np.abs(gt1[view_frame,:,:].numpy() - out[view_frame,:,:].cpu().numpy()), cmap='gray')
    divider5 = make_axes_locatable(axs[1,1])   #to match height of colorbar with the subfigures
    cax5 = divider5.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img5, cax=cax5)
    axs[1,1].set_title('Difference GT & Recon')
    axs[1,1].set_xticks([])  # Remove x-axis ticks
    axs[1,1].set_yticks([])  # Remove y-axis ticks

    plt.subplot(236)

    new_ref = torch.view_as_complex(ref.squeeze().permute(1,2,3,0).contiguous())
    img6 = axs[1,2].imshow(np.abs(new_ref[view_frame,:,:].detach().cpu().numpy()), cmap='gray', vmin=0, vmax=np.abs(new_ref[view_frame,:,:].detach().cpu().numpy()).max() * 0.5)
    
    divider6 = make_axes_locatable(axs[1,2])   #to match height of colorbar with the subfigures
    cax6 = divider6.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img6, cax=cax6)
    #axs[1,2].set_title('Difference GT & inp')
    axs[1,2].set_title('Reference')
    axs[1,2].set_xticks([])  # Remove x-axis ticks
    axs[1,2].set_yticks([])  # Remove y-axis ticks
    
    # Add the PSNR value at the bottom center of the figure
    plt.figtext(0.27, 0.001, f'Avg PSNR = {avg_psnr:.3f}', ha='center', fontsize=12)
    plt.figtext(0.07, 0.001, f'PSNR = {psnr:.3f}', ha='center', fontsize=12)
    plt.figtext(0.47, 0.001, f'SSIM = {ssim_score:.3f}', ha='center', fontsize=12)
    plt.figtext(0.67, 0.001, f'L1 norm = {np.mean(l1_norm):.3f}', ha='center', fontsize=12)
    plt.figtext(0.87, 0.001, f'RMSE = {rmse:.3f}', ha='center', fontsize=12)
    
    
    axs[1, 2].axis('off')  # Turn off axis for this empty subplot
    #plt.tight_layout()
    plt.show()


    plt.imshow(np.log(np.abs(pred_ksp[view_frame,0,:,:].detach().cpu())), cmap='gray')
    plt.title('Kspace of predicted img')
    plt.show()


def train(ksp_full, mask_from_file, mps1, no_of_epoch, alpha, reference_guided = True):
    '''
    Training loop for sg-dip
    
    Parameters
    ----------
    -- ksp_full : torch
        Fully sampled kspace, shape (#frames, #coils, xdim, ydim)
    -- mask_from_file: torch
        Sampling mask, shape (#frame, xdim, ydim)
    -- msp1 : torch
        Sensitivity maps, shape (#frames, #coils, xdim, ydim)
    -- no_of_epoch : int
    -- alpha : float  (explicit prior multiplier)

    Returns
    -------
    None
    '''
    
    losses = []
    psnrs = []
    avg_psnrs = []
    rmse_list = []
    l1_list = []
    out_alliter = []
    
    #### Hyperparameters ####
    exp_weight = .95
    learning_rate = 5e-4
    show_every = 50

    
    #####################
    ## Network loading ##
    #####################
    #net = UNet3D(2,2).to(device) ##
    #net = ResidualUNetSE3D(2,2).to(device)
    net = ResidualUNet3D(2,2).to(device)
    
    nx = mask_from_file.shape[-2]
    ny = mask_from_file.shape[-1]
    no_of_frame = mask_from_file.shape[0] ##
    
    if reference_guided:
        #### Reference guided DIP #####
        acquired = torch.unsqueeze(mask_from_file.to(device), axis = 1) * ksp_full.to(device)  
        #acquired_img = ksp_and_mps_to_gt(acquired, torch.unsqueeze(mps1.to(device), axis = 0))  #one mps set for all frames
        acquired_img = ksp_and_mps_to_gt(acquired, mps1.to(device)) 
        real_part = acquired_img.real
        imag_part = acquired_img.imag
        two_channel_tensor = torch.stack((real_part, imag_part), dim=0)
        ref = torch.unsqueeze(two_channel_tensor,axis=0)
        ref = ref.to(torch.float32)
        ref = Variable(ref.to(device), requires_grad=True)
        
    else:
        ref = Variable(torch.rand((1,2,no_of_frame,nx,ny)).cuda(), requires_grad=True)  ##start from noise
    
    
    with torch.no_grad():
        scale_factor = torch.linalg.norm(net(ref.to(device)))/torch.linalg.norm(ksp_and_mps_to_gt(acquired, torch.unsqueeze(mps1.to(device), axis = 0)).to(device))
        target_ksp = scale_factor * acquired.to(device)
    
    
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    optimizer2 = optim.Adam([ref], lr = 1e-1)
        
    avg_ksp = torch.zeros_like(ksp_full).to(device)
    
    
    ###### GT and undersampled image ######
    gt1 = ksp_and_mps_to_gt(ksp_full, mps1)
    img_under = ksp_and_mps_to_gt(acquired, mps1.to(device))
    
    out_avg = torch.zeros_like(torch.abs(gt1))
        
    start_time = time.time()
    
    for epoch in tqdm(range(no_of_epoch)):
        optimizer.zero_grad()
        optimizer2.zero_grad()
        noise_max = torch.max(ref)/2
        random_smoothing_temp = torch.zeros_like(ref).to(device)
        
        for jj in range(3):   
            noise = noise_max * torch.rand(*ref.shape, device=device)   #uniform noise
            net_output = net(ref + noise).squeeze()
            random_smoothing_temp += net_output
        net_output_final = random_smoothing_temp/3
        
        net_output_final = torch.view_as_complex(net_output_final.squeeze().permute(1,2,3,0).contiguous())
        
        #pred_ksp = mps_and_gt_to_ksp(torch.unsqueeze(mps1.to(device), axis = 1), torch.unsqueeze(net_output_final, axis = 1)) #one set mps for all frames
        pred_ksp = mps_and_gt_to_ksp(mps1.to(device), torch.unsqueeze(net_output_final, axis = 1)) 
        
    
        beta = 1
        loss = beta * torch.linalg.norm(torch.unsqueeze(mask_from_file.to(device), axis = 1) * target_ksp - torch.unsqueeze(mask_from_file.to(device), axis = 1) * pred_ksp.squeeze()) \
            + alpha * torch.linalg.norm(ref - net_output_final) 
        
        loss.backward()
        
        optimizer.step()
        optimizer2.step()
        
        
        with torch.no_grad():
            out = torch.abs(ksp_and_mps_to_gt(pred_ksp.detach(), mps1.to(device))).squeeze().cpu()  ##
            out /= torch.max(out) ##normalizing
        
            gt1 = torch.abs(gt1)/torch.max(torch.abs(gt1)) ##normalizing
            out_avg = out_avg * exp_weight + out * (1 - exp_weight)
            
            avg_ksp = avg_ksp * exp_weight + pred_ksp * (1 - exp_weight)
            
            ## Evaluation metrics
            psnr = compute_psnr(np.array(gt1, dtype=np.float32), np.array(out))  #, data_range=data_range)
            l1_norm = np.mean(np.linalg.norm(np.array(gt1, dtype=np.float32) - np.array(out), ord=1, axis=(-2, -1)))
            rmse = np.sqrt(np.mean((np.array(gt1, dtype=np.float32) - np.array(out)) ** 2))
            ssim_score, _ = ssim(np.array(gt1, dtype=np.float32), np.array(out), data_range=1.0, full=True)
            avg_psnr = compute_psnr(np.array(gt1), np.array(out_avg)/float(out_avg.max().item())) #,data_range=data_range)  ##changed
            
            avg_psnrs.append(avg_psnr)
            psnrs.append(psnr)
            rmse_list.append(rmse)
            l1_list.append(l1_norm)
            losses.append(loss.item())
    
            #########################
            ## Visualizing results ##
            #########################
            
            if epoch%show_every == 0:
                view_frame = 20  ##which frame to show in all epochs
                show_results(epoch, view_frame, out, img_under, gt1, out_avg, ref, pred_ksp, avg_psnr, psnr, ssim_score, l1_norm, rmse)
                
            torch.cuda.empty_cache()
        out_alliter.append(out.detach()) 
    
    end_t = time.time()
    print(f'total time: {end_t - start_time}')


def main(mask, pat, center, acc, slicee, alpha, scanner, epoch):
    '''
    Parameters
    ----------
    -- mask: torch, shape (#frames, xdim, ydim)
    -- center : int   (The number of center of the folder in data directory)
    -- pat : string  (The number of patient number of the folder in data directory) 
    -- slicee: int (slice number)
    -- alpha: float (explicit prior parameter)
    -- scanner : string   (The scanner number of the folder in data directory)
    -- epoch : int  (total number of epoch)
    
    Returns
    -------
    None
    '''

    direc = f'/v/raid1b/backup/maistiak/cmrxrecon25/ChallengeData/MultiCoil/Perfusion/TrainingSet/FullSample/Center00{center}/{scanner}/{pat}/perfusion.mat'
    ksp_full, mask = data_loading(direc, mask, acc, slicee, 5, 45)  #please put the clipped start frame and last frame if it doesn't fit to gpu with all frames
    sens_map = calc_sensmap(ksp_full)
    train(ksp_full, mask, sens_map, epoch, alpha)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mask", type=str, default = 'u', help='types of mask (u: uniform; g: gaussian; r: radial)')
    ap.add_argument("-pat", "--patient", type=str, default = 'P011', help='number in patient folder')
    ap.add_argument("-c", "--cent", type=int, default = 1, help='non zero digits in center folder')
    ap.add_argument("-acc", "--acf", type=int, default = 24, help='acceleration factor (8/16/24)')
    ap.add_argument("-sl", "--slice", type=int, default = 0, help='slice number')
    ap.add_argument("-al", "--alpha", type=float, default = 1.5, help='alpha factor for explicit prior')
    ap.add_argument("-sn", "--scname", type=str, default = 'UIH_30T_umr780', help='Scanner name in the folder')
    ap.add_argument("-ep", "--epoch", type=int, default = 1500, help='number of epoch')
    args = vars(ap.parse_args())
    
    main(args["mask"], args["patient"], args["cent"], args["acf"], args["slice"], args["alpha"], args["scname"], args["epoch"])