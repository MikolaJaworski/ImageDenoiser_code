import os
import time
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from skimage.transform import resize
from PIL import Image

class ID_dataset(Dataset):
    def __init__(self, mode, load_mode, train_path, test_path, patch_n=None, patch_size=None, transform=None):

        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform
        self.load_mode = load_mode
        

        if mode == 'train':
            input_ = sorted(glob(os.path.join(train_path, '*_input.npy')))
            target_ = sorted(glob(os.path.join(train_path, '*_target.npy')))
            if load_mode == 0: # batch data load
                self.input_ = input_
                self.target_ = target_
            else: # all data load
                self.input_ = [io.imread(f) for f in input_]
                self.target_ = [io.imread(f) for f in target_]
        else: # mode =='test'
            input_ = sorted(glob(os.path.join(test_path, '*_input.npy')))
            target_ = sorted(glob(os.path.join(test_path, '*_target.npy')))
            if load_mode == 0: # batch data load
                self.input_ = input_
                self.target_ = target_
            else: # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

   
    def __len__(self):
        return len(self.target_)
    

    def __getitem__(self, idx):
        input_img = self.input_[idx]
        target_img = self.target_[idx]
        
        if self.load_mode == 0:
            input_img = np.load(input_img)
            target_img = np.load(target_img)
        if self.transform:
                input_img = resize(input_img, (254, 254))
                target_img = resize(target_img, (254, 254))
        if self.patch_size:
            input_patches, target_patches = get_patch(input_img, target_img, self.patch_n, self.patch_size)
            return (input_patches, target_patches)
        else:
            return (input_img, target_img)    



def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    patch_input_imgs = image.extract_patches_2d(full_input_img, (patch_size, patch_size), max_patches=patch_n, random_state=42)
    patch_target_imgs = image.extract_patches_2d(full_target_img, (patch_size, patch_size), max_patches=patch_n, random_state=42)
    return patch_input_imgs, patch_target_imgs


def get_loader(mode='train', load_mode=0,
               train_path='./TrainingDataset/', test_path='./TestDataset/',
               patch_n=20, patch_size=64,
               transform=None, batch_size=5, num_workers=4):
    dataset_ = ID_dataset(mode, load_mode, train_path, test_path, patch_n, patch_size, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader

########### FORMUŁA ITERACJI PO DANYCH #########
'''
patch_size=64
x = get_loader()
for i, (x, y) in enumerate(x):
    x = x.view(-1, 3, patch_size, patch_size)
    y = y.view(-1, 3, patch_size, patch_size)
    print("iter: {}, x: {}, y: {}".format(i, x.size(), y.size()))

'''



################ TESTY ###################
'''
mode='train'
load_mode=0
train_path='./TrainingDataset/'
test_path='./TestDataset/',
patch_n = 10
patch_size = 64
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = None
x = ID_dataset(mode, load_mode, train_path, test_path, patch_n, patch_size, transform=transform)
i, t = x[1]

patch_inp_imgs = []
patch_trg_imgs = []
for i in range(len(x)):
    inpu, targ = x[1]
    for p in range(len(inpu)):
        inpu_pat = inpu[p]
        targ_pat = targ[p]
        patch_inp_imgs.append(inpu_pat)
        patch_trg_imgs.append(targ_pat)

#inpu, targ = x
#print(patch_inp_imgs[1].shape)
#print('Input shape: ' + str(inpu.shape) + ' Target shape: ' + str(targ.shape))

patch_size = 64
total_iters = 0
data_loader = get_loader()
for iter_, (x, y) in enumerate(data_loader):
    
    total_iters += 1
    if total_iters % 49 == 0:
        print('Input: ' + str(x.size()) + ' Target: ' + str(y.size()))
        print(x[1])

        x = x.permute(0, 1, 4, 2, 3)
        y = y.permute(0, 1, 4, 2, 3)
        print('Input1: ' + str(x.size()) + ' Target1: ' + str(y.size()))
        print(x[1])

        if patch_size: # patch training
            x = x.view(-1, 3, patch_size, patch_size)
            y = y.view(-1, 3, patch_size, patch_size)
        print('Input2: ' + str(x.size()) + ' Target2: ' + str(y.size()))
        print(x[1])

print(total_iters)


        inpu = x.numpy()
        targ = y.numpy()
        figu = plt.figure()
        figu.add_subplot(1, 2, 1)
        plt.imshow(inpu[0])
        figu.add_subplot(1, 2, 2)
        plt.imshow(targ[0])
        plt.show()

'''

##### SIEĆ ####

class ID_net(nn.Module):
    def __init__(self, out_ch=96):
        super(ID_net, self).__init__()
        self.conv_first = nn.Conv2d(3, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t_last = nn.ConvTranspose2d(out_ch, 3, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x.clone()
        out = self.relu(self.conv_first(x))
        out = self.relu(self.conv(out))
        residual_2 = out.clone()
        out = self.relu(self.conv(out))
        out = self.relu(self.conv(out))
        residual_3 = out.clone()
        out = self.relu(self.conv(out))

        # decoder
        out = self.conv_t(out)
        out += residual_3
        out = self.conv_t(self.relu(out))
        out = self.conv_t(self.relu(out))
        out += residual_2
        out = self.conv_t(self.relu(out))
        out = self.conv_t_last(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# net = ID_net().float().to(device)
# trainData = get_loader()
# num_epochs = 5
# patch_size = 64

def train(net, device, num_epochs, trainData, patch_size):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    train_losses = []
    total_iters = 0
    start_time = time.time()
    print('Let\'s start the training session')
    for epoch in range(1, num_epochs):
        net.train(True)
        print('Epoch: {}'.format(epoch))

        for iter_, (x, y) in enumerate(trainData):
            total_iters += 1
            print('Total iter: {}'.format(total_iters))
            # poukładanie tensora
            x = x.permute(0, 1, 4, 2, 3).float().to(device)
            y = y.permute(0, 1, 4, 2, 3).float().to(device)

            if patch_size: # patch training
                x = x.view(-1, 3, patch_size, patch_size)
                y = y.view(-1, 3, patch_size, patch_size)
            print('Prediction...')
            pred = net(x)
            print('Loss...')
            loss = criterion(pred, y)
            net.zero_grad()
            print('Optimizer...')
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # print
            if total_iters % 3 == 0:
                print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        num_epochs, iter_+1, 
                                                                                                        len(trainData), loss.item(), 
                                                                                                        time.time() - start_time))
    PATH = './ID_net_2.pth'
    torch.save(net.state_dict(), PATH)

def compute_measure(x, y, pred, data_range):
    original_psnr = compute_PSNR(x, y, data_range)
    original_rmse = compute_RMSE(x, y)
    pred_psnr = compute_PSNR(pred, y, data_range)
    pred_rmse = compute_RMSE(pred, y)
    return (original_psnr, original_rmse), (pred_psnr, pred_rmse)

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))

def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)

def save_figure(x, y, pred, original_result, pred_result, iter):
    x = x.permute(0, 2, 3, 1)
    y = y.permute(0, 2, 3, 1)
    pred = pred.permute(0, 2, 3, 1)
    x = x.numpy()
    y = y.numpy()
    pred = pred.numpy()
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(x[0])
    ax[0].set_title('Noisy image')
    ax[0].set_xlabel("PSNR: {:.4f}\nRMSE: {:.4f}".format(original_result[0], original_result[1]))
    ax[1].imshow(pred[0])
    ax[1].set_title('Denoise image')
    ax[1].set_xlabel("PSNR: {:.4f}\nRMSE: {:.4f}".format(pred_result[0], pred_result[1]))
    ax[2].imshow(y[0])
    ax[2].set_title('Orginal image')
    f.savefig(os.path.join('/home/mj417302/PracaMagisterska/ImageDenoiser/result', 'result_{}.png'.format(iter)))
    plt.close()

def test(net):
    testData = get_loader(mode='test', load_mode=1, transform=True ,batch_size=1, num_workers=4, patch_n=None, patch_size=None)
    total_iters = 0
    ori_psnr_avg, ori_rmse_avg = 0, 0
    pred_psnr_avg, pred_rmse_avg = 0, 0

    with torch.no_grad():
        for iter_, (x, y) in enumerate(testData):
            total_iters += 1    
            x = x.permute(0, 3, 1, 2).float()
            y = y.permute(0, 3, 1, 2).float()
            pred = net(x)
            original_result, pred_result = compute_measure(x, y, pred, 1)
            ori_psnr_avg += original_result[0]
            ori_rmse_avg += original_result[1]
            pred_psnr_avg += pred_result[0]
            pred_rmse_avg += pred_result[1]
            save_figure(x, y, pred, original_result, pred_result, iter_)
    print(total_iters)
    print('\n')
    print('Original\nPSNR avg: {:.4f}\nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(testData), ori_rmse_avg/len(testData)))
    print('After learning\nPSNR avg: {:.4f}\nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(testData), pred_rmse_avg/len(testData)))            

if __name__ == "__main__":

    # train

    net = ID_net().float().to(device)
    trainData = get_loader()  
    num_epochs = 5
    patch_size = 64
    train(net, device, num_epochs, trainData, patch_size)

    # test
    '''
    PATH = './ID_net_2.pth'
    device = torch.device('cpu')
    net = ID_net()
    net.load_state_dict(torch.load(PATH, map_location=device))
    print(net)
    test(net)
    '''

    # Koniec kodu
    # Autor: Mikołaj Jaworski