from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
import os

# Custom Dataset
class DFDataset(Dataset):
    def __init__(self, img_dir, seg_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.img_files = os.listdir(img_dir)
        self.seg_files = os.listdir(seg_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        seg_path = os.path.join(self.seg_dir, self.seg_files[idx])

        image = read_image(img_path) / 255.0
        seg_image = read_image(seg_path) / 255.0

        X = image
        Y_origin = image
        Y_seg = seg_image

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            Y_origin = self.target_transform(Y_origin)
            Y_seg = self.target_transform(Y_seg)

        Y = (Y_origin, Y_seg)

        return X, Y


def train_partial(dataloader, DF_part, loss_fn, optimizer, history):
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        Y_img = Y[0].to(device)
        Y_seg = Y[1].to(device)
        Y = Y_img, Y_seg

        # Compute prediction error
        DF_part.to(device)
        pred = DF_part(X)
        pred_img = pred[0].to(device)
        pred_seg = pred[1].to(device)
        pred = pred_img, pred_seg

        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            history.append(loss)

def save_checkpoint_DF(epoch, DF_network, DF_src, DF_dst, optimizer_src, optimizer_dst, filename):
    state = {
        'Epoch': epoch,
        'State_dict_all': DF_network.state_dict(),
        'State_dict_src': DF_src.state_dict(),
        'State_dict_dst': DF_dst.state_dict(),
        'optimizer_src': optimizer_src.state_dict(),
        'optimizer_dst': optimizer_dst.state_dict()
    }
    torch.save(state, filename)