import torch
import torch.nn as nn
from network import DF, DF_Src, DF_Dst
from Train_util import DFDataset, train_partial, save_checkpoint_DF
import matplotlib.pyplot as plt
from DFLoss import MSE_DISSIM_Loss
from Conversion_util import conversion, zipdir

DF_sample = DF(3, 64, 128, 128, 256, 64, 16)
DF_sample_src = DF_Src(DF_sample)
DF_sample_dst = DF_Dst(DF_sample)

dfdata_src = DFDataset('./DFDataset/img_i_2', './DFDataset/seg_i_2')
dfdata_dst = DFDataset('./DFDataset/img_e', './DFDataset/seg_e')

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

BATCH = 32


train_src_dataloader = DataLoader(dfdata_src, batch_size=BATCH, shuffle=True)
train_dst_dataloader = DataLoader(dfdata_dst, batch_size=BATCH, shuffle=True)

mse_dissim = MSE_DISSIM_Loss().to(device)
optimizer_src = torch.optim.Adam(DF_sample_src.parameters(), lr=1e-4)
optimizer_dst = torch.optim.Adam(DF_sample_dst.parameters(), lr=1e-4)

DF_sample = DF_sample.to(device)
DF_sample_src = DF_sample_src.to(device)
DF_sample_dst = DF_sample_dst.to(device)

src_history = []
dst_history = []

for i in range(30):
    print('epoch ', i)
    print('SRC partial training')
    train_partial(train_src_dataloader, DF_sample_src, mse_dissim, optimizer_src, src_history)
    print('\nDST partial training')
    train_partial(train_dst_dataloader, DF_sample_dst, mse_dissim, optimizer_dst, dst_history)

plt.plot(src_history, color='red', label='Src')
plt.plot(dst_history, color='blue', label='Dst')

plt.title('Change of Loss while Training')
plt.xlabel('Step')
plt.ylabel('MSE + DISSIM')
plt.legend()
plt.savefig('Loss.png')
plt.show()

save_checkpoint_DF(30, DF_sample, DF_sample_src, DF_sample_dst, optimizer_src, optimizer_dst, 'DFcheckpoint_fin_ver_30')

conversion('./DFDataset/img_e', './Conversion_data', DF_sample_src)
zipf = zipfile.ZipFile('conversion.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('Conversion_data/', zipf)
zipf.close()