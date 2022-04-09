import cv2
import numpy as np
import torch
import torch.utils.data as tud
from pydicom import dcmread

def gather(data,index):
    return list(map(lambda x: x[index],data))


class CTMRI_PairDataset(tud.IterableDataset):
    def __init__(self,imgs_anno_path_list,
                 mask_type,
                 transform=None,
                 device="cuda:0",
                 ways=1,shots=3,repeats=3):
        self.imgs_anno_path_list = imgs_anno_path_list
        if type(imgs_anno_path_list) != list:
            raise ValueError('Need Input a list')
        self.transform = transform
        self.mask_type = mask_type
        self.ways=ways
        self.shots=shots
        self.repeats=repeats
        self.device=device

#     def to_tensor(self,x):
#         return torch.tensor(x,dtype=torch.float32,device=self.device)
    def getitem(self, idx):
        # now = time.time()
        img_anno_path = self.imgs_anno_path_list[idx]
        img_path = img_anno_path[0]
        mask_path = img_anno_path[1]
        
#         '''
#         根據獲得的img/ mask路徑讀取檔案
#         在讀取之前先確定讀到的是.dcm，否則raise Error例外
#         '''
        if img_path.__contains__('.dcm'):    
            # pydcm read image
            ds = dcmread(img_path)
            image = ds.pixel_array
            image = image.astype('uint8') # 調整格式以配合albumentation套件需求
        else:
            raise ValueError(f'img path: {img_path} unknown')

#         '''
#         在這邊讀取mask，不論是CT或是MRI讀取到的都是png，
#         要注意的是cv2預設讀近來是float16，
#         '''
        # cv2 read mask(Ground)
        mask = cv2.imread(mask_path)[...,0].astype('uint8')# 調整格式以配合albumentation套件需求
        
#         '''
#         如果讀到MRI，我們只考慮所有value=63的像素，他代表肝臟的標示
#         其他像素點設置成0
#         '''
        # check mask type
        if self.mask_type == 'MRI':
            mask[mask!=63] = 0
            mask[mask==63] = 1
        elif self.mask_type == 'CT':
            mask[mask!=0] = 1
        else:
            raise ValueError('Non-support mask_type')

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image,mask = transformed['image'],transformed['mask']
        
        image = np.stack((image, image, image), axis=0)
        image = torch.Tensor(image)
        mask = torch.Tensor(mask)
        return image, mask
    def __iter__(self):
        samples=self.ways*self.shots+1
        
        order=np.concatenate(
            [np.random.permutation(len(self.imgs_anno_path_list))
             for _ in range(self.repeats)])
        for o in range(len(order)//samples):
            tmp=[*map(self.getitem,order[o*samples:(o+1)*samples])]
            
            yield gather(tmp,0),gather(tmp,1)


def collate_batch(batch,ways,shots):
#     """
#     Args:
#         supp_imgs: support images
#             way x shot x [B x 3 x H x W], list of lists of tensors
#         fore_mask: foreground masks for support images
#             way x shot x [B x H x W], list of lists of tensors
#         back_mask: background masks for support images
#             way x shot x [B x H x W], list of lists of tensors
#         qry_imgs: query images
#             N x [B x 3 x H x W], list of tensors
#     """
    xx, yy = gather(batch,0),gather(batch,1)
    supp_imgs=[[torch.stack(
                gather(xx,w*shots+s))
            for s in range(shots)] for w in range(ways)]
    fore_mask=[[torch.stack(
                gather(yy,w*shots+s))
            for s in range(shots)]for w in range(ways)]
    back_mask=[[
                1-torch.stack(gather(yy,w*shots+s))
            for s in range(shots)]for w in range(ways)]

    qry_imgs=[torch.stack(gather(xx,ways*shots))]# last one
    query_labels=[torch.stack(gather(yy,ways*shots))]# last one
    return dict(
            zip(("supp_imgs","fore_mask","back_mask","qry_imgs","query_labels"),
                (supp_imgs,fore_mask,back_mask,qry_imgs,query_labels)))
        
#         return dict(supp_imgs=supp_imgs,
#                    qry_imgs=qry_imgs,
#                    fore_mask=fore_mask)
            