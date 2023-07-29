import cv2
import numpy as np
from os.path import join
import torch
import torch.utils.data as tud
from pydicom import dcmread
from pycocotools.coco import COCO

def gather(data,index):
    return list(map(lambda x: x[index],data))

class CTMRI_MultiClassDataset(tud.IterableDataset):
    def __init__(self,anno_file,
                 root_dir,
                 ways=2,shots=3,tasks=500,
                 
                 transform=None):
        
        self.root_dir=root_dir
        self.ways=ways
        self.shots=shots
        self.tasks=tasks
        self.transform=transform
        
        self.coco_obj=COCO(anno_file)
        self.n_cats=len(self.coco_obj.cats)
        self.cat_ids=list(self.coco_obj.cats.keys())
        self.n_cats=len(self.cat_ids)
        
    def getitem(self,
                annoId,
                issupport=True,
                sup_cids=None):
        ann_obj=self.coco_obj.anns[annoId]
        cat_id=ann_obj["category_id"]
        mask_filename=join(self.root_dir,ann_obj['file_name'])
        dcm_filename=join(self.root_dir,self.coco_obj.imgs[ann_obj['image_id']]['file_name'])
        if dcm_filename.__contains__('.dcm'):    
            # pydcm read image
            ds = dcmread(dcm_filename)
            image = ds.pixel_array
            image = image.astype('uint8') # 調整格式以配合albumentation套件需求
        else:
            raise ValueError(f'img path: {dcm_filename} unknown')

        mask = cv2.imread(mask_filename)[...,0].astype('uint8')# 調整格式以配合albumentation套件需求
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image,mask = transformed['image'],transformed['mask']
        image = np.stack((image, image, image), axis=0)
        image = torch.Tensor(image)
        
        if issupport:
            formask=np.zeros_like(mask)
            backmask=np.zeros_like(mask)
            formask[mask==cat_id] = 1
            backmask[mask==0] =1
            formask = torch.Tensor(formask)
            backmask = torch.Tensor(backmask)
            return image, formask,backmask
        else:
            formask=np.full_like(mask,0)
            formask[mask>0]=255
            for i,sid in enumerate(sup_cids):
                formask[mask==sid]=i+1
            formask = torch.Tensor(formask)
            return image,formask
            
    def __iter__(self):
        # Get anootation IDs for support and query data
        support_cid=[np.random.choice(self.cat_ids,self.ways,replace=False)
             for _ in range(self.tasks)]
        query_cid=[support_cid[t][np.random.choice(self.ways,replace=False)]
                  for t in range(self.tasks)]
        # Get anootation objects for support and query data
        support_anno=[[np.random.permutation(self.coco_obj.getAnnIds(catIds=j))[:self.shots] 
                       for j in i] for i in support_cid]
        query_anno=[np.random.choice(self.coco_obj.getAnnIds(catIds=j)) for j in query_cid]

        for s_annoId,q_annoId,sup_cids in zip(support_anno,query_anno,support_cid):
            supports=[[*map(self.getitem,ids)] for ids in s_annoId]
            support_xs=[gather(way,0) for way in supports]
            support_fore_ys=[gather(way,1) for way in supports]
            support_back_ys=[gather(way,2) for way in supports]
            query_x,query_y=self.getitem(q_annoId,issupport=False,sup_cids=sup_cids)
            yield support_xs,support_fore_ys,support_back_ys,query_x,query_y
def collate_batch(batch):
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
    sup_xs,sup_fys,sup_bys,q_x,q_y=map(lambda i: gather(batch,i),range(5))
    ways=len(sup_xs[0])
    shots=len(sup_xs[0][0])
    supp_imgs=[[torch.stack([samples[w][s] for samples in sup_xs]) 
                for s in range(shots)] for w in range(ways)]
    
    fore_mask=[[torch.stack([samples[w][s] for samples in sup_fys]) 
                for s in range(shots)]for w in range(ways)]
    
    back_mask=[[torch.stack([samples[w][s] for samples in sup_bys]) 
                for s in range(shots)]for w in range(ways)]
    qry_imgs=[torch.stack(q_x)]
    query_labels=[torch.stack(q_y)]
    return dict(
            zip(("supp_imgs","fore_mask","back_mask","qry_imgs","query_labels"),
                (supp_imgs,fore_mask,back_mask,qry_imgs,query_labels)))




## Dataset info
# === CT Sets ===
# --All labels of CT sets were reviewed. Some ground truth data might be slightly different from the first published set. Please use the last version of the sets.
# --All distinguishable "vena cava inferior" areas were excluded from the liver in ground truth data.
# --All gallbladder "vena cava inferior" areas were excluded from the liver in ground truth data.
# --Labeles of the four abdomen organs in the ground data are represented by four different pixel values ranges. These ranges are:
# Torsal: 32 (28<<<35)
# Liver: 63 (55<<<70)

# === MR Sets ===
# --All labels of MR sets were reviewed. Some ground truth data might be slightly different from the first published set. Please use the last version of the sets.
# --The In-phase and Out-phase images have same UID in the T1DUAL sequences. Therefore they were stored under two folder.
# --The ground images in T1DUAL folder represents both In-phase and Out-phase images.
# --The anonymization method of the MR sequences was changed to prevent UID data in DICOM images.
# --All distinguishable "vena cava inferior" areas were excluded from the liver in ground truth data.
# --All gallbladder "vena cava inferior" areas were excluded from the liver in ground truth data.
# --The shape of the kidneys are determined elliptical as much as possible. Veins, small artifacts are included to the kidney if they are inside of the kidneys elliptical contour.
# --Labeles of the four abdomen organs in the ground data are represented by four different pixel values ranges. These ranges are:
# Torsal: 32 (28<<<35)
# Liver: 63 (55<<<70)
# Right kidney: 126 (110<<<135)
# Left kidney: 189 (175<<<200)
# Spleen: 252 (240<<<255)

