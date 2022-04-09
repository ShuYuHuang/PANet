import matplotlib.pyplot as plt
import torch

def renorm(x):
    mx=x.max()
    mn=x.min()
    return (x-mn)/(mx-mn)

def show_image_annot(images,masks,showPics = 5):
    imgs, msks = images[:showPics,0].numpy(),masks[:showPics,0].numpy()
    print("BATCHSIZE=",len(msks))
    for i in (imgs,msks):
        assert i.__class__.__name__ == 'ndarray', 'input data type should be ndarray'
    
    print(imgs.shape,'\n',msks.shape)
    plt.figure(figsize=(20,6))
    for i, (img,msk) in enumerate(zip(imgs,msks)):
        plt.subplot(2,len(imgs),i+1)
        plt.imshow(img, cmap='gray')
        plt.subplot(2,len(msks),i+len(imgs)+1)
        plt.imshow(msk, cmap='gray')
    plt.show()
    plt.close()
    

def show_image_metabatch(dataloader,shownorm=True):
    d = next(iter(dataloader))
    supp_imgs,fore_mask,_,qry_imgs,qry_label=map(d.__getitem__,
                                               ['supp_imgs', 'fore_mask', 'back_mask', 'qry_imgs','query_labels'])
    
    
    imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                        + [torch.cat(qry_imgs, dim=0),], dim=0).numpy()
    msks_concat = torch.cat([torch.cat(way, dim=0) for way in fore_mask]
                        + [torch.cat(qry_label, dim=0),], dim=0).numpy()
    cfg=dict(n_ways= len(supp_imgs),
        n_shots= len(supp_imgs[0]),
        n_queries= len(qry_imgs),
        batch_size = supp_imgs[0][0].shape[0],
        img_size = supp_imgs[0][0].shape[-2:],
        total_len=len(imgs_concat),
        support_len=len(supp_imgs)*len(supp_imgs[0])*supp_imgs[0][0].shape[0])
    print(cfg)
    plt.figure(figsize=(20,2))
    for i,img in enumerate(imgs_concat):
        assert img.__class__.__name__ == 'ndarray', 'input data type should be ndarray'
        plt.subplot(2,cfg['total_len'],i+1)
        if shownorm:
            plt.imshow(renorm(img.transpose(1,2,0)))
        else:
            plt.imshow(img.transpose(1,2,0))
        plt.axis("off")
        if i<cfg["support_len"]:
            plt.title("Support")
    for i,msk in enumerate(msks_concat):
        plt.subplot(2,cfg['total_len'],cfg['total_len']+i+1)
        plt.imshow(msk)
        
        plt.axis("off")
    plt.show()
    plt.close()
