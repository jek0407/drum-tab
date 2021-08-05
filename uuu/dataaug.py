from torchvision import transforms

def DataAug(dataloader,aug_num,aug_path='./data/train/aug/'):
    cnt = 0
    for num in range(aug_num):
        for img,mask in dataloader:
            p_img = transforms.ToPILImage()(img[0][0])
            p_img.save(aug_path+str(cnt)+'_img.png', format='png')
            p_mask = transforms.ToPILImage()(mask[0][0])
            p_mask.save(aug_path+str(cnt)+'_mask.png', format='png')
            cnt += 1
    return print(str(aug_num)+" times Augmentation at '"+aug_path+"' path")
