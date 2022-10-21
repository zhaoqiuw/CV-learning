import os
import argparse
import yaml
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pprint import pprint
from dataset import *
from augmentation import *
from model import PetModel
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
from matplotlib import colors
import torch


def get_dataset(dataset_cfg,model_cfg):
    image_train_dir = os.path.join(dataset_cfg['DATA_DIR_TRAIN'], 'image')
    label_train_dir = os.path.join(dataset_cfg['DATA_DIR_TRAIN'], 'label')
    image_valid_dir = os.path.join(dataset_cfg['DATA_DIR_TEST'], 'image')
    label_valid_dir = os.path.join(dataset_cfg['DATA_DIR_TEST'], 'label')
    preprocessing_fn = smp.encoders.get_preprocessing_fn(model_cfg['ENCODER_NAME'], model_cfg['ENCODER_WEIGHTS'])
    #print("@@@@@@@@",preprocessing_fn)
    train_dataset = Dataset(
    image_train_dir, 
    label_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn), 
)
    valid_dataset = Dataset(
    image_valid_dir, 
    label_valid_dir, 
    augmentation=None, 
    preprocessing=get_preprocessing(preprocessing_fn),
)
    
    print(len(train_dataset),len(valid_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=dataset_cfg['TRAIN_BATCH'], shuffle=True, num_workers=dataset_cfg['NUM_WORKERS'],pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=dataset_cfg['VAL_BATCH'], shuffle=False, num_workers=dataset_cfg['NUM_WORKERS'],pin_memory=True)
    # batch = next(iter(valid_dataloader))
    # print(batch['image'].size(),type(batch['image']))
    return train_dataloader ,valid_dataloader
    

def get_model(model_cfg):
    model = PetModel(model_cfg['MODEL_TYPE'], model_cfg['ENCODER_NAME'], in_channels=model_cfg['IN_CHANNLES'], out_classes=model_cfg['NUM_CLASSES'],lr=model_cfg['OPTIMIZER_LR'])
    return model
    
def train(model_cfg,dataset_cfg,train_cfg):
    train_dataloader ,valid_dataloader = get_dataset(dataset_cfg,model_cfg)
    model = get_model(model_cfg)
    
    print('Model config = %s', str(dict(model_cfg)))
    print('Dataset config = %s', str(dict(dataset_cfg)))
    print('Train config = %s', str(dict(train_cfg)))
    print('Train data loader len = %d', len(train_dataloader))
    print('Valid data loader len = %d', len(valid_dataloader))
    tensorboard_logger = TensorBoardLogger(train_cfg['LOG_PATH'], name=model_cfg['ENCODER_NAME']+'-'+model_cfg['MODEL_TYPE'],version = train_cfg['LOG_VERSION'])
    csv_logger = CSVLogger(train_cfg['LOG_PATH'], name=model_cfg['ENCODER_NAME']+'-'+model_cfg['MODEL_TYPE'],version = train_cfg['LOG_VERSION'])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=train_cfg['MONITOR'], 
                        dirpath=train_cfg['LOG_PATH']+'/'+model_cfg['ENCODER_NAME']+'-'+model_cfg['MODEL_TYPE'],
                        filename="Version="+str(train_cfg['LOG_VERSION'])+'-{epoch:02d}-{valid_dataset_iou:.2f}',
                        mode ='max',
                        save_top_k = 1)
    trainer = pl.Trainer(
    gpus=train_cfg['GPUS'], 
    max_epochs=train_cfg['MAX_EPOCHS'],
    callbacks=[checkpoint_callback],
    logger = [tensorboard_logger,csv_logger]
)
    trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader,
)
    return 
    
def test(model_cfg,dataset_cfg,train_cfg):
    _ ,test_dataloader = get_dataset(dataset_cfg,model_cfg)
    #model = get_model(model_cfg)
    trainer = pl.Trainer(
    gpus=train_cfg['GPUS'], 
    max_epochs=train_cfg['MAX_EPOCHS'],
)
    model = PetModel.load_from_checkpoint(train_cfg['CHECKPOINTS'],arch = model_cfg['MODEL_TYPE'], encoder_name = model_cfg['ENCODER_NAME'], in_channels=model_cfg['IN_CHANNLES'], out_classes=model_cfg['NUM_CLASSES'],lr=model_cfg['OPTIMIZER_LR'])
    
    import time 
 
    start = time.time()
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    end =time.time()
    print("==========time using=============")
    print(end-start)
    pprint(test_metrics)


def visualize(model_cfg,dataset_cfg,train_cfg):
    from PIL import Image
    path  = "./result"+"/"+model_cfg['MODEL_TYPE']+"_"+model_cfg['ENCODER_NAME']
    label_path = path +"/label"
    pred_path = path +"/pred"
    image_path = path + "/image"
    if not os.path.exists("./result"):
        os.mkdir("./result")
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(label_path)
        os.mkdir(pred_path)
        os.mkdir(image_path)
        
    _ ,test_dataloader = get_dataset(dataset_cfg,model_cfg)
    batch = next(iter(test_dataloader))
    model = PetModel.load_from_checkpoint(train_cfg['CHECKPOINTS'],arch = model_cfg['MODEL_TYPE'], encoder_name = model_cfg['ENCODER_NAME'], in_channels=model_cfg['IN_CHANNLES'], out_classes=model_cfg['NUM_CLASSES'],lr=model_cfg['OPTIMIZER_LR'])
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    with torch.no_grad():
        model.eval()
        for i,batch in enumerate(iter(test_dataloader)):
            logits = model(batch["image"])
            
            pr_masks = logits.sigmoid()
            pr_masks = (pr_masks > 0.5).float()
    # colormap = [
    #     "#231F20",
    #     "#DB5F57",]
    # cmap = colors.ListedColormap(colormap)
    
            j = 0
            for image, gt_mask, pr_mask,logit in zip(batch["image"], batch["mask"], pr_masks,logits):
                # #####        
                #####
                # image = image.numpy().transpose(1, 2, 0)*std+mean
                # im = Image.fromarray(np.uint8(image*255))
                # im = im.convert('RGB')
                # im.save(image_path+"/"+str(i)+"_" +str(j)+'.png')
                # im = Image.fromarray(gt_mask.numpy().squeeze()*255)
                # im = im.convert('RGB')
                # im.save(label_path+"/"+str(i)+"_" +str(j)+'.png')
                im = Image.fromarray(pr_mask.numpy().squeeze()*255)
                im = im.convert('RGB')
                im.save(pred_path+"/"+str(i)+"_" +str(j)+'.png')
                j = j+1
        
        # if i ==1:
        #     break
 
        # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3 * 10))
        # mean=[0.485, 0.456, 0.406]
        # std=[0.229, 0.224, 0.225]
        # image = image.numpy().transpose(1, 2, 0)*std+mean
        # cv2.imwrite(label_path+'filename.png', image)
        # cv2.imwrite(label_path)
        # axs[0].set_title("Image")
        # axs[0].imshow(image)
        # axs[0].axis("off")
        # axs[1].set_title("Ground truth")
        # axs[1].imshow(gt_mask.numpy().squeeze(), cmap=cmap, interpolation=None)
        # axs[1].axis("off")
        # axs[2].set_title("Prediction")
        # axs[2].imshow(pr_mask.numpy().squeeze(), cmap=cmap, interpolation=None)
        # axs[2].axis("off")
        # i = i+1

     
        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 3, 1)
        # mean=[0.485, 0.456, 0.406]
        # std=[0.229, 0.224, 0.225]
        # image = image.numpy().transpose(1, 2, 0)*std+mean
        # plt.imshow(image)  # convert CHW -> HWC
        # plt.title("Image")
        # plt.axis("off")

        # plt.subplot(1, 3, 2)
        # plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        # plt.title("Ground truth")
        # plt.axis("off")

        # plt.subplot(1, 3, 3)
        # plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        # plt.title("Prediction")
        # plt.axis("off")
       
    # plt.savefig('squares_plot.png', bbox_inches='tight')

    # plt.show()

def main():
    parser = argparse.ArgumentParser(description='Landslip_Seg')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help='Path of config files')
    parser.add_argument('--action', type=str, required=True, help="Input file directory")
    args = parser.parse_args()
    yaml_path = args.cfg_file
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    model_cfg = config['MODEL']
    dataset_cfg = config['DATASET']
    train_cfg = config['TRAIN']
    f.close()
    if args.action =="train":
        train(model_cfg,dataset_cfg,train_cfg)
    elif args.action == "test":
        test(model_cfg,dataset_cfg,train_cfg)
    elif args.action == "visual":
        visualize(model_cfg,dataset_cfg,train_cfg)
if __name__ =="__main__":
    main()

    