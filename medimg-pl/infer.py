import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torch
import numpy as np
from monai.config import print_config
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
import pandas as pd
import pydicom
import glob
import cv2 as cv
import os
import torch
from torch.utils.data import Dataset, DataLoader
import os.path
from pathlib import Path

from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)
class HandFracDataset(Dataset):

    def __init__(self, image_files, transforms):
        self.image_files = image_files
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

 
    def __getitem__(self, index):
        return self.transforms(self.image_files[index])


class ModelInference():
    def __init__(self, db_fields):
        self.fields = db_fields
        self.dcm_png_dir = ''
        self.dcm_csv=''
        self.dcm_csv_pred =''
        self.test_png_f =''

    def dcmtopng(self, test_dicoms):
        outdir = 'testdicomToPNG'

        for f in test_dicoms:
            ds = pydicom.read_file(f) # read dicom image
            img = ds.pixel_array # get image array
            dirName = f.split("/")[-3:][0] + "/" + f.split("/")[-3:][1] + "/"
            try:
                os.makedirs(outdir + dirName)
                print(outdir + dirName)
            except:
                pass
                cv.imwrite(outdir + dirName + f.split("/")[-1].replace('.dcm','.png'), img) # write png 
        print('created png files')
        return outdir

    def preprocess_imgs(self, test_png_f):
        for f in test_png_f:
            im = cv.imread(f)
            im = cv.resize(im, (224, 224))
            final_img = cv.normalize(im,  im, 0, 255, cv.NORM_MINMAX)
            cv.imwrite(f, final_img[:, :, 1]) # save image in gray format 

    def get_dcm_data(self, test_dicoms):
        all_data = []



        for i in test_dicoms: 
           dcm = pydicom.dcmread(i)
           fields = [dcm.PatientID, int(dcm.PatientAge[1:3]), dcm.PatientBirthDate, dcm.PatientSex, dcm.StudyDate, dcm.AcquisitionDate, dcm.Modality, dcm.BodyPartExamined,    dcm.ViewPosition, dcm.StudyDescription,   dcm.InstitutionName]
           all_data.append(fields)
        mydata = pd.DataFrame(all_data, 
			    columns = ['PatientID','PatientAge','PatientBirthDate', 'PatientSex','StudyDate','AcquisitionDate', 'Modality','BodyPartExamined', 'ViewPosition','StudyDescription',  'InstitutionName'])


        
        mydata['StudyDescription'] = mydata['StudyDescription'].map(lambda x : x.replace('Handgelenk links', 'Left wrist')
                                                        .replace('Handgelenk rechts','Right wrist'))
        mydata['StudyDate'] = pd.to_datetime(mydata['StudyDate'])
        mydata['AcquisitionDate'] = pd.to_datetime(mydata['AcquisitionDate'])
        mydata['PatientBirthDate'] = pd.to_datetime(mydata['PatientBirthDate'])

        mydata["Image"] = self.test_png_f
        csv_path = self.dcm_png_dir + "test_data.csv"
        self.dcm_csv = csv_path
        mydata.to_csv(csv_path)
        return csv_path


    def infer(self, model,test_loader, device, weight_path):
      model.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage))
      model.eval()
      y_pred = list()
      with torch.no_grad():
          for test_data in test_loader:
              test_images = test_data.to(device)
              pred = model(test_images).argmax(dim=1)
              for i in range(len(pred)):
                  y_pred.append(pred[i].item())
      return y_pred

    # Provide test dir 


    def generate_test_csv(self, test_dir):
        test_dicoms = glob.glob(test_dir + "/*/*.dcm")
	# Get the dir storing png files
        outdir = self.dcmtopng(test_dicoms)
        self.dcm_png_dir = outdir
        print(outdir)
        self.test_png_f = glob.glob(self.dcm_png_dir + "*/*/*.png")

        # Preprocess the png files
        self.preprocess_imgs(self.test_png_f)
        return( self.get_dcm_data(test_dicoms))

    ## Structure is: 
    ## - test_set
    ## -- folder_of_dicom_file 
    ## --- dicom_file


    def batchInference(self ):

        testdir = "testing_set"
        testcsv = "testdicomToPNGtest_data.csv"
        #if( Path(testcsv).exists()): 
        #    print('Test csv file exists++++')
        #    test_csv =  testcsv
        #    dcm_png_dir='testdicomToPNG'
        #else:
        #    print('Test csv file soes not exist, creating ...')
        #    generate_test_csv(testdir)


        self.generate_test_csv(testdir)
	# Get dicom data
        test_df = pd.read_csv(testcsv)

        class_names = ["Yes", "No"]
        num_class = len(class_names)

        image_file_list = test_df["Image"]
        image_width, image_height = Image.open(image_file_list[0]).size

        # Perform transforms
        val_transforms = Compose([
	    LoadImage(image_only=True),
	    AddChannel(),
	    ScaleIntensity(),
	    ToTensor()
	])

        act = Activations(softmax=True)

        # Read test data as batch
        test_ds = HandFracDataset(image_file_list, val_transforms)
        test_loader = DataLoader(test_ds, batch_size=32, num_workers=0)

        # define network
        device = torch.device('cpu')
        model = DenseNet121(
	  spatial_dims=2,
	  in_channels=1,
	  out_channels=num_class, 
	  pretrained = True).to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), 1e-4)#Adam 1e-5
        epoch_num = 30
        val_interval = 1

	# Perform prediction
        weight_path = 'model/best_metric_model_v0f.pth'
        y_pred = self.infer(model, test_loader, device, weight_path)

        # Store prediction in test df
        test_df["FracturePrediction"] = [class_names[pred] for pred in y_pred]
        test_df.drop(columns="Unnamed: 0", inplace=True)
        # Save df file as csv
        self.dcm_csv_pred = self.dcm_png_dir + "test_data_pred.csv"
        test_df.to_csv(self.dcm_csv_pred)


#db_fields = ['PatientAge', 'PatientSex']
#model_inf = ModelInference(db_fields)    
#model_inf.batchInference()
