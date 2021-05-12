import pandas as pd
import pydicom
from pydicom import dcmread
from pydicom.fileset import FileSet
from pydicom.data import get_testdata_file
import jinja2
from pydantic import BaseModel, Field
from typing import Optional,List
from infer import ModelInference
import os
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

#from pydantic import BaseModel

class Filter(BaseModel):
    platform : List[str] = None



#class Report(BaseModel):
#    title: str
#    timestamp: datetime
#    description: Optional[str] = None


class InfEngine():
    def __init__(self):
        self.meta_db = {}
        self.display_fields = {
				    'PatientID' : 'Patient ID', 'PatientAge': 'Patient Age',
				    'PatientBirthDate' : 'Patient Birth Date', 'PatientSex': 'Patient Sex',
				    'Modality': 'Modality', 'FracturePrediction': 'Fracture Prediction',
				    'BodyPartExamined': 'Body Part Examined', 'StudyDate':'Study Date'

			    }

        self.db_features =    ['PatientID', 'PatientAge','PatientBirthDate', 'PatientSex', 'StudyDate','AcquisitionDate','Modality', 'BodyPartExamined', 'StudyDescription',  'ViewPosition','InstitutionName'] 

#'FracturePrediction'

        self.features_list = ['PatientID','PatientAge','PatientBirthDate', 'PatientSex','Modality', 'FracturePrediction','BodyPartExamined','StudyDate', 'StudyDescription', 'ViewPosition'   ] 
        self.report_fields = ['PatientID','PatientAge','PatientBirthDate', 'PatientSex','Modality', 'FracturePrediction','BodyPartExamined','StudyDate' ] 
        self.test_results = [] 
        self.results_df = pd.DataFrame() 
        self.fracture_data = ['Y', 'N', 'Y', 'N', 'N', 'N', 'Y', 'N', 'N', 'N', 'Y', 'N', 'Y', 'N', 'N', 'N', 'Y', 'N', 'N', 'N' ]
        self.model_inf =  ModelInference(self.db_features)    

    def runBatchInference(self):
        self.model_inf.batchInference()
        self.results_df = pd.read_csv(self.model_inf.dcm_csv_pred)
        self.results_df  = self.results_df.drop(columns=['Unnamed: 0' ])
        self.results_df =  self.results_df.dropna(axis=1)

        print(self.results_df.head())
        return("Success")
 
    def getBatchSummary(self):
        print('Btach summary results')
        print(self.results_df)
        return(self.results_df.to_dict())

    def getDbFeatures(self):
        return(self.db_features) 
    
    def getReport(self, patID):
        print(patID)
        if(len(self.results_df)): 
            exam = self.results_df.loc[self.results_df['PatientID'] == patID]
            print(exam.columns) 
            print(self.report_fields)
            print('1')
            report_data = exam.iloc[0][self.report_fields]
            print('2')
            print(report_data)
            report_dict= {}
            for field in self.report_fields:
                report_dict[self.display_fields[field]]= str(report_data[field])
            report_dict['image']= "https://drive.google.com/thumbnail?id=1J-ThKUPOnd_aJOLnT5wrQ5zXWLaKj5k1"
            print( report_dict)
            json_compatible_item_data = jsonable_encoder(report_dict)
            return JSONResponse(content=json_compatible_item_data)
          
    def generateReport(self, patID):
        #patId = request.PatientID
        print(patID)
        if(len(self.results_df)): 
            exam = self.results_df.loc[self.results_df['PatientID'] == patID]
            print(exam.columns) 
            print(self.report_fields)
            report_data = exam.iloc[0][self.report_fields]
            print(report_data)
            report_dict= {}
            for field in self.report_fields:
                report_dict[self.display_fields[field]]= report_data[field]

            cwd = os.getcwd()
            rpdict_image = 'file:'+cwd+'/'+ exam.iloc[0]['Image']
            rpdict_image =  exam.iloc[0]['Image']
            print('rpdict img:'+ rpdict_image)
            print("Dictionary")
            print(report_dict)
      
           
            templateLoader = jinja2.FileSystemLoader(searchpath="./")
            templateEnv = jinja2.Environment(loader=templateLoader)
            TEMPLATE_FILE = "rep_template.html"
            template = templateEnv.get_template(TEMPLATE_FILE)
            report = template.render(rpdict=report_dict, rp_image= rpdict_image )
            print(report)
            return(report)
        return(0)

 


