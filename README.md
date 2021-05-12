# AI Healthcare Hackathon

<img src="figures/doc.jpg" width="600">


Find out more about this project at its [Devpost page](https://devpost.com/submit-to/12485-ai-for-healthcare-hackathon/manage/submissions/229798-track-1-challenge-5-medical-imaging-communication-tools/project_details/).

## Meet the team

This repository contais the work of the **AI Wonder Girls** team! 

<img src="figures/logo.png" width="200">

<img src="figures/team.png" width="400">


## Inspiration 
The **AI wonder Girls** team members are experienced data scientists and take great pleasure in contributing to the Healthcare Domain.

## What our tool does
The AI Medical Imaging application is developed with batch inference on the wrist fracture detection, automatic report generation, analytics dashboard, and configuration capabilities. 

## How we built it
The **implementation** of our model was done with *DenseNet121* is pretrained using the *MONAI library* on the given data set for hand images. The model is tuned optimized to an accuracy of 80% on the test dataset. The inference pipeline is developed with batch inference and updated in the database. The User interface based on *Streamlit*, is developed with the configuration of database and report features, which are used for analytics. The report is automatically generated for each imaging modality exam with *DICOM* features and AI model prediction data. 

## Challenges we ran into
- Understanding the radiology bottlenecks, use cases, constraints.  
- Notifications to multiple devices like SMS, email, mobile, desktop as options do not come in a single framework that can be made use of.

## Accomplishments that we're proud of
- Training a model with 80% accuracy, 
- Inference engine pipeline in Python with batch inference, 
- Automatic report generation, 
- Application User interface with Streamlit and Python
- A clear strategy for further development of the tool 

<img src="figures/dash.png" width="700">

*Prototype Application: report generation application*

<img src="figures/app6.jpg" width="700">

*Potential Application: patient analytics*


## What we learned
Healthcare needs are domain-specific and so need expert input to build an efficient AI product. Knowledge in both domains can help solve the bottlenecks in the workflows. 

## 4 pillars of our project

**Impact:** The AI report generation will ease out the bottlenecks in the workflows in the radiology and ensure faster diagnosis which is highly desirable in critical health problems. The hospitals with good infrastructure can provide this service to remote places with lesser resources by collecting the exams to be studied, AI generated reports can be processed further with the teleconsultations, without the patient need for travel. This can meet the need in high surge in diagnosis as in current covid pandemic and in countries where health care resources are in a crunch. 

**Feasibility:** The application , inference pipeline, model are built with python, REST APIs and opensource libraries, frameworks. 

**Extensibility:** The application can be scaled to AI models used for CT , MRI, Ultrasound, X-ray scans and can be integrated well with PACS and other applications in healthcare practice. The entire engine can be run on-premise servers to ensure privacy. It can integrate well with private clouds with cluster of hospitals under an umbrella, with multiple sources of data linked.  

**Innovation:** The application and inference pipeline are built with standard *REST APIs* and *opensource libraries* frameworks which can integrate well with the existing radiology systems like PACS and AI models currently in use. This helps improving greatly the integration of AI into healthcare systems enhancing the extension of the benefits to a large population. The possible identification of  high risk health problems with this application will allow doctors to prioritize patients and ultimately save more lives.


## What's next for Track 1 Challenge 5 Medical Imaging Communication Tools
- Notifications of critical cases of AI diagnosis to the radiologist,  
- Extending to other AI models in practice. 
