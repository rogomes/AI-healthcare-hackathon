from fastapi import FastAPI
from model import InfEngine
#from model import InfEngine,  Request
from starlette.responses import HTMLResponse



app = FastAPI()

inf_pl = InfEngine()
#report = inf_pl.generateReport('e0ac71efecc68f3330d1aa6cfa')
#print("Generated Report ..")

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/startBatchInference")
async def startBatchInference():
    result = inf_pl.runBatchInference()
    if(result=='Success'):
        summary= inf_pl.getBatchSummary()
        return summary
    else:
        return result

@app.get("/report/{patientId}")
async def get_report(patientId: str):
    print(patientId)
    report_data= inf_pl.getReport(patientId)
    print(report_data)
    return(report_data)
    #return HTMLResponse(content=html_content, status_code=200)
    #html_content = inf_pl.generateReport(patientId)
    #return HTMLResponse(content=html_content, status_code=200)

@app.get("/summary")
async def getSummary():
    print('in BatchSummary')
    return inf_pl.getBatchSummary()



