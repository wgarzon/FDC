import FDC_RF as fdc
from utils import *
from aggregator import *
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
global dsName, outDir, site, DTs, shareMDir, mdlName, best_dts

def initializeEnv(file):
    #read config values at site
    confValues = read_config_file(file)
    global dsName, outDir, site, DTs, shareMDir, mdlName, best_dts, dirFigs
    dsName, outDir, site, DTs, shareMDir = confValues['ds_dir'] + confValues['ds_name'], confValues['out_dir'], confValues['id'], int(confValues['ntrees']), confValues['share_models']
    mdlName, best_dts = confValues['model_name'], float(confValues['best_dts'])

    if not os.path.exists(outDir): os.makedirs(outDir)
    if not os.path.exists(outDir+"figs"): os.makedirs(outDir+"figs")
    if not os.path.exists(outDir + "outcomes"): os.makedirs(outDir + "outcomes")
    if not os.path.exists(shareMDir): os.makedirs(shareMDir)


def genLocalModel(Xtrain, ytrain):
    print("... Generating Local Model")

    #build model according to param file
    fdcRF = fdc.buildModel(mdlName,DTs,Xtrain,ytrain)
    fdcMdl = fdcRF.getModel()

    infoM1, infoM2, maxFeat = fdc.getInfoDTs(fdcMdl, fdcRF.nFeats)
    OOB_Err, OOB_Acc, AggAcc = fdc.ParOOBErrorTree(fdcMdl, Xtrain, ytrain)
    bestDTs, thresh = fdc.getBestByPerc(OOB_Acc, best_dts)
    mdlsByAcc, featsByAcc = fdc.splitMdlsByAcc(bestDTs, infoM1,[Xtrain,ytrain],mdlName)
    #print("ModelsByAccuracy",mdlsByAcc)
    saveMdlDisk(fdcMdl,outDir,shareMDir, "CentralModel.smodel")
    saveMdlsByAcc(mdlsByAcc, featsByAcc, site, outDir, shareMDir)

    return mdlsByAcc, fdcMdl


def evaluateLocalModel(file="central.ini",site="Central"):
    initializeEnv(file)
    print("... Generating Local Model")

    Xtrain, Xtest, ytrain, ytest = loadAggData(shareMDir+"Ensembled/")
    if Xtrain != None:
        dim = "Central Model:"+",Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)+ ",Xtrain:"+ str(Xtrain.shape)
    else:
        dim = "Central Model:"+",Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)

    print("Testing ", dim)
    CompleteModel = load(shareMDir+"CentralModel.smodel")
    fdc.ScoreModel("Complete"+site, site, shareMDir, CompleteModel,Xtest,ytest,dim)


def buildLocalModel(file="central.ini"):
    initializeEnv(file)
    print("***** Processing " + site + " *****")
    sections = read_config_file(file)
    fromRepo = sections["fromrepo"]
    dataFileName = sections["ds_name"]
    dataDir = sections["data_dir"]

        #download data from link provided
    if fromRepo =="Yes":
        if not getGDriveData(sections["linkid"],dataFileName,dataDir):
            raise "Error downloading the file " +dataFileName

    t1 = datetime.now()
    Xtrain, Xtest, ytrain, ytest = preprocess_ds(dataDir+dataFileName)
    mdlsByAcc, fdcMdl = genLocalModel(Xtrain, ytrain)
    t2 = datetime.now()
    delta = t2 - t1
    if not os.path.exists(shareMDir + "outcomes/"): os.makedirs(shareMDir + "outcomes/")
    f = open(shareMDir + "outcomes/Training_Time_Central.txt", "a")
    f.write("Site" + str(site) + "," + str(delta.total_seconds()) + "\n")
    f.close()
    return Xtest, ytest, Xtrain, fdcMdl

def main():
	if len(sys.argv) > 1: opc =  int(sys.argv[1])
	if opc==1:
		buildLocalModel()
	else:
		evaluateLocalModel()

main()