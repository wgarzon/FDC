import FDC_RF as fdc
from utils import *
from aggregator import *
import warnings

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

    #Get details about the model into dictionaries structure to be merged
    infoM1, infoM2, maxFeat = fdc.getInfoDTs(fdcMdl, fdcRF.nFeats)
    # Print information about DTs and stored them into .dot file
    #fdc.ShowInfoDTs(fdcMdl, site)
    #Parallelizable method to compute OOB Error
    OOB_Err, OOB_Acc, AggAcc = fdc.ParOOBErrorTree(fdcMdl, Xtrain, ytrain)
    #select best p decision trees
    bestDTs, thresh = fdc.getBestByPerc(OOB_Acc, best_dts)
    #bestDTs, thresh = fdc.getBestByThres(OOB_Acc,0.6)
    mdlsByAcc, featsByAcc = fdc.splitMdlsByAcc(bestDTs, infoM1,[Xtrain,ytrain],mdlName)
    #print("ModelsByAccuracy",mdlsByAcc)
    saveMdlDisk(fdcMdl,outDir,shareMDir,site+ "Complete.smodel")
    saveMdlsByAcc(mdlsByAcc, featsByAcc, site, outDir, shareMDir)

    return mdlsByAcc, fdcMdl


def localMdl(Xtrain, Xtest, ytrain, ytest):
    print("... Generating Local Model")

    #build model according to param file
    fdcRF = fdc.buildModel(mdlName,DTs,Xtrain,ytrain)
    fdcMdl = fdcRF.getModel()

    #Get details about the model into dictionaries structure to be merged
    infoM1, infoM2, maxFeat = fdc.getInfoDTs(fdcMdl, fdcRF.nFeats)
    # Print information about DTs and stored them into .dot file
    #fdc.ShowInfoDTs(fdcMdl, site)
    #Parallelizable method to compute OOB Error
    OOB_Err, OOB_Acc, AggAcc = fdc.ParOOBErrorTree(fdcMdl, Xtrain, ytrain)

    #select best p decision trees
    bestDTs, thresh = fdc.getBestByPerc(OOB_Acc, best_dts)
    #bestDTs, thresh = fdc.getBestByThres(OOB_Acc,0.6)

    mdlsByAcc, featsByAcc = fdc.splitMdlsByAcc(bestDTs, infoM1,[Xtrain,ytrain],mdlName)
    #print("ModelsByAccuracy",mdlsByAcc)

    # Print information of local model
    score, estimators, oob_score = fdc.ScoreModel(site, fdcMdl, Xtest, ytest)

    # save features by level of importance
    save_VI_file(fdcMdl, Xtrain, site, outDir)
    #plot OOBData
    plot_OOB(OOB_Acc, "OOB Accuracy", site, outDir)

    #save models local and by accuracy level: best and bad decision trees
    saveMdlDisk(fdcMdl,outDir,shareMDir,site+ "Complete.smodel")
    saveMdlsByAcc(mdlsByAcc, featsByAcc, site, outDir, shareMDir)

    return mdlsByAcc, score, estimators, oob_score, fdcMdl, AggAcc

def getLocalModel():
    localmodel = loadLocalModel(shareMDir, site, "smodel")
    return localmodel

# Create a collaborative model by Join
def ensembleJoin(Xtest, ytest, Xtrain):
    print("... Generating Joining Model")
    globalModel = mergeModelsJoin(shareMDir, "smodel")
    #store on disk the ensemble model by joining (Collaborative Model)
    saveMdlDisk(globalModel, outDir, shareMDir, site + "JoinCollab.jcmodel")
    score, NoEst, oob_score = fdc.ScoreModel("Join" + site, globalModel, Xtest, ytest)
    save_VI_file(globalModel, Xtrain, "Join"+site, outDir)

    return score, NoEst, oob_score

#Ensemble the models, by the best performance
def ensembleBestDTs(Xtest,ytest,Xtrain):
    print("... Generating Best Model")
    BestGlobModel = mergeModelsAcc(shareMDir, "Accmodel")
    #store on disk the ensemble model by joining (Collaborative Model)
    saveMdlDisk(BestGlobModel, outDir, shareMDir, site + "BestCollab.bcmodel")
    score, NoEst, oob_score = fdc.ScoreModel("Best"+site, BestGlobModel,Xtest,ytest)
    save_VI_file(BestGlobModel, Xtrain, "Best"+site, outDir)
    return score, NoEst, oob_score

def main(file,opc=0):
    initializeEnv(file)
    print("***** Processing " + site + " *****")
    Xtrain, Xtest, ytrain, ytest = preprocess_ds(dsName)

    #opc = int(input("1.Local Model, 2. Ensemble Collab Model (Join&Best), 3.None??:"))
    #Create a local Model
    if opc == 1:
        mdlsByAcc, score, estimators, oob_score, fdcMdl, _ = localMdl(Xtrain, Xtest, ytrain, ytest)
    elif opc == 2:
        #apply ensemble process by join and best estrategies
        ensembleJoin(Xtest, ytest, Xtrain)
        ensembleBestDTs(Xtest, ytest, Xtrain)


#Ensemble the models, by the best performance
def ensembleJnCentral(Xtest,ytest,Xtrain, idd):
    print("... Uploading Central Model")
    name = "TimeXcentralComplete.smodel".replace("X",str(idd))
    centrMdl = get_centModel("centralModels/"+name)
    #store on disk the ensemble model by joining (Collaborative Model)
    #saveMdlDisk(globalModel, outDir, shareMDir, site + "JoinCollab.jcmodel")
    score, NoEst, oob_score = fdc.ScoreModel("Central" + site, centrMdl, Xtest, ytest)
    #save_VI_file(centrMdl, Xtrain, "Join"+site, outDir)

    return score, NoEst, oob_score

#Ensemble the models, by the best performance
def ensembleBestCentral(Xtest,ytest,Xtrain, idd):
    print("... Uploading Best Central Model")
    name = "TimeXcentralModel_Best.Accmodel".replace("X",str(idd))
    bbest = get_centModel("centralModels/"+name)
    #store on disk the ensemble model by joining (Collaborative Model)
    #saveMdlDisk(globalModel, outDir, shareMDir, site + "JoinCollab.jcmodel")
    score, NoEst, oob_score = fdc.ScoreModel("BestCentral" + site, bbest, Xtest, ytest)
    #save_VI_file(centrMdl, Xtrain, "Join"+site, outDir)

    return score, NoEst, oob_score


def mainLocal(file):
    initializeEnv(file)
    print("***** Processing " + site + " *****")
    Xtrain, Xtest, ytrain, ytest = preprocess_ds(dsName)
    mdlsByAcc, fdcMdl = genLocalModel(Xtrain, ytrain)
    return Xtest, ytest, Xtrain, fdcMdl


def mainTestModel(locMdls, Xtest,ytest,Xtrain, file, idd):
    print("Xtest",Xtest.shape, "ytest",ytest.shape, "Xtrain",Xtrain.shape)
    initializeEnv(file)

    #compute score of local model using all test dataset
    for ss in range(len(locMdls)):
        score, estimators, oob_score = fdc.ScoreModel("Site"+str(ss+1), locMdls[ss], Xtest, ytest)

    #evaluated test data set over two collaborative models
    ensembleJoin(Xtest, ytest, Xtrain)
    ensembleBestDTs(Xtest, ytest, Xtrain)

    #test data set using central model
    ensembleJnCentral(Xtest, ytest, Xtrain,idd)
    ensembleBestCentral(Xtest, ytest, Xtrain,idd)
