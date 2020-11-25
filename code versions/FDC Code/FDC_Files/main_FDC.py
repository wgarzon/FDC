import FDC_RF as fdc
from utils import *
from aggregator import *
import warnings
from joblib import load

warnings.filterwarnings("ignore")
global dsName, outDir, site, DTs, shareMDir, mdlName, best_dts

def initializeEnv(file):
    #read config values at site
    confValues = read_config_file(file)
    global dsName, outDir, site, DTs, shareMDir, mdlName, best_dts, dirFigs
    if file == "mainc.ini":
        shareMDir = confValues['share_models']
        if not os.path.exists(shareMDir): os.makedirs(shareMDir)
        if not os.path.exists(shareMDir+"Ensembled"): os.makedirs(shareMDir+"Ensembled")
        return

    dsName, outDir, site, DTs, shareMDir = confValues['ds_dir'] + confValues['ds_name'], confValues['out_dir'], confValues['id'], int(confValues['ntrees']), confValues['share_models']
    mdlName, best_dts = confValues['model_name'], float(confValues['best_dts'])

    if not os.path.exists(outDir): os.makedirs(outDir)
    if not os.path.exists(shareMDir + "outcomes"): os.makedirs(shareMDir + "outcomes")
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
    saveMdlDisk(fdcMdl,outDir,shareMDir, "Site"+site+ "Complete.smodel")
    saveMdlsByAcc(mdlsByAcc, featsByAcc, "Site"+site, outDir, shareMDir)

    return mdlsByAcc, fdcMdl


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
    save_VI_file(globalModel, Xtrain, "Join"+site, shareMDir)

    return score, NoEst, oob_score

#Ensemble the models, by the best performance
def ensembleBestDTs(Xtest,ytest,Xtrain):
    print("... Generating Best Model")
    BestGlobModel = mergeModelsAcc(shareMDir, "Accmodel")
    #store on disk the ensemble model by joining (Collaborative Model)
    saveMdlDisk(BestGlobModel, outDir, shareMDir, site + "BestCollab.bcmodel")
    score, NoEst, oob_score = fdc.ScoreModel("Best"+site, BestGlobModel,Xtest,ytest)
    save_VI_file(BestGlobModel, Xtrain, "Best"+site, shareMDir)
    return score, NoEst, oob_score

#Ensemble the models, by the best performance
def ensembleJnCentral(Xtest,ytest,Xtrain, idd):
    print("... Uploading Central Model")
    name = "TimeXcentralComplete.smodel".replace("X",str(idd))
    centrMdl = get_centModel("centralModels/"+name)
    score, NoEst, oob_score = fdc.ScoreModel("Central" + site, centrMdl, Xtest, ytest)

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

#Build the local model
def localModel(file, site):
    initializeEnv(file)
    print("***** Processing " + site + " *****")
    Xtrain, Xtest, ytrain, ytest = preprocess_ds(dsName, shareMDir, site, True, True)
    mdlsByAcc, fdcMdl = genLocalModel(Xtrain, ytrain)
    return Xtest, ytest, Xtrain, fdcMdl

#join all datasets into one CSV to evaluate the models
def generateAllDataSets(file):
    initializeEnv(file)
    aggDataSets(shareMDir)

#function to evaluate collaborative models using aggdata
def evalCollabModels(file,site):
    initializeEnv(file)
    nPath = shareMDir+"Ensembled/"

    Xtrain, Xtest, ytrain, ytest = loadAggData(shareMDir+"Ensembled/")
    if Xtrain != None:
        dim = "Collab Model:"+",Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)+ ",Xtrain:"+ str(Xtrain.shape)
    else:
        dim = "Collab Model:"+",Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)
        
    print("Testing ", dim)
    nPath = shareMDir+"Ensembled/"
    print("... Evaluating Join and Best Collaborative Model")
    globalModel = load(nPath+"JoinCollab.jcmodel")
    fdc.ScoreModel("Join" + site,site,shareMDir, globalModel, Xtest, ytest,dim)

    BestGlobModel = load(nPath+"BestCollab.bcmodel")
    fdc.ScoreModel("Best"+site, site, shareMDir, BestGlobModel,Xtest,ytest,dim)


#ensemble the models per site into two collaborative models
def ensembleModels(file):
    initializeEnv(file)
    nPath = shareMDir+"Ensembled/"
    print("... Generating Joining Model")
    globalModel = mergeModelsJoin(shareMDir, "smodel")
    saveMdlDisk(globalModel, "", nPath, "JoinCollab.jcmodel")

    print("... Generating Best Model")
    BestGlobModel = mergeModelsAcc(shareMDir, "Accmodel")
    saveMdlDisk(BestGlobModel, "", nPath, "BestCollab.bcmodel")
    print("Two models generated Successful!!!")


#evaluate the local and collaborative model (optional)
def TestModels(file, site, lcMdl=True, colMdl=False, VIM=False):
    initializeEnv(file)

    if lcMdl:
        Xtrain, Xtest, ytrain, ytest = loadLocalData(shareMDir, site)
        dim = "Local Model:"+"Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)+ ",Xtrain:"+ str(Xtrain.shape)
        print("Testing ", dim)
        nn = "SiteXComplete.smodel".replace("X",str(site))
        localModel = load(outDir+nn)
        fdc.ScoreModel("Site"+str(site),str(site), shareMDir,localModel, Xtest, ytest, dim)
        if VIM: save_VI_file(localModel, Xtrain, "Local"+site, shareMDir)

    if colMdl:
        Xtrain, Xtest, ytrain, ytest = loadAggData(shareMDir+"Ensembled/")
        dim = "Collab Model:"+",Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)+ ",Xtrain:"+ str(Xtrain.shape)
        print("Testing ", dim)
        nPath = shareMDir+"Ensembled/"
        print("... Evaluating Join and Best Collaborative Model")
        globalModel = load(nPath+"JoinCollab.jcmodel")
        fdc.ScoreModel("Join" + site,site,shareMDir, globalModel, Xtest, ytest,dim)

        BestGlobModel = load(nPath+"BestCollab.bcmodel")
        fdc.ScoreModel("Best"+site, site, shareMDir, BestGlobModel,Xtest,ytest,dim)



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
