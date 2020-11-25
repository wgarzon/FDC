from utils import *
from aggregator import *
import main_test
from copy import deepcopy

nSites = 3
configs = ["site" + str(val) + ".ini" for val in range(1,nSites+1)]

print("Total Sites:"+ str(len(configs)))
print(configs)

xtestDS, ytestDS, ytrainDS, localMdls  = [], [], [], []
xtestRes, ytestRes, xtrainRes = None, None, None

for val in range(10):
    xtestDS, ytestDS, ytrainDS, localMdls  = [], [], [], []
    xtestRes, ytestRes, xtrainRes = None, None, None

    print("Timeee:",val+1)
    for ini in configs:
        #Generate local models
        Xtest, ytest, Xtrain, localMdl = main_test.mainLocal(ini)
        aa = deepcopy(localMdl)
        xtestDS.append(Xtest)
        ytrainDS.append(Xtrain)
        ytestDS.append(ytest.to_frame())
        localMdls.append(aa)

    xtestRes = pd.concat(xtestDS)
    ytestRes = pd.concat(ytestDS)
    xtrainRes = pd.concat(ytrainDS)

    #for ini in configs:
        #main_test.main(ini, 2)

    main_test.mainTestModel(localMdls,xtestRes,ytestRes,xtrainRes,configs[0],val)


print("Finishhhh")
