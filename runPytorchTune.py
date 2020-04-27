#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import sys
import socket
import configurations
import funPytorch as fun
import notifier
from ax.service.managed_loop import optimize
import pickle
from ray import tune

device = "cuda:0"

def createTrainEval(conf):
    def trainEval(param):
        pconf = conf(param)
        #print("======= CREATE MODEL")
        model, optim = fun.makeModel(pconf, device)
        #print("======= LOAD DATA")
        dataloaders, _ = fun.processData(pconf)
        #print("======= TRAIN MODEL")
        fun.runTrain(pconf, model, optim, dataloaders, tuneLog=True)

    return trainEval

if len(sys.argv) != 2:
    print("Use {} configName".format(sys.argv[0]))
else:
    conf = getattr(sys.modules['configurations'], sys.argv[1])

    print("====================")
    print("RUN USING {}".format(sys.argv[1]))
    print("====================")

    analysis = tune.run(createTrainEval(conf.conf), config=conf.param,
                        resources_per_trial=conf.trialResources,
                        local_dir=conf.resultsDir,
                        )

    print("====================")
    print("BEST PARAMETERS")
    print(analysis.get_best_config(metric="metric"))

    analysis.dataframe().to_pickle(conf.saveFile)

    notifier.sendMessage("Training of {} finished on {}".format(sys.argv[1], socket.gethostname()))

