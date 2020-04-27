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

device = "cuda:0"

def createTrainEval(conf):
    def trainEval(param):
        pconf = conf(param)
        #print("======= CREATE MODEL")
        model, optim = fun.makeModel(pconf, device)
        #print("======= LOAD DATA")
        dataloaders, _ = fun.processData(pconf)
        #print("======= TRAIN MODEL")
        return fun.runTrain(pconf, model, optim, dataloaders, returnValidMetric=True)

    return trainEval

if len(sys.argv) != 2:
    print("Use {} configName".format(sys.argv[0]))
else:
    conf = getattr(sys.modules['configurations'], sys.argv[1])

    print("====================")
    print("RUN USING {}".format(sys.argv[1]))
    print("====================")

    opt = {}
    #opt['best_parameters'], opt['values'], opt['experiment'], opt['model'] = optimize(
    opt['best_parameters'], opt['values'], _, _ = optimize(
        parameters=conf["param"],
        evaluation_function=createTrainEval(conf["conf"]),
        objective_name=conf["checkMetric"],
        total_trials=conf["trials"],
        arms_per_trial=1,
    )

    print("====================")
    print("BEST PARAMETERS")
    print(opt['best_parameters'])
    pickle.dump(opt, open(conf["saveFile"], 'wb'))

    notifier.sendMessage("Training of {} finished on {}".format(sys.argv[1], socket.gethostname()))

