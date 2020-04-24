#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import sys
import socket
import configurations
import funPytorch as fun
import notifier

device = "cuda:0"

if len(sys.argv) != 2:
    print("Use {} configName".format(sys.argv[0]))
else:
    conf = getattr(sys.modules['configurations'], sys.argv[1])

    print("====================")
    print("RUN USING {}".format(sys.argv[1]))
    print("====================")
    print("======= CREATE MODEL")
    model,optim = fun.makeModel(conf, device)
    print("======= LOAD DATA")
    dataloaders, _ = fun.processData(conf)
    print("======= TRAIN MODEL")
    fun.runTrain(conf, model, optim, dataloaders, verbose=True)

    notifier.sendMessage("Training of {} finished on {}".format(sys.argv[1], socket.gethostname()))

