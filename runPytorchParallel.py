#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import time
import sys
import socket
import configurations
import funPytorch as fun
import notifier
import torch.multiprocessing as mp

device = "cuda:0"

def myRun(conf, k, lock):
    valid = False
    while not valid:
        valid = True
        lock.acquire()
        try:
            print("======= CREATE MODEL {}".format(k))
            model, optim = fun.makeModel(conf[k], device)
            print("======= LOAD DATA {}".format(k))
            dataloaders, _ = fun.processData(conf[k])
            print("======= TRAIN MODEL {}".format(k))
        except RuntimeError as e:
            print("======= RETRY MODEL {}".format(k))
            valid = False
        lock.release()
        if valid:
            fun.runTrain(conf[k], model, optim, dataloaders)
            lock.acquire()
            print("======= FINISH MODEL {}".format(k))
            lock.release()
        else:
            time.sleep(120)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Use {} configName".format(sys.argv[0]))
    else:
        conf = getattr(sys.modules['configurations'], sys.argv[1])

        print("====================")
        print("RUN USING {}".format(sys.argv[1]))
        print("====================")

        #mp.set_start_method('spawn')
        lock = mp.Lock()
        processes = [mp.Process(target=myRun, args=(conf, k, lock)) for k in conf]
        for k in conf:
            processes[k].start()
        for k in conf:
            processes[k].join()

        notifier.sendMessage("Training of {} finished on {}".format(sys.argv[1], socket.gethostname()))

        print("====================")
        print("FINISH ALL")
        print("====================")

