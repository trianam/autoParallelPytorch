import copy

class Conf:
    def __init__(self, confDict):
        self.__dict__ = confDict

    def copy(self, updateDict={}):
        newConf = copy.copy(self)
        for k in updateDict:
            newConf.__dict__[k] = updateDict[k]

        return newConf

    def has(self, key):
        return key in self.__dict__

