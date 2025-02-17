import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

# write a class
def DatasetInfo(name = "PROTEINS"):  
        if name.lower() == "graph-twitter":
            return {"num_classes": 3,"num_features": 768,'initSize':100}
        if name.lower() == "graph-sst2":
            return {"num_classes": 2,"num_features": 768,'initSize':100}
        if name.lower() == "graph-sst5":
            return {"num_classes": 5,"num_features": 768,'initSize':100}        
        if 'drugood' in name:
            return {"num_classes": 2,"num_features": 39,'initSize':100}
        if "SPMotif" in name:
            return {"num_classes": 3,"num_features": 4,'initSize':100}
        if name == "CIFAR10":
            return {"num_classes": 10,"num_features": 5,'initSize':10}
        elif name == "COLLAB":
            return {"num_classes": 3,"num_features": 65}
        elif name == "PATTERN":
            return {"num_classes": 2,"num_features": 3}
        elif "IMDB-BINARY" in name:
            return {"num_classes": 2,"num_features": 65,'initSize':6}
        elif "REDDIT-BINARY" in name:
            return {"num_classes": 2,"num_features": 65,'initSize':20}    
        elif name == "IMDB-MULTI":
            return {"num_classes": 3,"num_features": 65,'initSize':8}
        elif name == "IMDB-BINARY":
            return {"num_classes": 2,"num_features": 65,'initSize':8}
        elif name == "REDDIT-MULTI-5K":
            return {"num_classes": 5,"num_features": 65}
        elif name == "REDDIT-MULTI-12K":
            return {"num_classes": 11,"num_features": 65}       
        elif name == "PROTEINS":
            return {"num_classes": 2,"num_features": 3,'initSize':6}
        elif name == "ENZYMES":
            return {"num_classes": 6,"num_features": 3}
        elif name == "NCI1":
            return {"num_classes": 2,"num_features": 37,'initSize':18}
        elif name == "NCI109":
            return {"num_classes": 2,"num_features": 37,'initSize':18}
        elif name=="BZR":
            return {"num_classes": 2,"num_features": 53,'initSize':6}
        elif name=="DD":
            return {"num_classes": 2,"num_features": 89,'initSize':10}
        elif name=="PTC_FM":
            return {"num_classes": 2,"num_features": 18,'initSize':4}
        elif "MUTAG" in name:
            return {"num_classes": 2,"num_features": 7,'initSize':4}
        
        