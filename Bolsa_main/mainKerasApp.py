#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function
Check help for argpaser: $python mainBolsa.py -h 
OR check README.txt 

"""
import argparse
import trainModels as tm
import loadModels_and_test as lmat

class Train_Model:
    def __init__(self, model, savefile = None , inputPATH_DATA = None, nbEpochs=None):
        """
            model: string name of the model used.
            savefile: name of the model .h5 to be saved
            inputPATH_DATA: path of the dataset folder 
            nbEpochs= nb of epochs / default = 20
        """
        self.model = model
        self.inputPATH_DATA=inputPATH_DATA
        self.savefile=savefile
        self.epochs=nbEpochs
    def train(self):
        #FIRST TWO ARE FOR THE MNIST; THIRD ONE IS MORE GENERAL (SO FAR FOR BINARY CLASSIFICATION...)
        if self.model=="default":
            tm.train_default_mnist(self.savefile) #self.img)
        elif self.model=="CNN":
            tm.train_CNN_mnist(self.savefile)#self.img)
            
        else: #TRANSFER LEARNING(XCEPTION)
            if(self.inputPATH_DATA!=None):
                tm.train_transfer_learning(self.savefile, self.inputPATH_DATA,self.epochs)
            else:
                print(">>>YOU FORGOT TO ENTER THE PATH OF THE DATASET TO TRAIN!")


###############################################################################
class Load_then_Classify:
    def __init__(self, model,  input_path = None, output_path=None, modeCSV=None,mnist=None):
        """
            model: string name of the model used.
            input_path: path of the single image to classify or the path of the whole folder containing the imgs;
            output_path: path of the csv file to write/append (based on the modeCSV variable) the results.
            modeCSV: "a" for append the csv file / "w" to write(erase what was written before)
        """
        self.model = model
        self.pathFolder = input_path
        self.pathResultFile= output_path
        self.modeCSV=modeCSV
        self.mnistClassifiersCalled = mnist #temporary solution..
    def loadAndTest(self):
        if(self.mnistClassifiersCalled):#if we want to classify mnist stuff...
            if self.model == None:
                print("No model path given...So we're using default model...")
                lmat.loadModelMnist('Pretrained_models/default_model.h5', self.pathFolder,self.pathResultFile,self.modeCSV)
            else:
                lmat.loadModelMnist(self.model, self.pathFolder,self.pathResultFile,self.modeCSV)

        else: # Generalize option binary
            print("HERE")
            lmat.loadModel(self.model, self.pathFolder, self.pathResultFile,self.modeCSV)
            

###############################################################################

#ARGPARSER:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mainBolsa.py')
    subparsers = parser.add_subparsers(help='sub-command help',dest='subparser_name')

###############################################################################

    # creation of parser for the training action...
    parser_train = subparsers.add_parser("train", help = "Choose your classifier to be trained.")
    
    parser_train.add_argument("--model", choices = ["default","CNN","TransferLearning"],
                        default = None, required=True, help = "Choose your classifier to be trained.")
    
    parser_train.add_argument("--save", metavar = "FILE", 
                              required=False,
                              default = None,
                              help = "Enter the path and name of the model (without .h5 in the end) to be saved, example: modeltest")
    
    parser_train.add_argument("--inputPATH_DATA", metavar="FILE",required=False,
                                 default=None,
                                 help="Enter path of the DATASET FOLDER.")
    
    parser_train.add_argument("--epochs",type=int,required=False,
                                 default=20,
                                 help="Enter the number of epochs.")
    """
    parser_train.add_argument("--inputIMG", metavar="FILE",required=False,
                                 default=None,
                                 help="Enter path of the image file")
    """


########################################################################################################################
    
    # creation of parser for the classifying&testing action...
    parser_classify = subparsers.add_parser("classify", help = "Choose the classifier to be loaded and tested.")
    
    parser_classify.add_argument("--model", metavar="FILE",
                        default = None, help = "Enter path of the classifier to be trained.")
    
    parser_classify.add_argument("--inputPATH", metavar="FILE",required=False,
                                 default=None,
                                 help="Enter path of the folder/images to classify.")

    parser_classify.add_argument("--outputPATH",metavar="FILE",required=False,
                                 default=None,
                                 help="Enter the path of the csv file to write the results")
    
    parser_classify.add_argument("--modeCSV",type=str,required=False,
                                 default='a',
                                 help="Enter mode for csv file management, w for write or a for append file.")
    
    parser_classify.add_argument("--mnist",type=bool,required=False,
                                 default=False,
                                 help="Enter this parameter true for mnist classifiers...")
#########################################################################################################################
    print(parser.parse_args())
    args = parser.parse_args()
    
    #print(args.input_image)
    
    
    if( args.subparser_name=='train'):
        print("YOU CHOSE TRAINING")
        training = Train_Model(model = args.model , savefile= args.save,  inputPATH_DATA=args.inputPATH_DATA, nbEpochs=args.epochs)
        training.train()
        
    else:
        print("YOU CHOSE CLASSIFICATION")
        loadclassify = Load_then_Classify(model = args.model , input_path = args.inputPATH, output_path = args.outputPATH, modeCSV=args.modeCSV, mnist=args.mnist)
        loadclassify.loadAndTest()
    