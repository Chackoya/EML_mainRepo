USAGE OF THE ARGUMENT PARSER ON THE CURRENT SETTINGS...

Check help with cmd: $python mainBolsa.py -h

Here are some usage examples of commands:

>>>FOR TRAINING & SAVING TRAINED MODEL
	Example 1 (terminal cmd to call the function for training with default model and saved on the path chosen
	(NOTE:DONT NEED TO ADD .h5! in the end):

	$python mainBolsa.py train --model default --save Pretrained_models/default_model
																
	----
	
	Example 2 (cmd for training CNN model and save on the path (AGAIN... DONT NEED TO ADD .h5)

	$python mainBolsa.py train --model CNN --save Pretrained_models/CNN_model

	----
	
	Example 3: it's also possible (but not required) to test an image directly after training (to avoid having to make another cmd right after...)
	$python mainBolsa.py train --model default --save Pretrained_models/default_model --inputIMG StockImg/img8.jpg


>>>FOR LOADING&TESTING:
	Example 1 (cmd to call the function in classify mode with the pretrained model default, also indicate the path of the img to classify
	(NOTE: this time we put the .h5 ).
	$python mainBolsa.py classify --model Pretrained_models/default_model.h5 --inputIMG StockImg/img8.jpg
	---
	
	Example 2 (same but for CNN)
	
	$python mainBolsa.py classify --model Pretrained_models/CNN_model.h5 --inputIMG StockImg/img8.jpg
	
	----

	Notes: for loading&testing, the arg --model and --inputIMG are optional, so we end up with something like this:
	$python mainBolsa.py classify
	This will result with the loading of the default model and just a print of the accuracy on a test mniist dataset

--------------------------------------
TODO: 
-add the pretrained model from google or something 
-maybe add some extra features to the parser (like configurations / nb epochs for example etc)...
- folders (test multiple iamges) -> OK
-csv file:  model used, nb_classes, img , classification value(get all values 10 cols) or list of outputs vals(put the list string format) , max value from array,  class;  -> OK
 
modo append(argparse); -> OK
dia 9 , 14h

-------------------------------------------------------------------------------------------------------------------------
#UPDATE USAGE EXAMPLE:


CLASSIFY:
python mainKerasApp.py classify --model Pretrained_models/xception_dogcat.h5 --inputPATH /home/gama/BolsaStuff/EML_mainRep/TestImg --outputPATH results.csv 



TRAIN:
python mainKerasApp.py train --model TransferLearning --save Pretrained_models/xception_dogcat --inputPATH /home/gama/BolsaStuff/EML_mainRep/PetImages --epochs 5


