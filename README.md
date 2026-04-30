# Digital RF Seek Source block
This is a block that can be used for GNU Radio. It allows you to "jump" around in a Digital_RF file to see different time points in the file. 

## How to use in standard GNU Radio
1. To use this block you will need to drop an "Embedded Python Block" into GNU-Radio. Then open up the editor and copy and paste the 
code from ```code.py``` if you want to use the **digital_rf_relseek_source** block. If you want the **binary_relseek_source** block copy, the code from ```binary_code.py``` and the process is exactly the same.  
2. You will also need a ```throttle``` block, a ```QT GUI Range``` block, and a sink of some sort. In our example we use ```QT GUI Sink```. Finally you will need a ```Message Strobe``` block.
3. In the parameters of the relseek block you will need to input the filepath to the dataset directory, the name of the channel that you are looking at, and the start seconds. (Please note that the channel must be within the dataset directory. For example if the dataset directory is ```/home/dataset``` the path to the channel must be ```/home/dataset/channel```.) If there is more than one dataset in the channel, you may specify this in the "subchannel" parameter. Otherwise you may leave this parameter blank.
4. In the ```QT GUI Range``` block, in the ID parameter, call it ```seek_seconds```.
5. Finally connect the ```Message Strobe``` strobe to the ```relseek_source``` seek. Connect the out of this source block to the in of the ```Throttle ``` and connect the out of this block to the in of the desired sink. You can see where you are seeking to in the logs output. 
#### Example Usage of the Block 
![Logo](Misc/digital_rf_reader_block.png)

## How to use in command line
1. You need to download two files. ```reading_MEP_epy_block_0.py``` and ```script.py```. Make sure that these two files are within the same directory.
2. Make sure that GNU Radio is installed on your computer. 
3. Open up a terminal and naviate to the directory where these two files are contained. Run this command: ``` python3 /home/script.py --data-dir "/home/data_folder" --channel "chA" --start-sec 100 --home-dir "/home/data_folder" ``` (Please note that chA should be within the data_folder directory)
4. GNU Radio should open up and run the program. 
