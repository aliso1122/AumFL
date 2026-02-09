# Guidance to the project

The top-layer parameters  are defined in  <font color=#008000 >control/Enums.py</font>, i.e. FLFramework, FLSubtype, LearningType and FLEnvironment. Browsing this file as the start can help you have a quick overview of the project.

The only main function of the project is written in <font color=#008000 >main.py</font>. Here we provide 4 scripts to run it with different settings.
```bash
python3 main.py --dataset_name CMUMOSEI --fl_environmet hetero012 --model_id ul-mmulfedaa --purpose generate_data
python3 main.py --dataset_name CMUMOSEI --fl_environmet hetero012 --model_id ul-mmulfedaa --purpose train
python3 main.py --dataset_name CMUMOSEI --fl_environmet single012 --model_id sl-mmulfedaa --purpose generate_data
python3 main.py --dataset_name CMUMOSEI --fl_environmet single012 --model_id sl-mmulfedaa --purpose train
```

The first 2 scripts are to **train encoders** under the simulated modality-heterogeneous FL environment named hetero012. The last 2 scripts are to apply the pre-trained encoders into a **supervised task**. The task is performed in a special FL environment with only one dataset, which is named single012. More parameters are defined in the config files: <font color=#008000 >trained_models/CMUMOSEI/hetero012/ul-mmulfedaa/config.json</font>, <font color=#008000 >trained_models/CMUMOSEI/single012/sl-mmulfedaa/config.json</font>(A default config will be created when the program runs).  The trained model will be saved in the same directory with the config file.

Before run the scripts, you need to download the standard dataset and put it <font color=#008000 >data/CMUMOSEI/raw</font> as required by <font color=#008000 >data/CMUMOSEI/raw</font>. All the datasets of the project are public and you might click the link for downloading: [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), [UR Fall](https://github.com/zkashef/ECE535-FederatedLearning) and [CMU-MOSEI](https://github.com/pliang279/MultiBench).

Lastly, we present the architecture of the project:

* **control/Enums** Top-layer definitions of the project.
* **control/config_parameters** Parameters to run the project.

* **control/Server.py** The codes of server for canonical FL.
* **control/Client.py** The codes of client for canonical FL.
* **control/global_training_algorithm.py** The algorithms for canonical FL, which are run by the server.

 
* **control/MMULFED/ServerMMULFED.py** The codes of Server for GAMAFedAC.
* **control/MMULFED/ClientMMULFED.py** The codes of client for GAMAFedAC.
* **control/MMULFED/global_training_algorithm.py** The algorithms for GAMAFedAC, which are run by the server.

* **control/Manager.py** Every client has a manager for local training and the manager can be used by all types of clients in the project.

* **data_generating/data_preprocess.py** Codes to create client datasets for simulating FL environment.
* **data_generating/CMUMOSEI.py** Codes to preprocess CMU-MOSEI dataset.
* **data_generating/HAR.py** Codes to preprocess UCI-HAR dataset.
* **data_generating/URFALL.py** Codes to preprocess UR Fall dataset.

* **models/** The directory for defined models.
* **main.py** The only entrance of the project.
