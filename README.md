# Machine Learning
- This repository contains various projects which have been solved with the help of machine learning
- This [link](https://github.com/shahidzk1/Machine_learning/blob/main/HP_OPT_class/HP_OPT_class_code.py) has a class that uses Optuna package to optimize the hyperparameters of transformers, MLP, XGB, CNN using the TPE algorithm.

## Higgs Boson Challenge
  - The [class](https://github.com/shahidzk1/Machine_learning/blob/main/HP_OPT_class/HP_OPT_class_code.py) was created to be used for the hyperparameters optimization in the [google colab](https://colab.research.google.com/drive/1I4HS7SZduw426C-YuxboArrfI9QiT6OV?usp=sharing). Google's colab is a free-of-cost Platform as a service cloud model.
  - The colab uses the above 4 different ML algorithms for the Higgs selection from the background and the data comes from the [HiggsChallenge](https://www.kaggle.com/competitions/higgs-boson/data)
    - The colab also shows the inner workings of the ML model by using the Shapley score using the SHAP library. The SHAP method fits a simplified model on the ML model locally. The contribution of an individual variable is the difference between its presence and absence while predicting a label for a class.
  - The offline notebook is available at the [link](https://github.com/shahidzk1/HP_OPT/blob/main/Higgs_challenge/HiggsBosonChallenge.ipynb)
## Flowers classification
 - To classify flowers from 5 different species CNN is used [google colab](https://colab.research.google.com/drive/1GqXfQ9thSFojbgLURxic5kIFXJL0l0xB?usp=sharing)
## Running the code on your personal computer
  - If you want to run it on your personal computer e.g. on Visual Studio or terminal then
  - cd /directory/on/your/personal/computer
    - Clone the repository:
      ```
      git clone https://github.com/shahidzk1/Machine_learning.git
      ```
    - cd /directory/on/your/personal/computer/Machine_learning/
      ```
      git pull origin main
      pip install -r requirements.txt
      python setup.py install
      ```
    - cd /directory/on/your/personal/computer/Machine_learning/test/
    - For unit testing run
       - python [test_hp_opt_class_code.py](https://github.com/shahidzk1/Machine_learning/blob/main/HP_OPT_class/test_hp_opt_class_code.py)
       - The output should end with
          ```
          Ran 4 tests in xs (x is time in sec)
          OK
          ```
    - Run [the notebook](https://github.com/shahidzk1/Machine_learning/blob/main/Higgs_challenge/HiggsBosonChallenge_personal_PC.ipynb)
### Azure Cloud
 - Running on Azure cloud, Azure AI machine learning studio, can lead to warnings about the scikit-learn version, which can be ignored.
 - If optuna and optuna-integration packages are not found after requirements installation then simply use the following
    ```
          pip install optuna
          pip install optuna-integration
    ```
    
