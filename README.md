# MultiPRIDE-Evalita-2026
MultiPRIDE: Multilingual Automatic Detection of Reclamation of Slurs in the LGBTQ+ Context

Access to the data is available through [MultiPRIDE Organizers](https://multipride-evalita.github.io/).

```commandline
git clone https://github.com/rbg-research/MultiPRIDE-Evalita-2026.git
cd MultiPRIDE-Evalita-2026
```

# Notebooks
## Contents

| S.No |      Division       |                                                              Description                                                               |                                       Link                                        |
|:----:|:-------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|
|  1   | Frequency Analysis  |                          Distributional Analysis of Data and Label, Class Specific Feature Importance Scoring                          |                                [Link](notebooks/0_Descriptive-Analytics-Data.ipynb)                                 |
|  2   | Frequency Analysis  | Baseline setup with Conventional ML Algorithms, Language Specific and Multi-lingual Classification, Stratified 5-fold Cross Validation |   [Link](notebooks/1_Conventional-ML-Baseline.ipynb)   |
|  3   | Contextual Analysis |    Deep Embedding methods followed by Conventional ML Algorithms, Multi-lingual Classification, Stratified 5-fold Cross Validation     |     |
|  4   | Contextual Analysis |                              Fine-tuning Multilingual Language Models, Stratified 5-fold Cross Validation                              |               |
|  5   | Contextual Analysis |                                   Prompt Analysis w.r.t Zero-shot Multilingual Large Language Models                                   |          |
|  6   | Contextual Analysis |                                   Prompt Analysis w.r.t Few-shot Multilingual Large Language Models                                    |  |



# Label Distribution
![Label Distribution](figures/label_distribution.svg)

# Frequency Analysis: Chi-Square Feature Selection
![Frequency Analysis](figures/chi2_features_by_language.svg)


# Installation
#### Python and Libraries
```commandline
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-dev python3.10-distutils
wget https://bootstrap.pypa.io/get-pip.py
source ~/.bashrc
python3.10 get-pip.py
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
pip3.10 install -U pip
rm get-pip.py
```

#### Virtual Environment
```commandline
pip3.10 install virtualenv
virtualenv --python=python3.10 "$HOME"/environments/multipride
source "$HOME"/environments/multipride/bin/activate
```

#### Jupyter Notebook
```commandline
pip install jupyter
pip install ipykernel
python -m ipykernel install --user --name=multipride
```

#### Requirements
```commandline
pip install -r requirements.txt
```
