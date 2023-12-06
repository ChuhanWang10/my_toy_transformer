# my_toy_transformer

## Set up environment
```
conda create -n toy_projects python=3.9
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Download data
```
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```