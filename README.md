# MEPOI  
**MEPOI: A Multi-Modal Explainable POI Recommendation Framework**

## Dataset  
The datasets used in this project can be downloaded from the following link:  
[Google Drive - MEPOI Datasets](https://drive.google.com/drive/folders/1DDFQzEtOiC60-AR_SqyDJmYKt7a2bmmE)

## Code Structure  

- The `data/` folder contains scripts for dataset preprocessing.  
- `buildgraph.py` is used to construct the trajectory flow graph.  
- `pretrain.py` is used to pre-train the model and save the pre-trained weights.  
- `pretrainalter.py` loads the pre-trained model and performs alternating training and testing for performance evaluation.

## Requirements  

- Python 3.8+  
- PyTorch 2.4.0  

