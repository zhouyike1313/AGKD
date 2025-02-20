# Preprocess

Import the experimental virtual environment in conda
```
conda env -f AGKD.yaml
```
then enter the environment:
```
conda activate AGKD
```

### Process raw pathology images

Download the WSI images through their website( [TCGA](https://portal.gdc.cancer.gov/analysis_page?app=Projects),[camelyon16](https://camelyon17.grand-challenge.org/Data/)), put the WSI that downloaded to the `AGKD/WSI/` directory, then preprocess the original image:
```
cd pre
python run_preprocess.py
```
The storage format of the original data set can refer to  `AGKD/WSI/example ` : 
> AGKD/WSI/dataset_name/\*/slide-idx1.svs
> 
> ...
> 
> AGKD/WSI/dataset_name/\*/slide-idxn.svs


  
### Divide the dataset
```
cd pre
python pro_csv.py
```

The format of the label file of the original dataset can refer to `AGKD/csv-stad/example/sheet/total.csv`:
| File Name | Sample Type |
|--|--|
| slide-idxi.svs | Primary Tumor |
| slide-idxj.svs |Solid Tissue Normal  |
|....|...|

| wsi | bag_id | x | y | type |
|-----|--------|---|---|------|
| home/zhouyike/AGKD/zhouyike/AGKD/STAD/TCGA-STAD/7ebda02d-8615-44f3-86b1-5605c9a6b7f3/TCGA-BR-4370-11A-01-BS1.9159a0b8-332d-44c3-96aa-9a9e873e975e.svs | 29 | 18432 | 2048 | 0 |
| home/zhouyike/AGKD/zhouyike/AGKD/STAD/TCGA-STAD/7ebda02d-8615-44f3-86b1-5605c9a6b7f3/TCGA-BR-4370-11A-01-BS1.9159a0b8-332d-44c3-96aa-9a9e873e975e.svs | 30 | 20480 | 2048 | 0 |
| .... | ... | ... | ... | ... |

Where, 'wsi' is the absolute path to the '.svs' file


# Train and test

### train AGKD

```
cd AGKD
python main.py
```
`AGKD.yaml` is used to set parameters.

### test
```
python test.py
```
The `predict.csv` of the directory `"output_dir"` records the prediction results on the test set.
The `main.py` file also defaults to testing the model as soon as it is trained.

# Interpretability experiment

```
cd interpretability
python top_bag_top_patch.py
```
`"output_dir"/tb_tp/*/predict.csv` records the prediction results on the test set.
