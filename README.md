# DeepANIS

Predicting antibody paratope by bidirectional long-short-term memory neural networks and transformer encoder with concatenated CDR sequences

## Dependencies

+ cuda >= 9.0
+ cudnn >= 7.0
+ tensorflow >= 1.9.0
+ keras >= 2.2.4
+ numpy == 1.19.1
+ scikit-learn == 0.23.2

## Dataset

277 antibody/antigen complexes -> 277 concatenated CDR sequences.

can be refered in the 'DeepANIS/data/'.

## Preprocess

You can download the source dataset from SAbDab (http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/).

Then install keras_transformer:
`cd keras_transformer`
`python setup.py install`

In the folder `./data`, we have preprocessed dataset by the corresponding jupyter notebook. All source datasets can be refered in the `./data/source/`, and all preprocess files can be refered in the `./Data/preprocess/`.

## Models

The trained models can be refered in the `./trained_model/<model>`.

## Predicting

We provide a example using sequence_model. You can predict paratope using:
`python .py`

## Example output

Input:

`Heavy chain:` 

`EVQLQESGPGLVKPYQSLSLSCTVTGYSITSDYAWNWIRQFPGNKLEWMGYITYSGTTDYNPSLKSRISITRDTSKNQFFLQLNSVTTEDTATYYCARYYYGYWYFDVWGQGTTLTVSS`

`Light chain:`

`DIQMTQSPAIMSASPGEKVTMTCSASSSVSYMYWYQQKPGSSPRLLIYDSTNLASGVPVRFSGSGSGTSYSLTISRMEAEDAATYYCQQWSTYPLTFGAGTKLELK`

Output:

| CDR chain | GYSITSD | ITYSG | YCARYYYG | SASSSVSYMYW | STNLASG | QWSTYPLTF | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Prediction | [0.22 0.77 0.44 0.01 0.47 0.61 0.69] | [0.29 0.36 0.91 0.57 0.84] | [0.97 0.71 0.13 0.23 0.21 0.24 0.19 0.08] | [0.19 0.06 0.33 0.32 0.39 0.40 0.66 0.89 0.35 0.86 0.08] | [0.11 0.2 0.32 0.42 0.22 0.37 0.21] | [0.04 0.34 0.46 0.70 0.88 0.59 0.70 0.61 0.66]|

















