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
`Heavy chain: EVQLQESGPGLVKPYQSLSLSCTVTGYSITSDYAWNWIRQFPGNKLEWMGYITYSGTTDYNPSLKSRISITRDTSKNQFFLQLNSVTTEDTATYYCARYYYGYWYFDVWGQGTTLTVSS`
`Light chain: DIQMTQSPAIMSASPGEKVTMTCSASSSVSYMYWYQQKPGSSPRLLIYDSTNLASGVPVRFSGSGSGTSYSLTISRMEAEDAATYYCQQWSTYPLTFGAGTKLELK`

Output:

CDR chain	GYSITSD	ITYSG	YCARYYYG	SASSSVSYMYW	STNLASG	QWSTYPLTF
Prediction	[0.226 0.777 0.444 0.096 0.474 0.613 0.689]	[0.288 0.36 0.905 0.571 0.837]	[0.969 0.71 0.13 0.226 0.216 0.242 0.189 0.075]	[0.191 0.062 0.331 0.316 0.385 0.404 0.656 0.896 0.35 0.864 0.083]	[0.108 0.2 0.321 0.424 0.215 0.368 0.205]	[0.045 0.34 0.46 0.699 0.879 0.589 0.703 0.608 0.663]
















