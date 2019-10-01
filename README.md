# Action segmentation using TCN models for pytorch

Personally refined code for　[Multi-Stage TCN (CVPR2019)](https://arxiv.org/pdf/1903.01945.pdf)


## Datasets  

Need to download the dataset before starting.
Download the data directory from the [link](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) which is in the [original code](https://github.com/yabufarha/ms-tcn) and place it in the top directory level, same as main.py.

Data consists 3 datasets

・GTEA , 50salads , breakfast


## Training  

```
python main.py [arguements(e.g. template.yaml)] train [split(1-4 for breakfast and gtea, 1-5 for 50salads)] --[device(e.g. cuda:0)]
```  

 check the template.yaml for example of settings  

## Evaluation  

```
python main.py [arguements(e.g. template.yaml)] test [split(1-4 for breakfast and gtea, 1-5 for 50salads)] --[device(e.g. cuda:0)]
```  

to output the predictions in the result directory  

then

```
python eval.py [arguements(e.g. template.yaml)] [split(1-4 for breakfast and gtea, 1-5 for 50salads)] 
```  

to show the evaluation results


## Features　　

・For extracting features we use the I3D model , which are usually used in action segmentation tasks

## References

[Multi-Stage TCN (CVPR2019)](https://arxiv.org/pdf/1903.01945.pdf)