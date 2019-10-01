# Action segmentation using TCN models for pytorch

Originally refined code for　[Multi-Stage TCN (CVPR2019)](https://arxiv.org/pdf/1903.01945.pdf)

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

## Datasets  

・GTEA , 50salads , breakfast

## Features　　

・For extracting features we use the I3D model , which are usually used in action segmentation tasks

## References

[Multi-Stage TCN (CVPR2019)](https://arxiv.org/pdf/1903.01945.pdf)