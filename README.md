# Cross Modality Knowledge Distillation for Robust VRU detection in low light and adverse weather conditions

The goal of this project is to use knowledge distillation techniques to improve the performance and robustness of object detectors in adverse weather and low light conditions without adding any extra sensors or cameras in inference time (RGB camera only). 

Paper: 
[Cross Modality Knowledge Distillation for Robust Pedestrian Detection in Low Light and Adverse Weather Conditions](https://ieeexplore.ieee.org/abstract/document/10095353)

<img width="466" alt="image" src="https://github.com/AlirezaRahimpour/Cross-Modality-Knowledge-Distillation-for-Robust-Pedestrian-Detection/assets/18356361/24f00c12-913c-40db-b344-98fd569a6900">



This implementation is based on [Faster R-CNN](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf) with ResNet50-FPN backone in Pytorch using [Seeing Through Fog](https://www.cs.princeton.edu/~fheide/AdverseWeatherFusion/) dataset.

Trained and tested with python: 3.9.7, Ubuntu:18.04.5 LTS, Cuda: 11.2, Pytorch:1.11, GPU: Geforce RTX 3090

## Usage
- Install [PyTorch](https://pytorch.org/).
- Download the data from [here](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EQiY_z8k_1FOnYtWzN-JljcB0k96HO5azGNu_rZsPq4jIg?e=TUhJtb) and extract the ZIP file in `data/` folder.
- Download the [trained teacher network](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EQbkqtMSPXRHmkirHyYfStUBd5ktb0Mh4Q81noLXhx2tOQ?e=boom6E) or train it by running this comment. The teacher network is trained using both RGB images and 3 Gated slices in the dataset.
```
python train_teacher.py
``` 
- train Cross Modality Knowledge Distillation (CMKD) method based on Mean Squred Error (MSE) of backnone features by running this comment:
```
python train_cmkd_mse.py
```
- train Cross Modality Knowledge Distillation (CMKD) method based on adversarial training of backnone features by running this comment:
```![1082](https://media.github.ford.com/user/45972/files/ec8d6458-b5a2-42c9-ba4d-3451e1f0bfc1)

python train_cmkd_adv.py
```
- The trained network can be evaluated using val and test sets by running this comment:
```
python evaluate.py
```
- The baseline network can be trained by running this comment. Baseline is trained using only RGB images without CMKD. 
```
python train_baseline.py
```
- Visual detection examples can be seen by running this comment: 
```
visual_detect.py
```
## Results & Pretrained Weights
|Model|COCO mAP val set| COCO mAP test set| Trained model|
|---|---|---|---|
Teacher|25.8|27.5|[download](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EQbkqtMSPXRHmkirHyYfStUBd5ktb0Mh4Q81noLXhx2tOQ?e=boom6E)
|Baseline|22.5|24.2|[download](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EfTjUsojmxJJmSXrIaX7b98Bdv3NmER5iJ6UOG9DV0t8qA?e=FVdD5X)
|CMKD-MSE|23.6|25.4|[download](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EcyNYGUdSVVHmldwy9ytTXABwXw1loMY9uomx4iFRsrFMw?e=hkwPqP)
|CMKD-Adv|24.2|26.0|[download](https://azureford-my.sharepoint.com/:u:/g/personal/arahimpo_ford_com/EcJ5AiKKSKZGgnR9q2NzmYABnYYqeN9v7gwxfm-0wGGBSA?e=AxDSbc)
## Visual detection examples
knowledge distillation successfully detects pedestrians that baseline does not detect.
![1082](https://media.github.ford.com/user/45972/files/de381a27-98ec-4fab-a98e-8e39feddc932)
![886](https://media.github.ford.com/user/45972/files/d24e1f33-1019-45f0-8fd8-4d29d0fbf28e)

knowledge distillation removes the false positive that baseline has. 
![243](https://media.github.ford.com/user/45972/files/5ff19077-99f9-44ab-b3d9-061e024663a3)
![504](https://media.github.ford.com/user/45972/files/ff9c7ff0-46cb-4f0c-bd55-cddde34c3300)


## Citation 

If you find this code useful, please consider citing:  

```bibtex
@INPROCEEDINGS{10095353,
  author={Hnewa, Mazin and Rahimpour, Alireza and Miller, Justin and Upadhyay, Devesh and Radha, Hayder},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Cross Modality Knowledge Distillation for Robust Pedestrian Detection in Low Light and Adverse Weather Conditions}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095353}}
}
```
