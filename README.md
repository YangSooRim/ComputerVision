# ComputerVision
## 챌린지 개요
본 챌린지는 "Visual Place Recognition"을 수행하고 있습니다.
시각적 장소 인식 기술이란 이미지의 feature를 추출하여 데이터베이스에 저장하고, 
새롭게 들어오는 질의 이미지에 대해서 동일하게 feature를 추출하여 데이터베이스와 비교하여 가장 유사한 이미지를 검색합니다. 
이를 통해 질의 이미지가 촬영된 곳의 위치 혹은 촬영된 각도 등을 추정할 수 있습니다. 

본 챌린지에서는 VPR 중 NetVLAD를 베이스라인으로 설정하였습니다.
NetVLAD와 관련된 자세한 내용은 [발표영상](https://drive.google.com/file/d/1jzSB-qtzxrzhHOEfR0Xeu_p7L_jc2yu7/view?usp=sharing)을 참고하시기 바랍니다.

## 베이스라인 관련
- 베이스라인 방법론 : NetVLAD
- ![image](https://user-images.githubusercontent.com/88366891/168052630-2573742e-a019-4eef-bed3-5017abdabcf6.png)
- 베이스라인 논문 : 
[NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/abs/1511.07247)

## 참고
- [Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition](https://arxiv.org/abs/2103.01486)
- [Patch-NetVLAD Github](https://github.com/QVPR/Patch-NetVLAD.git)

## 데이터셋
- [Pittsburgh30k](https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/)

# Visual Place Recognition with NetVLAD
## How to Run
### 이미지 데이터셋 다운
```
git clone https://github.com/QVPR/Patch-NetVLAD.git
```
```
conda create -n patchnetvlad python=3.8 numpy pytorch-gpu torchvision natsort tqdm opencv pillow scikit-learn faiss matplotlib-base -c conda-forge
conda activate patchnetvlad
pip3 install --no-deps -e .
```
```
cd Patch-NetVLAD/patchnetvlad
mkdir dataset
cd dataset
```
```
unzip query.zip
unzip refer.zip
```

### Feature 추출
#### Refer Feature Extract
```
python feature_extract.py \
--config_path patchnetvlad/configs/performance.ini \
--dataset_file_path=pits_refer.txt \
--dataset_root_dir=patchnetvlad/dataset/refer \
--output_features_dir patchnetvlad/output_features/pits_refer
```
#### Query Feature Extract
```
 python feature_extract.py \
 --config_path patchnetvlad/configs/performance.ini \
 --dataset_file_path=pits_query.txt \
 --dataset_root_dir=patchnetvlad/dataset/pits_refer \
 --output_features_dir patchnetvlad/output_features/pits_query
```
# Quick Start
## Feature Matching
```
  python feature_match.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_root_dir=patchnetvlad/dataset \
  --query_file_path=pits_query.txt \
  --index_file_path=pits_refer.txt \
  --query_input_features_dir patchnetvlad/output_features/pits_query \
  --index_input_features_dir patchnetvlad/output_features/pits_index \
  --ground_truth_path patchnetvlad/dataset_gt_files/pitts30k_test.npz
  --result_save_folder patchnetvlad/results/matching_hw
```
