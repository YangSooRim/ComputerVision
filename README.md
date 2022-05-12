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
- ![image](https://user-images.githubusercontent.com/88366891/168052765-9a1b4c5a-f5ca-4fec-8233-24b13fcc4af0.png)
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
Pittsburgh250k에서 query와 refer 이미지 데이터셋을 다운 받습니다. 
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
NetVLAD를 활용하여 이미지의 global feature를 추출하고 feature database를 생성합니다
```
python feature_extract.py \
--config_path patchnetvlad/configs/performance.ini \
--dataset_file_path=pits_refer.txt \
--dataset_root_dir=patchnetvlad/dataset/refer \
--output_features_dir patchnetvlad/output_features/pits_refer
```
#### Query Feature Extract
질의 이미지도 NetVLAD를 활용하여 global feature를 추출합니다.
```
 python feature_extract.py \
 --config_path patchnetvlad/configs/performance.ini \
 --dataset_file_path=pits_query.txt \
 --dataset_root_dir=patchnetvlad/dataset/pits_refer \
 --output_features_dir patchnetvlad/output_features/pits_query
```
# Quick Start
## Feature Matching
페이스북에서 제공하는 gpu를 활용하는 faiss 라이브러리를 활용하여 벡터 간 유사도를 확인하여 데이터베이스에서 가장 유사한 feature를 찾습니다.
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
## Eval
출력은 .json 형식으로 받습니다.
```
{
    "NetVLAD": [
        {
            "id": 0,
            "positive": [
                0,
                11,
                96,
                12,
                1
            ]
        },
        {
            "id": 1,
            "positive": [
                1,
                2,
                13,
                3,
                4
            ]
        },
```
다음과 같은 형식으로 작성해주시기 바랍니다. json 파일 추출 시 참고 코드입니다.
```
def feature_match(eval_set, device, opt, config):
    input_query_local_features_prefix = join(opt.query_input_features_dir, 'patchfeats')
    input_query_global_features_prefix = join(opt.query_input_features_dir, 'globalfeats.npy')
    input_index_local_features_prefix = join(opt.index_input_features_dir, 'patchfeats')
    input_index_global_features_prefix = join(opt.index_input_features_dir, 'globalfeats.npy')

    qFeat = np.load(input_query_global_features_prefix)
    pool_size = qFeat.shape[1]
    dbFeat = np.load(input_index_global_features_prefix)

    if dbFeat.dtype != np.float32:
        qFeat = qFeat.astype('float32')
        dbFeat = dbFeat.astype('float32')

    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    # noinspection PyArgumentList
    faiss_index.add(dbFeat)

    _, predictions = faiss_index.search(dbFeat, 5)


    file_path = "./submit.json"

    data = {}
    data['NetVLAD'] = list()

    for i in range(len(predictions)) :
        data_t = [("id",i),("positive",predictions[i].tolist())]
        data_t = dict(data_t)
        data['NetVLAD'].append(data_t)
    
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
   ```
   
