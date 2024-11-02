-----------------------------------
운영 체제 (OS) : Windows 11 (Visual Studio Code)
Python 버전 : 3.10.14
CPU 아키텍처 : x86
-----------------------------------

# 파일 설명

[파일 구조] - 파일 구조를 유지해주셔야 실행에 문제가 없습니다.
1. input - 모델 예측에 사용할 데이터 저장소
    - ewp02
    - ewp04
    - train, test ldaps
    - train_y
(모든파일을 데이터타입 최적화 후 parquet 포맷으로 변경 -> 메모리 최적화를 위함)
2. models - 모델 저장소
    - lgb_gj_1029_total.pkl
    - lgb_yg_1029_total.pkl
3. notebooks - 분석 및 모델링에 사용한 주피터 노트북 저장소
    - data_optimize (데이터 최적화 주피터 노트북)
    - final (최종 제출 주피터 노트북)
4. src - 각종 함수들 및 config 정보 저장소
    - config (input, output 경로, 모델 파라미터 등 구성요소)
    - data_processor (데이터 전처리 함수)
    - metric (NMAE 지표 수식)
    - utils (데이터 불러오기, 최적화 등 함수)

[기타 파일]
5. requirements.txt - 본 공모전 과정에서 최종적으로 사용한 패키지들 정리
(설치가 안될 경우 python.exe -m pip install -r requirements.txt 로 실행)
6. 씽씽 불어라팀 최종 제출.csv - 최종 제출 예측 결과 csv 파일
7. 예측 발전량 시각화 - 경주, 영광에 대한 예측 발전량 시각화 png 파일
