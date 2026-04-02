# LocalizedWeather Triton 실행 가이드

이 문서는 기존 프로젝트 README와 별도로, Triton 추론 서버와 GUI를 다시 실행할 수 있도록 정리한 전용 안내서입니다.

## 목적

이 저장소의 Triton 실행 흐름은 다음 두 가지입니다.

1. `weather_model`을 Triton Python backend로 서빙
2. 실제 2023 MADIS/ERA5 NetCDF 데이터를 사용해 브라우저 GUI에서 추론 결과 확인

## 구성 파일

- [Triton 모델 백엔드](triton_model_repository/weather_model/1/model.py)
- [Triton 설정 파일](triton_model_repository/weather_model/config.pbtxt)
- [Triton 이미지용 Dockerfile](triton_model_repository/Dockerfile)
- [Triton 스모크 테스트](triton_model_repository/test_infer_weather.py)
- [실데이터 GUI](gui/triton_gui.py)

## 필요한 데이터

GUI는 아래 파일을 사용합니다.

- `WindDataNE-US/madis/processed/Meta-2019-2023/madis_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc`
- `WindDataNE-US/ERA5/Processed/era5_2023_e2m_8_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc`
- `WindDataNE-US/Shapefiles/Regions/northeastern_buffered.shp`

주의:

- 저장소를 다른 위치로 옮기면 [gui/triton_gui.py](gui/triton_gui.py)의 데이터 경로를 함께 수정해야 합니다.
- GUI의 경계선은 `geopandas`가 없어도 동작하지만, 그 경우 외곽선만 생략됩니다.

## 설치

기본 Python 의존성은 다음 명령으로 설치합니다.

```bash
python -m pip install -r requirements.txt
```

추가로 GUI 실행 환경에는 최소한 아래 패키지가 필요합니다.

- `netCDF4`
- `geopandas`

현재 작업 환경에서 `netCDF4`가 없으면 아래처럼 설치할 수 있습니다.

```bash
python -m pip install netCDF4
```

`geopandas`는 경계선 표시용입니다. 없어도 GUI는 실행되지만 지도 외곽선은 표시되지 않습니다.

## Triton 서버 실행 순서

### 1. Triton 이미지 빌드

```bash
cd triton_model_repository
docker build -t weather-triton-server:latest .
```

이 이미지는 Triton 서버에 PyTorch와 torch-geometric 계열 의존성을 포함합니다.

### 2. Triton 서버 시작

```bash
docker run --gpus=all --rm --name localized-weather-triton \
  -p 18000:8000 -p 18001:8001 -p 18002:8002 \
  -v "$PWD:/models" \
  weather-triton-server:latest \
  tritonserver --model-repository=/models
```

실행 위치는 `triton_model_repository/`여야 합니다. 그래야 `"$PWD"`가 올바른 모델 저장소를 가리킵니다.

### 3. 모델 스모크 테스트

```bash
python triton_model_repository/test_infer_weather.py
```

기대 결과:

- Triton에 샘플 요청 전송
- `pred` 출력 반환
- 응답 JSON 출력

이 단계가 통과해야 GUI도 정상 동작합니다.

## GUI 실행 순서

### 1. GUI 시작

```bash
python gui/triton_gui.py
```

기본 주소는 다음과 같습니다.

- `http://127.0.0.1:8080`

### 2. 브라우저에서 확인

GUI 화면에서 할 수 있는 작업은 다음과 같습니다.

- 채널 선택: `u`, `v`, `temp`, `dewpoint`
- `Run inference`: Triton에 실제 추론 요청 전송
- `Reset map`: 기본 지도 상태로 복귀

## API 요약

GUI는 내부 확인용 엔드포인트도 제공합니다.

### `GET /api/ready`

Triton 연결 여부 확인용입니다.

예시:

```json
{"ready": true}
```

### `GET /api/boundary`

지도 외곽선과 bounds 정보를 반환합니다.

### `POST /api/infer`

실데이터 기반 추론 요청입니다.

예시:

```bash
curl -s -X POST http://127.0.0.1:8080/api/infer \
  -H 'Content-Type: application/json' \
  -d '{"channel":"v"}'
```

응답에는 아래 정보가 들어갑니다.

- `model_name`
- `model_version`
- `sample_time`
- `stations`
- `outputs`

## Triton 입력 형식

`config.pbtxt`에 정의된 입력은 다음과 같습니다.

- `madis_x`
- `madis_lon`
- `madis_lat`
- `edge_index`
- `ex_lon`
- `ex_lat`
- `ex_x`
- `edge_index_e2m`

출력은 다음 하나입니다.

- `pred`

입력 형태나 차원을 바꾸면 `config.pbtxt`와 `triton_model_repository/weather_model/1/model.py`를 같이 수정해야 합니다.

## 종료 방법

### Docker로 실행한 Triton 종료

```bash
docker stop localized-weather-triton
```

### GUI 종료

GUI를 실행한 터미널에서 `Ctrl+C`를 누르면 됩니다.

### 포트 점유 확인

`8080` 포트가 이미 사용 중이면 아래처럼 확인할 수 있습니다.

```bash
ss -ltnp | grep ':8080'
```

## 자주 나는 문제

### 1. `ModuleNotFoundError: netCDF4`

- GUI 실행 환경에 `netCDF4`를 설치해야 합니다.

### 2. `ModuleNotFoundError: geopandas`

- 경계선 표시용입니다.
- 설치하지 않아도 GUI는 실행됩니다.

### 3. `Address already in use`

- 이미 다른 GUI 프로세스가 `8080` 포트를 사용 중입니다.
- 해당 프로세스를 종료한 뒤 다시 실행합니다.

### 4. Triton이 응답하지 않음

- Triton 서버가 먼저 올라와 있어야 합니다.
- `http://localhost:18000/v2/health/ready`를 확인합니다.

## 다시 실행할 때의 최소 절차

1. Triton 이미지 빌드
2. Triton 서버 실행
3. `test_infer_weather.py`로 점검
4. `gui/triton_gui.py` 실행
5. 브라우저에서 `http://127.0.0.1:8080` 접속
