import xarray as xr
from pathlib import Path

# 다운로드된 파일 중 하나만 지정하여 테스트
dataPath = Path('/projects3/home/flag0220/LocalizedWeather/WindDataNE-US')
targetFolder = dataPath / 'madis' / 'raw'

# 2019년 1월 폴더에서 gz 파일 하나 찾기
test_files = list((targetFolder / '2019/1/mesonet/').glob('*.gz'))

if test_files:
    test_file = test_files[0]
    print(f"테스트 파일 경로: {test_file}")
    
    try:
        # 엔진을 지정하지 않고 열기 시도
        with xr.open_dataset(test_file) as d:
            print("성공적으로 파일을 읽었습니다!")
            print(d.data_vars)
    except Exception as e:
        print(f"파일 열기 에러 발생: {e}")
        
        # 만약 에러가 난다면, scipy 엔진으로 다시 시도 (NetCDF3 gz 파일인 경우)
        try:
            print("\nscipy 엔진으로 다시 시도합니다...")
            with xr.open_dataset(test_file, engine='scipy') as d2:
                print("scipy 엔진으로 읽기 성공!")
        except Exception as e2:
            print(f"scipy 엔진 열기 에러: {e2}")
else:
    print("해당 경로에 테스트할 .gz 파일이 없습니다.")