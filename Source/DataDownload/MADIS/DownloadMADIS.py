from dateutil import rrule
from datetime import datetime
import multiprocessing
from pathlib import Path
import xarray as xr
import warnings
import pandas as pd
import tqdm
from IPython.display import clear_output
import urllib.request

dataPath = Path('/projects3/home/flag0220/LocalizedWeather/WindDataNE-US')
targetFolder = dataPath / 'madis' / 'raw'
targetFolder.mkdir(parents=True, exist_ok=True)

# 1. 안전한 다운로드 파트 (urllib 및 타임아웃 적용)
mesonetUrl = lambda year, month, day, hour: f'https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/{year}/{month:02d}/{day:02d}/LDAD/mesonet/netCDF/{year}{month:02d}{day:02d}_{hour:02d}00.gz'
startDate = '2019-01-01'
endDate = '2024-01-01'
dt = rrule.HOURLY

dates = list(rrule.rrule(dt, dtstart=datetime.strptime(startDate, '%Y-%m-%d'),
                      until=datetime.strptime(endDate, '%Y-%m-%d')))[:-1]

def getDataForDate(date):
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    url = mesonetUrl(year, month, day, hour)
    outputFile = targetFolder/f'{year}/{month}/mesonet/{year}{month:02d}{day:02d}_{hour:02d}00.gz'
    
    if outputFile.exists() and outputFile.stat().st_size > 0:
        return
    
    outputFile.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response, open(outputFile, 'wb') as out_file:
            out_file.write(response.read())
    except Exception:
        if outputFile.exists():
            outputFile.unlink()
        return

print("1단계: 기상 데이터 다운로드를 시작합니다...")
# 서버 차단을 막기 위해 Pool 사이즈는 32으로 제한
with multiprocessing.Pool(32) as pool:
    list(tqdm.tqdm(pool.imap_unordered(getDataForDate, dates), total=len(dates), desc="Downloading"))


# 2. 메모리 안정성을 확보한 전처리 병합 파트 (순차 처리 및 가비지 컬렉션)
var_list = ['stationId', 'reportTime', 'latitude', 'longitude', 'elevation', 'dewpoint', 
            'dewpointDD', 'temperature', 'temperatureDD', 'windSpeed', 'windSpeedDD', 
            'windDir', 'windDirDD', 'relHumidity', 'relHumidityDD', 'solarRadiation']

inputPath = lambda year, month: targetFolder / f'{year}/{month}/mesonet/'
targetPath = lambda year, month: dataPath / 'madis' / 'raw_monthly' / 'mesonet' / str(year)

def process_and_save_monthly(year, month):
    out_path = targetPath(year, month) / f'{month}.nc'
    if out_path.exists():
        return
        
    datafiles = list(inputPath(year, month).glob('*.gz'))
    if not datafiles:
        return

    monthly_dfs = []
    
    for file in tqdm.tqdm(datafiles, desc=f"Processing {year}-{month:02d}"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with xr.open_dataset(file) as d:
                    existing_vars = [v for v in var_list if v in d.data_vars or v in d.coords]
                    d_filtered = d[existing_vars]
                    
                    df = d_filtered.to_dataframe().reset_index()
                    
                    if 'reportTime' in df.columns:
                        df.insert(0, 'minute', df['reportTime'].dt.minute)
                        df.insert(0, 'hour', df['reportTime'].dt.hour)
                        df.insert(0, 'day', df['reportTime'].dt.day)
                        df.insert(0, 'month', df['reportTime'].dt.month)
                        df.insert(0, 'year', df['reportTime'].dt.year)
                        df = df.drop(columns='reportTime')
                    
                    # byte 문자열을 일반 문자열로 디코딩 (에러 방지)
                    for col in df.columns:
                        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
                            df[col] = df[col].str.decode('utf-8')

                    monthly_dfs.append(df)
                    
        except Exception:
            continue

    if monthly_dfs:
        try:
            final_df = pd.concat(monthly_dfs, ignore_index=True)
            final_ds = final_df.to_xarray()
            
            out_path.parent.mkdir(parents=True, exist_ok=True)
            final_ds.to_netcdf(out_path)
            
            # 병합 완료 후 즉시 메모리 해제
            final_ds.close()
            del final_df
            del monthly_dfs
            
        except Exception as e:
            print(f"저장 에러 발생 ({year}-{month:02d}): {e}")

print("\n2단계: 월별 데이터 전처리 및 병합을 시작합니다...")
for year in range(2019, 2024):
    for month in range(1, 13):
        process_and_save_monthly(year, month)

print("\n모든 데이터 처리가 완료되었습니다.")