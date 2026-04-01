import argparse
import json
import os
import sys
from pathlib import Path

# GPU 설정을 먼저 읽어 환경 변수에 등록합니다.
temp_parser = argparse.ArgumentParser(add_help=False)
temp_parser.add_argument('--gpus', default='0', type=str)
temp_args, _ = temp_parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = temp_args.gpus

import Main
from PostProcessInputs import PostProcessArgs
from Settings.Settings import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', default='gnn_test', type=str)
    parser.add_argument('--gpus', default='0', type=str)

    parser.add_argument('--model_type', default=ModelType.GNN.value, type=int)
    parser.add_argument('--madis_vars_i', default=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                                   EnvVariables.dewpoint.value), type=int, nargs='+')
    parser.add_argument('--madis_vars_o', default=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value,
                                                   EnvVariables.dewpoint.value), type=int, nargs='+')

    # 기존 LocalLauncher와 동일하게 외부 변수에서는 이슬점(dewpoint)을 제외했습니다.
    parser.add_argument('--external_vars', default=(EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value), type=int, nargs='+')

    ## Graph
    parser.add_argument('--network_construction_method', default=NetworkConstructionMethod.KNN.value, type=int)

    ## Dataset
    parser.add_argument('--coords', default=None, type=float, nargs='*')
    parser.add_argument('--shapefile_path', default='Shapefiles/Regions/northeastern_buffered.shp', type=str)
    parser.add_argument('--madis_control_ratio', default=.9, type=float)

    # Experiment Hyperparameters
    parser.add_argument('--lead_hrs', default=12, type=int)
    parser.add_argument('--back_hrs', default=48, type=int)
    parser.add_argument('--n_neighbors_m2m', default=4, type=int)
    parser.add_argument('--n_neighbors_e2m', default=4, type=int)
    parser.add_argument('--n_neighbors_h2m', default=0, type=int)

    parser.add_argument('--hrrr_analysis_only', default=0, type=int)
    parser.add_argument('--n_years', default=5, type=int)
    parser.add_argument('--interpolation_type', default=InterpolationType.none.value, type=int)

    # Model Hyperparameters
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--loss_function_type', default=LossFunctionType.CUSTOM.value, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--n_passing', default=4, type=int)

    # code setup
    parser.add_argument('--show_progress_bar', default=1, type=int)

    # file systems
    parser.add_argument('--data_path', default='/projects3/home/flag0220/LocalizedWeather/WindDataNE-US/', type=str)
    parser.add_argument('--output_saving_path', default=None, type=str)

    parser.add_argument('--use_wb', default=0, type=int)

    ## process args
    args = parser.parse_args()
    
    # 정수로 받은 0과 1을 불리언(True/False)으로 변환합니다.
    args.hrrr_analysis_only = args.hrrr_analysis_only == 1
    args.show_progress_bar = args.show_progress_bar == 1
    
    # Name 파라미터를 기반으로 새로운 저장 경로 지정
    args.output_saving_path = f'ModelOutputs/{args.name}'

    save_args = vars(args).copy()

    for k in save_args.keys():
        v = save_args[k]
        if issubclass(type(v), Enum):
            save_args[k] = v.value

    # save config to json
    outputPath = (Path(args.data_path) / args.output_saving_path / 'params.json')
    # ensure output directory exists before writing
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    with open(outputPath, 'w') as f:
        json.dump(save_args, f)

    args = PostProcessArgs(args)

    args.use_wb = args.use_wb == 1

    # run main code
    Main.Main(args).Run()