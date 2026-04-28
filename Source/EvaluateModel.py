# Author: Qidong Yang & Jonathan Giezendanner

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from Settings.Settings import ModelType, LossFunctionType, EnvVariables


class EvaluateModel:

    def __init__(self, model, data_loaders, madis_norm_dict, external_norm_dict, device, lead_hrs, madis_vars_i,
                 madis_vars_o, madis_vars, external_vars, loss_function=None, loss_function_report=None,
                 save_metrics_types=None, save_metrics_functions=None, per_variable_metrics_types=None,
                 per_variable_metrics=None, model_type=ModelType.GNN, show_progress_bar=False, optimizer=None,
                 sensor_dropout=False, sensor_dropout_ratio=0.1, past_only=False):
        self.model = model
        self.data_loaders = data_loaders
        self.madis_norm_dict = madis_norm_dict
        self.external_norm_dict = external_norm_dict
        self.device = device
        self.lead_hrs = lead_hrs
        self.madis_vars_i = madis_vars_i
        self.madis_vars_o = madis_vars_o
        self.madis_vars = madis_vars
        self.external_vars = external_vars
        self.loss_function = loss_function
        self.loss_function_report = loss_function_report
        self.save_metrics_types = save_metrics_types
        self.save_metrics_functions = save_metrics_functions
        self.per_variable_metrics_types = per_variable_metrics_types
        self.per_variable_metrics = per_variable_metrics
        self.model_type = model_type
        self.show_progress_bar = show_progress_bar
        self.optimizer = optimizer
        self.sensor_dropout = sensor_dropout
        self.sensor_dropout_ratio = sensor_dropout_ratio
        self.past_only = past_only

    def call_evaluate(self, station_type='train', save=False):
        data_loader = self.data_loaders[station_type]
        self.is_train = station_type == 'train'

        per_variable_loss = dict()
        nb_obs = dict()

        for madis_var in self.madis_vars_o:
            nb_obs[madis_var] = 0
            per_variable_loss[madis_var] = dict()
            for per_variable_metrics_type in self.per_variable_metrics_types:
                per_variable_loss[madis_var][per_variable_metrics_type] = 0

        loss_report = 0
        n_report = 0

        if save:
            Pred_list = []
            Target_list = []
            atten_n = None
            atten_sum = None
            atten_max = None
            time_list = []
            Pred_seen_list = []
            Target_seen_list = []
            Pred_ghost_list = []
            Target_ghost_list = []

        save_metrics_dict = dict()
        save_metrics_dict_seen = dict()
        save_metrics_dict_ghost = dict()
        save_metrics_norm = dict()
        save_metrics_norm_seen = dict()
        save_metrics_norm_ghost = dict()
        save_metric_count = dict()
        save_metric_count_seen = dict()
        save_metric_count_ghost = dict()
        for save_metric in self.save_metrics_types:
            save_metrics_dict[save_metric] = 0
            save_metrics_dict_seen[save_metric] = 0
            save_metrics_dict_ghost[save_metric] = 0
            save_metrics_norm[save_metric] = np.nan
            save_metrics_norm_seen[save_metric] = np.nan
            save_metrics_norm_ghost[save_metric] = np.nan
            save_metric_count[save_metric] = 0
            save_metric_count_seen[save_metric] = 0
            save_metric_count_ghost[save_metric] = 0

        per_variable_loss_seen = dict()
        per_variable_loss_ghost = dict()
        nb_obs_seen = dict()
        nb_obs_ghost = dict()
        for madis_var in self.madis_vars_o:
            nb_obs_seen[madis_var] = 0
            nb_obs_ghost[madis_var] = 0
            per_variable_loss_seen[madis_var] = dict()
            per_variable_loss_ghost[madis_var] = dict()
            for per_variable_metrics_type in self.per_variable_metrics_types:
                per_variable_loss_seen[madis_var][per_variable_metrics_type] = 0
                per_variable_loss_ghost[madis_var][per_variable_metrics_type] = 0

        if self.is_train:
            self.model.train()
        else:
            self.model.eval()

        print(f'runing {station_type}', flush=True)

        with torch.set_grad_enabled(self.is_train):
            loopItems = tqdm(data_loader) if self.show_progress_bar else data_loader
            for sample_k, sample in enumerate(loopItems):
                self.sample_k = sample_k
                self.sample = sample

                edge_index_m2m, madis_lat, madis_lon, madis_x, y = self.ProcessSampleMadis(self.sample)

                if self.external_norm_dict is not None:
                    edge_index_ex2m, external_lat, external_lon, external_x = self.GetERA5Sample()
                    edge_index_ex2m = edge_index_ex2m.to(self.device)
                    external_lat = external_lat.to(self.device)
                    external_lon = external_lon.to(self.device)
                    external_x = external_x.to(self.device)
                else:
                    external_lon = None
                    external_lat = None
                    external_x = None
                    edge_index_ex2m = None

                if self.is_train:
                    self.optimizer.zero_grad()

                out, alphas = self.RunModel(edge_index_ex2m, edge_index_m2m, external_lat, external_lon, external_x,
                                            madis_lat, madis_lon, madis_x)

                is_real = self.GetIsReal()
                seen_station_mask, ghost_station_mask = self.GetStationMasks()
                train_station_mask = seen_station_mask
                if self.is_train and self.sensor_dropout:
                    sensor_dropout_mask = self.GetSensorDropoutMask(seen_station_mask)
                    train_station_mask = seen_station_mask & (~sensor_dropout_mask)

                loss_mask = is_real & train_station_mask
                eval_seen_mask = is_real & seen_station_mask
                eval_ghost_mask = is_real & ghost_station_mask

                if self.is_train:
                    ls = self.loss_function(out, y, loss_mask)
                    ls.backward()
                    if type(self.model) is nn.DataParallel:
                        torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                loss_report += self.loss_function_report(out.detach(), y.detach()).cpu().numpy()
                n_report += out[..., 0].numel()

                for save_metric in self.save_metrics_types:
                    save_metrics_dict[save_metric] += self.save_metrics_functions[save_metric](
                        out.detach(), y.detach(), is_real).cpu().numpy()
                    save_metrics_dict_seen[save_metric] += self.save_metrics_functions[save_metric](
                        out.detach(), y.detach(), eval_seen_mask).cpu().numpy()
                    save_metrics_dict_ghost[save_metric] += self.save_metrics_functions[save_metric](
                        out.detach(), y.detach(), eval_ghost_mask).cpu().numpy()
                    save_metric_count[save_metric] += self.GetMetricDenominator(save_metric, is_real)
                    save_metric_count_seen[save_metric] += self.GetMetricDenominator(save_metric, eval_seen_mask)
                    save_metric_count_ghost[save_metric] += self.GetMetricDenominator(save_metric, eval_ghost_mask)

                for k, madis_var in enumerate(self.madis_vars_o):
                    y[..., k] = self.madis_norm_dict[madis_var].decode(y[..., k])
                    out[..., k] = self.madis_norm_dict[madis_var].decode(out[..., k])

                if save:
                    out_save, y_save = self.GetPredictionAndTargetForSaving(out, y)

                    if alphas is not None:
                        alphas = alphas.detach().cpu().numpy()
                        if atten_sum is None:
                            atten_sum = alphas.sum(axis=0)
                            atten_max = alphas.max(axis=0, keepdims=True)
                            atten_n = alphas.shape[0]
                        else:
                            atten_sum += alphas.sum(axis=0)
                            atten_max = np.max(np.concatenate([atten_max, alphas], axis=0), axis=0, keepdims=True)
                            atten_n += alphas.shape[0]

                    Pred_list.append(out_save)
                    Target_list.append(y_save)
                    time_list.append(self.sample['time'].numpy())

                    seen_selector = seen_station_mask.squeeze(-1)
                    ghost_selector = ghost_station_mask.squeeze(-1)
                    if seen_selector.any():
                        Pred_seen_list.append(out[seen_selector].detach().cpu().numpy())
                        Target_seen_list.append(y[seen_selector].detach().cpu().numpy())
                    if ghost_selector.any():
                        Pred_ghost_list.append(out[ghost_selector].detach().cpu().numpy())
                        Target_ghost_list.append(y[ghost_selector].detach().cpu().numpy())

                y = y.detach()
                out = out.detach()
                metric_station_mask = train_station_mask.squeeze(-1) if self.is_train else seen_station_mask.squeeze(-1)
                for k, madis_var in enumerate(self.madis_vars_o):
                    out_k, y_k = self.GetPerVariableTargetAndPredictionFromMatrixForMonitoring(k, out, y)
                    out_k = out_k[metric_station_mask]
                    y_k = y_k[metric_station_mask]
                    if torch.numel(out_k) == 0:
                        continue
                    nb_obs[madis_var] += torch.numel(out_k)
                    for per_variable_metrics_type in self.per_variable_metrics_types:
                        per_variable_loss[madis_var][per_variable_metrics_type] += np.sum(
                            self.per_variable_metrics[per_variable_metrics_type](out_k, y_k).cpu().numpy())

                    seen_k_mask = eval_seen_mask[..., k]
                    if torch.any(seen_k_mask):
                        out_k_seen = out[..., k][seen_k_mask]
                        y_k_seen = y[..., k][seen_k_mask]
                        nb_obs_seen[madis_var] += torch.numel(out_k_seen)
                        for per_variable_metrics_type in self.per_variable_metrics_types:
                            per_variable_loss_seen[madis_var][per_variable_metrics_type] += np.sum(
                                self.per_variable_metrics[per_variable_metrics_type](out_k_seen,
                                                                                      y_k_seen).cpu().numpy())

                    ghost_k_mask = eval_ghost_mask[..., k]
                    if torch.any(ghost_k_mask):
                        out_k_ghost = out[..., k][ghost_k_mask]
                        y_k_ghost = y[..., k][ghost_k_mask]
                        nb_obs_ghost[madis_var] += torch.numel(out_k_ghost)
                        for per_variable_metrics_type in self.per_variable_metrics_types:
                            per_variable_loss_ghost[madis_var][per_variable_metrics_type] += np.sum(
                                self.per_variable_metrics[per_variable_metrics_type](out_k_ghost,
                                                                                      y_k_ghost).cpu().numpy())

        for madis_var in self.madis_vars_o:
            for per_variable_metrics_type in self.per_variable_metrics_types:
                if nb_obs[madis_var] > 0:
                    per_variable_loss[madis_var][per_variable_metrics_type] /= nb_obs[madis_var]
                else:
                    per_variable_loss[madis_var][per_variable_metrics_type] = np.nan

                if nb_obs_seen[madis_var] > 0:
                    per_variable_loss_seen[madis_var][per_variable_metrics_type] /= nb_obs_seen[madis_var]
                else:
                    per_variable_loss_seen[madis_var][per_variable_metrics_type] = np.nan

                if nb_obs_ghost[madis_var] > 0:
                    per_variable_loss_ghost[madis_var][per_variable_metrics_type] /= nb_obs_ghost[madis_var]
                else:
                    per_variable_loss_ghost[madis_var][per_variable_metrics_type] = np.nan

        loss_report /= n_report

        if save:
            self.Preds = np.concatenate(Pred_list, axis=0)
            self.Targets = np.concatenate(Target_list, axis=0)
            self.Times = np.concatenate(time_list, axis=0)
            self.Preds_Seen = np.concatenate(Pred_seen_list, axis=0) if len(Pred_seen_list) > 0 else None
            self.Targets_Seen = np.concatenate(Target_seen_list, axis=0) if len(Target_seen_list) > 0 else None
            self.Preds_Ghost = np.concatenate(Pred_ghost_list, axis=0) if len(Pred_ghost_list) > 0 else None
            self.Targets_Ghost = np.concatenate(Target_ghost_list, axis=0) if len(Target_ghost_list) > 0 else None

            if atten_sum is not None:
                self.Attns_Mean = atten_sum / atten_n
                self.Attns_Max = atten_max.squeeze()
            else:
                self.Attns_Mean = None
                self.Attns_Max = None

        self.save_metric_dict = save_metrics_dict
        self.save_metric_dict_seen = save_metrics_dict_seen
        self.save_metric_dict_ghost = save_metrics_dict_ghost
        self.save_metric_dict_norm = save_metrics_norm
        self.save_metric_dict_norm_seen = save_metrics_norm_seen
        self.save_metric_dict_norm_ghost = save_metrics_norm_ghost
        self.per_variable_loss_seen = per_variable_loss_seen
        self.per_variable_loss_ghost = per_variable_loss_ghost

        for save_metric in self.save_metrics_types:
            if save_metric_count[save_metric] > 0:
                self.save_metric_dict_norm[save_metric] = save_metrics_dict[save_metric] / save_metric_count[save_metric]
            if save_metric_count_seen[save_metric] > 0:
                self.save_metric_dict_norm_seen[save_metric] = (
                    save_metrics_dict_seen[save_metric] / save_metric_count_seen[save_metric])
            if save_metric_count_ghost[save_metric] > 0:
                self.save_metric_dict_norm_ghost[save_metric] = (
                    save_metrics_dict_ghost[save_metric] / save_metric_count_ghost[save_metric])

        return loss_report, per_variable_loss

    def GetERA5Sample(self):
        external_lon = self.sample['external_lon']
        external_lat = self.sample['external_lat']
        edge_index_ex2m = self.sample['ex2m_edge_index']
        external_vals_dict = dict()
        for external_var in self.external_vars:
            external_vals_dict[external_var] = self.sample['ext_' + external_var.name]
            external_vals_dict[external_var] = self.external_norm_dict[external_var].encode(
                external_vals_dict[external_var]).unsqueeze(3)
        external_x = torch.cat(list(external_vals_dict.values()), dim=-1)
        return edge_index_ex2m, external_lat, external_lon, external_x

    def GetIsReal(self):
        is_real_list = []
        for var in self.madis_vars_o:
            target_is_real_key = 'target_' + var.name + '_is_real'
            if self.past_only and (target_is_real_key in self.sample):
                is_real = self.sample.get(target_is_real_key).to(self.device)
                if is_real.dim() == 1:
                    is_real = is_real.unsqueeze(0)
                is_real = (is_real == 1).unsqueeze(-1)
            else:
                is_real = self.sample.get(var.name + '_is_real').to(self.device)
                # Keep only the forecast target timestep to align with y/out shape: (batch, station, variable)
                if is_real.dim() == 2:
                    is_real = is_real.unsqueeze(0)
                is_real = (is_real[:, :, -1] == 1).unsqueeze(-1)
            is_real_list.append(is_real)
        return torch.cat(is_real_list, dim=-1)

    def GetStationMasks(self):
        seen_station_mask = self.sample['seen_station_mask'].to(self.device)
        ghost_station_mask = self.sample['ghost_station_mask'].to(self.device)

        reference = self.sample.get(self.madis_vars_o[0].name + '_is_real')
        if reference.dim() == 2:
            batch_size = 1
        else:
            batch_size = reference.shape[0]

        if seen_station_mask.dim() == 1:
            seen_station_mask = seen_station_mask.unsqueeze(0)
        if ghost_station_mask.dim() == 1:
            ghost_station_mask = ghost_station_mask.unsqueeze(0)

        # Defensive alignment for cases where collate returns (station, batch)
        if seen_station_mask.dim() == 2 and seen_station_mask.shape[0] != batch_size and seen_station_mask.shape[1] == batch_size:
            seen_station_mask = seen_station_mask.transpose(0, 1)
        if ghost_station_mask.dim() == 2 and ghost_station_mask.shape[0] != batch_size and ghost_station_mask.shape[1] == batch_size:
            ghost_station_mask = ghost_station_mask.transpose(0, 1)

        seen_station_mask = (seen_station_mask == 1).unsqueeze(-1)
        ghost_station_mask = (ghost_station_mask == 1).unsqueeze(-1)
        return seen_station_mask, ghost_station_mask

    def GetSensorDropoutMask(self, seen_station_mask):
        random_keep = torch.rand_like(seen_station_mask.float())
        dropped = random_keep < self.sensor_dropout_ratio
        return dropped & seen_station_mask

    def GetMetricDenominator(self, save_metric_type, logical_mask):
        valid_count = int(torch.sum(logical_mask).item())
        if valid_count == 0:
            return 0

        if save_metric_type == LossFunctionType.CUSTOM:
            u_index = np.array(self.madis_vars_o) == EnvVariables.u
            v_index = np.array(self.madis_vars_o) == EnvVariables.v
            other_index = ~(u_index | v_index)
            n_other = int(np.sum(other_index))
            return valid_count * (1 + n_other)

        return valid_count

    def ProcessSampleMadis(self, sample):
        madis_lon = sample['madis_lon'].to(self.device)
        madis_lat = sample['madis_lat'].to(self.device)
        edge_index_m2m = sample['k_edge_index'].to(self.device)
        madis_vals_dict = dict()
        for madis_var in self.madis_vars:
            madis_vals_dict[madis_var] = sample[madis_var]
            madis_vals_dict[madis_var] = self.madis_norm_dict[madis_var].encode(madis_vals_dict[madis_var]).unsqueeze(3)
        y = self.GetTarget(madis_vals_dict, sample).to(self.device)
        madis_x = self.GetMadisInputs(madis_vals_dict).to(self.device)
        return edge_index_m2m, madis_lat, madis_lon, madis_x, y

    def GetPredictionAndTargetForSaving(self, out, y):
        out_save = out.detach().cpu().numpy()
        y_save = y.detach().cpu().numpy()
        return out_save, y_save

    def GetPerVariableTargetAndPredictionFromMatrixForMonitoring(self, k, out, y):
        out_k = out[..., k]
        y_k = y[..., k]
        return out_k, y_k

    def GetMadisInputs(self, madis_vals_dict):
        madis_x = torch.cat(list(map(madis_vals_dict.get, self.madis_vars_i)), dim=-1)
        if self.past_only:
            return madis_x
        madis_matrix_len = madis_x.shape[2]
        madis_x = madis_x[:, :, :madis_matrix_len - self.lead_hrs, :]
        return madis_x

    def GetTarget(self, madis_vals_dict, sample):
        target_key = 'target_' + self.madis_vars_o[0].name
        if self.past_only and (target_key in sample):
            target_list = []
            for var in self.madis_vars_o:
                target_val = sample['target_' + var.name]
                if target_val.dim() == 1:
                    target_val = target_val.unsqueeze(0)
                target_list.append(self.madis_norm_dict[var].encode(target_val).unsqueeze(-1))
            return torch.cat(target_list, dim=-1)
        y = torch.cat(list(map(lambda var: madis_vals_dict.get(var)[:, :, -1, :], self.madis_vars_o)), dim=-1)
        return y

    def RunModel(self, edge_index_ex2m, edge_index_m2m, external_lat, external_lon, external_x, madis_lat, madis_lon,
                 madis_x):
        alphas = None
        if self.model_type == ModelType.ViT:
            out, alphas = self.model(madis_x, external_x, return_attn=False)
        elif self.model_type == ModelType.GNN:
            out = self.model(madis_x,
                             madis_lon,
                             madis_lat,
                             edge_index_m2m,
                             external_lon,
                             external_lat,
                             external_x,
                             edge_index_ex2m)
        else:
            raise NotImplementedError
        return out, alphas
