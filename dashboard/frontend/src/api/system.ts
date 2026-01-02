import apiClient from './client';
import { SystemStatus, KillSwitchResponse } from '../types';

export const getSystemStatus = async (): Promise<SystemStatus> => {
  const response = await apiClient.get<SystemStatus>('/status');
  return response.data;
};

export const toggleKillSwitch = async (activate: boolean): Promise<KillSwitchResponse> => {
  const response = await apiClient.post<KillSwitchResponse>('/kill-switch', { activate });
  return response.data;
};
