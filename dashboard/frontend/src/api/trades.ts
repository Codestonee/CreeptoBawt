import apiClient from './client';
import { TradesResponse } from '../types';

export const getTrades = async (): Promise<TradesResponse> => {
  const response = await apiClient.get<TradesResponse>('/trades');
  return response.data;
};
