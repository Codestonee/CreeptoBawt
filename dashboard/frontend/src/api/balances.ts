import apiClient from './client';
import { BalancesResponse } from '../types';

export const getBalances = async (): Promise<BalancesResponse> => {
  const response = await apiClient.get<BalancesResponse>('/balances');
  return response.data;
};
