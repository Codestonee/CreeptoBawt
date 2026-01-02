import apiClient from './client';
import { PnL } from '../types';

export const getPnL = async (): Promise<PnL> => {
  const response = await apiClient.get<PnL>('/pnl');
  return response.data;
};
