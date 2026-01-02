import apiClient from './client';
import { PositionsResponse } from '../types';

export const getPositions = async (): Promise<PositionsResponse> => {
  const response = await apiClient.get<PositionsResponse>('/positions');
  return response.data;
};
