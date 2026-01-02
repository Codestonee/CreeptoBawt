import apiClient from './client';
import { OrdersResponse } from '../types';

export const getOrders = async (): Promise<OrdersResponse> => {
  const response = await apiClient.get<OrdersResponse>('/orders');
  return response.data;
};

export const cancelOrder = async (orderId: string): Promise<{ success: boolean; order_id: string }> => {
  const response = await apiClient.delete(`/orders/${orderId}`);
  return response.data;
};
