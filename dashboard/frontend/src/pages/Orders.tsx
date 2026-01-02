import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getOrders, cancelOrder } from '../api/orders';
import { OrdersTable } from '../components/dashboard/OrdersTable';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { REFRESH_INTERVAL } from '../utils/constants';

export const Orders: React.FC = () => {
  const queryClient = useQueryClient();

  const { data: orders, isLoading } = useQuery({
    queryKey: ['orders'],
    queryFn: getOrders,
    refetchInterval: REFRESH_INTERVAL,
  });

  const cancelOrderMutation = useMutation({
    mutationFn: cancelOrder,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });

  const handleCancelOrder = (orderId: string) => {
    if (window.confirm('Are you sure you want to cancel this order?')) {
      cancelOrderMutation.mutate(orderId);
    }
  };

  if (isLoading || !orders) {
    return <LoadingSpinner />;
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Orders</h1>
      <OrdersTable orders={orders.orders} onCancelOrder={handleCancelOrder} />
    </div>
  );
};
