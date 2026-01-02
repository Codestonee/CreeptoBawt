import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { getTrades } from '../api/trades';
import { TradesTable } from '../components/dashboard/TradesTable';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { REFRESH_INTERVAL } from '../utils/constants';

export const Trades: React.FC = () => {
  const { data: trades, isLoading } = useQuery({
    queryKey: ['trades'],
    queryFn: getTrades,
    refetchInterval: REFRESH_INTERVAL,
  });

  if (isLoading || !trades) {
    return <LoadingSpinner />;
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Trades</h1>
      <TradesTable trades={trades.trades} />
    </div>
  );
};
