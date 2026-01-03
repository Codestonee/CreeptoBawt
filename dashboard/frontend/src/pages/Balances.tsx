import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { getBalances } from '../api/balances';
import { BalanceCard } from '../components/dashboard/BalanceCard';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { REFRESH_INTERVAL } from '../utils/constants';

export const Balances: React.FC = () => {
  const { data: balances, isLoading } = useQuery({
    queryKey: ['balances'],
    queryFn: getBalances,
    refetchInterval: REFRESH_INTERVAL,
  });

  if (isLoading || !balances) {
    return <LoadingSpinner />;
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Balances</h1>
      <BalanceCard
        balances={balances.balances}
        totalUsdValue={balances.total_usd_value}
      />
    </div>
  );
};
