import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { getPositions } from '../api/positions';
import { PositionsTable } from '../components/dashboard/PositionsTable';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { REFRESH_INTERVAL } from '../utils/constants';

export const Positions: React.FC = () => {
  const { data: positions, isLoading } = useQuery({
    queryKey: ['positions'],
    queryFn: getPositions,
    refetchInterval: REFRESH_INTERVAL,
  });

  if (isLoading || !positions) {
    return <LoadingSpinner />;
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Positions</h1>
      <PositionsTable positions={positions.positions} />
    </div>
  );
};
