import React from 'react';
import { formatCurrency } from '../../utils/formatters';
import { Balance } from '../../types';

interface BalanceCardProps {
  balances: Balance[];
  totalUsdValue: string;
}

export const BalanceCard: React.FC<BalanceCardProps> = ({ balances, totalUsdValue }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Balances</h3>
      <div className="space-y-3">
        {balances.map((balance) => (
          <div key={balance.currency} className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">{balance.currency}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Available: {balance.available} | Locked: {balance.locked}
              </p>
            </div>
            <p className="font-semibold text-gray-900 dark:text-white">
              {formatCurrency(balance.usd_value)}
            </p>
          </div>
        ))}
        <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <p className="font-semibold text-gray-900 dark:text-white">Total</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">
              {formatCurrency(totalUsdValue)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
