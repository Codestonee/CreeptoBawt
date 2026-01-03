import React from 'react';
import { Badge } from '../common/Badge';
import { formatCurrency, formatNumber, formatDate } from '../../utils/formatters';
import { Trade } from '../../types';

interface TradesTableProps {
  trades: Trade[];
}

export const TradesTable: React.FC<TradesTableProps> = ({ trades }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Trades</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-900">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Symbol
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Side
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Price
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Quantity
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Fee
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                P&L
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Executed
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {trades.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-6 py-4 text-center text-gray-500 dark:text-gray-400">
                  No recent trades
                </td>
              </tr>
            ) : (
              trades.map((trade) => {
                const pnl = parseFloat(trade.realized_pnl);
                const isProfit = pnl >= 0;
                
                return (
                  <tr key={trade.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                      {trade.symbol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <Badge variant={trade.side === 'buy' ? 'success' : 'error'}>
                        {trade.side.toUpperCase()}
                      </Badge>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900 dark:text-white">
                      {formatCurrency(trade.price)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900 dark:text-white">
                      {formatNumber(trade.quantity, 4)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900 dark:text-white">
                      {formatCurrency(trade.fee)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right">
                      <span className={isProfit ? 'text-green-600' : 'text-red-600'}>
                        {formatCurrency(trade.realized_pnl)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {formatDate(trade.executed_at)}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
