import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { TrendingUp, ShoppingCart, DollarSign } from 'lucide-react';
import { StatsCard } from '../components/dashboard/StatsCard';
import { BalanceCard } from '../components/dashboard/BalanceCard';
import { PositionsTable } from '../components/dashboard/PositionsTable';
import { OrdersTable } from '../components/dashboard/OrdersTable';
import { TradesTable } from '../components/dashboard/TradesTable';
import { KillSwitch } from '../components/dashboard/KillSwitch';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { getBalances } from '../api/balances';
import { getPositions } from '../api/positions';
import { getOrders, cancelOrder } from '../api/orders';
import { getTrades } from '../api/trades';
import { getPnL } from '../api/pnl';
import { toggleKillSwitch, getSystemStatus } from '../api/system';
import { formatCurrency, formatPercent } from '../utils/formatters';
import { REFRESH_INTERVAL } from '../utils/constants';

export const Dashboard: React.FC = () => {
  const queryClient = useQueryClient();

  const { data: balances } = useQuery({
    queryKey: ['balances'],
    queryFn: getBalances,
    refetchInterval: REFRESH_INTERVAL,
  });

  const { data: positions } = useQuery({
    queryKey: ['positions'],
    queryFn: getPositions,
    refetchInterval: REFRESH_INTERVAL,
  });

  const { data: orders } = useQuery({
    queryKey: ['orders'],
    queryFn: getOrders,
    refetchInterval: REFRESH_INTERVAL,
  });

  const { data: trades } = useQuery({
    queryKey: ['trades'],
    queryFn: getTrades,
    refetchInterval: REFRESH_INTERVAL,
  });

  const { data: pnl } = useQuery({
    queryKey: ['pnl'],
    queryFn: getPnL,
    refetchInterval: REFRESH_INTERVAL,
  });

  const { data: status } = useQuery({
    queryKey: ['status'],
    queryFn: getSystemStatus,
    refetchInterval: REFRESH_INTERVAL,
  });

  const killSwitchMutation = useMutation({
    mutationFn: toggleKillSwitch,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['status'] });
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });

  const cancelOrderMutation = useMutation({
    mutationFn: cancelOrder,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });

  const handleKillSwitch = (activate: boolean) => {
    killSwitchMutation.mutate(activate);
  };

  const handleCancelOrder = (orderId: string) => {
    if (window.confirm('Are you sure you want to cancel this order?')) {
      cancelOrderMutation.mutate(orderId);
    }
  };

  if (!balances || !positions || !orders || !trades || !pnl || !status) {
    return <LoadingSpinner />;
  }

  const totalPnl = parseFloat(pnl.total_pnl);
  const todayPnl = parseFloat(pnl.today_pnl);

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total P&L"
          value={formatCurrency(pnl.total_pnl)}
          change={formatPercent(pnl.today_pnl_pct)}
          positive={totalPnl >= 0}
          icon={<DollarSign className="h-8 w-8" />}
        />
        <StatsCard
          title="Today P&L"
          value={formatCurrency(pnl.today_pnl)}
          positive={todayPnl >= 0}
          icon={<TrendingUp className="h-8 w-8" />}
        />
        <StatsCard
          title="Open Positions"
          value={positions.positions.length.toString()}
          icon={<TrendingUp className="h-8 w-8" />}
        />
        <StatsCard
          title="Open Orders"
          value={orders.orders.length.toString()}
          icon={<ShoppingCart className="h-8 w-8" />}
        />
      </div>

      {/* Balance and Kill Switch */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <BalanceCard
            balances={balances.balances}
            totalUsdValue={balances.total_usd_value}
          />
        </div>
        <KillSwitch
          isActive={status.kill_switch_active}
          onToggle={handleKillSwitch}
        />
      </div>

      {/* Positions */}
      <PositionsTable positions={positions.positions} />

      {/* Orders */}
      <OrdersTable orders={orders.orders} onCancelOrder={handleCancelOrder} />

      {/* Recent Trades */}
      <TradesTable trades={trades.trades.slice(0, 5)} />
    </div>
  );
};
