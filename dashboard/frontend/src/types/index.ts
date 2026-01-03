export interface Balance {
  currency: string;
  available: string;
  locked: string;
  total: string;
  usd_value: string;
}

export interface BalancesResponse {
  balances: Balance[];
  total_usd_value: string;
}

export interface Position {
  symbol: string;
  side: string;
  quantity: string;
  entry_price: string;
  current_price: string;
  unrealized_pnl: string;
  unrealized_pnl_pct: string;
}

export interface PositionsResponse {
  positions: Position[];
}

export interface Order {
  id: string;
  symbol: string;
  side: string;
  type: string;
  price: string;
  quantity: string;
  filled_quantity: string;
  status: string;
  created_at: string;
}

export interface OrdersResponse {
  orders: Order[];
}

export interface Trade {
  id: string;
  symbol: string;
  side: string;
  price: string;
  quantity: string;
  fee: string;
  realized_pnl: string;
  executed_at: string;
}

export interface TradesResponse {
  trades: Trade[];
}

export interface PnL {
  total_pnl: string;
  realized_pnl: string;
  unrealized_pnl: string;
  today_pnl: string;
  today_pnl_pct: string;
  fees_paid: string;
  sharpe_ratio: string;
  max_drawdown: string;
}

export interface SystemStatus {
  status: string;
  mode: string;
  uptime_seconds: number;
  kill_switch_active: boolean;
  connected_exchanges: string[];
  active_strategies: string[];
}

export interface KillSwitchRequest {
  activate: boolean;
}

export interface KillSwitchResponse {
  success: boolean;
  kill_switch_active: boolean;
  orders_cancelled: number;
}
