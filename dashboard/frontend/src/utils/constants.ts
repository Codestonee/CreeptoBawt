export const REFRESH_INTERVAL = 5000; // 5 seconds

export const STATUS_COLORS = {
  healthy: 'text-green-500',
  degraded: 'text-yellow-500',
  unhealthy: 'text-red-500',
  running: 'text-green-500',
  stopped: 'text-gray-500',
};

export const SIDE_COLORS = {
  buy: 'text-green-600',
  sell: 'text-red-600',
  long: 'text-green-600',
  short: 'text-red-600',
};

export const STATUS_BADGES = {
  open: 'bg-blue-100 text-blue-800',
  filled: 'bg-green-100 text-green-800',
  cancelled: 'bg-gray-100 text-gray-800',
  rejected: 'bg-red-100 text-red-800',
  partial: 'bg-yellow-100 text-yellow-800',
};
