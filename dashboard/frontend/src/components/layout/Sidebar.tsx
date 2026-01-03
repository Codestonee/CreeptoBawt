import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { BarChart3, Wallet, TrendingUp, List, ShoppingCart, Settings } from 'lucide-react';

const navItems = [
  { path: '/', label: 'Dashboard', icon: BarChart3 },
  { path: '/balances', label: 'Balances', icon: Wallet },
  { path: '/positions', label: 'Positions', icon: TrendingUp },
  { path: '/orders', label: 'Orders', icon: List },
  { path: '/trades', label: 'Trades', icon: ShoppingCart },
  { path: '/settings', label: 'Settings', icon: Settings },
];

export const Sidebar: React.FC = () => {
  const location = useLocation();
  
  return (
    <aside className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 h-full">
      <nav className="p-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-blue-50 text-blue-600 dark:bg-blue-900 dark:text-blue-300'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              <Icon className="h-5 w-5" />
              <span className="font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
};
