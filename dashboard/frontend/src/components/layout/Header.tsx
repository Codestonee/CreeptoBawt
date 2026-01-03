import React from 'react';
import { Activity } from 'lucide-react';

interface HeaderProps {
  status?: string;
  mode?: string;
  onKillSwitch?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ status = 'running', mode = 'paper', onKillSwitch }) => {
  return (
    <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
      <div className="px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Activity className="h-8 w-8 text-blue-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            CreeptoBawt Dashboard
          </h1>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Mode:</span>
            <span className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm font-medium">
              {mode.toUpperCase()}
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className={`h-2 w-2 rounded-full ${status === 'running' ? 'bg-green-500' : 'bg-gray-500'}`} />
            <span className="text-sm text-gray-600 dark:text-gray-400">{status}</span>
          </div>
          
          {onKillSwitch && (
            <button
              onClick={onKillSwitch}
              className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors text-sm font-medium"
            >
              🛑 STOP
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
