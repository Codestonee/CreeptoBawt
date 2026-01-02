import React, { useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Button } from '../common/Button';

interface KillSwitchProps {
  isActive: boolean;
  onToggle: (activate: boolean) => void;
}

export const KillSwitch: React.FC<KillSwitchProps> = ({ isActive, onToggle }) => {
  const [showConfirm, setShowConfirm] = useState(false);

  const handleClick = () => {
    if (!isActive) {
      setShowConfirm(true);
    } else {
      onToggle(false);
    }
  };

  const handleConfirm = () => {
    onToggle(true);
    setShowConfirm(false);
  };

  const handleCancel = () => {
    setShowConfirm(false);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-center space-x-3 mb-4">
        <AlertTriangle className="h-6 w-6 text-red-600" />
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Emergency Kill Switch</h3>
      </div>
      
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        {isActive
          ? 'Kill switch is ACTIVE. All trading is halted.'
          : 'Immediately stop all trading activity and cancel open orders.'}
      </p>

      {!showConfirm ? (
        <Button
          variant={isActive ? 'secondary' : 'danger'}
          onClick={handleClick}
          className="w-full"
        >
          {isActive ? 'Deactivate Kill Switch' : '🛑 ACTIVATE KILL SWITCH'}
        </Button>
      ) : (
        <div className="space-y-3">
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded">
            <p className="text-sm text-red-800 dark:text-red-300 font-medium">
              Are you sure you want to activate the kill switch?
            </p>
            <p className="text-xs text-red-600 dark:text-red-400 mt-1">
              This will immediately cancel all open orders and halt trading.
            </p>
          </div>
          <div className="flex space-x-3">
            <Button variant="danger" onClick={handleConfirm} className="flex-1">
              Yes, Activate
            </Button>
            <Button variant="secondary" onClick={handleCancel} className="flex-1">
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
