import React from 'react';
import { Header } from './Header';
import { Sidebar } from './Sidebar';

interface LayoutProps {
  children: React.ReactNode;
  onKillSwitch?: () => void;
  status?: string;
  mode?: string;
}

export const Layout: React.FC<LayoutProps> = ({ children, onKillSwitch, status, mode }) => {
  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
      <Header status={status} mode={mode} onKillSwitch={onKillSwitch} />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
};
