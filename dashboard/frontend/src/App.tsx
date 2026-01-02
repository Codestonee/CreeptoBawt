import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/layout/Layout';
import { Dashboard } from './pages/Dashboard';
import { Positions } from './pages/Positions';
import { Orders } from './pages/Orders';
import { Trades } from './pages/Trades';
import { Settings } from './pages/Settings';
import { useQuery } from '@tanstack/react-query';
import { getSystemStatus } from './api/system';
import { toggleKillSwitch } from './api/system';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

const AppContent: React.FC = () => {
  const { data: status } = useQuery({
    queryKey: ['status'],
    queryFn: getSystemStatus,
    refetchInterval: 5000,
  });

  const handleKillSwitch = async () => {
    if (window.confirm('Are you sure you want to activate the kill switch?')) {
      await toggleKillSwitch(true);
      queryClient.invalidateQueries({ queryKey: ['status'] });
    }
  };

  return (
    <Layout
      status={status?.status}
      mode={status?.mode}
      onKillSwitch={handleKillSwitch}
    >
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/balances" element={<Dashboard />} />
        <Route path="/positions" element={<Positions />} />
        <Route path="/orders" element={<Orders />} />
        <Route path="/trades" element={<Trades />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  );
};

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
