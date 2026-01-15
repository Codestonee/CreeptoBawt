"""
Historical data loader for backtesting.

Supports multiple sources:
1. Binance API (free, limited history)
2. CSV files (user-provided)
3. Cached parquet files (fast reload)
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from binance import AsyncClient
import asyncio

logger = logging.getLogger("Backtesting.DataLoader")


class DataLoader:
    """Load and cache historical market data."""
    
    def __init__(self, cache_dir: str = "data/backtest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client: Optional[AsyncClient] = None
    
    async def load_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1m",
        source: str = "binance",
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical OHLCV data.
        
        Args:
            symbols: List of symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            interval: Kline interval (1m, 5m, 15m, 1h, 1d)
            source: Data source ('binance', 'csv')
            force_refresh: If True, bypass cache
        
        Returns:
            Dict of {symbol: DataFrame} with OHLCV data
        """
        logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        data = {}
        
        for symbol in symbols:
            # Check cache first
            if not force_refresh:
                cached = self._load_from_cache(symbol, start_date, end_date, interval)
                if cached is not None:
                    logger.info(f"   ✓ {symbol}: Loaded from cache ({len(cached)} bars)")
                    data[symbol] = cached
                    continue
            
            # Load from source
            if source == "binance":
                df = await self._load_from_binance(symbol, start_date, end_date, interval)
            elif source == "csv":
                df = self._load_from_csv(symbol, start_date, end_date, interval)
            else:
                raise ValueError(f"Unknown source: {source}")
            
            if df is None or len(df) == 0:
                raise ValueError(f"No data loaded for {symbol}")
            
            # Cache for next time
            self._save_to_cache(df, symbol, start_date, end_date, interval)
            
            logger.info(f"   ✓ {symbol}: Loaded {len(df)} bars from {source}")
            data[symbol] = df
        
        return data
    
    async def _load_from_binance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """Load data from Binance API."""
        
        if self.client is None:
            self.client = await AsyncClient.create()
        
        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        # Binance API limits: 1000 klines per request
        all_klines = []
        current_start = start_ts
        
        while current_start < end_ts:
            try:
                klines = await self.client.futures_klines(
                    symbol=symbol.upper(),
                    interval=interval,
                    startTime=current_start,
                    endTime=end_ts,
                    limit=1000
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Update start time for next batch
                current_start = klines[-1][6] + 1  # Close time + 1ms
                
                # Rate limit protection
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching klines for {symbol}: {e}")
                break
        
        if not all_klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Set index and clean
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def _load_from_csv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        
        csv_path = self.cache_dir / f"{symbol}_{interval}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Filter date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        df = df[(df.index >= start) & (df.index <= end)]
        
        return df
    
    def _load_from_cache(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Load from parquet cache."""
        
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.parquet"
        
        if not cache_file.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ):
        """Save to parquet cache."""
        
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.parquet"
        
        try:
            df.to_parquet(cache_file, compression='snappy')
            logger.debug(f"Cached {symbol} to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    async def close(self):
        """Cleanup resources."""
        if self.client:
            await self.client.close_connection()
