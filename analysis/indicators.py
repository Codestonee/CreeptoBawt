import numpy as np

def calculate_tr(high, low, close):
    """True Range för ATR och ADX."""
    h_l = high - low
    h_pc = np.abs(high - np.roll(close, 1))
    l_pc = np.abs(low - np.roll(close, 1))
    
    # Första värdet blir felaktigt pga roll, sätt till h-l
    h_pc[0] = h_l[0]
    l_pc[0] = h_l[0]
    
    return np.maximum(h_l, np.maximum(h_pc, l_pc))

def calculate_atr(high, low, close, period=14):
    """Average True Range (Volatilitet)."""
    tr = calculate_tr(high, low, close)
    return rma(tr, period)

def rma(x, n):
    """Rolling Moving Average (Wilder's Smoothing)."""
    a = np.full_like(x, np.nan)
    if len(x) < n:
        return a
    a[n-1] = x[:n].mean()
    for i in range(n, len(x)):
        a[i] = (a[i-1] * (n - 1) + x[i]) / n
    return a

def calculate_adx(high, low, close, period=14):
    """Average Directional Index (Trendstyrka)."""
    if len(close) < period * 2:
        return np.nan

    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    up_move[0] = 0
    down_move[0] = 0

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr = calculate_tr(high, low, close)
    
    # Smooth
    atr = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / (atr + 1e-9) # Epsilon för säkerhet
    minus_di = 100 * rma(minus_dm, period) / (atr + 1e-9)
    
    # Förhindra division med noll
    sum_di = plus_di + minus_di
    dx = 100 * np.abs(plus_di - minus_di) / (sum_di + 1e-9)
    adx = rma(dx, period)
    
    return adx[-1] # Returnera senaste värdet