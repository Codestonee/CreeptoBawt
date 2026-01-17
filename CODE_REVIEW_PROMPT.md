# Code Review Request: Market Making Bot Order Sizing Issues

## Project Overview

This is a cryptocurrency market-making bot using the Avellaneda-Stoikov strategy, trading on Binance Spot API with ~$50 balance across 3 pairs (LTCUSDC, XRPUSDC, DOGEUSDC).

## Current Problem

Orders are being rejected because their USD value falls just below minimum thresholds. Despite multiple fixes, order sizes consistently come in slightly under the required $5 minimum.

### Example of the issue:

```
[ENG] BUY 36.4 DOGEUSDC @ $0.14 -> binance
[BIN] ❌ 🛑 RISK REJECT: Value $4.93 < Min $5.0
```

The strategy calculates 36.4 DOGE, but at actual price ~$0.1355, that's only $4.93.

## What I need reviewed:

### 1. `strategies/avellaneda_stoikov.py` - `_calculate_sizes()` method (lines 1062-1190)

- The method calculates minimum quantity to meet MIN_NOTIONAL_USD
- Uses ceiling-based rounding to ensure sizes round UP
- Still produces values below minimum

### 2. Order flow from strategy → risk gatekeeper

- Strategy outputs order size
- Risk gatekeeper in `execution/risk_gatekeeper.py` validates against MIN_NOTIONAL
- There may be a price discrepancy between calculation time and validation time

### 3. Key settings in `config/settings.py`:

- MIN_NOTIONAL_USD = 5.0
- RISK_MIN_NOTIONAL_USD = 5.0
- MAX_POSITION_USD = 50.0

## Specific questions:

1. Why does the min_qty calculation in the strategy produce values that fail the risk gatekeeper check?
2. Is there a race condition or price staleness issue between quote calculation and order validation?
3. Should the strategy use a higher MIN_NOTIONAL (e.g., $6) to ensure buffer, or is there a logic bug?

## Key files to review:

- `strategies/avellaneda_stoikov.py` - Main strategy with order sizing
- `execution/risk_gatekeeper.py` - Order validation
- `execution/binance_executor.py` - Order execution
- `config/settings.py` - Configuration values
