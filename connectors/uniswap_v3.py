"""
Uniswap V3 DEX Connector - On-chain liquidity and price data.

Features:
- Direct on-chain queries (no Graph dependency)
- sqrtPriceX96 to human-readable conversion
- Pool slot0 monitoring
- Multicall batching for efficiency
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog

from core.events import MarketEvent

log = structlog.get_logger()


# Uniswap V3 Pool ABI (minimal subset)
POOL_ABI = [
    {
        "inputs": [],
        "name": "slot0",
        "outputs": [
            {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"internalType": "int24", "name": "tick", "type": "int24"},
            {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
            {"internalType": "uint8", "name": "feeProtocol", "type": "uint8"},
            {"internalType": "bool", "name": "unlocked", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "liquidity",
        "outputs": [{"internalType": "uint128", "name": "", "type": "uint128"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "token0",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "token1",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "fee",
        "outputs": [{"internalType": "uint24", "name": "", "type": "uint24"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Multicall3 ABI
MULTICALL_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"name": "target", "type": "address"},
                    {"name": "callData", "type": "bytes"},
                ],
                "name": "calls",
                "type": "tuple[]",
            }
        ],
        "name": "aggregate",
        "outputs": [
            {"name": "blockNumber", "type": "uint256"},
            {"name": "returnData", "type": "bytes[]"},
        ],
        "stateMutability": "view",
        "type": "function",
    }
]

# Common addresses
ADDRESSES = {
    "ethereum": {
        "multicall3": "0xcA11bde05977b3631167028862bE2a173976CA11",
        "uniswap_factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "quoter_v2": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
    },
    "polygon": {
        "multicall3": "0xcA11bde05977b3631167028862bE2a173976CA11",
        "uniswap_factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    },
    "arbitrum": {
        "multicall3": "0xcA11bde05977b3631167028862bE2a173976CA11",
        "uniswap_factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    },
}

# Common Uniswap V3 pools (Ethereum mainnet)
COMMON_POOLS = {
    "WETH-USDC-500": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
    "WETH-USDC-3000": "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",
    "WETH-USDT-3000": "0x4e68Ccd3E89f51C3074ca5072bbAC773960dFa36",
    "WBTC-WETH-500": "0x4585FE77225b41b697C938B018E2Ac67Ac5a20c0",
    "WBTC-WETH-3000": "0xCBCdF9626bC03E24f779434178A73a0B4bad62eD",
}


@dataclass
class UniswapConfig:
    """Uniswap V3 connector configuration."""
    rpc_url: str = "http://localhost:8545"  # Local node preferred
    chain_id: int = 1  # Ethereum mainnet
    poll_interval: float = 1.0  # Seconds between polls
    use_multicall: bool = True  # Batch RPC calls


class UniswapV3Connector:
    """
    Uniswap V3 on-chain connector.
    
    Queries pool state directly from blockchain.
    Optimized with Multicall batching.
    """
    
    def __init__(self, config: Optional[UniswapConfig] = None) -> None:
        self.config = config or UniswapConfig()
        self._w3 = None
        self._running = False
        self._pools: Dict[str, Any] = {}
        self._poll_task: Optional[asyncio.Task] = None
        self._handlers: List = []
    
    async def connect(self) -> None:
        """Connect to Ethereum node."""
        try:
            from web3 import Web3
            from web3.middleware import geth_poa_middleware
            
            self._w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
            
            # Add PoA middleware for non-mainnet chains
            if self.config.chain_id != 1:
                self._w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not self._w3.is_connected():
                raise ConnectionError("Failed to connect to Ethereum node")
            
            block = self._w3.eth.block_number
            log.info("uniswap_connected", block=block, chain_id=self.config.chain_id)
            
        except ImportError:
            log.error("web3_not_installed")
            raise
    
    async def disconnect(self) -> None:
        """Stop polling."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
    
    async def start(self) -> None:
        """Start polling pools."""
        if self._running:
            return
        
        await self.connect()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
    
    async def stop(self) -> None:
        """Stop the connector."""
        await self.disconnect()
    
    def add_handler(self, handler) -> None:
        """Add event handler."""
        self._handlers.append(handler)
    
    async def add_pool(self, pool_address: str, symbol: str) -> None:
        """Add a pool to monitor."""
        if not self._w3:
            await self.connect()
        
        pool_contract = self._w3.eth.contract(
            address=self._w3.to_checksum_address(pool_address),
            abi=POOL_ABI,
        )
        
        # Get token info
        token0 = pool_contract.functions.token0().call()
        token1 = pool_contract.functions.token1().call()
        fee = pool_contract.functions.fee().call()
        
        self._pools[pool_address] = {
            "contract": pool_contract,
            "symbol": symbol,
            "token0": token0,
            "token1": token1,
            "fee": fee,
        }
        
        log.info(
            "pool_added",
            address=pool_address,
            symbol=symbol,
            fee=fee,
        )
    
    async def _poll_loop(self) -> None:
        """Poll pools periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.config.poll_interval)
                
                if self.config.use_multicall and len(self._pools) > 1:
                    await self._poll_with_multicall()
                else:
                    await self._poll_individually()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("poll_error", error=str(e))
    
    async def _poll_individually(self) -> None:
        """Poll each pool separately."""
        for address, pool_info in self._pools.items():
            try:
                contract = pool_info["contract"]
                slot0 = contract.functions.slot0().call()
                liquidity = contract.functions.liquidity().call()
                
                price = self.sqrt_price_x96_to_price(
                    slot0[0],
                    token0_decimals=18,  # Assuming standard ERC20
                    token1_decimals=6,   # USDC
                )
                
                event = MarketEvent(
                    event_type="trade",  # Using trade type for DEX price updates
                    exchange="uniswap_v3",
                    symbol=pool_info["symbol"],
                    timestamp_exchange=int(time.time() * 1_000_000),
                    timestamp_received=int(time.time() * 1_000_000),
                    price=price,
                    quantity=Decimal(str(liquidity)),
                    side="buy",
                    metadata={
                        "pool_address": address,
                        "tick": slot0[1],
                        "fee": pool_info["fee"],
                    },
                )
                
                for handler in self._handlers:
                    handler(event)
                    
            except Exception as e:
                log.warning("pool_poll_error", address=address, error=str(e))
    
    async def _poll_with_multicall(self) -> None:
        """Poll all pools with single multicall."""
        if not self._pools or not self._w3:
            return
        
        chain = "ethereum" if self.config.chain_id == 1 else "polygon"
        multicall_address = ADDRESSES.get(chain, {}).get("multicall3")
        
        if not multicall_address:
            await self._poll_individually()
            return
        
        # Build multicall
        multicall = self._w3.eth.contract(
            address=self._w3.to_checksum_address(multicall_address),
            abi=MULTICALL_ABI,
        )
        
        calls = []
        pool_addresses = list(self._pools.keys())
        
        for address in pool_addresses:
            pool = self._pools[address]["contract"]
            calls.append({
                "target": address,
                "callData": pool.encodeABI(fn_name="slot0"),
            })
        
        try:
            _, results = multicall.functions.aggregate(calls).call()
            
            for i, address in enumerate(pool_addresses):
                pool_info = self._pools[address]
                
                # Decode slot0 result
                slot0 = self._w3.codec.decode(
                    ["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"],
                    results[i],
                )
                
                price = self.sqrt_price_x96_to_price(
                    slot0[0],
                    token0_decimals=18,
                    token1_decimals=6,
                )
                
                event = MarketEvent(
                    event_type="trade",
                    exchange="uniswap_v3",
                    symbol=pool_info["symbol"],
                    timestamp_exchange=int(time.time() * 1_000_000),
                    timestamp_received=int(time.time() * 1_000_000),
                    price=price,
                    quantity=Decimal("0"),
                    side="buy",
                    metadata={
                        "pool_address": address,
                        "tick": slot0[1],
                    },
                )
                
                for handler in self._handlers:
                    handler(event)
                    
        except Exception as e:
            log.error("multicall_error", error=str(e))
            await self._poll_individually()
    
    @staticmethod
    def sqrt_price_x96_to_price(
        sqrt_price_x96: int,
        token0_decimals: int = 18,
        token1_decimals: int = 6,
    ) -> Decimal:
        """
        Convert Uniswap V3 sqrtPriceX96 to human-readable price.
        
        sqrtPriceX96 = sqrt(price) * 2^96
        price = (sqrtPriceX96 / 2^96)^2 * 10^(token0_decimals - token1_decimals)
        """
        Q96 = 2 ** 96
        
        # Calculate price as Decimal for precision
        sqrt_price = Decimal(sqrt_price_x96) / Decimal(Q96)
        price = sqrt_price * sqrt_price
        
        # Adjust for token decimals
        decimal_adjustment = Decimal(10) ** (token0_decimals - token1_decimals)
        price = price * decimal_adjustment
        
        return price
    
    @staticmethod
    def price_to_sqrt_price_x96(
        price: Decimal,
        token0_decimals: int = 18,
        token1_decimals: int = 6,
    ) -> int:
        """Convert human-readable price to sqrtPriceX96."""
        Q96 = 2 ** 96
        
        # Adjust for decimals
        decimal_adjustment = Decimal(10) ** (token0_decimals - token1_decimals)
        adjusted_price = price / decimal_adjustment
        
        # Calculate sqrt and scale
        import math
        sqrt_price = Decimal(math.sqrt(float(adjusted_price)))
        sqrt_price_x96 = int(sqrt_price * Q96)
        
        return sqrt_price_x96
    
    @staticmethod
    def tick_to_price(tick: int, token0_decimals: int, token1_decimals: int) -> Decimal:
        """Convert Uniswap V3 tick to price."""
        import math
        price = Decimal(1.0001 ** tick)
        decimal_adjustment = Decimal(10) ** (token0_decimals - token1_decimals)
        return price * decimal_adjustment
