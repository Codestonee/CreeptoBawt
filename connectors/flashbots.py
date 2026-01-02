"""
Flashbots Integration - MEV protection for DEX trades.

Enables private transaction submission to prevent frontrunning and sandwich attacks
on Ethereum DEX trades.

Features:
- Bundle construction for atomic multi-transaction execution
- Private mempool submission (no public mempool exposure)
- Target block selection (current + 1 to current + 3)
- Miner bribe mechanism (paid only if bundle succeeds)
- Revert protection (no gas spent if bundle fails)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass
class FlashbotsTransaction:
    """Individual transaction in a Flashbots bundle."""
    to: str  # Recipient address
    data: str  # Transaction data (hex)
    value: int = 0  # ETH value in wei
    gas: int = 300000  # Gas limit
    max_fee_per_gas: int | None = None
    max_priority_fee_per_gas: int | None = None
    nonce: int | None = None


@dataclass
class FlashbotsBundle:
    """Bundle of transactions to be atomically executed."""
    transactions: list[FlashbotsTransaction]
    target_block: int
    min_timestamp: int | None = None
    max_timestamp: int | None = None
    revert_protection: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        return {
            "transactions": [
                {
                    "to": tx.to,
                    "data": tx.data,
                    "value": hex(tx.value),
                    "gas": hex(tx.gas),
                    "maxFeePerGas": hex(tx.max_fee_per_gas) if tx.max_fee_per_gas else None,
                    "maxPriorityFeePerGas": hex(tx.max_priority_fee_per_gas) if tx.max_priority_fee_per_gas else None,
                    "nonce": hex(tx.nonce) if tx.nonce is not None else None,
                }
                for tx in self.transactions
            ],
            "targetBlock": self.target_block,
            "minTimestamp": self.min_timestamp,
            "maxTimestamp": self.max_timestamp,
            "revertProtection": self.revert_protection,
        }


@dataclass
class FlashbotsBundleResult:
    """Result of bundle submission."""
    bundle_hash: str
    target_block: int
    submitted_at: float = field(default_factory=time.time)
    included: bool = False
    block_number: int | None = None
    transaction_hashes: list[str] = field(default_factory=list)


class FlashbotsConnector:
    """
    Connector for Flashbots MEV protection.

    Submits transaction bundles to Flashbots relay instead of public mempool
    to prevent MEV extraction.
    """

    def __init__(
        self,
        relay_url: str = "https://relay.flashbots.net",
        network: str = "mainnet",
    ) -> None:
        """
        Initialize Flashbots connector.

        Args:
            relay_url: Flashbots relay endpoint
            network: Network identifier ("mainnet", "goerli", etc.)
        """
        self.relay_url = relay_url
        self.network = network
        self._flashbots_provider = None
        self._web3 = None
        self._signer = None

        log.info(
            "flashbots_connector_initialized",
            relay_url=relay_url,
            network=network,
        )

    async def connect(
        self,
        web3_provider: Any,
        signer_private_key: str,
    ) -> None:
        """
        Connect to Flashbots relay.

        Args:
            web3_provider: Web3 provider instance
            signer_private_key: Private key for signing bundles
        """
        try:
            # Import flashbots library
            from eth_account import Account
            from eth_account.signers.local import LocalAccount
            from flashbots import flashbot

            self._web3 = web3_provider

            # Create signer account
            self._signer: LocalAccount = Account.from_key(signer_private_key)

            # Initialize Flashbots provider
            self._flashbots_provider = flashbot(
                self._web3,
                self._signer,
                self.relay_url,
            )

            log.info(
                "flashbots_connected",
                signer_address=self._signer.address,
            )

        except ImportError as e:
            log.error(
                "flashbots_library_not_installed",
                error=str(e),
                message="Install with: pip install flashbots",
            )
            raise

        except Exception as e:
            log.error("flashbots_connect_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Disconnect from Flashbots."""
        self._flashbots_provider = None
        self._web3 = None
        self._signer = None
        log.info("flashbots_disconnected")

    async def send_bundle(
        self,
        transactions: list[FlashbotsTransaction],
        target_block_offset: int = 1,
        miner_bribe_wei: int = 0,
        revert_protection: bool = True,
    ) -> FlashbotsBundleResult:
        """
        Send a bundle to Flashbots relay.

        Args:
            transactions: List of transactions to bundle
            target_block_offset: Blocks ahead to target (1-3 recommended)
            miner_bribe_wei: Bribe amount in wei (paid only if bundle succeeds)
            revert_protection: Whether to protect against reverts

        Returns:
            Bundle submission result

        Raises:
            RuntimeError: If not connected or submission fails
        """
        if not self._flashbots_provider or not self._web3:
            raise RuntimeError("Not connected to Flashbots")

        try:
            # Get current block
            current_block = self._web3.eth.block_number
            target_block = current_block + target_block_offset

            # Validate target block offset
            if target_block_offset < 1 or target_block_offset > 3:
                log.warning(
                    "suboptimal_target_block",
                    offset=target_block_offset,
                    message="Recommended offset: 1-3",
                )

            # Add miner bribe transaction if specified
            if miner_bribe_wei > 0:
                bribe_tx = self._create_bribe_transaction(miner_bribe_wei)
                transactions = transactions + [bribe_tx]

            # Create bundle
            bundle = FlashbotsBundle(
                transactions=transactions,
                target_block=target_block,
                revert_protection=revert_protection,
            )

            # Sign and submit bundle
            signed_txs = [self._sign_transaction(tx) for tx in bundle.transactions]

            result = await self._submit_bundle(signed_txs, target_block)

            log.info(
                "flashbots_bundle_submitted",
                bundle_hash=result.bundle_hash,
                target_block=target_block,
                tx_count=len(transactions),
            )

            return result

        except Exception as e:
            log.error("flashbots_bundle_submission_failed", error=str(e))
            raise

    async def simulate_bundle(
        self,
        transactions: list[FlashbotsTransaction],
        block_number: int | None = None,
    ) -> dict[str, Any]:
        """
        Simulate bundle execution without submitting.

        Args:
            transactions: Transactions to simulate
            block_number: Block to simulate against (current if None)

        Returns:
            Simulation results
        """
        if not self._flashbots_provider or not self._web3:
            raise RuntimeError("Not connected to Flashbots")

        try:
            if block_number is None:
                block_number = self._web3.eth.block_number

            signed_txs = [self._sign_transaction(tx) for tx in transactions]

            simulation = self._flashbots_provider.simulate(
                signed_txs,
                block_tag=block_number,
            )

            log.info(
                "bundle_simulation_complete",
                block=block_number,
                success=simulation.get("success", False),
            )

            return simulation

        except Exception as e:
            log.error("bundle_simulation_failed", error=str(e))
            raise

    async def get_bundle_stats(
        self,
        bundle_hash: str,
        target_block: int,
    ) -> dict[str, Any]:
        """
        Get statistics for a submitted bundle.

        Args:
            bundle_hash: Bundle hash from submission
            target_block: Target block number

        Returns:
            Bundle statistics
        """
        if not self._flashbots_provider:
            raise RuntimeError("Not connected to Flashbots")

        try:
            stats = self._flashbots_provider.get_bundle_stats(
                bundle_hash,
                target_block,
            )

            return stats

        except Exception as e:
            log.error("get_bundle_stats_failed", error=str(e))
            return {}

    async def get_user_stats(self) -> dict[str, Any]:
        """
        Get user statistics from Flashbots.

        Returns:
            User statistics including bundles submitted, included, etc.
        """
        if not self._flashbots_provider:
            raise RuntimeError("Not connected to Flashbots")

        try:
            stats = self._flashbots_provider.get_user_stats()
            return stats

        except Exception as e:
            log.error("get_user_stats_failed", error=str(e))
            return {}

    def _create_bribe_transaction(self, amount_wei: int) -> FlashbotsTransaction:
        """
        Create a miner bribe transaction.

        Args:
            amount_wei: Bribe amount in wei

        Returns:
            Bribe transaction
        """
        # Flashbots builder address (gets bribe if bundle succeeds)
        flashbots_builder = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

        return FlashbotsTransaction(
            to=flashbots_builder,
            data="0x",
            value=amount_wei,
            gas=21000,  # Simple transfer
        )

    def _sign_transaction(self, tx: FlashbotsTransaction) -> dict[str, Any]:
        """
        Sign a transaction.

        Args:
            tx: Transaction to sign

        Returns:
            Signed transaction dict
        """
        if not self._signer or not self._web3:
            raise RuntimeError("Signer not initialized")

        # Build transaction dict
        tx_dict = {
            "to": tx.to,
            "data": tx.data,
            "value": tx.value,
            "gas": tx.gas,
            "chainId": self._web3.eth.chain_id,
        }

        # Add gas prices if specified
        if tx.max_fee_per_gas:
            tx_dict["maxFeePerGas"] = tx.max_fee_per_gas
        if tx.max_priority_fee_per_gas:
            tx_dict["maxPriorityFeePerGas"] = tx.max_priority_fee_per_gas

        # Add nonce if specified, otherwise get current
        if tx.nonce is not None:
            tx_dict["nonce"] = tx.nonce
        else:
            tx_dict["nonce"] = self._web3.eth.get_transaction_count(self._signer.address)

        # Sign transaction
        signed_tx = self._signer.sign_transaction(tx_dict)

        return {
            "signed_transaction": signed_tx.rawTransaction.hex(),
        }

    async def _submit_bundle(
        self,
        signed_transactions: list[dict[str, Any]],
        target_block: int,
    ) -> FlashbotsBundleResult:
        """
        Submit signed bundle to Flashbots.

        Args:
            signed_transactions: List of signed transactions
            target_block: Target block number

        Returns:
            Bundle result
        """
        # Extract raw transactions
        raw_txs = [tx["signed_transaction"] for tx in signed_transactions]

        # Submit to Flashbots
        result = self._flashbots_provider.send_bundle(
            raw_txs,
            target_block_number=target_block,
        )

        # Parse result
        bundle_hash = result.bundle_hash() if hasattr(result, "bundle_hash") else "unknown"

        return FlashbotsBundleResult(
            bundle_hash=bundle_hash,
            target_block=target_block,
        )

    def create_swap_bundle(
        self,
        router_address: str,
        swap_data: str,
        value_wei: int = 0,
        gas_limit: int = 300000,
    ) -> list[FlashbotsTransaction]:
        """
        Create a bundle for a DEX swap.

        Args:
            router_address: DEX router contract address
            swap_data: Encoded swap function call
            value_wei: ETH value for swap (if needed)
            gas_limit: Gas limit for swap transaction

        Returns:
            List containing swap transaction
        """
        swap_tx = FlashbotsTransaction(
            to=router_address,
            data=swap_data,
            value=value_wei,
            gas=gas_limit,
        )

        return [swap_tx]

    def calculate_optimal_bribe(
        self,
        gas_price_gwei: Decimal,
        estimated_gas: int,
        mev_profit_wei: int,
        bribe_percentage: Decimal = Decimal("0.9"),
    ) -> int:
        """
        Calculate optimal miner bribe.

        Args:
            gas_price_gwei: Current gas price in gwei
            estimated_gas: Estimated gas for bundle
            mev_profit_wei: Expected MEV profit in wei
            bribe_percentage: Percentage of MEV profit to offer (default 90%)

        Returns:
            Recommended bribe in wei
        """
        # Calculate gas cost
        gas_price_wei = int(gas_price_gwei * Decimal("1e9"))
        gas_cost_wei = gas_price_wei * estimated_gas

        # Calculate bribe as percentage of MEV profit minus gas cost
        net_profit = mev_profit_wei - gas_cost_wei
        if net_profit <= 0:
            return 0

        bribe_wei = int(net_profit * bribe_percentage)

        log.info(
            "bribe_calculated",
            gas_cost_wei=gas_cost_wei,
            mev_profit_wei=mev_profit_wei,
            bribe_wei=bribe_wei,
            bribe_percentage=str(bribe_percentage),
        )

        return bribe_wei


# Convenience functions

def create_flashbots_connector(
    relay_url: str = "https://relay.flashbots.net",
    network: str = "mainnet",
) -> FlashbotsConnector:
    """Create a Flashbots connector instance."""
    return FlashbotsConnector(relay_url=relay_url, network=network)
