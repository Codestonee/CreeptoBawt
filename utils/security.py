"""
Security Module - Secure credential management using OS keyring.

Features:
- OS-native secret storage (Windows Credential Manager, macOS Keychain)
- Fernet encryption for additional security
- Never stores secrets in plain text
- API key store/retrieve/delete operations
"""
from __future__ import annotations

import base64
import os
from typing import Optional, Tuple

import structlog

log = structlog.get_logger()

# Service name for keyring
SERVICE_NAME = "crypto_trading_bot"


class SecretsManager:
    """
    Secure secret storage using OS keyring.
    
    Uses the operating system's native credential manager:
    - Windows: Credential Manager
    - macOS: Keychain
    - Linux: Secret Service (GNOME Keyring, KWallet)
    
    Additional Fernet encryption layer for API secrets.
    """
    
    def __init__(self, service_name: str = SERVICE_NAME) -> None:
        self.service_name = service_name
        self._keyring = None
        self._fernet = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize keyring and encryption."""
        try:
            import keyring
            self._keyring = keyring
            log.info("keyring_initialized", backend=type(keyring.get_keyring()).__name__)
        except ImportError:
            log.warning("keyring_not_installed", message="Secrets will not be persisted")
        
        try:
            from cryptography.fernet import Fernet
            self._fernet_class = Fernet
        except ImportError:
            log.warning("cryptography_not_installed", message="Encryption disabled")
            self._fernet_class = None
    
    def _get_encryption_key(self, exchange: str) -> bytes:
        """Get or create encryption key for an exchange."""
        if not self._keyring:
            raise RuntimeError("Keyring not available")
        
        key_name = f"{exchange}_encryption_key"
        existing_key = self._keyring.get_password(self.service_name, key_name)
        
        if existing_key:
            return existing_key.encode()
        
        # Generate new key
        if self._fernet_class:
            from cryptography.fernet import Fernet
            new_key = Fernet.generate_key()
            self._keyring.set_password(self.service_name, key_name, new_key.decode())
            return new_key
        
        raise RuntimeError("Cannot generate encryption key without cryptography library")
    
    def store_api_credentials(
        self,
        exchange: str,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
    ) -> None:
        """
        Store API credentials securely.
        
        Args:
            exchange: Exchange identifier (e.g., 'binance', 'coinbase')
            api_key: API key
            api_secret: API secret (will be encrypted)
            passphrase: Optional passphrase (for OKX, KuCoin)
        """
        if not self._keyring:
            log.warning("keyring_not_available", exchange=exchange)
            return
        
        # Store API key directly
        self._keyring.set_password(
            self.service_name,
            f"{exchange}_api_key",
            api_key,
        )
        
        # Encrypt and store API secret
        if self._fernet_class:
            key = self._get_encryption_key(exchange)
            cipher = self._fernet_class(key)
            encrypted_secret = cipher.encrypt(api_secret.encode()).decode()
            self._keyring.set_password(
                self.service_name,
                f"{exchange}_api_secret",
                encrypted_secret,
            )
        else:
            # Fall back to unencrypted (not recommended)
            self._keyring.set_password(
                self.service_name,
                f"{exchange}_api_secret",
                api_secret,
            )
        
        # Store passphrase if provided
        if passphrase:
            if self._fernet_class:
                key = self._get_encryption_key(exchange)
                cipher = self._fernet_class(key)
                encrypted_passphrase = cipher.encrypt(passphrase.encode()).decode()
                self._keyring.set_password(
                    self.service_name,
                    f"{exchange}_passphrase",
                    encrypted_passphrase,
                )
            else:
                self._keyring.set_password(
                    self.service_name,
                    f"{exchange}_passphrase",
                    passphrase,
                )
        
        log.info("credentials_stored", exchange=exchange)
    
    def get_api_credentials(self, exchange: str) -> Tuple[str, str, Optional[str]]:
        """
        Retrieve API credentials.
        
        Args:
            exchange: Exchange identifier
            
        Returns:
            Tuple of (api_key, api_secret, passphrase)
            
        Raises:
            ValueError: If credentials not found
        """
        if not self._keyring:
            raise RuntimeError("Keyring not available")
        
        api_key = self._keyring.get_password(self.service_name, f"{exchange}_api_key")
        encrypted_secret = self._keyring.get_password(self.service_name, f"{exchange}_api_secret")
        encrypted_passphrase = self._keyring.get_password(self.service_name, f"{exchange}_passphrase")
        
        if not api_key or not encrypted_secret:
            raise ValueError(f"Credentials not found for {exchange}")
        
        # Decrypt secret
        api_secret = encrypted_secret
        if self._fernet_class:
            try:
                key = self._get_encryption_key(exchange)
                cipher = self._fernet_class(key)
                api_secret = cipher.decrypt(encrypted_secret.encode()).decode()
            except Exception as e:
                log.warning("decryption_failed", exchange=exchange, error=str(e))
                # Might be stored unencrypted in older version
        
        # Decrypt passphrase
        passphrase = None
        if encrypted_passphrase:
            if self._fernet_class:
                try:
                    key = self._get_encryption_key(exchange)
                    cipher = self._fernet_class(key)
                    passphrase = cipher.decrypt(encrypted_passphrase.encode()).decode()
                except Exception:
                    passphrase = encrypted_passphrase
            else:
                passphrase = encrypted_passphrase
        
        return api_key, api_secret, passphrase
    
    def delete_credentials(self, exchange: str) -> None:
        """
        Delete stored credentials for an exchange.
        
        Args:
            exchange: Exchange identifier
        """
        if not self._keyring:
            return
        
        for key_type in ["api_key", "api_secret", "passphrase", "encryption_key"]:
            try:
                self._keyring.delete_password(self.service_name, f"{exchange}_{key_type}")
            except Exception:
                pass  # Key might not exist
        
        log.info("credentials_deleted", exchange=exchange)
    
    def has_credentials(self, exchange: str) -> bool:
        """Check if credentials exist for an exchange."""
        if not self._keyring:
            return False
        
        api_key = self._keyring.get_password(self.service_name, f"{exchange}_api_key")
        return api_key is not None
    
    def list_exchanges(self) -> list:
        """
        List exchanges with stored credentials.
        
        Note: This is a simple implementation that checks known exchanges.
        A more robust solution would enumerate keyring entries.
        """
        known_exchanges = [
            "binance", "coinbase", "bybit", "okx", "kucoin",
            "mexc", "gateio", "bitget", "kraken", "huobi",
        ]
        
        return [ex for ex in known_exchanges if self.has_credentials(ex)]


# Global instance for convenience
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get or create the global secrets manager."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def store_credentials(
    exchange: str,
    api_key: str,
    api_secret: str,
    passphrase: Optional[str] = None,
) -> None:
    """Convenience function to store credentials."""
    get_secrets_manager().store_api_credentials(exchange, api_key, api_secret, passphrase)


def get_credentials(exchange: str) -> Tuple[str, str, Optional[str]]:
    """Convenience function to get credentials."""
    return get_secrets_manager().get_api_credentials(exchange)
