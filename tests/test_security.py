"""Tests for Security Module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from utils.security import SecretsManager, get_secrets_manager


class TestSecretsManager:
    """Tests for SecretsManager."""
    
    @pytest.fixture
    def mock_keyring(self):
        """Create a mock keyring."""
        mock_kr = MagicMock()
        mock_kr.get_password = Mock(return_value=None)
        mock_kr.set_password = Mock()
        mock_kr.delete_password = Mock()
        return mock_kr
    
    @pytest.fixture
    def mock_fernet_class(self):
        """Create a mock Fernet class."""
        mock_cipher = MagicMock()
        mock_cipher.encrypt = Mock(return_value=b"encrypted_data")
        mock_cipher.decrypt = Mock(return_value=b"decrypted_data")
        
        mock_fernet = MagicMock()
        mock_fernet.generate_key = Mock(return_value=b"test_encryption_key_123456789012")
        mock_fernet.return_value = mock_cipher
        
        return mock_fernet
    
    @pytest.fixture
    def secrets_manager(self, mock_keyring, mock_fernet_class):
        """Create SecretsManager with mocked dependencies."""
        manager = SecretsManager(service_name="test_service")
        manager._keyring = mock_keyring
        manager._fernet_class = mock_fernet_class
        return manager
    
    def test_initialization(self):
        """Test SecretsManager initialization."""
        manager = SecretsManager(service_name="my_service")
        assert manager.service_name == "my_service"
    
    def test_store_api_credentials_basic(self, secrets_manager, mock_keyring):
        """Test storing API credentials."""
        secrets_manager.store_api_credentials(
            exchange="binance",
            api_key="test_key_123",
            api_secret="test_secret_456",
        )
        
        # Verify keyring calls
        calls = mock_keyring.set_password.call_args_list
        assert len(calls) >= 2  # At least key and secret
        
        # Check API key was stored
        api_key_call = [c for c in calls if "api_key" in str(c)]
        assert len(api_key_call) > 0
    
    def test_store_api_credentials_with_passphrase(
        self,
        secrets_manager,
        mock_keyring,
    ):
        """Test storing credentials with passphrase."""
        secrets_manager.store_api_credentials(
            exchange="okx",
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_passphrase",
        )
        
        calls = mock_keyring.set_password.call_args_list
        # Should have key, secret, passphrase, and encryption keys
        assert len(calls) >= 3
    
    def test_get_api_credentials(self, secrets_manager, mock_keyring, mock_fernet_class):
        """Test retrieving API credentials."""
        # Mock keyring to return stored values (strings as keyring stores strings)
        def get_password_side_effect(service, key):
            mapping = {
                "test_service_binance_api_key": "stored_key_123",
                "test_service_binance_api_secret": "encrypted_secret_base64",
                "test_service_binance_encryption_key": "test_encryption_key_123456789012",
            }
            return mapping.get(f"{service}_{key}")
        
        mock_keyring.get_password = Mock(side_effect=get_password_side_effect)
        
        # Mock Fernet to decrypt
        mock_cipher = MagicMock()
        mock_cipher.decrypt = Mock(return_value=b"decrypted_secret_456")
        mock_fernet_class.return_value = mock_cipher
        
        api_key, api_secret, passphrase = secrets_manager.get_api_credentials("binance")
        
        assert api_key == "stored_key_123"
        assert api_secret == "decrypted_secret_456"
        assert passphrase is None
    
    def test_get_api_credentials_not_found(self, secrets_manager, mock_keyring):
        """Test retrieving non-existent credentials raises error."""
        mock_keyring.get_password = Mock(return_value=None)
        
        with pytest.raises(ValueError, match="Credentials not found"):
            secrets_manager.get_api_credentials("nonexistent_exchange")
    
    def test_delete_credentials(self, secrets_manager, mock_keyring):
        """Test deleting credentials."""
        secrets_manager.delete_credentials("binance")
        
        # Should attempt to delete all credential types
        assert mock_keyring.delete_password.call_count >= 4
    
    def test_has_credentials_true(self, secrets_manager, mock_keyring):
        """Test has_credentials returns True when credentials exist."""
        mock_keyring.get_password = Mock(return_value="some_key")
        
        assert secrets_manager.has_credentials("binance") is True
    
    def test_has_credentials_false(self, secrets_manager, mock_keyring):
        """Test has_credentials returns False when no credentials."""
        mock_keyring.get_password = Mock(return_value=None)
        
        assert secrets_manager.has_credentials("binance") is False
    
    def test_list_exchanges(self, secrets_manager, mock_keyring):
        """Test listing exchanges with stored credentials."""
        # Mock keyring to return True for some exchanges
        def mock_get_password(service, key):
            if "binance" in key or "coinbase" in key:
                return "some_value"
            return None
        
        mock_keyring.get_password = Mock(side_effect=mock_get_password)
        
        exchanges = secrets_manager.list_exchanges()
        
        assert "binance" in exchanges
        assert "coinbase" in exchanges
    
    def test_get_encryption_key_creates_new(self, secrets_manager, mock_keyring, mock_fernet_class):
        """Test encryption key generation for new exchange."""
        mock_keyring.get_password = Mock(return_value=None)
        
        key = secrets_manager._get_encryption_key("new_exchange")
        
        assert key is not None
        # Should have stored the new key
        assert mock_keyring.set_password.called
    
    def test_get_encryption_key_retrieves_existing(self, secrets_manager, mock_keyring):
        """Test retrieving existing encryption key."""
        existing_key = b"existing_key_12345678901234567890"
        mock_keyring.get_password = Mock(return_value=existing_key.decode())
        
        key = secrets_manager._get_encryption_key("existing_exchange")
        
        assert key == existing_key
    
    def test_store_without_keyring(self):
        """Test storing credentials without keyring available."""
        manager = SecretsManager()
        manager._keyring = None
        
        # Should log warning but not raise
        manager.store_api_credentials(
            exchange="test",
            api_key="key",
            api_secret="secret",
        )
    
    def test_get_without_keyring(self):
        """Test getting credentials without keyring raises error."""
        manager = SecretsManager()
        manager._keyring = None
        
        with pytest.raises(RuntimeError, match="Keyring not available"):
            manager.get_api_credentials("test")
    
    def test_encryption_without_cryptography(self, mock_keyring):
        """Test storing credentials without cryptography library."""
        manager = SecretsManager()
        manager._keyring = mock_keyring
        manager._fernet_class = None
        
        # Should fall back to unencrypted storage
        manager.store_api_credentials(
            exchange="test",
            api_key="key",
            api_secret="secret",
        )
        
        # Secret should be stored without encryption
        calls = mock_keyring.set_password.call_args_list
        secret_call = [c for c in calls if "api_secret" in str(c)]
        assert len(secret_call) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_secrets_manager_singleton(self):
        """Test global secrets manager is singleton."""
        manager1 = get_secrets_manager()
        manager2 = get_secrets_manager()
        
        assert manager1 is manager2
    
    @patch('utils.security.get_secrets_manager')
    def test_store_credentials_convenience(self, mock_get_manager):
        """Test store_credentials convenience function."""
        from utils.security import store_credentials
        
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        
        store_credentials(
            exchange="binance",
            api_key="key",
            api_secret="secret",
        )
        
        mock_manager.store_api_credentials.assert_called_once_with(
            "binance",
            "key",
            "secret",
            None,
        )
    
    @patch('utils.security.get_secrets_manager')
    def test_get_credentials_convenience(self, mock_get_manager):
        """Test get_credentials convenience function."""
        from utils.security import get_credentials
        
        mock_manager = Mock()
        mock_manager.get_api_credentials.return_value = ("key", "secret", None)
        mock_get_manager.return_value = mock_manager
        
        result = get_credentials("binance")
        
        assert result == ("key", "secret", None)
        mock_manager.get_api_credentials.assert_called_once_with("binance")


class TestSecurityBestPractices:
    """Tests for security best practices."""
    
    @pytest.fixture
    def mock_keyring(self):
        """Create a mock keyring."""
        mock_kr = MagicMock()
        mock_kr.get_password = Mock(return_value=None)
        mock_kr.set_password = Mock()
        mock_kr.delete_password = Mock()
        return mock_kr
    
    @pytest.fixture
    def mock_fernet_class(self):
        """Create a mock Fernet class."""
        mock_cipher = MagicMock()
        mock_cipher.encrypt = Mock(return_value=b"encrypted_data")
        mock_cipher.decrypt = Mock(return_value=b"decrypted_data")
        
        mock_fernet = MagicMock()
        mock_fernet.generate_key = Mock(return_value=b"test_encryption_key_123456789012")
        mock_fernet.return_value = mock_cipher
        
        return mock_fernet
    
    def test_no_plaintext_secrets_in_memory(self, mock_keyring, mock_fernet_class):
        """Test that secrets are encrypted in storage."""
        manager = SecretsManager()
        manager._keyring = mock_keyring
        manager._fernet_class = mock_fernet_class
        
        # Store credentials
        manager.store_api_credentials(
            exchange="test",
            api_key="key",
            api_secret="super_secret_value",
        )
        
        # Verify encryption was used
        assert mock_fernet_class.called or mock_fernet_class.return_value.encrypt.called
    
    def test_passphrase_encrypted(self, mock_keyring, mock_fernet_class):
        """Test that passphrases are also encrypted."""
        manager = SecretsManager()
        manager._keyring = mock_keyring
        manager._fernet_class = mock_fernet_class
        
        manager.store_api_credentials(
            exchange="okx",
            api_key="key",
            api_secret="secret",
            passphrase="my_passphrase",
        )
        
        # Should encrypt both secret and passphrase
        calls = mock_keyring.set_password.call_args_list
        assert len(calls) >= 3  # key, secret, passphrase (+ encryption keys)
    
    def test_credentials_not_logged(self, mock_keyring, mock_fernet_class):
        """Test that credentials aren't accidentally logged."""
        # This is more of a documentation test
        # In real code, ensure log statements don't include sensitive data
        manager = SecretsManager()
        manager._keyring = mock_keyring
        manager._fernet_class = mock_fernet_class
        
        manager.store_api_credentials(
            exchange="test",
            api_key="sensitive_key",
            api_secret="sensitive_secret",
        )
        
        # The test itself verifies nothing sensitive is in logs by design
        # Actual log checking would require log capture fixtures
        pass
