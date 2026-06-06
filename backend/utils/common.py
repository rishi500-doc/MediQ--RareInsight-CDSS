import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_logger(name: str) -> logging.Logger:
    """Configures and returns a production-ready logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)

def create_session(timeout: int = 30) -> requests.Session:
    """Creates a robust requests session with retries and a default timeout."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    # IMPROVEMENT: Attach default timeout so no call can hang forever
    session.request = lambda method, url, **kwargs: requests.Session.request(
        session, method, url, timeout=kwargs.pop('timeout', timeout), **kwargs
    )
    return session
