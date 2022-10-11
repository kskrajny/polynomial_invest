import time
import logging
import re

from collections import OrderedDict

from multiprocessing import Lock
from threading import RLock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session


def get_default_logger():
    """Base logging class. Logs currently only to stream on a debug level
    :return: Returns logger object
    """

    logger = logging.getLogger("__main__")

    # _log_path = './logs/data_downloader.log'
    # _fh = logging.FileHandler(_log_path)
    # _fh.setLevel(logging.ERROR)

    _sh = logging.StreamHandler()
    _sh.setLevel(logging.DEBUG)

    _formatter = logging.Formatter('%(asctime)s - %(process)d - %(thread)d - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    _sh.setFormatter(_formatter)

    logger.addHandler(_sh)
    # self.logger.addHandler(_fh)
    return logger


class TTLOrderedDict(OrderedDict):
    """ Some code reused from main aii code
    OrderedDict with TTL
    Extra args and kwargs are passed to initial .update() call
    """

    def __init__(self, default_ttl, max_elements=30, *args, **kwargs):
        """
        Be warned, if you use this with Python versions earlier than 3.6
        when passing **kwargs order is not preseverd.
        """
        assert isinstance(default_ttl, int)
        self._default_ttl = default_ttl
        self._max_elements = max_elements
        self._lock = RLock()
        super().__init__()
        self.update(*args, **kwargs)

    def __repr__(self):
        return '<TTLOrderedDict@%#08x; ttl=%r, OrderedDict=%r;>' % (
            id(self), self._default_ttl, self.items())

    def __len__(self):
        with self._lock:
            self._purge()
            return super().__len__()

    def set_ttl(self, key, ttl, now=None):
        """Set TTL for the given key"""
        if now is None:
            now = time.time()
        with self._lock:
            value = self[key]
            super().__setitem__(key, (now + ttl, value))

    def get_ttl(self, key, now=None):
        """Return remaining TTL for a key"""
        if now is None:
            now = time.time()
        with self._lock:
            expire, _value = super().__getitem__(key)
            return expire - now

    def expire_at(self, key, timestamp):
        """Set the key expire timestamp"""
        with self._lock:
            value = self.__getitem__(key)
            super().__setitem__(key, (timestamp, value))

    def is_expired(self, key, now=None):
        """ Check if key has expired, and return it if so"""
        with self._lock:
            if now is None:
                now = time.time()

            try:
                expire, _value = super().__getitem__(key)
            except KeyError as e:
                return None

            if expire:
                if expire < now:
                    return key

    def _purge(self):
        _keys = list(super().__iter__())
        _remove = [key for key in _keys if self.is_expired(key)]  # noqa
        [self.__delitem__(key) for key in _remove]

    def _clear(self):
        _keys = list(super().__iter__())
        [self.__delitem__(key) for key in _keys]

    def __iter__(self):
        """
        Yield only non expired keys, without purging the expired ones
        """
        with self._lock:
            for key in super().__iter__():
                if not self.is_expired(key):
                    yield key

    def __setitem__(self, key, value):
        with self._lock:
            if self._default_ttl is None:
                expire = None
            else:
                expire = time.time() + self._default_ttl
            super().__setitem__(key, (expire, value))

    def __delitem__(self, key):
        with self._lock:
            item = super().__getitem__(key)[1]
            super().__delitem__(key)

    def __getitem__(self, key):
        with self._lock:
            if self.is_expired(key):
                self.__delitem__(key)
                raise KeyError

            try:
                item = super().__getitem__(key)[1]
                return item
            except KeyError as e:
                raise KeyError

    def keys(self):
        with self._lock:
            self._purge()
            return super().keys()

    def items(self):
        with self._lock:
            self._purge()
            _items = list(super(OrderedDict, self).items())
            return [(k, v[1]) for (k, v) in _items]

    def values(self):
        with self._lock:
            self._purge()
            _values = list(super(OrderedDict, self).values())
            return [v[1] for v in _values]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class SQLSessionConnector:
    """Some code reused from main aii code"""

    def __init__(self):
        self.lock = Lock()
        self.sessions: TTLOrderedDict = TTLOrderedDict(5 * 60)
        self.host_and_port_regex = "^.*@(.*)\/.*$"

    def sql_session(self, db_connection_string, process_name) -> Session:
        session_key = self.get_session_key(db_connection_string, process_name)
        try:
            session = self.sessions[session_key]
            if session.is_active:
                return session
        except:
            pass
        session = self.create_db_session(db_connection_string)
        self.sessions[session_key] = session
        return session

    def create_db_session(self, db_conection_string: str, timeout: int = 3600) -> Session:
        with self.lock:
            DBSession = sessionmaker(self.create_db_engine(db_conection_string, timeout))
            return DBSession()

    def create_db_engine(self, db_conection_string: str, timeout: int):
        return create_engine(db_conection_string, isolation_level='READ_UNCOMMITTED',
                             pool_pre_ping=True, pool_recycle=7200, pool_size=10,
                             connect_args={'connect_timeout': timeout})  # echo=True for SQL queries

    def invalidate_all_sessions(self):
        for session_key in self.sessions:
            session = self.sessions[session_key]
            session.expire_all()
            session.invalidate()
        self.sessions.clear()

    def invalidate_session(self, db_connection_string, process_name):
        session_key = self.get_session_key(db_connection_string, process_name)
        session = self.sessions[session_key]
        if session is not None:
            session.expire_all()
            session.invalidate()

    def get_session_key(self, db_connection_string, process_name):
        session_key = db_connection_string + "_" + process_name
        re_object = re.match(self.host_and_port_regex, db_connection_string, re.M | re.I)
        if re_object:
            session_key = re_object.group(1) + "_" + process_name
        return session_key
