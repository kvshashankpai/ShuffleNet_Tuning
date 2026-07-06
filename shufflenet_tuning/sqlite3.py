"""Local shim to bypass a broken user-site sqlite3 package."""

from pysqlite3 import dbapi2 as dbapi2  # noqa: F401
from pysqlite3.dbapi2 import *  # noqa: F401,F403
