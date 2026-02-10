import os


def parse_bool_env(name, default='0'):
    """Return True when the environment variable equals '1', ignoring trailing semicolons."""
    value = os.environ.get(name, default)
    if value is None:
        value = default
    value = value.strip()
    if value.endswith(';'):
        value = value[:-1].rstrip()
    return value == '1'
