import os


def env_bool(name: str, default: bool = False):
    """
    Get an environment variable and convert it to a boolean.
    """
    value = os.getenv(name)

    if value is None:
        return default

    return value.lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def env_list(name: str, default: str = ""):
    """
    Get a comma-separated environment variable and convert it to a list.
    """
    value = os.getenv(name, default)

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]
