from django.conf import settings

# Message lookup structure: LANGUAGE_CODE -> app namespace -> stable message key
_MESSAGES = {
    "en-us": {},
    "vi": {},
}

MESSAGES = _MESSAGES[settings.LANGUAGE_CODE]
