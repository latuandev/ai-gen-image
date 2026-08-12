import random
from enum import Enum


class EnumChoices(Enum):
    """
    Base enum class providing utility methods for enum choices and values.

    Example:
        class Status(EnumChoices):
            ACTIVE = 1
            INACTIVE = 0
    """

    @classmethod
    def choices(cls):
        """
        Return enum members as a list of (value, name) tuples.

        Example:
            Status.choices()
            # [(1, 'ACTIVE'), (0, 'INACTIVE')]
        """
        return [(choice.value, choice.name) for choice in cls]

    @classmethod
    def random(cls, not_values=None):
        """
        Return a random enum value, excluding specified values if provided.

        Examples:
            Status.random()
            # 1 or 0

            Status.random(not_values=[0])
            # 1
        """
        if not_values is None:
            not_values = []

        choices = [
            choice.value for choice in cls if choice.value not in not_values
        ]

        return random.choice(choices)

    @classmethod
    def names(cls):
        """
        Return a list of all enum member names.

        Example:
            Status.names()
            # ['ACTIVE', 'INACTIVE']
        """
        return [choice.name for choice in cls]

    @classmethod
    def values(cls):
        """
        Return a list of all enum member values.

        Example:
            Status.values()
            # [1, 0]
        """
        return [choice.value for choice in cls]
