from common.utils.enum_choices import EnumChoices

# --------------------------------|
# Session for Class choice enums. |
# --------------------------------|

class AgentRunStatus(str, EnumChoices):
    """
    Agent run status enum.
    """
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


# -------------------------|
# Session for constants.   |
# -------------------------|
