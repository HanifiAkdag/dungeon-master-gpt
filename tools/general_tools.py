from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_core.tools import ToolException
import random


def roll_dice(max_value=6):
    return random.randint(1, max_value)

class RollDiceInput(BaseModel):
    max_value: int = Field(description="The maximum value of the dice. This value is also included in the possible outcomes.")

roll_dice_tool = StructuredTool.from_function(
    func=roll_dice,
    name="Roll_Dice",
    description="Roll a dice with the maximum value you specify. Generates a random number between 1 and the maximum value.",
    args_schema=RollDiceInput,
    handle_tool_error=False,
    return_direct=False,
    # coroutine= ... <- you can specify an async method if desired as well
)
