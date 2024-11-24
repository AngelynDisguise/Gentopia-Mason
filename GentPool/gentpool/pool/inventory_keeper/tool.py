### Define your custom tool here. Check prebuilts in gentopia.tool (:###
from gentopia.tools import *
from gentopia.tools import WriteFile

path = "~/Desktop/cs478/assignments/ass5/CS478-A5/CS478-A5/Gentopia-Mason/GentPool/gentpool/pool/ramsay"

class WriteToInventory(BaseTool):
    name = "write_inventory"
    description = "A tool to write JSON objects to the food inventory file."
    args_schema: Optional[Type[BaseModel]] = create_model("WriteToInventoryArgs", text=(str, ...))

    def _run(self, text: AnyStr) -> Any:
        ans = WriteFile()._run("food_inventory.json", text)
        return ans

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    
class ReadFromInventory(BaseTool):
    name = "read_inventory"
    description = "A tool to read JSON objects from the food inventory file."
    args_schema: Optional[Type[BaseModel]] = create_model("ReadFromInventoryArgs", text=(str, ...))

    def _run(self, text: AnyStr) -> Any:
        ans = ReadFile()._run("food_inventory.json")
        return ans

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError