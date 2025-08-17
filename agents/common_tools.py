from langchain_core.tools import StructuredTool
from agents.utils import ReadTxtFileInput, read_txt_file

read_text_file_tool = StructuredTool.from_function(
    func=read_txt_file,
    name="read_txt_file",
    description="Read a file and return its contents with automatic encoding fix for special characters.",
    args_schema=ReadTxtFileInput,
)
