from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_core.tools import ToolException
import fandom

fandom.set_wiki("lotr")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn", truncation=True, max_length=512, model_max_length=512, device=device)
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

def summarize(text, max_length=700, min_length=350, length_penalty=2.0, num_beams=5):
    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    # Don't summarize if text is too short
    if inputs.shape[1] < max_length:
        return text
    # Generate summary
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=True)
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary



def lotr_search(query: str) -> str:
    try:
        search_results = fandom.search(query, results = 10)
    except fandom.error.FandomError as e:
        raise ToolException(f"Error occurred: {e}")

    return search_results

def lotr_get_subsections(page_name: str) -> str:
    try:
        page = fandom.page(page_name)
        sections = page.sections
    except fandom.error.FandomError as e:
        raise ToolException(f"Error occurred: {e}")

    return sections

def process_section(section, level=2):
    # Generate the appropriate number of hash symbols for the current level
    hash_symbols = '#' * level
    title = section['title']
    content = summarize(section['content'])
    markdown = f"{hash_symbols} {title}\n\n{content}\n\n"

    # Process subsections if they exist
    if 'sections' in section:
        for subsection in section['sections']:
            markdown += process_section(subsection, level + 1)

    return markdown

def generate_markdown(data):
    markdown_output = ""

    for item in data:
        title = item['title']
        content = summarize(item['content'])
        markdown_output += f"# {title}\n\n{content}\n\n"

        # Process sections if they exist
        if 'sections' in item:
            for section in item['sections']:
                markdown_output += process_section(section)

    return markdown_output

def find_and_summarize_sections(sections, sections_wanted, level=2):
    markdown_output = ""
    for section in sections:
        if section['title'] in sections_wanted:
            if 'sections' in section:
                subsections = [e['title'] for e in section['sections']]
                sections_wanted = [e for e in sections_wanted if e not in subsections]
            markdown_output += process_section(section, level)
        if 'sections' in section:
            markdown_output += find_and_summarize_sections(section['sections'], sections_wanted, level + 1)
    return markdown_output

def summarize_sections(page: str, sections_wanted: list[str]):
    try:
        page = fandom.page(page)
    except fandom.error.FandomError as e:
        raise ToolException(f"Error occurred: {e}")

    return find_and_summarize_sections(page.content['sections'], sections_wanted)


class SearchInput(BaseModel):
    query: str = Field(description="The topic you want get the explanation for.")

class GetSubsectionsInput(BaseModel):
    page_name: str = Field(description="The page you want to get the subsections from.")

class SummarizeSectionsInput(BaseModel):
    page: str = Field(description="The page you want to summarize.")
    sections_wanted: list[str] = Field(description="The sections you want to summarize.")

search_tool = StructuredTool.from_function(
    func=lotr_search,
    name="Search",
    description="Search for possible relevant pages on LOTR fandom. The pages can be about a character, an event, a place etc. You must use this function first to select the page you are looking for. Then use Get_Subsections tool.",
    args_schema=SearchInput,
    handle_tool_error=True,
    return_direct=False,
    # coroutine= ... <- you can specify an async method if desired as well
)

get_subsections_tool = StructuredTool.from_function(
    func=lotr_get_subsections,
    name="Get_Subsections",
    description="Get the subsections of a page on LOTR fandom. You must select a list of subsections that you need. You must use the Search tool first. If the page isn't what you were looking for search again. Then you can use Summarize_Sections tool.",
    args_schema=GetSubsectionsInput,
    handle_tool_error=True,
    return_direct=False,
    # coroutine= ... <- you can specify an async method if desired as well
)

summarize_tool = StructuredTool.from_function(
    func=summarize_sections,
    name="Summarize_Sections",
    description="Summarize the sections of a page on LOTR fandom. You must select a list of sections that you need. Returns the summary of the sections in the markdown format. You must use the Get_Subsections tool first.",
    args_schema=SummarizeSectionsInput,
    handle_tool_error=True,
    return_direct=False,
    # coroutine= ... <- you can specify an async method if desired as well
)