import llama_cpp
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

import instructor

from pydantic import BaseModel
from typing import List
from rich.console import Console


llama = llama_cpp.Llama(
    model_path="./models/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
    n_gpu_layers=-1,
    chat_format="llama-3",
    n_ctx=2048,
    draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10),  # (1)!
    logits_all=True,
    verbose=False,
)


create = instructor.patch(
    create=llama.create_chat_completion_openai_v1,
    mode=instructor.Mode.JSON_SCHEMA,  # (2)!
)


text_block = """
In our recent online meeting, participants from various backgrounds joined to discuss
the upcoming tech conference. The names and contact details of the participants were as follows:

- Name: John Doe, Email: johndoe@email.com, Twitter: @TechGuru44
- Name: Jane Smith, Email: janesmith@email.com, Twitter: @DigitalDiva88
- Name: Alex Johnson, Email: alexj@email.com, Twitter: @CodeMaster2023

During the meeting, we agreed on several key points. The conference will be held on March 15th, 2024,
at the Grand Tech Arena located at 4521 Innovation Drive. Dr. Emily Johnson, a renowned AI researcher,
will be our keynote speaker.

The budget for the event is set at $50,000, covering venue costs, speaker fees, and promotional activities.
Each participant is expected to contribute an article to the conference blog by February 20th.

A follow-up meetingis scheduled for January 25th at 3 PM GMT to finalize the agenda and confirm the list of speakers.
"""


# class User(BaseModel):
#     name: str
#     email: str
#     twitter: str

class User(BaseModel):
    forename: str
    surname: str
    email: str
    twitter: str


class MeetingInfo(BaseModel):
    users: List[User]
    iso_8601_date: str
    location: str
    budget: int
    deadline: str


extraction_stream = create(
    response_model=instructor.Partial[MeetingInfo],  # (3)!
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the meeting and the users {text_block}",
        },
    ],
    stream=True,
    max_retries=2
)


console = Console()

for extraction in extraction_stream:
    obj = extraction.model_dump()
    console.clear()  # (4)!
    console.print(obj)