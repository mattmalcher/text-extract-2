import instructor

from pydantic import BaseModel
from typing import List
from rich.console import Console

from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

oai_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base
    )

client = instructor.from_openai(
    client=oai_client,  
    mode=instructor.Mode.JSON
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


class User(BaseModel):
    name: str
    email: str
    twitter: str


class MeetingInfo(BaseModel):
    users: List[User]
    date: str
    location: str
    budget: int
    deadline: str


extraction_stream = client.chat.create(
    model = "bartowski/Meta-Llama-3-8B-Instruct-AWQ",
    response_model=instructor.Partial[MeetingInfo],  # (3)!
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the meeting and the users {text_block}",
        },
    ],
    stream=True,
    response_format={ "type": "json_object" }
)


console = Console()

for extraction in extraction_stream:
    obj = extraction.model_dump()
    console.clear()  # (4)!
    console.print(obj)