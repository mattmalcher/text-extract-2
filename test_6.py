import llama_cpp
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

import instructor

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional
from rich.console import Console

from datetime import datetime

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
2 Bedroom Apartment  For Sale 

    A beautifully presented two bedroom, two bathroom top-floor apartment in a sought-after modern development.The spacious property includes an expansive open-plan living area that flows on to a generous sun-trap balcony. This apartment is not overlooked by any other buildings; all windows look out at the treetops of the nearby green spaces, giving the property a sense of calm and privacy rarely found in this well-connected area. The master bedroom benefits from an en-suite shower room, and the generous bathroom makes this property perfect for families or sharers. The property comes with an allocated parking space in the building’s secure underground car park. The building also benefits from a lift service and communal grounds.Crown Dale is primarily served by Gipsy Hill, West Norwood, and Crystal Palace rail links. A wealth of shopping and leisure amenities are nearby at the Crystal Palace Triangle, also West Norwood High Street and Gipsy Parade.EPC: C | Council Tax Band: C | Lease: 109 years remaining | SC: £285pm | GR: £260 | BI: £2,295
            Tenure: Leasehold
"""


class Ownership(str, Enum):
    leasehold = "leasehold"
    freehold = "freehold"
    share_of_freehold = "share_of_freehold"

class Alphabetical(str, Enum):
    a = "A"
    b = "B"
    c = "C"
    d = "D"
    e = "E"
    f = "F"
    g = "G"
    h = "H" # EPC doesnt go this far but ctax does


class Property(BaseModel):
    id: str = Field(default=None)

    name: str

    price: int

    ownership_type: Optional[Ownership] = None
    council_tax_band: Optional[Alphabetical] = None

    epc : Optional[Alphabetical] = None

    size_m2: Optional[int] = None
    size_sq_ft: Optional[int] = None

    lat: Optional[float] = None
    lon: Optional[float] = None

    service_charge: Optional[str] = None
    ground_rent: Optional[str] = None
    building_insurance: Optional[str] = None

    def augment(self, data: dict):

        # Existing items that are not empty + new items.
        update = data | {k:v for k, v in self.model_dump().items() if v is not None}

        # check each one is ok, then set it.
        for k, v in (
            self.model_validate(update).model_dump(exclude_defaults=True).items()
        ):

            setattr(self, k, v)
        return self

    # define eq and hash so we can use sets
    def __eq__(self, other):
        return isinstance(other, Property) and self.id == other.id

    def __hash__(self):
        return hash(self.id)
    
initial_property = Property(
    id="blah",
    name="test",
    price=123_456
)


additional_data = create(
    response_model=Property,  # (3)!
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the property {text_block}",
        },
    ],
    stream=False,
    max_retries=2
)

initial_property.augment(additional_data.model_dump())

print("done")