from dataclasses import dataclass
import random

@dataclass
class Customer:

    customer_id:str

    age:int

    tenure_years:int

    preferred_channel:str

    risk_level:str

CHANNELS=[

    "VoiceBot",

    "IVR",

    "Mobile App"

]

RISK=[

    "LOW",

    "MEDIUM",

    "HIGH"

]


def generate_customer():

    cid=f"CUST_{random.randint(1000,9999)}"

    return Customer(

        customer_id=cid,

        age=random.randint(22,70),

        tenure_years=random.randint(1,18),

        preferred_channel=random.choice(CHANNELS),

        risk_level=random.choice(RISK)

    )