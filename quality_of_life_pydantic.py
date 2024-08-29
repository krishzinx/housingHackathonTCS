from pydantic import BaseModel


class SenseOfControl(BaseModel):
    cost_of_living: int
    safety: int
    influence_and_contribution: int


class HealthEquity(BaseModel):
    housing_standard: int
    air_noise_light: int
    food_choice: int


class ConnectionToNature(BaseModel):
    housing_standard: int
    air_noise_light: int
    food_choice: int


class SenseOfWonder(BaseModel):
    distinctive_design_and_culture: int
    play_and_recreation: int


class GettingAround(BaseModel):
    walking_and_cycling: int
    public_transport: int
    car: int


class ConnectedCommunities(BaseModel):
    belonging: int
    local_business_and_jobs: int


class QualityOfLifeIndexes(BaseModel):
    a_sense_of_control: SenseOfControl
    a_sense_of_wonder: SenseOfWonder
    connected_communities: ConnectedCommunities
    connection_to_nature: ConnectionToNature
    getting_around: GettingAround
    health_equity: HealthEquity