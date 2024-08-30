import os

import pandas as pd
from openai import OpenAI

from config import key
from quality_of_life_pydantic import QualityOfLifeIndexes
from questions import questions
from responses import response1 as response

os.environ["OPENAI_API_KEY"] = key


def get_openai_response(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 500,
                        temperature: float = 0.2) -> QualityOfLifeIndexes:
    """
    Connects to the OpenAI API and submits a prompt, returning the response.

    Parameters:
    - prompt (str): The text prompt to submit to OpenAI.
    - model (str): The model to use, e.g., "gpt-4o". Defaults to "gpt-4".
    - max_tokens (int): The maximum number of tokens in the response. Defaults to 500.
    - temperature (float): The creativity level of the response. Defaults to 0.7.

    Returns:
    - str: The response from the OpenAI API.
    """

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        chat_completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are an analyst assistant which helps generate scores between 0 to 100 on various aspects of quality of life based on questions around quality of life indexes and user responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=QualityOfLifeIndexes
        )

        # Extracting the text response
        return chat_completion.choices[0].message.parsed

    except Exception as e:
        raise Exception(f"An error occurred: {e}")


def create_prompt_user_response_matrix(user_response: dict) -> str:
    """
    Creates the prompt for converting user responses to qol matrix.
    Parameters:
    - user_response (dict): The response from the OpenAI API.
    Returns:
    - prompt (str): The prompt created for extracting qol matrix from user response.
    """
    prompt = f"""
        We are designing a tool which gathers user responses on questions around quality of life aspects for housing and recommends them areas to live in.
        We have decided to use 6 indexes of measuring quality of life with each one of them having multiple factors. Alongside these indexes to understand user demographics we also get user information like number of adults, children and pets in the family, whether the user has a preferred city or county where the user would like to move, and whether the user wants to live in an urban, semi-urban or rural environment. We also get additional context from user which captures important information like must haves for user and whether there is a deal breaker or non negotiable item for user

        Here is the information for the indexes
        First index is sense of control which has 4 factors in it cost of living, essential services, safety and influence and contribution
        Second index is a sense of wonder which has 2 factors play & recreation and distinctive design & culture
        Third index is connected communities which has 2 factors local business and jobs and belonging
        Fourth index is connection to nature which has 3 factors biodiversity, climate resilience and adaptation, and green and blue spaces
        Fifth index is getting around which has 3 factors cars, public transport and walking and cycling
        Sixth index is health equity which has 3 factors air noise light, food choice, and housing standards

        Here are the questions and responses gathered from a user on these indexes and factors:

        Demographics Information
        Question: {user_response['demographics']['number_of_family_members']['question']}
        User response: {user_response['demographics']['number_of_family_members']['response']}

        Question: {user_response['demographics']['city_or_county_preference']['question']}
        User response: {user_response['demographics']['city_or_county_preference']['response']}

        Question: {user_response['demographics']['urban_or_rural_preference']['question']}
        User response: {user_response['demographics']['urban_or_rural_preference']['response']}

        Index name: A sense of control
        Factors: cost of living
        Question: {user_response['a_sense_of_control']['cost_of_living']['question']}
        User response: {user_response['a_sense_of_control']['cost_of_living']['response']}

        Index name: A sense of control
        Factors: essential services
        Question: {user_response['a_sense_of_control']['essential_services']['question']}
        User response: {user_response['a_sense_of_control']['essential_services']['response']}

        Index name: A sense of control
        Factors: safety
        Question: {user_response['a_sense_of_control']['safety']['question']}
        User response: {user_response['a_sense_of_control']['safety']['response']}

        Index name: A sense of control
        Factors: influence and contribution
        Question: {user_response['a_sense_of_control']['influence_and_contribution']['question']}
        User response: {user_response['a_sense_of_control']['influence_and_contribution']['response']}

        Index name: A sense of wonder
        Factors: distinctive design and culture
        Question: {user_response['a_sense_of_wonder']['distinctive_design_and_culture']['question']}
        User response: {user_response['a_sense_of_wonder']['distinctive_design_and_culture']['response']}

        Index name: A sense of wonder
        Factors: play and recreation
        Question: {user_response['a_sense_of_wonder']['play_and_recreation']['question']}
        User response: {user_response['a_sense_of_wonder']['play_and_recreation']['response']}

        Index name: Connected Communities
        Factors: local business & jobs and belonging
        Question: {user_response['connected_communities']['local_business_and_jobs']['question']}
        User response: {user_response['connected_communities']['local_business_and_jobs']['response']}

        Index name: Connection to nature
        Factors: biodiversity, climate resilience & adaptation and green & blue spaces
        Question: {user_response['connection_to_nature']['biodiversity']['question']}
        User response: {user_response['connection_to_nature']['biodiversity']['response']}

        Index name: Getting Around
        Factors: car
        Question: {user_response['getting_around']['car']['question']}
        User response: {user_response['getting_around']['car']['response']}

        Index name: Getting Around
        Factors: public transport
        Question: {user_response['getting_around']['public_transport']['question']}
        User response: {user_response['getting_around']['public_transport']['response']}

        Index name: Getting Around
        Factors: walking and cycling
        Question: {user_response['getting_around']['walking_cycling']['question']}
        User response: {user_response['getting_around']['walking_cycling']['response']}

        Index name: Health Equity
        Factors: air, noise and light
        Question: {user_response['health_equity']['air_noise_light']['question']}
        User response: {user_response['health_equity']['air_noise_light']['response']}

        Index name: Health Equity
        Factors: food choice
        Question: {user_response['health_equity']['food_choice']['question']}
        User response: {user_response['health_equity']['food_choice']['response']}

        Index name: Health Equity
        Factors: housing standard
        Question: {user_response['health_equity']['housing_standard']['question']}
        User response: {user_response['health_equity']['housing_standard']['response']}

        Index name: Health Equity
        Factors: allergies
        Question: {user_response['health_equity']['allergies']['question']}
        User response: {user_response['health_equity']['allergies']['response']}

        Additional Context
        Question: {user_response['additional_context']['must_have']['question']}
        User response: {user_response['additional_context']['must_have']['response']}

        Here is the task, you have to create a JSON object which contains index, factor and score of 0 to 100 for each factor based on the questions and user response from above data
        Here is an example output JSON, please use just the JSON structure for creating the output and do not use the numbers in JSON for your calculation

        {{
            "a_sense_of_control": {{
                "cost_of_living": 65,
                "safety": 78,
                "influence_and_contribution": 55
            }},
            "health_equity": {{
                "housing_standard": 58,
                "air_noise_light": 82,
                "food_choice": 62
            }},
            "connection_to_nature": {{
                "green_and_blue_spaces": 75,
                "biodiversity": 68,
                "climate_resilience_and_adaptation": 72
            }},
            "a_sense_of_wonder": {{
                "distinctive_design_and_culture": 60,
                "play_and_recreation": 70
            }},
            "getting_around": {{
                "walking_and_cycling": 52,
                "public_transport": 48,
                "car": 68
            }},
            "connected_communities": {{
                "belonging": 72,
                "local_business_and_jobs": 58
            }},
        }}
    """
    return prompt


def read_csv(path='data/irl_ed_qol.csv'):
    df = pd.read_csv(path)
    return df


def get_primary_score(row, user_qol_sorted):
    score = 0
    for qol_factor, qol_score in user_qol_sorted:
        score = score + (abs(qol_score - row[qol_factor]))
    return score


def get_secondary_score(row, user_qol_sorted):
    score = 0
    for qol_factor, qol_score in user_qol_sorted:
        score = score + (abs(qol_score - row[qol_factor]))
    return score


def get_top_locations(df_qol: pd.DataFrame, user_qol: QualityOfLifeIndexes) -> pd.DataFrame:
    """
    Function to get top locations based on a QualityOfLifeIndex dataframe and user responses
    :param df_qol: dataframe containing qol matrix for all regions
    :param user_qol: QualityOfLifeIndexes object containing user responses
    :return: dataframe containing qol matrix for top 5 matching regions
    """
    user_qol_dict = pd.json_normalize(user_qol.dict(), sep=":").to_dict(orient='records')[0]
    user_qol_sorted = sorted(user_qol_dict.items(), key=lambda x: x[1], reverse=True)

    df_qol['user_score_primary'] = df_qol.apply(lambda x: get_primary_score(x, user_qol_sorted[0:5]), axis=1)
    df_qol['user_score_secondary'] = df_qol.apply(lambda x: get_secondary_score(x, user_qol_sorted), axis=1)
    df_qol = df_qol.sort_values(by=['user_score_primary', 'user_score_secondary', 'QoL'])
    return df_qol.iloc[0:5]


user_response_dict = {
    "demographics": {
        "number_of_family_members": {
            "question": questions[0],
            "response": response[0]
        },
        "city_or_county_preference": {
            "question": questions[1],
            "response": response[1]
        },
        "urban_or_rural_preference": {
            "question": questions[2],
            "response": response[2]
        },
    },
    "a_sense_of_control": {
        "cost_of_living": {
            "question": questions[3],
            "response": response[3]
        },
        "essential_services": {
            "question": questions[4],
            "response": response[4]
        },
        "safety": {
            "question": questions[5],
            "response": response[5]
        },
        "influence_and_contribution": {
            "question": questions[6],
            "response": response[6]
        }
    },
    "a_sense_of_wonder": {
        "play_and_recreation": {
            "question": questions[7],
            "response": response[7]
        },
        "distinctive_design_and_culture": {
            "question": questions[8],
            "response": response[8]
        }
    },
    "connected_communities": {
        "local_business_and_jobs": {
            "question": questions[9],
            "response": response[9]
        }
    },
    "connection_to_nature": {
        "biodiversity": {
            "question": questions[10],
            "response": response[10]
        }
    },
    "getting_around": {
        "car": {
            "question": questions[11],
            "response": response[11]
        },
        "public_transport": {
            "question": questions[12],
            "response": response[12]
        },
        "walking_cycling": {
            "question": questions[13],
            "response": response[13]
        }
    },
    "health_equity": {
        "air_noise_light": {
            "question": questions[14],
            "response": response[14]
        },
        "food_choice": {
            "question": questions[15],
            "response": response[15]
        },
        "housing_standard": {
            "question": questions[16],
            "response": response[16]
        },
        "allergies": {
            "question": questions[17],
            "response": response[17]
        }
    },
    "additional_context": {
        "must_have": {
            "question": questions[18],
            "response": response[18]
        }
    }
}

prompt = create_prompt_user_response_matrix(user_response_dict)
parsed_response = get_openai_response(prompt)
df_qol = read_csv()
df_top_locations = get_top_locations(df_qol, parsed_response)
print(df_top_locations)
