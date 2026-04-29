from openai import OpenAI

client = OpenAI(api_key="sk-PqLX9onFofU-C4lEFbVtmAI9s3s10NvkJH4SclodW8T3BlbkFJ5Jp8fCep8lTloNk-ijHAGegT5nPVS__OR3HzgNtjEA")




def generate_response(system_prompt, input_text,gender):
    prompt = f"""
    
    System: {system_prompt}
    
    Thông tin user: 
    - Giới tính: {gender}
    User:  {input_text}
    
    Hãy xưng hô phù hợp với giới tính user.
    """
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )
    
    return response.output_text
