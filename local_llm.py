# ollama pull llama3

# ollama run llama3

#prompt ollama run llama3 "Write a fantasy story based on: a man riding a horse, fireworks in a stadium, a child watching the moon."


import subprocess

# prompt = """
# Write a story combining:
# - A man is riding a horse.
# - A stadium full of people watching fireworks.
# - A woman cooking dinner.
# """

def get_prompts(capitons):
    # prompt = get_prompts()
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=captions.encode(),
        capture_output=True
    )
    print(result.stdout.decode())
    return result
