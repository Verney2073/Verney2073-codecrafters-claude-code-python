import argparse
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read",
                            }
                        },
                        "required": ["file_path"],
                    },
                },
            }
        ],
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    if tool_calls := chat.choices[0].message.tool_calls:
        try:
            for tool in tool_calls:
                response = tool.function.arguments
                messages.append(
                    {"role": "tool", "tool_call_id": tool.id, "content": response}
                )
                client.chat.completions.create(
                    model="anthropic/claude-haiku-4.5",
                    messages=messages,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "description": "Read and return the contents of a file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {
                                            "type": "string",
                                            "description": "The path to the file to read",
                                        }
                                    },
                                    "required": ["file_path"],
                                },
                            },
                        }
                    ],
                )
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
    else:
        # TODO: Uncomment the following line to pass the first stage
        print(chat.choices[0].message.content)


if __name__ == "__main__":
    main()
