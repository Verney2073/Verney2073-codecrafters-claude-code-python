import argparse
import json
import os
import subprocess
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

    loop = True

    while loop:
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "description": "Write content to a file",
                        "parameters": {
                            "type": "object",
                            "required": ["file_path", "content"],
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path of the file to write to",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content to write to the file",
                                },
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Execute a shell command",
                        "parameters": {
                            "type": "object",
                            "required": ["command"],
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The command to execute",
                                }
                            },
                        },
                    },
                },
            ],
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        # You can use print statements as follows for debugging, they'll be visible when running tests.
        print("Logs from your program will appear here!", file=sys.stderr)

        chat_message = chat.choices[0].message

        tool_calls = chat_message.tool_calls

        messages.append(
            {
                "role": "assistant",
                "content": chat_message.content,
                "tool_calls": tool_calls,
            }
        )

        if tool_calls and len(tool_calls) > 0:
            for tool in tool_calls:
                if tool.function.name == "Read":
                    try:
                        args = json.loads(tool.function.arguments)
                        file_path = args["file_path"]
                        with open(file_path, "r") as f:
                            content = f.read()
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool.id,
                                    "content": content,
                                }
                            )
                    except Exception as e:
                        print(f"Error reading file: {e}", file=sys.stderr)
                elif tool.function.name == "Write":
                    try:
                        args = json.loads(tool.function.arguments)
                        file_path = args["file_path"]
                        content = args["content"]
                        with open(file_path, "w") as f:
                            f.write(content)
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool.id,
                                    "content": content,
                                }
                            )
                    except Exception as e:
                        print(f"Error writing file: {e}", file=sys.stderr)
                elif tool.function.name == "Bash":
                    try:
                        args = json.loads(tool.function.arguments)
                        command = args["command"]
                        completed = subprocess.run(
                            command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool.id,
                                "content": completed.stdout,
                            }
                        )
                    except Exception as e:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool.id,
                                "content": completed.stderr,
                            }
                        )
                        print(f"Error executing command: {e}", file=sys.stderr)
        else:
            # TODO: Uncomment the following line to pass the first stage
            print(chat_message.content)
            loop = False


if __name__ == "__main__":
    main()
