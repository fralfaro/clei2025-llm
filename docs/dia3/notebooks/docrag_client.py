import asyncio
from fastmcp import Client
import argparse

parser = argparse.ArgumentParser(description="...")
parser.add_argument('-q', '--question', type=str, required=True)
parser.add_argument("-r", "--remote", action="store_true", help="User HTTP protocol (default: False)")


async def call_tool(name: str):
    
    async with client:
        # Basic server interaction
        await client.ping()
        print("Server is reachable")

        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()

        print(tools)
        print(resources)
        print(prompts)

        result = await client.call_tool("search_docs", {"query": name})
        print(result.content[-1].text)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Running Python script ...") #, args)

    if args.question:
        if args.remote:
            print("Using remote client...")
            client = Client("http://localhost:8000/mcp")
        else:
            # Local client
            print("Using local client...")
            client = Client("docrag_server.py")

        asyncio.run(call_tool(args.question))
        
    print("--- Done ---")